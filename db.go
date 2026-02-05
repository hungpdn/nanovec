package nanovec

import (
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/hungpdn/nanovec/internal"
	"github.com/hungpdn/nanovec/internal/storage"
	"github.com/hungpdn/nanovec/pkg/errors"
	"github.com/hungpdn/nanovec/pkg/maths"
	"github.com/hungpdn/nanovec/pkg/types"
)

// DB represents the vector database
type DB struct {
	mu      sync.RWMutex // Ensure thread-safe
	path    string       // Path to save data
	config  Config
	index   internal.VectorIndex // Search engine (in RAM)
	storage internal.Storage     // Storage engine (on disk)

	// Reuse buffers for search queries to avoid allocation
	searchBufPool *sync.Pool
}

// Open initializes the database
func Open(path string, cfg *Config) (*DB, error) {
	if cfg == nil {
		cfg = &DefaultConfig
	}

	storePath := path + ".store"
	store, err := storage.NewBoltStorage(storePath)
	if err != nil {
		return nil, err
	}

	idx := cfg.GetVectorIndex()
	indexPath := path + ".idx"
	err = idx.Load(indexPath)
	if err != nil {
		// If load fails, we will rebuild
		idx = cfg.GetVectorIndex()
	}
	indexLoaded := err == nil

	if indexLoaded && idx.Dim() != cfg.Dimension {
		cfg.Dimension = idx.Dim()
	}

	storeVer, err := store.GetVersion()
	if err != nil {
		return nil, fmt.Errorf("failed to get storage version: %v", err)
	}

	idxVer := idx.GetVersion()

	// Logic: If Index is missing, corrupt, or OUTDATED -> Rebuild
	needRebuild := !indexLoaded || (idxVer != storeVer)

	// Self-Healing
	if needRebuild {
		reason := "Index missing"
		if indexLoaded && idxVer != storeVer {
			reason = fmt.Sprintf("Version mismatch (Index: %d, Store: %d)", idxVer, storeVer)
		}

		log.Printf("⚠️ %s. Rebuilding index from storage (SQLite-style recovery)...", reason)

		idx = cfg.GetVectorIndex()

		const batchSize = 1000
		var batchIDs []string
		var batchVecs []types.Vector
		var batchMetas []map[string]any
		count := 0
		start := time.Now()
		err := store.Scan(func(doc *types.Document) error {
			batchIDs = append(batchIDs, doc.ID)
			batchVecs = append(batchVecs, doc.Vector)
			batchMetas = append(batchMetas, doc.Metadata)

			if len(batchIDs) >= batchSize {
				if err := idx.AddBatch(batchIDs, batchVecs, batchMetas); err != nil {
					return err
				}
				batchIDs = batchIDs[:0]
				batchVecs = batchVecs[:0]
				batchMetas = batchMetas[:0]
			}
			count++
			return nil
		})

		if len(batchIDs) > 0 {
			if err := idx.AddBatch(batchIDs, batchVecs, batchMetas); err != nil {
				return nil, fmt.Errorf("failed to flush remaining batch: %v", err)
			}
		}
		if err != nil {
			return nil, fmt.Errorf("failed to rebuild index: %v", err)
		}

		idx.SetVersion(storeVer)

		if err := idx.Save(indexPath); err != nil {
			log.Printf("Warning: failed to save rebuilt index: %v", err)
		}

		log.Printf("✅ Restored %d vectors in %v. System is consistent.", count, time.Since(start))
	}

	return &DB{
		path:    path,
		config:  *cfg,
		index:   idx,
		storage: store,
		searchBufPool: &sync.Pool{
			New: func() any {
				return make([]float32, cfg.Dimension)
			},
		},
	}, nil
}

// Insert adds or updates a vector, ensures Atomicity by treating Storage as WAL/Master
func (db *DB) Insert(id string, vec []float32, meta map[string]any) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if len(vec) != db.config.Dimension {
		return errors.ErrDimMismatch
	}

	doc := &types.Document{ID: id, Vector: types.Vector(vec), Metadata: meta}

	newVer, err := db.storage.Put(doc)
	if err != nil {
		return fmt.Errorf("storage write failed: %v", err)
	}

	_ = db.index.Delete(id)
	if err := db.index.Add(id, types.Vector(vec), meta); err != nil {
		return fmt.Errorf("critical: memory index update failed: %v", err)
	}

	db.index.SetVersion(newVer)
	return nil
}

// InsertBatch adds multiple vectors
func (db *DB) InsertBatch(ids []string, vecs [][]float32, metas []map[string]any) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if len(ids) != len(vecs) || len(ids) != len(metas) {
		return errors.ErrBatchSizeMismatch
	}

	docs := make([]*types.Document, len(ids))
	typeVectors := make([]types.Vector, len(vecs))

	for i, id := range ids {
		if len(vecs[i]) != db.config.Dimension {
			return errors.ErrDimMismatch
		}
		typeVectors[i] = types.Vector(vecs[i])
		docs[i] = &types.Document{ID: id, Vector: typeVectors[i], Metadata: metas[i]}
	}

	newVer, err := db.storage.PutBatch(docs)
	if err != nil {
		return fmt.Errorf("storage batch write failed: %v", err)
	}

	for _, id := range ids {
		_ = db.index.Delete(id)
	}

	if err := db.index.AddBatch(ids, typeVectors, metas); err != nil {
		// Note: We don't rollback storage here as it's complex for batches.
		// The Open() self-healing will fix this sync drift on restart.
		return fmt.Errorf("index batch update failed: %v", err)
	}

	db.index.SetVersion(newVer)
	return nil
}

// Search finds similar vectors
func (db *DB) Search(query []float32, k int, filter types.FilterFunc) ([]types.SearchResult, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if len(query) != db.config.Dimension {
		return nil, errors.ErrQueryDimMismatch
	}

	buf := db.searchBufPool.Get().([]float32)
	defer db.searchBufPool.Put(buf)
	copy(buf, query)
	maths.NormalizeInPlace(buf)

	return db.index.Search(types.Vector(buf), k, filter)
}

// Update updates document
func (db *DB) Update(id string, newVec []float32, newMeta map[string]any) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if len(newVec) > 0 && len(newVec) != db.config.Dimension {
		return fmt.Errorf("dimension mismatch: expected %d, got %d", db.config.Dimension, len(newVec))
	}

	oldDoc, err := db.storage.Get(id)
	if err != nil {
		return err
	}

	finalVec := oldDoc.Vector
	if len(newVec) > 0 {
		finalVec = types.Vector(newVec)
	}
	finalMeta := oldDoc.Metadata
	if newMeta != nil {
		finalMeta = newMeta
	}

	newDoc := &types.Document{ID: id, Vector: finalVec, Metadata: finalMeta}

	newVer, err := db.storage.Put(newDoc)
	if err != nil {
		return err
	}

	_ = db.index.Delete(id)
	if err := db.index.Add(id, finalVec, finalMeta); err != nil {
		return fmt.Errorf("index update failed: %v", err)
	}

	db.index.SetVersion(newVer)
	return nil
}

// Delete removes a document
func (db *DB) Delete(id string) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	newVer, err := db.storage.Delete(id)
	if err != nil {
		return err
	}
	if err := db.index.Delete(id); err != nil {
		return err
	}

	db.index.SetVersion(newVer)
	return nil
}

// Exists checks if a document ID exists
func (db *DB) Exists(id string) bool {
	db.mu.RLock()
	defer db.mu.RUnlock()
	return db.storage.Has(id)
}

// Close closes the connection and persists the index
func (db *DB) Close() error {
	db.mu.Lock()
	defer db.mu.Unlock()

	indexPath := db.path + ".idx"
	if err := db.index.Save(indexPath); err != nil {
		return fmt.Errorf("failed to save index: %v", err)
	}
	return db.storage.Close()
}
