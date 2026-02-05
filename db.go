package nanovec

import (
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/hungpdn/nanovec/internal"
	"github.com/hungpdn/nanovec/internal/index"
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
	var finalConfig Config
	if cfg == nil {
		finalConfig = DefaultConfig
	} else {
		finalConfig = *cfg
	}

	if finalConfig.VacuumThreshold <= 0 || finalConfig.VacuumThreshold > 1.0 {
		finalConfig.VacuumThreshold = DefaultConfig.VacuumThreshold
	}

	cfg = &finalConfig

	storePath := path + ".store"
	store, err := storage.NewBoltStorage(storePath, cfg.ReadOnly)
	if err != nil {
		return nil, err
	}

	idx := cfg.GetVectorIndex()
	indexPath := path + ".idx"

	err = idx.Load(indexPath)
	if err != nil {
		if cfg.ReadOnly {
			_ = store.Close()
			return nil, fmt.Errorf("failed to load index in read-only mode (file missing?): %v", err)
		}
		idx = cfg.GetVectorIndex()
	}
	indexLoaded := err == nil

	if indexLoaded {

		if idx.Dim() != cfg.Dimension {
			cfg.Dimension = idx.Dim()
		}

		if hnsw, ok := idx.(*index.HNSWIndex[float32]); ok {
			if cfg.M != hnsw.M {
				log.Printf("ℹ️ Syncing Config M: %d -> %d (from disk)", cfg.M, hnsw.M)
				cfg.M = hnsw.M
			}
			if cfg.EfConstruction != hnsw.EfConstruction {
				cfg.EfConstruction = hnsw.EfConstruction
			}
		}

		if hnsw, ok := idx.(*index.HNSWIndex[uint8]); ok {
			if cfg.M != hnsw.M {
				log.Printf("ℹ️ Syncing Config M: %d -> %d (from disk)", cfg.M, hnsw.M)
				cfg.M = hnsw.M
			}
			if cfg.EfConstruction != hnsw.EfConstruction {
				cfg.EfConstruction = hnsw.EfConstruction
			}
		}
	}

	storeVer, err := store.GetVersion()
	if err != nil {
		return nil, fmt.Errorf("failed to get storage version: %v", err)
	}

	idxVer := idx.GetVersion()
	needRebuild := false

	docCount, err := store.Count()
	if err != nil {
		return nil, fmt.Errorf("failed to count storage docs: %v", err)
	}
	if docCount == 0 && !indexLoaded {
		idx.SetVersion(storeVer)
		if err := idx.Save(indexPath); err != nil {
			log.Printf("⚠️ Warning: Failed to init empty index file: %v", err)
		}
		goto final
	}

	if cfg.ReadOnly {
		goto final
	}

	// Self-Healing
	needRebuild = !indexLoaded || (idxVer != storeVer)
	if needRebuild {
		if count, _ := store.Count(); count > 0 {
			// Scan 1 document to peek dimension
			var detectedDim int
			_ = store.Scan(func(doc *types.Document) error {
				detectedDim = len(doc.Vector)
				return fmt.Errorf("stop") // Hack to stop scanning after 1 item
			})

			if detectedDim > 0 && detectedDim != cfg.Dimension {
				log.Printf("⚠️ Config dimension (%d) mismatch with Storage (%d). Auto-adjusting...", cfg.Dimension, detectedDim)
				cfg.Dimension = detectedDim
			}
		}

		reason := "Index missing"
		if indexLoaded && idxVer != storeVer {
			reason = fmt.Sprintf("Version mismatch (Index: %d, Store: %d)", idxVer, storeVer)
		}

		log.Printf("⚠️ %s. Rebuilding index from storage...", reason)

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
			if err = idx.AddBatch(batchIDs, batchVecs, batchMetas); err != nil {
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

final:
	return &DB{
		path:    path,
		config:  finalConfig,
		index:   idx,
		storage: store,
		searchBufPool: &sync.Pool{
			New: func() any {
				s := make([]float32, cfg.Dimension)
				return &s
			},
		},
	}, nil
}

// Insert adds or updates a vector, ensures Atomicity by treating Storage as WAL/Master
func (db *DB) Insert(id string, vec []float32, meta map[string]any) error {

	if db.config.ReadOnly {
		return errors.ErrReadOnly
	}

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
	if db.config.ReadOnly {
		return errors.ErrReadOnly
	}

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
		// The Open() self-healing will fix this sync drift on restart.
		log.Printf("CRITICAL: Index corrupted during batch insert. Persistence requires restart. Error: %v", err)
		panic("nanovec: memory index corrupted, restarting required to maintain consistency")
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

	bufPtr := db.searchBufPool.Get().(*[]float32)
	defer db.searchBufPool.Put(bufPtr)
	buf := *bufPtr
	copy(buf, query)
	maths.NormalizeInPlace(buf)

	return db.index.Search(types.Vector(buf), k, filter)
}

// Update updates document
func (db *DB) Update(id string, newVec []float32, newMeta map[string]any) error {
	if db.config.ReadOnly {
		return errors.ErrReadOnly
	}

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

	if len(newVec) == 0 {
		if err := db.index.UpdateMetadata(id, finalMeta); err != nil {
			return fmt.Errorf("index metadata update failed: %v", err)
		}
	} else {
		_ = db.index.Delete(id)
		if err := db.index.Add(id, finalVec, finalMeta); err != nil {
			return fmt.Errorf("index update failed: %v", err)
		}
	}

	db.index.SetVersion(newVer)
	return nil
}

// Delete removes a document
func (db *DB) Delete(id string) error {
	if db.config.ReadOnly {
		return errors.ErrReadOnly
	}

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

	if !db.config.ReadOnly {
		indexPath := db.path + ".idx"
		if err := db.index.Save(indexPath); err != nil {
			return fmt.Errorf("failed to save index: %v", err)
		}
	} else {
		_ = db.index.Close()
	}
	return db.storage.Close()
}

// Vacuum rebuilds the vector index from scratch to remove "ghost nodes" (deleted items)
// and optimize memory usage. It mimics SQLite's VACUUM command.
// It now includes a heuristic to SKIP if the garbage ratio is low (< 20%).
//
// ⚠️ BLOCKING OPERATION: This function holds a global lock (Stop-the-World) during the
// entire rebuild process. For a 1GB dataset, this might take 10-20 seconds.
//
// Recommendation: Run this function asynchronously or during maintenance windows.
func (db *DB) Vacuum() error {
	if db.config.ReadOnly {
		return errors.ErrReadOnly
	}

	db.mu.Lock()
	defer db.mu.Unlock()

	// Smart Vacuum Logic
	activeCount := db.index.Count()
	deletedCount := db.index.DeletedCount()

	if deletedCount == 0 {
		return nil
	}

	totalNodes := activeCount + deletedCount
	threshold := db.config.VacuumThreshold
	garbageRatio := float64(deletedCount) / float64(totalNodes)

	if garbageRatio < threshold {
		log.Printf("🧹 Vacuum skipped: Garbage ratio %.2f%% is below threshold %.0f%% (Deleted: %d, Active: %d)",
			garbageRatio*100, threshold*100, deletedCount, activeCount)
		return nil
	}

	log.Printf("🧹 Starting VACUUM... Garbage Ratio: %.2f%% (Deleted: %d, Total: %d)",
		garbageRatio*100, deletedCount, totalNodes)

	start := time.Now()

	docCount, _ := db.storage.Count()
	if docCount > 10000 {
		log.Printf("🧹 VACUUM started on %d items. System will be BLOCKED until completion...", docCount)
	} else {
		log.Println("🧹 VACUUM started...")
	}

	newIdx := db.config.GetVectorIndex()

	const batchSize = 1000
	var batchIDs []string
	var batchVecs []types.Vector
	var batchMetas []map[string]any

	count := 0

	err := db.storage.Scan(func(doc *types.Document) error {
		batchIDs = append(batchIDs, doc.ID)
		batchVecs = append(batchVecs, doc.Vector)
		batchMetas = append(batchMetas, doc.Metadata)

		if len(batchIDs) >= batchSize {
			if err := newIdx.AddBatch(batchIDs, batchVecs, batchMetas); err != nil {
				return err
			}
			batchIDs = batchIDs[:0]
			batchVecs = batchVecs[:0]
			batchMetas = batchMetas[:0]
		}
		count++
		return nil
	})
	if err != nil {
		return fmt.Errorf("vacuum scan failed: %v", err)
	}

	if len(batchIDs) > 0 {
		if err = newIdx.AddBatch(batchIDs, batchVecs, batchMetas); err != nil {
			return fmt.Errorf("vacuum flush failed: %v", err)
		}
	}

	storeVer, err := db.storage.GetVersion()
	if err != nil {
		return fmt.Errorf("failed to get storage version: %v", err)
	}
	newIdx.SetVersion(storeVer)

	db.index = newIdx

	indexPath := db.path + ".idx"
	if err := db.index.Save(indexPath); err != nil {
		log.Printf("⚠️ Warning: Vacuum save failed (RAM is OK): %v", err)
	}

	log.Printf("✅ VACUUM completed in %v. Index optimized (Items: %d).", time.Since(start), count)
	return nil
}

// Count returns the total number of documents in the database
func (db *DB) Count() (int, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()
	return db.storage.Count()
}

// ReadOnly returns true if the database is in read-only mode
func (db *DB) ReadOnly() bool {
	db.mu.RLock()
	defer db.mu.RUnlock()
	return db.config.ReadOnly
}
