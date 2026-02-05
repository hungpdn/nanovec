package nanovec

import (
	"fmt"
	"sync"

	"github.com/hungpdn/nanovec/internal"
	"github.com/hungpdn/nanovec/internal/storage"
	"github.com/hungpdn/nanovec/pkg/errors"
	"github.com/hungpdn/nanovec/pkg/types"
)

// DB represents the vector database
type DB struct {
	mu      sync.RWMutex // Ensure thread-safe
	path    string       // Path to save data
	config  Config
	index   internal.VectorIndex // Search engine (in RAM)
	storage internal.Storage     // Storage engine (on disk)
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
		// If load fails, we will rebuild, so just init a fresh index
		idx = cfg.GetVectorIndex()
	}
	indexLoaded := err == nil

	if indexLoaded && idx.Dim() != cfg.Dimension {
		cfg.Dimension = idx.Dim()
	}

	db := &DB{
		path:    path,
		config:  *cfg,
		index:   idx,
		storage: store,
	}

	storeCount, err := db.storage.Count()
	if err != nil {
		return nil, fmt.Errorf("failed to check storage integrity: %v", err)
	}

	// Self-Healing: rebuild if
	// - Index failed to load
	// - Counts mismatch (Sync Drift / Corruption)
	if !indexLoaded || idx.Count() != storeCount {
		reason := "Index missing or corrupt"
		if indexLoaded && idx.Count() != storeCount {
			reason = fmt.Sprintf("Sync drift detected (Index: %d, Store: %d)", idx.Count(), storeCount)
		}

		fmt.Printf("⚠️ %s. Rebuilding from Storage...\n", reason)

		// Reset Index to clean state before rebuilding
		if idx.Count() > 0 {
			db.index = cfg.GetVectorIndex()
		}

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
				if err := db.index.AddBatch(batchIDs, batchVecs, batchMetas); err != nil {
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
			if err := db.index.AddBatch(batchIDs, batchVecs, batchMetas); err != nil {
				return nil, fmt.Errorf("failed to flush remaining batch: %v", err)
			}
		}
		if err != nil {
			return nil, fmt.Errorf("failed to rebuild index: %v", err)
		}
		fmt.Printf("✅ Restored %d vectors from storage (Batched).\n", count)
	}

	return db, nil
}

// Insert adds or updates a vector (Upsert behavior)
func (db *DB) Insert(id string, vec []float32, meta map[string]any) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if len(vec) != db.config.Dimension {
		return errors.ErrDimMismatch
	}

	doc := &types.Document{
		ID:       id,
		Vector:   types.Vector(vec),
		Metadata: meta,
	}

	if err := db.storage.Put(doc); err != nil {
		return fmt.Errorf("storage write failed: %v", err)
	}

	_ = db.index.Delete(id)
	if err := db.index.Add(id, types.Vector(vec), meta); err != nil {
		_ = db.storage.Delete(id)
		return fmt.Errorf("index add failed (rolled back storage): %v", err)
	}

	return nil
}

// InsertBatch adds multiple vectors efficiently
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

		docs[i] = &types.Document{
			ID:       id,
			Vector:   typeVectors[i],
			Metadata: metas[i],
		}
	}

	if err := db.storage.PutBatch(docs); err != nil {
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

	return nil
}

// Search finds similar vectors
func (db *DB) Search(query []float32, k int, filter types.FilterFunc) ([]types.SearchResult, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if len(query) != db.config.Dimension {
		return nil, errors.ErrQueryDimMismatch
	}

	return db.index.Search(types.Vector(query), k, filter)
}

// Update updates document. Delete-then-Insert in Index.
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

	newDoc := &types.Document{
		ID:       id,
		Vector:   finalVec,
		Metadata: finalMeta,
	}

	if err := db.storage.Put(newDoc); err != nil {
		return err
	}

	_ = db.index.Delete(id)
	if err := db.index.Add(id, finalVec, finalMeta); err != nil {
		// Attempt to restore old doc to storage
		_ = db.storage.Put(oldDoc)
		return fmt.Errorf("index update failed (rolled back storage): %v", err)
	}

	return nil
}

// Delete removes a document
func (db *DB) Delete(id string) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if err := db.storage.Delete(id); err != nil {
		return err
	}

	if err := db.index.Delete(id); err != nil {
		return err
	}

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
