package nanovec

import (
	"fmt"
	"sync"

	"github.com/hungpdn/nanovec/internal"
	"github.com/hungpdn/nanovec/internal/index"
	"github.com/hungpdn/nanovec/internal/storage"
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

	idx := index.NewFlatIndex(cfg.Dimension)
	indexPath := path + ".idx"
	err = idx.Load(indexPath)
	indexLoaded := err == nil

	if indexLoaded {
		if idx.Dim != cfg.Dimension {
			fmt.Printf("⚠️ Config dimension (%d) matches disk index (%d). Updating config.\n", cfg.Dimension, idx.Dim)
			cfg.Dimension = idx.Dim
		}
	}

	db := &DB{
		path:    path,
		config:  *cfg,
		index:   idx,
		storage: store,
	}

	if !indexLoaded || idx.Count() == 0 {
		fmt.Println("⚠️ Index missing or empty. Rebuilding from Storage...")

		count := 0
		err := db.storage.Scan(func(doc *types.Document) error {
			if err := db.index.Add(doc.ID, doc.Vector); err != nil {
				return err
			}
			count++
			return nil
		})

		if err != nil {
			return nil, fmt.Errorf("failed to rebuild index: %v", err)
		}
		fmt.Printf("✅ Restored %d vectors from storage.\n", count)
	}

	return db, nil
}

// Insert adds or updates a vector (Upsert behavior)
func (db *DB) Insert(id string, vec []float32, meta map[string]interface{}) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if len(vec) != db.config.Dimension {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d", db.config.Dimension, len(vec))
	}

	doc := &types.Document{
		ID:       id,
		Vector:   types.Vector(vec),
		Metadata: meta,
	}

	// Write to Persistent Storage (Durability)
	if err := db.storage.Put(doc); err != nil {
		return fmt.Errorf("storage write failed: %v", err)
	}

	// Update Memory Index (Availability)
	// Note: If this fails (e.g. OOM), the DB is in a "Storage-Index mismatch" state.
	// A WAL replay on restart would fix this.
	_ = db.index.Delete(id)
	if err := db.index.Add(id, types.Vector(vec)); err != nil {
		return fmt.Errorf("index update failed: %v", err)
	}

	return nil
}

// Search finds similar vectors
func (db *DB) Search(query []float32, k int) ([]types.SearchResult, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if len(query) != db.config.Dimension {
		return nil, fmt.Errorf("query dimension mismatch")
	}

	ids, scores, err := db.index.Search(types.Vector(query), k)
	if err != nil {
		return nil, err
	}

	results := make([]types.SearchResult, 0, len(ids))
	for i, id := range ids {
		doc, err := db.storage.Get(id)
		if err != nil {
			// Determine policy: Skip or return partial?
			// Here we skip, assuming index might be slightly ahead/behind or consistency issue
			continue
		}
		results = append(results, types.SearchResult{
			ID:       doc.ID,
			Score:    scores[i],
			Metadata: doc.Metadata,
		})
	}

	return results, nil
}

// Update updates document. Delete-then-Insert in Index.
func (db *DB) Update(id string, newVec []float32, newMeta map[string]interface{}) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if len(newVec) > 0 && len(newVec) != db.config.Dimension {
		return fmt.Errorf("dimension mismatch: expected %d, got %d", db.config.Dimension, len(newVec))
	}

	oldDoc, err := db.storage.Get(id)
	if err != nil {
		return fmt.Errorf("document not found: %s", id)
	}

	finalVec := oldDoc.Vector
	vectorChanged := false

	if len(newVec) > 0 {
		finalVec = types.Vector(newVec)
		vectorChanged = true
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

	// Persist to Disk First
	if err := db.storage.Put(newDoc); err != nil {
		return err
	}

	// Update Index if Vector Changed
	if vectorChanged {
		// Attempt to remove old vector
		if err := db.index.Delete(id); err != nil {
			// Log this error in production.
			// The storage is updated, but index might still have old vector pointing to new data.
			return fmt.Errorf("index data inconsistency: failed to delete old vector: %v", err)
		}

		// Add new vector
		if err := db.index.Add(id, finalVec); err != nil {
			return fmt.Errorf("index data inconsistency: failed to add new vector: %v", err)
		}
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

// Exists checks if a document ID exists in the database.
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
