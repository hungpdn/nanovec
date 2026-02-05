package nanovec

import (
	"fmt"
	"sync"

	"github.com/hungpdn/nanovec/internal"
	"github.com/hungpdn/nanovec/pkg/types"
)

// Config configuration for DB
type Config struct {
	Path      string // Path to save data
	Dimension int    // Number of vector dimensions (e.g., 1536 for OpenAI)
}

// DB represents the vector database
type DB struct {
	mu      sync.RWMutex // Ensure thread-safe (Concurrency)
	config  Config
	index   internal.VectorIndex // Search engine (in RAM)
	storage internal.Storage     // Storage engine (on disk)
}

// Open initializes the database
func Open(cfg Config) (*DB, error) {
	// 1. Initialize Storage (e.g., mock or BoltDB)
	// store, err := storage.NewBoltStorage(cfg.Path)
	// Temporarily mock to make code run
	store := &mockStorage{data: make(map[string]*types.Document)}

	// 2. Initialize Index (e.g., HNSW)
	// idx := index.NewHNSW(cfg.Dimension)
	// Temporarily mock
	idx := &mockIndex{
		vectors: make(map[string]types.Vector),
		dim:     cfg.Dimension,
	}

	return &DB{
		config:  cfg,
		index:   idx,
		storage: store,
	}, nil
}

// Insert adds a new vector
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

	// 1. Save to disk first (WAL principle)
	if err := db.storage.Put(doc); err != nil {
		return err
	}

	// 2. Update the index in RAM
	if err := db.index.Add(id, types.Vector(vec)); err != nil {
		return err
	}

	return nil
}

// Search finds similar vectors
func (db *DB) Search(query []float32, k int) ([]types.SearchResult, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	// 1. Find ID and Score from Index (RAM) - Fast
	ids, scores, err := db.index.Search(types.Vector(query), k)
	if err != nil {
		return nil, err
	}

	// 2. Get Metadata from Storage (Disk) - Slower but doesn't consume RAM
	results := make([]types.SearchResult, 0, len(ids))
	for i, id := range ids {
		doc, err := db.storage.Get(id)
		if err != nil {
			continue // Skip if error
		}
		results = append(results, types.SearchResult{
			ID:       doc.ID,
			Score:    scores[i],
			Metadata: doc.Metadata,
		})
	}

	return results, nil
}

// Close closes the connection
func (db *DB) Close() error {
	return db.storage.Close()
}

// ---------------------------------------------------------
// Mock Implementation (This is just a demo code to get it working right away)
// ---------------------------------------------------------

type mockStorage struct {
	data map[string]*types.Document
}

func (m *mockStorage) Put(doc *types.Document) error          { m.data[doc.ID] = doc; return nil }
func (m *mockStorage) Get(id string) (*types.Document, error) { return m.data[id], nil }
func (m *mockStorage) Close() error                           { return nil }

type mockIndex struct {
	vectors map[string]types.Vector
	dim     int
}

func (m *mockIndex) Add(id string, vec types.Vector) error { m.vectors[id] = vec; return nil }
func (m *mockIndex) Search(vec types.Vector, k int) ([]string, []float32, error) {
	// Dummy implementation: Returns the first vector found.
	// In reality, you would compute Cosine Similarity here
	keys := make([]string, 0, k)
	scores := make([]float32, 0, k)
	for id := range m.vectors {
		keys = append(keys, id)
		scores = append(scores, 0.99) // Fake score
		if len(keys) >= k {
			break
		}
	}
	return keys, scores, nil
}
func (m *mockIndex) Save(path string) error { return nil }
func (m *mockIndex) Load(path string) error { return nil }
