package internal

import "github.com/hungpdn/nanovec/pkg/types"

// VectorIndex: Interface for search algorithms (HNSW, Flat...)
type VectorIndex interface {
	Add(id string, vec types.Vector) error
	Delete(id string) error
	Search(vec types.Vector, k int) ([]string, []float32, error) // Returns List ID and Score
	Save(path string) error
	Load(path string) error
}

// Storage: Interface for persistent storage
type Storage interface {
	Put(doc *types.Document) error // Upsert (Thêm hoặc Ghi đè)
	Get(id string) (*types.Document, error)
	Delete(id string) error // <--- Cần thêm cái này
	Close() error
}
