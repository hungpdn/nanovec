package internal

import "github.com/hungpdn/nanovec/pkg/types"

// VectorIndex: Interface for search algorithms (HNSW, Flat...)
type VectorIndex interface {
	Add(id string, vec types.Vector) error
	Search(vec types.Vector, k int) ([]string, []float32, error) // Returns List ID and Score
	Save(path string) error
	Load(path string) error
}

// Storage: Interface for persistent storage
type Storage interface {
	Put(doc *types.Document) error
	Get(id string) (*types.Document, error)
	Close() error
}
