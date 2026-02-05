package internal

import (
	"github.com/hungpdn/nanovec/pkg/types"
)

// VectorIndex: Interface for search algorithms (HNSW, Flat...)
type VectorIndex interface {
	Add(id string, vec types.Vector, meta map[string]any) error
	AddBatch(ids []string, vecs []types.Vector, metas []map[string]any) error
	Delete(id string) error
	Search(vec types.Vector, k int, filter types.FilterFunc) ([]types.SearchResult, error)
	Save(path string) error
	Load(path string) error
	Dim() int
	Count() int
}

// Storage: Interface for persistent storage
type Storage interface {
	Put(doc *types.Document) error // Upsert
	Get(id string) (*types.Document, error)
	Delete(id string) error
	Scan(fn func(doc *types.Document) error) error
	Has(id string) bool
	Count() (int, error)
	Close() error
}
