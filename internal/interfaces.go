package internal

import (
	"github.com/hungpdn/nanovec/pkg/types"
)

// VectorIndex: Interface for search algorithms (HNSW, Flat...)
type VectorIndex interface {
	Add(id string, vec types.Vector, meta map[string]any) error
	AddBatch(ids []string, vecs []types.Vector, metas []map[string]any) error
	UpdateMetadata(id string, meta map[string]any) error
	Delete(id string) error
	Search(vec types.Vector, k int, filter types.FilterFunc) ([]types.SearchResult, error)
	Save(path string) error
	Load(path string) error
	Dim() int
	Count() int
	SetVersion(v uint64)
	GetVersion() uint64
}

// Storage: Interface for persistent storage
type Storage interface {
	// Put inserts a document and returns the new DB Version (Atomic)
	Put(doc *types.Document) (uint64, error)
	// PutBatch inserts multiple documents and returns the new DB Version
	PutBatch(docs []*types.Document) (uint64, error)
	Get(id string) (*types.Document, error)
	Delete(id string) (uint64, error)
	Scan(fn func(doc *types.Document) error) error
	Has(id string) bool
	Count() (int, error)
	Close() error
	GetVersion() (uint64, error)
}
