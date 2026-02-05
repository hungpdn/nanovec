package internal

import (
	"github.com/hungpdn/nanovec/pkg/types"
)

// VectorIndex: Interface for search algorithms (HNSW, Flat...)
type VectorIndex interface {
	// Add a vector with associated metadata
	Add(id string, vec types.Vector, meta map[string]any) error
	// AddBatch adds multiple vectors with associated metadata
	AddBatch(ids []string, vecs []types.Vector, metas []map[string]any) error
	// UpdateMetadata updates metadata for a given vector ID
	UpdateMetadata(id string, meta map[string]any) error
	// Delete removes a vector by its ID
	Delete(id string) error
	// Search finds the k nearest neighbors to a vector, optionally applying a filter
	Search(vec types.Vector, k int, filter types.FilterFunc) ([]types.SearchResult, error)
	// Save persists the index to a file
	Save(path string) error
	// Load loads the index from a file
	Load(path string) error
	// Dim returns the dimensionality of the vectors in the index
	Dim() int
	//  Count returns the number of vectors in the index
	Count() int
	// SetVersion sets the version of the index
	SetVersion(v uint64)
	// GetVersion gets the version of the index
	GetVersion() uint64
}

// Storage: Interface for persistent storage
type Storage interface {
	// Put inserts a document and returns the new DB Version (Atomic)
	Put(doc *types.Document) (uint64, error)
	// PutBatch inserts multiple documents and returns the new DB Version
	PutBatch(docs []*types.Document) (uint64, error)
	// Get retrieves a document by its ID
	Get(id string) (*types.Document, error)
	// Delete removes a document by its ID and returns the new DB Version
	Delete(id string) (uint64, error)
	// Scan iterates over all documents, applying the provided function to each
	Scan(fn func(doc *types.Document) error) error
	// Has checks if a document exists by its ID
	Has(id string) bool
	// Count returns the total number of documents
	Count() (int, error)
	// Close closes the storage
	Close() error
	// GetVersion gets the current DB Version
	GetVersion() (uint64, error)
}
