package types

// Vector is a float32 array (most commonly used for embedding).
type Vector []float32

// Vector8 is a quantized vector using 1 byte per dimension (SQ8).
// It reduces RAM usage by 4x.
type Vector8 []uint8

// Document represents a record
type Document struct {
	ID       string                 `json:"id"`
	Vector   Vector                 `json:"vector"`
	Metadata map[string]interface{} `json:"metadata"` // Additional data (payload)
}

// SearchResult returned results from a search
type SearchResult struct {
	ID       string  `json:"id"`
	Score    float32 `json:"score"` // Similarity score
	Metadata map[string]interface{}
}

// FilterFunc defines a filter for metadata
type FilterFunc func(metadata map[string]interface{}) bool
