package types

// Number constraint for vector elements (Float32 or Uint8)
type Number interface {
	float32 | uint8
}

// Vector is a float32 array (most commonly used for embedding).
type Vector []float32

// Vector8 is a quantized vector using 1 byte per dimension (SQ8).
type Vector8 []uint8

// Document represents a record
type Document struct {
	ID       string         `json:"id"`
	Vector   Vector         `json:"vector"`
	Metadata map[string]any `json:"metadata"`
}

// SearchResult returned results from a search
type SearchResult struct {
	ID       string  `json:"id"`
	Score    float32 `json:"score"`
	Metadata map[string]any
}

// FilterFunc defines a filter for metadata
type FilterFunc func(metadata map[string]any) bool
