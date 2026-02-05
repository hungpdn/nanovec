package types

// Vector is a float32 array (most commonly used for embedding).
type Vector []float32

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
