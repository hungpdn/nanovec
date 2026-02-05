package nanovec

import (
	"github.com/hungpdn/nanovec/internal"
	"github.com/hungpdn/nanovec/internal/index"
)

// IndexType defines the type of vector index
type IndexType string

const (
	IndexTypeFlat IndexType = "FLAT"
	IndexTypeHNSW IndexType = "HNSW"
)

// Config configuration for DB
type Config struct {
	Dimension int // Number of vector dimensions (e.g., 1536 for OpenAI)

	IndexType IndexType // Index Selection

	// HNSW Parameters
	M              int // Max connections per node (Default: 16)
	EfConstruction int // Search beam size during build (Default: 200)
}

var DefaultConfig = Config{
	Dimension:      1536,
	IndexType:      IndexTypeFlat, // Default to Flat for small datasets
	M:              16,
	EfConstruction: 200,
}

func (cfg *Config) GetVectorIndex() internal.VectorIndex {
	if cfg.IndexType == IndexTypeHNSW {
		return index.NewHNSWIndex(cfg.Dimension, cfg.M, cfg.EfConstruction)
	} else {
		return index.NewFlatIndex(cfg.Dimension)
	}
}
