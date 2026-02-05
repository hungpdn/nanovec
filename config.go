package nanovec

import (
	"github.com/hungpdn/nanovec/internal"
	"github.com/hungpdn/nanovec/internal/index"
	"github.com/hungpdn/nanovec/internal/index/flat"
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

	IndexType    IndexType // Index Selection
	Quantization bool      // Enable SQ8 compression (4x RAM savings)

	// HNSW Parameters
	M              int // Max connections per node (Default: 16)
	EfConstruction int // Search beam size during build (Default: 200)
}

var DefaultConfig = Config{
	Dimension:      1536,
	IndexType:      IndexTypeFlat, // Default to Flat for small datasets
	Quantization:   false,
	M:              16,
	EfConstruction: 200,
}

func (cfg *Config) GetVectorIndex() internal.VectorIndex {
	if cfg.IndexType == IndexTypeHNSW {
		// Future: Support HNSW + SQ8
		return index.NewHNSWIndex(cfg.Dimension, cfg.M, cfg.EfConstruction)
	}
	// Flat Index Selection
	if cfg.Quantization {
		return flat.NewFlatIndexSQ8(cfg.Dimension)
	}
	return flat.NewFlatIndex(cfg.Dimension)
}
