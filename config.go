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
	Dimension int // Number of vector dimensions

	IndexType    IndexType // Index Selection
	Quantization bool      // Enable SQ8 compression

	// HNSW Parameters
	M              int // Max connections per node (Default: 16)
	EfConstruction int // Search beam size during build (Default: 200)
}

var DefaultConfig = Config{
	Dimension:      1536,
	IndexType:      IndexTypeFlat,
	Quantization:   false,
	M:              16,
	EfConstruction: 200,
}

// GetVectorIndex creates the appropriate index based on configuration
func (cfg *Config) GetVectorIndex() internal.VectorIndex {
	switch cfg.IndexType {
	case IndexTypeHNSW:
		if cfg.Quantization {
			return index.NewHNSWIndexSQ8(cfg.Dimension, cfg.M, cfg.EfConstruction)
		}
		return index.NewHNSWIndexFloat(cfg.Dimension, cfg.M, cfg.EfConstruction)

	default: // Default to Flat
		if cfg.Quantization {
			return index.NewFlatIndexSQ8(cfg.Dimension)
		}
		return index.NewFlatIndexFloat(cfg.Dimension)
	}
}
