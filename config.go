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

// GetVectorIndex creates the appropriate index based on configuration
// It acts as a Factory for Flat/HNSW and Float32/SQ8 variants
func (cfg *Config) GetVectorIndex() internal.VectorIndex {
	switch cfg.IndexType {
	case IndexTypeHNSW:
		if cfg.Quantization {
			// HNSW + SQ8 (Graph + Compression)
			return index.NewHNSWIndexSQ8(cfg.Dimension, cfg.M, cfg.EfConstruction)
		}
		// HNSW + Float32 (Graph + Precision)
		return index.NewHNSWIndex(cfg.Dimension, cfg.M, cfg.EfConstruction)

	default: // Default to Flat
		if cfg.Quantization {
			// Flat + SQ8 (Scan + Compression)
			return flat.NewFlatIndexSQ8(cfg.Dimension)
		}
		// Flat + Float32 (Scan + Precision)
		return flat.NewFlatIndex(cfg.Dimension)
	}
}
