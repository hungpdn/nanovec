package nanovec_test

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/hungpdn/nanovec"
)

func randomVector(dim int) []float32 {
	vec := make([]float32, dim)
	for i := 0; i < dim; i++ {
		vec[i] = rand.Float32()
	}
	return vec
}

// Measure the insertion speed of each item (worst case)
func BenchmarkDB_Insert_Sequential(b *testing.B) {
	dbPath := b.TempDir() + "/bench_seq.db"
	cfg := &nanovec.Config{
		Dimension: 128,
		IndexType: nanovec.IndexTypeHNSW,
	}
	db, _ := nanovec.Open(dbPath, cfg)
	defer db.Close()

	vec := randomVector(128)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		id := fmt.Sprintf("id_%d", i)
		db.Insert(id, vec, nil)
	}
}

// Measure the speed of the Insert Batch (Best case - Important for tuning Arena Buffer)
func BenchmarkDB_Insert_Batch_1000(b *testing.B) {
	dbPath := b.TempDir() + "/bench_batch.db"
	cfg := &nanovec.Config{
		Dimension: 128,
		IndexType: nanovec.IndexTypeHNSW,
	}
	db, _ := nanovec.Open(dbPath, cfg)
	defer db.Close()

	batchSize := 1000
	ids := make([]string, batchSize)
	vecs := make([][]float32, batchSize)
	metas := make([]map[string]any, batchSize)

	for i := 0; i < batchSize; i++ {
		ids[i] = fmt.Sprintf("id_%d", i)
		vecs[i] = randomVector(128)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for k := 0; k < batchSize; k++ {
			ids[k] = fmt.Sprintf("batch_%d_%d", i, k)
		}
		db.InsertBatch(ids, vecs, metas)
	}
}

// Measure End-to-End Search Speed
func BenchmarkDB_Search_HNSW(b *testing.B) {
	// 1. Setup Data
	dbPath := b.TempDir() + "/bench_search.db"
	cfg := &nanovec.Config{
		Dimension:      128,
		IndexType:      nanovec.IndexTypeHNSW,
		M:              16,
		EfConstruction: 200,
	}
	db, _ := nanovec.Open(dbPath, cfg)
	defer db.Close()

	// Pre-load 10,000 vectors
	count := 10000
	ids := make([]string, count)
	vecs := make([][]float32, count)
	metas := make([]map[string]any, count)
	for i := 0; i < count; i++ {
		ids[i] = fmt.Sprintf("id_%d", i)
		vecs[i] = randomVector(128)
	}
	db.InsertBatch(ids, vecs, metas)

	query := randomVector(128)
	b.ResetTimer()

	// 2. Run Benchmark
	for i := 0; i < b.N; i++ {
		db.Search(query, 10, nil)
	}
}
