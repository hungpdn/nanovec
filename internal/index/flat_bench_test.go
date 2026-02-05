package index

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/hungpdn/nanovec/pkg/types"
)

/*
➜  nanovec1 git:(main) ✗ go test -bench=. -benchmem ./internal/index/
goos: darwin
goarch: amd64
pkg: github.com/hungpdn/nanovec/internal/index
cpu: Intel(R) Core(TM) i7-8850H CPU @ 2.60GHz
BenchmarkFlat_Search_Float32-12       	     576	   2004854 ns/op	    2278 B/op	      18 allocs/op
BenchmarkFlat_Search_SQ8-12           	     835	   1758990 ns/op	    2150 B/op	      19 allocs/op
BenchmarkFlat_InsertBatch-12          	    1640	    630714 ns/op	 2747335 B/op	      27 allocs/op
BenchmarkFlat_Search_Concurrent-12    	    2073	   1045822 ns/op	    2602 B/op	      16 allocs/op
BenchmarkFlat_UpdateVector-12         	 2599023	       469.7 ns/op	       0 B/op	       0 allocs/op
BenchmarkFlat_UpdateMetadata-12       	23465242	        52.64 ns/op	       0 B/op	       0 allocs/op
PASS
ok  	github.com/hungpdn/nanovec/internal/index	22.994s
*/

// ensureRandomVector helper is available if running standalone
// (If you already have this in hnsw_test.go within the same package, you can remove this function)
func ensureRandomVector(dim int) types.Vector {
	vec := make(types.Vector, dim)
	for i := 0; i < dim; i++ {
		vec[i] = rand.Float32()
	}
	return vec
}

const (
	BenchDim   = 128
	BenchCount = 100000 // 100k vectors for search benchmark
)

// 1. Benchmark Search Performance (Float32)
// This measures the efficiency of the "Heap Optimization" and "SIMD DotProduct".
func BenchmarkFlat_Search_Float32(b *testing.B) {
	// Setup
	idx := NewFlatIndexFloat(BenchDim)

	// Pre-fill data
	ids := make([]string, BenchCount)
	vecs := make([]types.Vector, BenchCount)
	metas := make([]map[string]any, BenchCount)

	for i := 0; i < BenchCount; i++ {
		ids[i] = fmt.Sprintf("vec_%d", i)
		vecs[i] = ensureRandomVector(BenchDim)
	}

	// Bulk Insert
	_ = idx.AddBatch(ids, vecs, metas)

	query := ensureRandomVector(BenchDim)
	b.ResetTimer()

	// Run Benchmark
	for i := 0; i < b.N; i++ {
		// Search Top-10
		_, _ = idx.Search(query, 10, nil)
	}
}

// 2. Benchmark Search Performance (SQ8)
// Measures the speedup gained from smaller memory footprint (Cache friendly).
func BenchmarkFlat_Search_SQ8(b *testing.B) {
	idx := NewFlatIndexSQ8(BenchDim)

	ids := make([]string, BenchCount)
	vecs := make([]types.Vector, BenchCount)
	metas := make([]map[string]any, BenchCount)

	for i := 0; i < BenchCount; i++ {
		ids[i] = fmt.Sprintf("vec_%d", i)
		vecs[i] = ensureRandomVector(BenchDim)
	}

	_ = idx.AddBatch(ids, vecs, metas)

	query := ensureRandomVector(BenchDim)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, _ = idx.Search(query, 10, nil)
	}
}

// 3. Benchmark Insert Batch Speed
// Measures Memory Allocation efficiency and Data Copy speed.
func BenchmarkFlat_InsertBatch(b *testing.B) {
	// Prepare a batch of 1000 vectors
	batchSize := 1000
	ids := make([]string, batchSize)
	vecs := make([]types.Vector, batchSize)
	metas := make([]map[string]any, batchSize)

	for i := 0; i < batchSize; i++ {
		ids[i] = fmt.Sprintf("id_%d", i)
		vecs[i] = ensureRandomVector(BenchDim)
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		b.StopTimer() // Don't count initialization
		idx := NewFlatIndexFloat(BenchDim)
		b.StartTimer()

		// Measure Batch Insert
		_ = idx.AddBatch(ids, vecs, metas)
	}
}

// 4. Benchmark Parallel Search (Concurrency)
// Simulates a real-world server load
func BenchmarkFlat_Search_Concurrent(b *testing.B) {
	idx := NewFlatIndexFloat(BenchDim)

	// 50k items for concurrency test
	count := 50000
	ids := make([]string, count)
	vecs := make([]types.Vector, count)
	metas := make([]map[string]any, count)

	for i := 0; i < count; i++ {
		ids[i] = fmt.Sprintf("%d", i)
		vecs[i] = ensureRandomVector(BenchDim)
	}
	_ = idx.AddBatch(ids, vecs, metas)

	query := ensureRandomVector(BenchDim)
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, _ = idx.Search(query, 10, nil)
		}
	})
}

// 5. Benchmark Update Vector (Delete + Insert)
// Measures the cost of updating a document's vector.
// In FlatIndex, this involves:
// 1. Delete: Find ID (Map lookup) -> Swap-and-Pop (Memory Copy) -> Map Update
// 2. Insert: Check ID -> Append Vector (Memory Copy) -> Map Update
func BenchmarkFlat_UpdateVector(b *testing.B) {
	idx := NewFlatIndexFloat(BenchDim)

	// Pre-fill data
	ids := make([]string, BenchCount)
	vecs := make([]types.Vector, BenchCount)
	metas := make([]map[string]any, BenchCount)

	for i := 0; i < BenchCount; i++ {
		ids[i] = fmt.Sprintf("vec_%d", i)
		vecs[i] = ensureRandomVector(BenchDim)
	}
	_ = idx.AddBatch(ids, vecs, metas)

	// Pre-allocate a replacement vector to avoid allocation noise during bench
	updateVec := ensureRandomVector(BenchDim)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		// Round-robin selection to keep IDs valid and simulate random access
		// We use the pre-generated IDs list to ensure we always hit a valid key
		targetIdx := i % BenchCount
		id := ids[targetIdx]

		// 1. Delete (Swap-and-Pop)
		// This moves the last element to the deleted position, keeping the array packed.
		if err := idx.Delete(id); err != nil {
			b.Fatalf("Delete failed for %s: %v", id, err)
		}

		// 2. Add back (Append)
		// The ID is re-inserted at the end of the list.
		if err := idx.Add(id, updateVec, nil); err != nil {
			b.Fatalf("Add failed for %s: %v", id, err)
		}
	}
}

// 6. Benchmark Update Metadata Only
// Measures the cost of updating metadata without touching the vector data.
// Should be O(1) map lookup + slice assignment.
func BenchmarkFlat_UpdateMetadata(b *testing.B) {
	idx := NewFlatIndexFloat(BenchDim)

	ids := make([]string, BenchCount)
	vecs := make([]types.Vector, BenchCount)
	metas := make([]map[string]any, BenchCount)

	for i := 0; i < BenchCount; i++ {
		ids[i] = fmt.Sprintf("vec_%d", i)
		vecs[i] = ensureRandomVector(BenchDim)
		metas[i] = map[string]any{"v": 0}
	}
	_ = idx.AddBatch(ids, vecs, metas)

	newMeta := map[string]any{"v": 1, "updated": true}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		targetIdx := i % BenchCount
		id := ids[targetIdx]

		if err := idx.UpdateMetadata(id, newMeta); err != nil {
			b.Fatalf("UpdateMetadata failed: %v", err)
		}
	}
}
