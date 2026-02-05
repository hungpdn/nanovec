package index

import (
	"fmt"
	"testing"

	"github.com/hungpdn/nanovec/pkg/types"
)

/*
➜  nanovec git:(main) ✗ go test -bench=. -benchmem ./internal/index/

goos: darwin
goarch: amd64
pkg: github.com/hungpdn/nanovec/internal/index
cpu: Intel(R) Core(TM) i7-8850H CPU @ 2.60GHz
BenchmarkFlat_Search_Float32-12       	     536	   2194571 ns/op	    2284 B/op	      18 allocs/op
BenchmarkFlat_Search_SQ8-12           	     705	   1748422 ns/op	    2219 B/op	      19 allocs/op
BenchmarkFlat_InsertBatch-12          	    1806	    599377 ns/op	 2747335 B/op	      27 allocs/op
BenchmarkFlat_Search_Concurrent-12    	    1969	    639025 ns/op	    2704 B/op	      16 allocs/op
BenchmarkFlat_UpdateVector-12         	 2198317	       472.9 ns/op	       0 B/op	       0 allocs/op
BenchmarkFlat_UpdateMetadata-12       	18527338	        66.79 ns/op	       0 B/op	       0 allocs/op
BenchmarkHNSW_Build_Float32-12        	      25	  54656426 ns/op	11546245 B/op	   44240 allocs/op
BenchmarkHNSW_Search_Float32-12       	    1116	   1332555 ns/op	     320 B/op	       1 allocs/op
BenchmarkHNSW_Build_SQ8-12            	      37	  46874895 ns/op	10271961 B/op	   38914 allocs/op
BenchmarkHNSW_Search_SQ8-12           	    1350	   1362429 ns/op	     448 B/op	       2 allocs/op
BenchmarkHNSW_Search_Concurrent-12    	   10000	    101775 ns/op	     960 B/op	       6 allocs/op
BenchmarkHNSW_UpdateVector-12         	    9877	    120335 ns/op	   20075 B/op	      77 allocs/op
BenchmarkHNSW_UpdateMetadata-12       	11098572	        99.31 ns/op	       0 B/op	       0 allocs/op
PASS
ok  	github.com/hungpdn/nanovec/internal/index	201.565s
*/

// Constants specific for HNSW benchmarks
const (
	HNSWBenchDim   = 128
	HNSWBenchCount = 100000 // 100k vectors for SEARCH setup (Big Graph)
	HNSWBatchSize  = 1000   // 1k vectors for BUILD benchmark (Matching Flat for fair comparison)
	HNSWM          = 16
	HNSWEf         = 200
)

// 1. Benchmark Build Speed (Float32) - Batch 1000
// Measures how fast we can insert a batch of 1000 vectors into an empty graph.
func BenchmarkHNSW_Build_Float32(b *testing.B) {
	// Prepare data (Batch Size = 1000)
	ids := make([]string, HNSWBatchSize)
	vecs := make([]types.Vector, HNSWBatchSize)
	metas := make([]map[string]any, HNSWBatchSize)

	for i := 0; i < HNSWBatchSize; i++ {
		ids[i] = fmt.Sprintf("vec_%d", i)
		vecs[i] = randomVector(HNSWBenchDim)
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		// Stop timer to init index (we only measure AddBatch)
		b.StopTimer()
		idx := NewHNSWIndexFloat(HNSWBenchDim, HNSWM, HNSWEf)
		b.StartTimer()

		if err := idx.AddBatch(ids, vecs, metas); err != nil {
			b.Fatal(err)
		}
	}
}

// 2. Benchmark Search Performance (Float32) - Dataset 100k
// This is the most critical metric for HNSW (Query Latency).
func BenchmarkHNSW_Search_Float32(b *testing.B) {
	// Setup - Build Index once with FULL DATASET (100k)
	idx := NewHNSWIndexFloat(HNSWBenchDim, HNSWM, HNSWEf)

	ids := make([]string, HNSWBenchCount)
	vecs := make([]types.Vector, HNSWBenchCount)
	metas := make([]map[string]any, HNSWBenchCount)

	for i := 0; i < HNSWBenchCount; i++ {
		ids[i] = fmt.Sprintf("vec_%d", i)
		vecs[i] = randomVector(HNSWBenchDim)
	}
	_ = idx.AddBatch(ids, vecs, metas)

	query := randomVector(HNSWBenchDim)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, _ = idx.Search(query, 10, nil)
	}
}

// 3. Benchmark Build Speed (SQ8) - Batch 1000
// Measures build speed with Quantization overhead.
func BenchmarkHNSW_Build_SQ8(b *testing.B) {
	ids := make([]string, HNSWBatchSize)
	vecs := make([]types.Vector, HNSWBatchSize)
	metas := make([]map[string]any, HNSWBatchSize)

	for i := 0; i < HNSWBatchSize; i++ {
		ids[i] = fmt.Sprintf("vec_%d", i)
		vecs[i] = randomVector(HNSWBenchDim)
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		b.StopTimer()
		idx := NewHNSWIndexSQ8(HNSWBenchDim, HNSWM, HNSWEf)
		b.StartTimer()

		if err := idx.AddBatch(ids, vecs, metas); err != nil {
			b.Fatal(err)
		}
	}
}

// 4. Benchmark Search Performance (SQ8) - Dataset 100k
// Measures search speed on Quantized Graph.
func BenchmarkHNSW_Search_SQ8(b *testing.B) {
	idx := NewHNSWIndexSQ8(HNSWBenchDim, HNSWM, HNSWEf)

	ids := make([]string, HNSWBenchCount)
	vecs := make([]types.Vector, HNSWBenchCount)
	metas := make([]map[string]any, HNSWBenchCount)

	for i := 0; i < HNSWBenchCount; i++ {
		ids[i] = fmt.Sprintf("vec_%d", i)
		vecs[i] = randomVector(HNSWBenchDim)
	}
	_ = idx.AddBatch(ids, vecs, metas)

	query := randomVector(HNSWBenchDim)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, _ = idx.Search(query, 10, nil)
	}
}

// 5. Benchmark Parallel Search - Dataset 20k (Smaller for setup speed)
// Simulates high QPS environment.
func BenchmarkHNSW_Search_Concurrent(b *testing.B) {
	// Reduce count for concurrent test setup to avoid timeout if running locally
	concurrentCount := 50000
	idx := NewHNSWIndexFloat(HNSWBenchDim, HNSWM, HNSWEf)

	ids := make([]string, concurrentCount)
	vecs := make([]types.Vector, concurrentCount)
	metas := make([]map[string]any, concurrentCount)

	for i := 0; i < concurrentCount; i++ {
		ids[i] = fmt.Sprintf("%d", i)
		vecs[i] = randomVector(HNSWBenchDim)
	}
	_ = idx.AddBatch(ids, vecs, metas)

	query := randomVector(HNSWBenchDim)
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, _ = idx.Search(query, 10, nil)
		}
	})
}

// 6. Benchmark Update Vector (Delete + Insert) - Dataset 20k
func BenchmarkHNSW_UpdateVector(b *testing.B) {
	updateCount := 20000
	idx := NewHNSWIndexFloat(HNSWBenchDim, HNSWM, HNSWEf)

	ids := make([]string, updateCount)
	vecs := make([]types.Vector, updateCount)
	metas := make([]map[string]any, updateCount)

	for i := 0; i < updateCount; i++ {
		ids[i] = fmt.Sprintf("vec_%d", i)
		vecs[i] = randomVector(HNSWBenchDim)
	}
	_ = idx.AddBatch(ids, vecs, metas)

	updateVec := randomVector(HNSWBenchDim)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		targetIdx := i % updateCount
		id := ids[targetIdx]

		if err := idx.Delete(id); err != nil {
			b.Fatalf("Delete failed for %s: %v", id, err)
		}

		if err := idx.Add(id, updateVec, nil); err != nil {
			b.Fatalf("Add failed for %s: %v", id, err)
		}
	}
}

// 7. Benchmark Update Metadata Only - Dataset 20k
func BenchmarkHNSW_UpdateMetadata(b *testing.B) {
	updateCount := 20000
	idx := NewHNSWIndexFloat(HNSWBenchDim, HNSWM, HNSWEf)

	ids := make([]string, updateCount)
	vecs := make([]types.Vector, updateCount)
	metas := make([]map[string]any, updateCount)

	for i := 0; i < updateCount; i++ {
		ids[i] = fmt.Sprintf("vec_%d", i)
		vecs[i] = randomVector(HNSWBenchDim)
		metas[i] = map[string]any{"v": 0}
	}
	_ = idx.AddBatch(ids, vecs, metas)

	newMeta := map[string]any{"v": 1, "updated": true}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		targetIdx := i % updateCount
		id := ids[targetIdx]

		if err := idx.UpdateMetadata(id, newMeta); err != nil {
			b.Fatalf("UpdateMetadata failed: %v", err)
		}
	}
}
