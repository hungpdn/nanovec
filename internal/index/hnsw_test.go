package index

import (
	"container/heap"
	"fmt"
	"math/rand"
	"path/filepath"
	"sync"
	"testing"

	"github.com/hungpdn/nanovec/pkg/types"
)

// Helper: create random vector
func randomVector(dim int) types.Vector {
	vec := make(types.Vector, dim)
	for i := 0; i < dim; i++ {
		vec[i] = rand.Float32()
	}
	return vec
}

// 1. Basic Flow: Add, Search, Delete
func TestHNSW_BasicFlow(t *testing.T) {
	dim := 128
	idx := NewHNSWIndexFloat(dim, 16, 200)

	id := "vec1"
	vec := randomVector(dim)
	meta := map[string]any{"type": "test"}

	// Insert
	if err := idx.Add(id, vec, meta); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	if idx.Count() != 1 {
		t.Errorf("Expected count 1, got %d", idx.Count())
	}

	// Search for itself (Score must be very high, around 1.0, or the interval should be around 0.0)
	results, err := idx.Search(vec, 1, nil)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) == 0 {
		t.Fatal("Search returned no results")
	}
	if results[0].ID != id {
		t.Errorf("Expected top result %s, got %s", id, results[0].ID)
	}

	// Delete
	_ = idx.Delete(id)

	// Search again (so it's not visible due to Soft Delete / Tombstone)
	results, _ = idx.Search(vec, 1, nil)
	if len(results) > 0 && results[0].ID == id {
		t.Error("Deleted ID still appears in search results")
	}
}

// 2. Persistence (Save/Load)
func TestHNSW_Persistence(t *testing.T) {
	tmpDir := t.TempDir()
	idxPath := filepath.Join(tmpDir, "hnsw.idx")
	dim := 64
	count := 100

	// Create Index and add data
	idx1 := NewHNSWIndexFloat(dim, 16, 200)
	ids := make([]string, count)
	vecs := make([]types.Vector, count)
	metas := make([]map[string]any, count)

	for i := 0; i < count; i++ {
		ids[i] = fmt.Sprintf("vec_%d", i)
		vecs[i] = randomVector(dim)
		metas[i] = map[string]any{"val": i}
	}

	// Test AddBatch
	if err := idx1.AddBatch(ids, vecs, metas); err != nil {
		t.Fatalf("AddBatch failed: %v", err)
	}

	// Save to disk
	if err := idx1.Save(idxPath); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	// Load into new Index
	idx2 := NewHNSWIndexFloat(dim, 16, 200)
	if err := idx2.Load(idxPath); err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	// Verify Metadata & Count
	if idx2.Count() != count {
		t.Errorf("Count mismatch. Expected %d, got %d", count, idx2.Count())
	}

	// Verify Data by Search
	results, _ := idx2.Search(vecs[0], 1, nil)
	if len(results) == 0 || results[0].ID != ids[0] {
		t.Error("Persistence check failed: Cannot find vec_0 in loaded index")
	}

	// Verify Metadata Integrity
	if results[0].Metadata["val"] != float64(0) && results[0].Metadata["val"] != 0 {
		t.Errorf("Metadata mismatch: %v", results[0].Metadata)
	}
}

// 3. Concurrency (Stress Test)
func TestHNSW_Concurrency(t *testing.T) {
	dim := 32
	idx := NewHNSWIndexFloat(dim, 16, 200)

	var wg sync.WaitGroup
	workers := 10
	itemsPerWorker := 100

	// Run many goroutine Insert và Search concurrency
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for j := 0; j < itemsPerWorker; j++ {
				id := fmt.Sprintf("w%d_i%d", workerID, j)
				vec := randomVector(dim)

				// Insert
				_ = idx.Add(id, vec, nil)

				// Intermittent search (reading while writing)
				if j%10 == 0 {
					_, _ = idx.Search(vec, 5, nil)
				}
			}
		}(i)
	}

	wg.Wait()

	expectedTotal := workers * itemsPerWorker
	if idx.Count() != expectedTotal {
		t.Errorf("Concurrent insert lost data. Expected %d, got %d", expectedTotal, idx.Count())
	}
}

// 4. SQ8 (Quantization)
func TestHNSW_SQ8(t *testing.T) {
	dim := 128
	// Init Index SQ8
	idx := NewHNSWIndexSQ8(dim, 16, 200)

	targetVec := randomVector(dim)
	_ = idx.Add("target", targetVec, nil)

	// Add many vector
	noiseVec := randomVector(dim)
	_ = idx.Add("noise", noiseVec, nil)

	// Search exact targetVec
	results, _ := idx.Search(targetVec, 1, nil)

	if len(results) == 0 {
		t.Fatal("SQ8 Search returned empty")
	}

	// SQ8 rounds numbers, so the score may not be an absolute 1.0, but the ID must be correct.
	if results[0].ID != "target" {
		t.Errorf("SQ8 accuracy fail. Expected 'target', got '%s'", results[0].ID)
	}
}

// 5. Heaps
func TestHeaps(t *testing.T) {
	// 1. MinHeap Test (The smallest score is at the top)
	t.Run("MinHeap", func(t *testing.T) {
		h := &MinHeap{}
		heap.Init(h)

		// Messy push
		heap.Push(h, pqItem{id: 1, score: 0.5})
		heap.Push(h, pqItem{id: 2, score: 0.1}) // Smallest
		heap.Push(h, pqItem{id: 3, score: 0.9})

		if h.Len() != 3 {
			t.Errorf("Expected len 3, got %d", h.Len())
		}

		// Pop: Expect 0.1 -> 0.5 -> 0.9
		first := heap.Pop(h).(pqItem)
		if first.score != 0.1 {
			t.Errorf("MinHeap Failed: expected 0.1, got %f", first.score)
		}

		second := heap.Pop(h).(pqItem)
		if second.score != 0.5 {
			t.Errorf("MinHeap Failed: expected 0.5, got %f", second.score)
		}
	})

	// 2. Test MaxHeap (The highest score is at the top)
	t.Run("MaxHeap", func(t *testing.T) {
		h := &MaxHeap{}
		heap.Init(h)

		heap.Push(h, pqItem{id: 1, score: 0.5})
		heap.Push(h, pqItem{id: 2, score: 0.1})
		heap.Push(h, pqItem{id: 3, score: 0.9}) // Biggest

		// Pop: Expect 0.9 -> 0.5 -> 0.1
		first := heap.Pop(h).(pqItem)
		if first.score != 0.9 {
			t.Errorf("MaxHeap Failed: expected 0.9, got %f", first.score)
		}
	})
}

func BenchmarkHNSW_Search_Float(b *testing.B) {
	dim := 128
	idx := NewHNSWIndexFloat(dim, 16, 200)

	// Setup: Insert 10k vectors
	ids := make([]string, 10000)
	vecs := make([]types.Vector, 10000)
	for i := 0; i < 10000; i++ {
		ids[i] = fmt.Sprintf("id_%d", i)
		vecs[i] = randomVector(dim)
	}
	_ = idx.AddBatch(ids, vecs, make([]map[string]any, 10000))

	query := randomVector(dim)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, _ = idx.Search(query, 10, nil)
	}
}
