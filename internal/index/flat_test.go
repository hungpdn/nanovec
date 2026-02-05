package index

import (
	"fmt"
	"path/filepath"
	"sync"
	"testing"

	"github.com/hungpdn/nanovec/pkg/types"
)

// 1. Flat Index Float32 basic
func TestFlat_Float_Basic(t *testing.T) {
	dim := 4
	idx := NewFlatIndexFloat(dim)

	// Normalized vector (for easier dot product calculation)
	// A = [1, 0, 0, 0]
	// B = [0, 1, 0, 0]
	_ = idx.Add("A", []float32{1, 0, 0, 0}, nil)
	_ = idx.Add("B", []float32{0, 1, 0, 0}, nil)

	// Search A -> should find A (Score 1.0) and B (Score 0.0)
	results, err := idx.Search([]float32{1, 0, 0, 0}, 2, nil)
	if err != nil {
		t.Fatal(err)
	}

	if len(results) != 2 {
		t.Errorf("Expected 2 results, got %d", len(results))
	}
	if results[0].ID != "A" {
		t.Errorf("Top result mismatch. Expected A, got %s", results[0].ID)
	}
}

// 2. Flat Index SQ8 (Quantization Logic)
func TestFlat_SQ8_Basic(t *testing.T) {
	dim := 4
	idx := NewFlatIndexSQ8(dim)

	// In SQ8:
	// 1.0 -> 255
	// 0.0 -> ~127
	// -1.0 -> 0

	// Add vector A = [1, 1, 1, 1]
	_ = idx.Add("A", []float32{1, 1, 1, 1}, nil)

	// Add vector B = [-1, -1, -1, -1]
	_ = idx.Add("B", []float32{-1, -1, -1, -1}, nil)

	// Search A
	results, _ := idx.Search([]float32{1, 1, 1, 1}, 2, nil)

	if len(results) < 1 {
		t.Fatal("No results found")
	}
	if results[0].ID != "A" {
		t.Errorf("SQ8 logic error. Expected A, got %s", results[0].ID)
	}

	// The SQ8 score may be > 1.0 or slightly off due to the DotProductSQ8 formula,
	// but with A match A, the score must be the highest.
}

// 3. Batch Add aand Delete for Flat
func TestFlat_BatchAndDelete(t *testing.T) {
	dim := 2
	idx := NewFlatIndexFloat(dim)

	ids := []string{"1", "2", "3"}
	vecs := []types.Vector{
		{1, 0}, {0, 1}, {1, 1},
	}
	metas := []map[string]any{
		{"v": 1}, {"v": 2}, {"v": 3},
	}

	// Batch Add
	_ = idx.AddBatch(ids, vecs, metas)

	if idx.Count() != 3 {
		t.Errorf("Count mismatch. Expected 3, got %d", idx.Count())
	}

	// Delete item "2" (in the middle of the list)
	// Flat indexes usually use Swap-and-Pop, so item "3" will be moved to the position of "2"
	_ = idx.Delete("2")

	if idx.Count() != 2 {
		t.Errorf("Count mismatch after delete. Expected 2, got %d", idx.Count())
	}

	// Verify "2" is gone
	results, _ := idx.Search([]float32{0, 1}, 3, nil) // Search vector of "2" old
	for _, res := range results {
		if res.ID == "2" {
			t.Error("Deleted ID '2' still found")
		}
	}

	// Verify "3" still exists
	found3 := false
	for _, res := range results {
		if res.ID == "3" {
			found3 = true
			if res.Metadata["v"] != 3 { // check metadata consistency after swap
				t.Error("Metadata corrupted after delete swap")
			}
		}
	}
	if !found3 {
		t.Error("ID '3' missing after delete swap")
	}
}

// 4. Save/Load for Flat
func TestFlat_Persistence(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "flat.idx")

	idx1 := NewFlatIndexFloat(2)
	_ = idx1.Add("A", []float32{1, 0}, map[string]any{"foo": "bar"})

	if err := idx1.Save(path); err != nil {
		t.Fatal(err)
	}

	idx2 := NewFlatIndexFloat(2)
	if err := idx2.Load(path); err != nil {
		t.Fatal(err)
	}

	if idx2.Count() != 1 {
		t.Error("Load count mismatch")
	}

	// Check Data validity
	res, _ := idx2.Search([]float32{1, 0}, 1, nil)
	if len(res) == 0 || res[0].ID != "A" {
		t.Error("Data mismatch after load")
	}
	if res[0].Metadata["foo"] != "bar" {
		t.Error("Metadata mismatch after load")
	}
}

// 5. Concurrency for Flat Index
func TestFlat_Concurrency(t *testing.T) {
	dim := 32
	idx := NewFlatIndexFloat(dim)

	var wg sync.WaitGroup
	workers := 10
	itemsPerWorker := 100

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for j := 0; j < itemsPerWorker; j++ {
				id := fmt.Sprintf("w%d_i%d", workerID, j)
				vec := randomVector(dim)

				// Write
				_ = idx.Add(id, vec, nil)

				// Concurrent Read
				if j%10 == 0 {
					_, _ = idx.Search(vec, 5, nil)
				}
			}
		}(i)
	}

	wg.Wait()

	expectedTotal := workers * itemsPerWorker
	if idx.Count() != expectedTotal {
		t.Errorf("Concurrent Flat insert failed. Expected %d, got %d", expectedTotal, idx.Count())
	}
}
