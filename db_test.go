package nanovec_test

import (
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/hungpdn/nanovec"
)

// Helper to assert errors
func assertNoError(t *testing.T, err error, msg string) {
	t.Helper()
	if err != nil {
		t.Fatalf("%s: unexpected error: %v", msg, err)
	}
}

// Helper to assert equality
func assertEqual(t *testing.T, expected, actual any, msg string) {
	t.Helper()
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("%s: expected %v, got %v", msg, expected, actual)
	}
}

func TestCrashRecovery(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "crash_db_std")
	cfg := &nanovec.Config{Dimension: 3}

	// 1. Insert Data
	func() {
		db, err := nanovec.Open(dbPath, cfg)
		assertNoError(t, err, "Failed to open DB for setup")
		defer db.Close()

		ids := []string{"doc1", "doc2"}
		vecs := [][]float32{
			{1.0, 0.0, 0.0},
			{0.0, 1.0, 0.0},
		}
		metas := []map[string]any{
			{"name": "A", "val": 100},
			{"name": "B", "val": 200},
		}

		err = db.InsertBatch(ids, vecs, metas)
		assertNoError(t, err, "Failed to batch insert")
	}()

	// 2. Simulate Crash: Delete the .idx file
	idxPath := dbPath + ".idx"
	err := os.Remove(idxPath)
	assertNoError(t, err, "Failed to delete .idx file for simulation")

	// 3. Recovery
	db, err := nanovec.Open(dbPath, cfg)
	assertNoError(t, err, "Failed to re-open DB during recovery")
	defer db.Close()

	// 4. Verify Data matches
	results, err := db.Search([]float32{1.0, 0.0, 0.0}, 5, nil)
	assertNoError(t, err, "Search failed after recovery")

	if len(results) == 0 {
		t.Fatal("Expected results, got 0")
	}

	top := results[0]
	assertEqual(t, "doc1", top.ID, "Top result ID mismatch")

	if math.Abs(float64(top.Score)-1.0) > 0.0001 {
		t.Errorf("Expected score ~1.0, got %f", top.Score)
	}
}

func TestHNSWIndex(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "hnsw_test")

	cfg := &nanovec.Config{
		Dimension:      3,
		IndexType:      nanovec.IndexTypeHNSW,
		M:              16,
		EfConstruction: 100,
	}

	func() {
		db, err := nanovec.Open(dbPath, cfg)
		assertNoError(t, err, "Open HNSW DB")
		defer db.Close()

		err = db.Insert("vecA", []float32{1, 0, 0}, map[string]any{"type": "A"})
		assertNoError(t, err, "Insert A")
		err = db.Insert("vecB", []float32{0, 1, 0}, map[string]any{"type": "B"})
		assertNoError(t, err, "Insert B")

		results, err := db.Search([]float32{1, 0, 0}, 5, nil)
		assertNoError(t, err, "Search A")

		if len(results) == 0 {
			t.Fatal("HNSW returned 0 results")
		}
		assertEqual(t, "vecA", results[0].ID, "Should find vecA first")
	}()

	func() {
		db, err := nanovec.Open(dbPath, cfg)
		assertNoError(t, err, "Re-Open HNSW DB")
		defer db.Close()

		results, err := db.Search([]float32{0, 1, 0}, 5, nil)
		assertNoError(t, err, "Search B after restart")

		if len(results) == 0 {
			t.Fatal("HNSW persistence failed")
		}
		assertEqual(t, "vecB", results[0].ID, "Should find vecB after restart")
	}()
}

func TestSQ8Index(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "sq8_test")

	cfg := &nanovec.Config{
		Dimension:    3,
		IndexType:    nanovec.IndexTypeFlat,
		Quantization: true,
	}

	func() {
		db, err := nanovec.Open(dbPath, cfg)
		assertNoError(t, err, "Open SQ8 DB")
		defer db.Close()

		err = db.Insert("vecA", []float32{1, 0, 0}, map[string]any{"type": "A"})
		assertNoError(t, err, "Insert A")
		err = db.Insert("vecB", []float32{0, 1, 0}, map[string]any{"type": "B"})
		assertNoError(t, err, "Insert B")

		results, err := db.Search([]float32{1, 0, 0}, 5, nil)
		assertNoError(t, err, "Search A")

		if len(results) == 0 {
			t.Fatal("SQ8 returned 0 results")
		}
		assertEqual(t, "vecA", results[0].ID, "Should find vecA first")
		if results[0].Score < 0.98 {
			t.Errorf("Score too low for SQ8 match: %f", results[0].Score)
		}
	}()
}

// TestHNSWSQ8Index verifies the combination of Graph Search + Compression
func TestHNSWSQ8Index(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "hnsw_sq8_test")

	cfg := &nanovec.Config{
		Dimension:      3,
		IndexType:      nanovec.IndexTypeHNSW, // Enable Graph
		Quantization:   true,                  // Enable Compression
		M:              16,
		EfConstruction: 100,
	}

	// 1. Insert & Search
	func() {
		db, err := nanovec.Open(dbPath, cfg)
		assertNoError(t, err, "Open HNSW+SQ8 DB")
		defer db.Close()

		// Insert vectors
		// Use clearly separable vectors to test graph navigation
		err = db.Insert("vecA", []float32{1, 0, 0}, map[string]any{"type": "A"})
		assertNoError(t, err, "Insert A")
		err = db.Insert("vecB", []float32{0, 1, 0}, map[string]any{"type": "B"})
		assertNoError(t, err, "Insert B")
		err = db.Insert("vecC", []float32{0, 0, 1}, map[string]any{"type": "C"})
		assertNoError(t, err, "Insert C")

		// Search for A
		results, err := db.Search([]float32{0.9, 0.1, 0}, 5, nil)
		assertNoError(t, err, "Search A")

		if len(results) == 0 {
			t.Fatal("HNSW+SQ8 returned 0 results")
		}
		assertEqual(t, "vecA", results[0].ID, "Should find vecA first")

		// SQ8 + HNSW approximation might lose some precision, but for [0.9, 0.1, 0] vs [1,0,0]
		// score should still be very high (> 0.85)
		if results[0].Score < 0.85 {
			t.Errorf("Score too low for HNSW+SQ8 match: %f", results[0].Score)
		}
	}()

	// 2. Persistence
	func() {
		db, err := nanovec.Open(dbPath, cfg)
		assertNoError(t, err, "Re-Open HNSW+SQ8 DB")
		defer db.Close()

		// Verify data persisted and graph is navigable
		results, err := db.Search([]float32{0, 0.9, 0.1}, 5, nil) // Search for B
		assertNoError(t, err, "Search B after restart")

		if len(results) == 0 {
			t.Fatal("Persistence failed (0 results)")
		}
		assertEqual(t, "vecB", results[0].ID, "Should find vecB after restart")
	}()
}
