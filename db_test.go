package nanovec_test

import (
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/hungpdn/nanovec"
	"github.com/hungpdn/nanovec/internal/index"
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

func TestHNSWPersistence(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "hnsw_persist_test")

	cfg := &nanovec.Config{
		Dimension:      3,
		IndexType:      nanovec.IndexTypeHNSW,
		M:              16,
		EfConstruction: 100,
	}

	// 1. Insert Data
	func() {
		db, err := nanovec.Open(dbPath, cfg)
		assertNoError(t, err, "Open HNSW DB")
		defer db.Close()

		err = db.Insert("vecA", []float32{1, 0, 0}, map[string]any{"type": "A"})
		assertNoError(t, err, "Insert A")
		err = db.Insert("vecB", []float32{0, 1, 0}, map[string]any{"type": "B"})
		assertNoError(t, err, "Insert B")
	}()

	// 2. Restart and Verify
	func() {
		db, err := nanovec.Open(dbPath, cfg)
		assertNoError(t, err, "Re-Open HNSW DB")
		defer db.Close()

		// Search for B to ensure graph is navigable and data persisted
		results, err := db.Search([]float32{0, 1, 0}, 5, nil)
		assertNoError(t, err, "Search B after restart")

		if len(results) == 0 {
			t.Fatal("HNSW persistence failed: No results found")
		}
		assertEqual(t, "vecB", results[0].ID, "Should find vecB after restart")

		// Ensure metadata is intact
		if results[0].Metadata["type"] != "B" {
			t.Errorf("Metadata mismatch: expected B, got %v", results[0].Metadata["type"])
		}
	}()
}

func TestDeleteConsistency(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "delete_test")
	cfg := &nanovec.Config{Dimension: 3}

	// 1. Insert and Delete
	func() {
		db, err := nanovec.Open(dbPath, cfg)
		assertNoError(t, err, "Open DB")
		defer db.Close()

		// Insert
		err = db.Insert("doc1", []float32{1, 0, 0}, nil)
		assertNoError(t, err, "Insert doc1")

		// Verify exists
		if !db.Exists("doc1") {
			t.Fatal("doc1 should exist")
		}

		// Delete
		err = db.Delete("doc1")
		assertNoError(t, err, "Delete doc1")

		// Verify gone from RAM immediately
		if db.Exists("doc1") {
			t.Fatal("doc1 should be deleted from RAM")
		}
	}()

	// 2. Restart and Verify (Persistence check)
	func() {
		db, err := nanovec.Open(dbPath, cfg)
		assertNoError(t, err, "Re-Open DB")
		defer db.Close()

		// Verify gone from Disk/Index after restart
		if db.Exists("doc1") {
			t.Fatal("doc1 should remain deleted after restart (Zombie Data check)")
		}

		// Search should return empty
		results, err := db.Search([]float32{1, 0, 0}, 5, nil)
		assertNoError(t, err, "Search")
		if len(results) != 0 {
			t.Errorf("Expected 0 results for deleted item, got %d", len(results))
		}
	}()
}

func TestConfig_GetVectorIndex(t *testing.T) {
	tests := []struct {
		name         string
		cfg          nanovec.Config
		expectedType string // "FlatFloat", "FlatSQ8", "HNSWFloat", "HNSWSQ8"
	}{
		{
			name:         "Default Flat Float",
			cfg:          nanovec.Config{IndexType: nanovec.IndexTypeFlat, Quantization: false, Dimension: 128},
			expectedType: "FlatFloat",
		},
		{
			name:         "Flat SQ8",
			cfg:          nanovec.Config{IndexType: nanovec.IndexTypeFlat, Quantization: true, Dimension: 128},
			expectedType: "FlatSQ8",
		},
		{
			name:         "HNSW Float",
			cfg:          nanovec.Config{IndexType: nanovec.IndexTypeHNSW, Quantization: false, Dimension: 128, M: 16, EfConstruction: 200},
			expectedType: "HNSWFloat",
		},
		{
			name:         "HNSW SQ8",
			cfg:          nanovec.Config{IndexType: nanovec.IndexTypeHNSW, Quantization: true, Dimension: 128, M: 16, EfConstruction: 200},
			expectedType: "HNSWSQ8",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			idx := tt.cfg.GetVectorIndex()

			var actualType string
			switch idx.(type) {
			case *index.FlatIndex[float32]:
				actualType = "FlatFloat"
			case *index.FlatIndex[uint8]:
				actualType = "FlatSQ8"
			case *index.HNSWIndex[float32]:
				actualType = "HNSWFloat"
			case *index.HNSWIndex[uint8]:
				actualType = "HNSWSQ8"
			default:
				t.Fatalf("Unknown index type returned")
			}

			if actualType != tt.expectedType {
				t.Errorf("Expected %s, got %s", tt.expectedType, actualType)
			}
		})
	}
}

func TestDB_Vacuum(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "vacuum_test")
	cfg := &nanovec.Config{Dimension: 2}

	// 1. Setup Data
	func() {
		db, err := nanovec.Open(dbPath, cfg)
		assertNoError(t, err, "Open DB")
		defer db.Close()

		// Insert 3 items
		_ = db.Insert("A", []float32{1, 0}, nil)
		_ = db.Insert("B", []float32{0, 1}, nil)
		_ = db.Insert("C", []float32{1, 1}, nil)

		// Delete B
		_ = db.Delete("B")

		// Run Vacuum
		if err := db.Vacuum(); err != nil {
			t.Fatalf("Vacuum failed: %v", err)
		}

		// Verify internal integrity immediately
		if db.Exists("B") {
			t.Error("Deleted item 'B' resurrected after Vacuum")
		}
		if !db.Exists("A") || !db.Exists("C") {
			t.Error("Valid items lost after Vacuum")
		}
	}()

	// 2. Restart & Verify Persistence
	func() {
		db, err := nanovec.Open(dbPath, cfg)
		assertNoError(t, err, "Re-Open DB")
		defer db.Close()

		results, _ := db.Search([]float32{1, 0}, 5, nil)
		if len(results) != 2 { // Should be A and C only
			t.Errorf("Expected 2 results after Vacuum+Restart, got %d", len(results))
		}
	}()
}
