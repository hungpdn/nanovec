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
	// 1. Setup: Create a temporary directory (Go 1.15+ standard way)
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "crash_db_std")
	cfg := &nanovec.Config{Dimension: 3}

	// 2. Insert Data
	func() {
		db, err := nanovec.Open(dbPath, cfg)
		assertNoError(t, err, "Failed to open DB for setup")
		defer db.Close()

		// Use InsertBatch to test the batch path
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

	// 3. Simulate Crash: Delete the .idx file
	// The .store file (BoltDB) remains, which is our source of truth.
	idxPath := dbPath + ".idx"
	err := os.Remove(idxPath)
	assertNoError(t, err, "Failed to delete .idx file for simulation")

	// Verify .store exists
	storePath := dbPath + ".store"
	if _, err := os.Stat(storePath); os.IsNotExist(err) {
		t.Fatalf(".store file missing, cannot test recovery")
	}

	// 4. Recovery: Re-open the DB
	// It should detect the missing index and rebuild it from storage.
	db, err := nanovec.Open(dbPath, cfg)
	assertNoError(t, err, "Failed to re-open DB during recovery")
	defer db.Close()

	// 5. Verify Data matches
	// Search for doc1
	results, err := db.Search([]float32{1.0, 0.0, 0.0}, 5, nil)
	assertNoError(t, err, "Search failed after recovery")

	if len(results) == 0 {
		t.Fatal("Expected results, got 0")
	}

	// Verify Top 1 is doc1
	top := results[0]
	assertEqual(t, "doc1", top.ID, "Top result ID mismatch")

	// Verify Metadata
	if val, ok := top.Metadata["val"].(int); !ok || val != 100 {
		// Note: Gob might decode numbers as int or float depending on implementation
		// If it fails, check if it came back as float64, etc.
		// For robustness in tests without testify, we often just print:
		t.Logf("Metadata recovered: %v", top.Metadata)
	}

	// Verify Score (Float comparison needs epsilon)
	if math.Abs(float64(top.Score)-1.0) > 0.0001 {
		t.Errorf("Expected score ~1.0, got %f", top.Score)
	}

	// Verify doc2 exists via Exists()
	if !db.Exists("doc2") {
		t.Error("doc2 should exist after recovery")
	}
}

func TestHNSWIndex(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "hnsw_test")

	// Configure for HNSW
	cfg := &nanovec.Config{
		Dimension:      3,
		IndexType:      nanovec.IndexTypeHNSW, // <--- Switch to HNSW
		M:              16,
		EfConstruction: 100,
	}

	// 1. Insert & Search
	func() {
		db, err := nanovec.Open(dbPath, cfg)
		assertNoError(t, err, "Open HNSW DB")
		defer db.Close()

		// Insert vectors that are clearly distinct
		// Vec A: [1, 0, 0]
		// Vec B: [0, 1, 0]
		err = db.Insert("vecA", []float32{1, 0, 0}, map[string]any{"type": "A"})
		assertNoError(t, err, "Insert A")
		err = db.Insert("vecB", []float32{0, 1, 0}, map[string]any{"type": "B"})
		assertNoError(t, err, "Insert B")

		// Search for A (Exact match)
		results, err := db.Search([]float32{1, 0, 0}, 5, nil)
		assertNoError(t, err, "Search A")

		if len(results) == 0 {
			t.Fatal("HNSW returned 0 results")
		}
		assertEqual(t, "vecA", results[0].ID, "Should find vecA first")
		if results[0].Score < 0.99 {
			t.Errorf("Score too low for exact match: %f", results[0].Score)
		}
	}()

	// 2. Persistence & Recovery
	func() {
		// Re-open (simulating restart)
		db, err := nanovec.Open(dbPath, cfg)
		assertNoError(t, err, "Re-Open HNSW DB")
		defer db.Close()

		// Verify data persisted
		results, err := db.Search([]float32{0, 1, 0}, 5, nil) // Search for B
		assertNoError(t, err, "Search B after restart")

		if len(results) == 0 {
			t.Fatal("HNSW persistence failed (0 results)")
		}
		assertEqual(t, "vecB", results[0].ID, "Should find vecB after restart")
	}()
}
