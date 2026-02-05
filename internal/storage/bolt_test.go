package storage

import (
	"path/filepath"
	"testing"

	"github.com/hungpdn/nanovec/pkg/types"
)

// TestBoltStorage_BasicOperations verifies CRUD operations and metadata consistency
func TestBoltStorage_BasicOperations(t *testing.T) {
	// Setup temporary database
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test_basic.db")

	store, err := NewBoltStorage(dbPath)
	if err != nil {
		t.Fatalf("Failed to open storage: %v", err)
	}
	defer store.Close()

	doc := &types.Document{
		ID:       "doc1",
		Vector:   []float32{1.0, 0.0, 0.0},
		Metadata: map[string]any{"name": "test_item"},
	}

	// 1. Test Put (Insert)
	version1, err := store.Put(doc)
	if err != nil {
		t.Fatalf("Put failed: %v", err)
	}

	// 2. Test Count (O(1) counter from metadata)
	count, _ := store.Count()
	if count != 1 {
		t.Errorf("Expected count 1, got %d", count)
	}

	// 3. Test Get and Data Integrity
	retrieved, err := store.Get("doc1")
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	if retrieved.ID != doc.ID || len(retrieved.Vector) != 3 || retrieved.Vector[0] != 1.0 {
		t.Errorf("Data mismatch after retrieval: %+v", retrieved)
	}

	// 4. Test Versioning
	currentVersion, _ := store.GetVersion()
	if version1 != currentVersion {
		t.Errorf("Version mismatch: expected %d, got %d", version1, currentVersion)
	}

	// 5. Test Delete (Idempotency and Consistency)
	versionAfterDelete, err := store.Delete("doc1")
	if err != nil {
		t.Errorf("Delete failed: %v", err)
	}
	if versionAfterDelete <= version1 {
		t.Errorf("Version should increment after a successful delete")
	}

	finalCount, _ := store.Count()
	if finalCount != 0 {
		t.Errorf("Expected count 0 after delete, got %d", finalCount)
	}

	// 6. Deleting a non-existent ID should NOT increment version
	versionNoOp, _ := store.Delete("non_existent")
	if versionNoOp != versionAfterDelete {
		t.Errorf("Version should NOT increment when deleting a non-existent ID")
	}
}

// TestBoltStorage_BatchArena verifies high-performance batch inserts using the Arena Buffer
func TestBoltStorage_BatchArena(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test_batch.db")
	store, _ := NewBoltStorage(dbPath)
	defer store.Close()

	docs := []*types.Document{
		{ID: "b1", Vector: []float32{0.1, 0.2, 0.3}, Metadata: map[string]any{"val": 1}},
		{ID: "b2", Vector: []float32{0.4, 0.5, 0.6}, Metadata: map[string]any{"val": 2}},
		{ID: "b3", Vector: []float32{0.7, 0.8, 0.9}, Metadata: map[string]any{"val": 3}},
	}

	// PutBatch uses internal batchBuf to minimize allocations
	_, err := store.PutBatch(docs)
	if err != nil {
		t.Fatalf("PutBatch failed: %v", err)
	}

	count, _ := store.Count()
	if count != 3 {
		t.Errorf("Expected 3 documents, got %d", count)
	}

	// Verify Scan functionality
	foundDocs := make(map[string]bool)
	err = store.Scan(func(doc *types.Document) error {
		foundDocs[doc.ID] = true
		return nil
	})
	if err != nil {
		t.Errorf("Scan failed: %v", err)
	}
	if len(foundDocs) != 3 {
		t.Errorf("Scan missed some documents: %v", foundDocs)
	}
}

// TestBoltStorage_MemorySafety ensures protection against Out-of-Memory attacks
func TestBoltStorage_MemorySafety(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test_safety.db")
	store, _ := NewBoltStorage(dbPath)
	defer store.Close()

	// Vector exceeding MaxAllowedDim (32768)
	hugeDoc := &types.Document{
		ID:     "invalid_dim",
		Vector: make([]float32, MaxAllowedDim+1),
	}

	_, err := store.Put(hugeDoc)
	if err == nil {
		t.Error("Put should have failed for vectors exceeding MaxAllowedDim")
	}
}

// TestBoltStorage_SelfHealing verifies the counter is repaired upon startup
func TestBoltStorage_SelfHealing(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test_healing.db")

	// 1. Initial Insert
	store, _ := NewBoltStorage(dbPath)
	_, _ = store.Put(&types.Document{ID: "d1", Vector: []float32{1.0}})
	store.Close()

	// 2. Re-open Database
	// NewBoltStorage automatically calls SyncDocCount()
	store2, err := NewBoltStorage(dbPath)
	if err != nil {
		t.Fatalf("Failed to re-open DB: %v", err)
	}
	defer store2.Close()

	count, err := store2.Count()
	if err != nil {
		t.Fatalf("Count failed: %v", err)
	}
	if count != 1 {
		t.Errorf("Self-healing failed to restore correct count. Got %d", count)
	}
}
