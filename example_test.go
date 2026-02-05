package nanovec_test

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/hungpdn/nanovec"
)

// ExampleOpen demonstrates the basic lifecycle of the database:
// Open, Insert, Search, Count and Close.
func ExampleOpen() {
	// 1. Setup temporary path for the example
	tmpDir, err := os.MkdirTemp("", "nanovec_example_*")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(tmpDir) // Cleanup on exit
	dbPath := filepath.Join(tmpDir, "example.db")

	// 2. Configure the database
	cfg := &nanovec.Config{
		Dimension: 3,
		IndexType: nanovec.IndexTypeFlat, // Use Flat for exact results
	}

	// 3. Open the database
	db, err := nanovec.Open(dbPath, cfg)
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	// 4. Insert a document
	// Vector: [1.0, 0.0, 0.0]
	err = db.Insert("vec1", []float32{1.0, 0.0, 0.0}, map[string]any{"tag": "example"})
	if err != nil {
		log.Fatal(err)
	}

	// 5. Search for the nearest neighbor
	// Query: [1.0, 0.0, 0.0] should perfectly match "vec1"
	results, err := db.Search([]float32{1.0, 0.0, 0.0}, 1, nil)
	if err != nil {
		log.Fatal(err)
	}

	if len(results) > 0 {
		fmt.Printf("Found ID: %s, Score: %.1f\n", results[0].ID, results[0].Score)
	}

	// Output:
	// Found ID: vec1, Score: 1.0
}

// ExampleDB_InsertBatch demonstrates how to ingest data efficiently using batching.
// This is significantly faster than inserting items one by one.
func ExampleDB_InsertBatch() {
	// Setup
	tmpDir, _ := os.MkdirTemp("", "nanovec_batch_*")
	defer os.RemoveAll(tmpDir)
	dbPath := filepath.Join(tmpDir, "batch.db")

	db, _ := nanovec.Open(dbPath, &nanovec.Config{Dimension: 2})
	defer db.Close()

	// Prepare Batch Data
	ids := []string{"A", "B", "C"}
	vecs := [][]float32{
		{1.0, 0.0},
		{0.0, 1.0},
		{1.0, 1.0},
	}
	metas := []map[string]any{
		{"type": "primary"},
		{"type": "secondary"},
		{"type": "mixed"},
	}

	// Execute Batch Insert
	if err := db.InsertBatch(ids, vecs, metas); err != nil {
		log.Fatal(err)
	}

	count, _ := db.Count()
	fmt.Printf("Successfully inserted %d documents.\n", count)

	// Search verification
	results, _ := db.Search([]float32{1.0, 0.0}, 1, nil)
	fmt.Printf("Top result: %s\n", results[0].ID)

	// Output:
	// Successfully inserted 3 documents.
	// Top result: A
}

// ExampleDB_Vacuum shows how to optimize the database storage and index
// after deleting data.
func ExampleDB_Vacuum() {
	// Setup
	tmpDir, _ := os.MkdirTemp("", "nanovec_vacuum_*")
	defer os.RemoveAll(tmpDir)
	dbPath := filepath.Join(tmpDir, "vacuum.db")

	db, _ := nanovec.Open(dbPath, &nanovec.Config{Dimension: 2})
	defer db.Close()

	// Insert and Delete
	db.Insert("temp_item", []float32{0.5, 0.5}, nil)
	db.Delete("temp_item")

	// Run Vacuum to reclaim space and rebuild index
	// In a real application, this removes "ghost" nodes from the HNSW graph.
	if err := db.Vacuum(); err != nil {
		log.Fatal(err)
	}

	fmt.Println("Vacuum completed.")

	// Output:
	// Vacuum completed.
}
