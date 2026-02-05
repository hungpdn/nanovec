package main

import (
	"fmt"
	"log"
	"os"

	"github.com/hungpdn/nanovec"
	"github.com/hungpdn/nanovec/pkg/types"
)

func main() {
	// Cleanup previous run for clean demo
	_ = os.RemoveAll("./data")
	_ = os.MkdirAll("./data", 0755)

	fmt.Println("--- 1. Initializing Database ---")
	path := "./data/mydata.db"
	cfg := nanovec.Config{
		Dimension: 3,
	}

	db, err := nanovec.Open(path, &cfg)
	if err != nil {
		log.Fatal("Open DB failed: ", err)
	}
	defer db.Close()

	fmt.Println("\n--- 2. Batch Inserting Data ---")
	// Demonstrate the new High-Performance Batch Insert
	ids := []string{"doc1", "doc2", "doc3"}
	vecs := [][]float32{
		{0.1, 0.2, 0.3},
		{0.8, 0.9, 0.1},
		{0.1, 0.2, 0.35},
	}
	metas := []map[string]any{
		{"title": "Hello World", "category": "greeting"},
		{"title": "Advanced Go", "category": "tech"},
		{"title": "Vector DBs", "category": "tech"},
	}

	if err := db.InsertBatch(ids, vecs, metas); err != nil {
		log.Fatal("Batch insert failed:", err)
	}
	fmt.Println("-> Batch Inserted 3 documents")

	fmt.Println("\n--- 3. Searching (category='tech') ---")
	// Define a filter
	techFilter := func(m map[string]any) bool {
		return m["category"] == "tech"
	}

	// Search
	results, err := db.Search([]float32{0.1, 0.2, 0.35}, 5, techFilter)
	if err != nil {
		log.Fatal(err)
	}
	printResults(results)

	fmt.Println("\n--- 4. Updating doc1 ---")
	newMeta := map[string]any{
		"title":    "Hello World v2",
		"category": "greeting_updated",
	}
	// Note: Empty vector slice means "keep existing vector"
	if err := db.Update("doc1", nil, newMeta); err != nil {
		log.Fatal(err)
	}
	fmt.Println("-> Updated doc1 metadata only")

	fmt.Println("\n--- 5. Verify Persistence (Simulate Restart) ---")
	db.Close() // Force save

	// Re-open
	db2, err := nanovec.Open(path, &cfg)
	if err != nil {
		log.Fatal(err)
	}
	defer db2.Close()

	if db2.Exists("doc1") {
		fmt.Println("-> 'doc1' confirmed to exist after restart.")
	}
}

func printResults(results []types.SearchResult) {
	if len(results) == 0 {
		fmt.Println("No results found.")
		return
	}
	for _, res := range results {
		fmt.Printf("   [Found] ID: %-5s | Score: %.4f | Meta: %v\n", res.ID, res.Score, res.Metadata)
	}
}
