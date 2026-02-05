package main

import (
	"fmt"
	"log"
	"os"

	"github.com/hungpdn/nanovec"
)

func main() {
	_ = os.RemoveAll("./data")
	_ = os.MkdirAll("./data", 0755)

	fmt.Println("--- 1. Initializing Database (HNSW Mode) ---")
	path := "./data/mydata.db"

	// Switch to HNSW for large datasets
	cfg := nanovec.Config{
		Dimension:      3,
		IndexType:      nanovec.IndexTypeHNSW,
		M:              16,  // Graph connections
		EfConstruction: 200, // Build accuracy
	}

	db, err := nanovec.Open(path, &cfg)
	if err != nil {
		log.Fatal("Open DB failed: ", err)
	}
	defer db.Close()

	fmt.Println("\n--- 2. Batch Inserting ---")
	ids := []string{"doc1", "doc2", "doc3"}
	vecs := [][]float32{
		{0.1, 0.2, 0.3},
		{0.8, 0.9, 0.1},
		{0.1, 0.2, 0.35},
	}
	metas := []map[string]any{
		{"name": "A"}, {"name": "B"}, {"name": "C"},
	}

	if err := db.InsertBatch(ids, vecs, metas); err != nil {
		log.Fatal(err)
	}
	fmt.Println("-> Inserted 3 docs into HNSW Graph")

	fmt.Println("\n--- 3. Approximate Search ---")
	results, err := db.Search([]float32{0.1, 0.2, 0.35}, 2, nil)
	if err != nil {
		log.Fatal(err)
	}

	for _, res := range results {
		fmt.Printf("   ID: %-5s | Score: %.4f\n", res.ID, res.Score)
	}
}
