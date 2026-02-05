package main

import (
	"fmt"
	"log"

	"github.com/hungpdn/nanovec"
)

func main() {
	// 1. Initialize database
	cfg := nanovec.Config{
		Path:      "./mydata.db",
		Dimension: 3, // vector 3 dimensions
	}

	db, err := nanovec.Open(cfg)
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	// 2. Add data
	err = db.Insert("doc1", []float32{0.1, 0.2, 0.3}, map[string]interface{}{
		"title": "Hello World",
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Inserted doc1 successfully")

	// 3. Search
	results, err := db.Search([]float32{0.1, 0.2, 0.35}, 1)
	if err != nil {
		log.Fatal(err)
	}

	for _, res := range results {
		fmt.Printf("Found: %s (Score: %.2f) - Meta: %v\n", res.ID, res.Score, res.Metadata)
	}
}
