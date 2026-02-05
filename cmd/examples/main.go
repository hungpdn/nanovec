package main

import (
	"fmt"
	"log"

	"github.com/hungpdn/nanovec"
	"github.com/hungpdn/nanovec/pkg/types"
)

func main() {

	fmt.Println("--- 1. Initializing Database ---")
	path := "./data/mydata.db"
	cfg := nanovec.Config{
		Dimension: 3,
	}

	db, err := nanovec.Open(path, &cfg)
	if err != nil {
		log.Fatal("Open DB failed:", err)
	}
	defer db.Close()

	fmt.Println("\n--- 2. Inserting Data ---")
	err = db.Insert("doc1", []float32{0.1, 0.2, 0.3}, map[string]interface{}{
		"title":    "Hello World",
		"category": "greeting",
	})
	if err != nil {
		log.Fatal("Insert doc1 failed:", err)
	}

	err = db.Insert("doc2", []float32{0.8, 0.9, 0.1}, map[string]interface{}{
		"title":    "Advanced Go",
		"category": "tech",
	})
	if err != nil {
		log.Fatal("Insert doc2 failed:", err)
	}
	fmt.Println("-> Inserted doc1 and doc2")

	fmt.Println("\n--- 3. Searching (Before Update) ---")
	results, err := db.Search([]float32{0.1, 0.2, 0.35}, 2, nil)
	if err != nil {
		log.Fatal(err)
	}
	printResults(results)

	fmt.Println("\n--- 4. Updating doc1 ---")
	newVec := []float32{0.15, 0.25, 0.35}
	newMeta := map[string]interface{}{
		"title":    "Hello World v2 (Updated)",
		"category": "greeting_updated",
	}

	if db.Exists("doc1") {
		fmt.Println("doc1 already exists. performing update...")
	}

	err = db.Update("doc1", newVec, newMeta)
	if err != nil {
		log.Fatal("Update doc1 failed:", err)
	}
	fmt.Println("-> Updated doc1 successfully")

	fmt.Println("\n--- 5. Searching (After Update) ---")
	results, err = db.Search([]float32{0.1, 0.2, 0.35}, 1, nil)
	if err != nil {
		log.Fatal(err)
	}
	printResults(results)

	fmt.Println("\n--- 6. Deleting doc1 ---")
	err = db.Delete("doc1")
	if err != nil {
		log.Fatal("Delete doc1 failed:", err)
	}
	fmt.Println("-> Deleted doc1")

	fmt.Println("\n--- 7. Searching (After Delete) ---")
	results, err = db.Search([]float32{0.1, 0.2, 0.35}, 5, nil)
	if err != nil {
		log.Fatal(err)
	}

	if len(results) == 0 {
		fmt.Println("-> No results found (Correct!)")
	} else {
		printResults(results)
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
