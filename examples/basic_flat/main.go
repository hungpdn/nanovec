package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/hungpdn/nanovec"
)

func main() {
	// 1. Setup a temporary directory for safety
	tmpDir, err := os.MkdirTemp("", "nanovec_basic_*")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(tmpDir) // Auto-cleanup on exit

	dbPath := filepath.Join(tmpDir, "vectors.db")
	fmt.Printf("📂 Database initialized at: %s\n", dbPath)

	// 2. Configuration: Dimension=3, Type=Flat (Exact Search)
	cfg := nanovec.Config{
		Dimension: 3,
		IndexType: nanovec.IndexTypeFlat,
	}

	db, err := nanovec.Open(dbPath, &cfg)
	if err != nil {
		log.Fatal("Failed to open DB:", err)
	}
	defer db.Close()

	// 3. Insert Semantic Data
	// Let's pretend dimensions represent: [Redness, Roundness, Sweetness]
	ids := []string{"apple", "banana", "lime"}
	vecs := [][]float32{
		{0.9, 0.9, 0.8}, // Apple: Red, Round, Sweet
		{0.1, 0.2, 0.6}, // Banana: Not Red, Not Round, Sweet
		{0.0, 0.9, 0.1}, // Lime: Not Red, Round, Sour
	}
	metas := []map[string]any{
		{"type": "fruit", "color": "red"},
		{"type": "fruit", "color": "yellow"},
		{"type": "fruit", "color": "green"},
	}

	fmt.Println("🚀 Batch inserting data...")
	if err := db.InsertBatch(ids, vecs, metas); err != nil {
		log.Fatal(err)
	}

	// 4. Search: Find something "Red and Round" (like an Apple)
	query := []float32{0.8, 0.8, 0.0} // Query vector
	fmt.Println("🔍 Searching for 'Red and Round' object...")

	results, err := db.Search(query, 2, nil)
	if err != nil {
		log.Fatal(err)
	}

	for _, res := range results {
		fmt.Printf("   - Found ID: %-10s | Score: %.4f | Meta: %v\n",
			res.ID, res.Score, res.Metadata)
	}
}
