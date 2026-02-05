package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/hungpdn/nanovec"
)

func main() {
	tmpDir, _ := os.MkdirTemp("", "nanovec_persist_*")
	defer os.RemoveAll(tmpDir) // Remove this if you want to inspect files manually
	dbPath := filepath.Join(tmpDir, "persist.db")

	cfg := nanovec.Config{Dimension: 2, IndexType: nanovec.IndexTypeFlat}

	// --- Session 1: Create and Modify ---
	fmt.Println("--- Session 1: Writing Data ---")
	{
		db, _ := nanovec.Open(dbPath, &cfg)

		// Insert A, B, C
		db.Insert("A", []float32{1.0, 0.0}, nil)
		db.Insert("B", []float32{0.0, 1.0}, nil)
		db.Insert("C", []float32{1.0, 1.0}, nil)

		// Delete B
		fmt.Println("🗑️  Deleting 'B'...")
		db.Delete("B")

		// Vacuum (Rebuild index to remove 'B' fully and optimize file size)
		fmt.Println("🧹 Running Vacuum...")
		if err := db.Vacuum(); err != nil {
			log.Fatal(err)
		}

		db.Close()
	}

	// --- Session 2: Restart and Verify ---
	fmt.Println("\n--- Session 2: Restarting DB ---")
	{
		db, err := nanovec.Open(dbPath, &cfg)
		if err != nil {
			log.Fatal(err)
		}
		defer db.Close()

		// Verify B is gone
		if db.Exists("B") {
			log.Fatal("❌ Error: 'B' should have been deleted!")
		} else {
			fmt.Println("✅ Verified: 'B' is deleted.")
		}

		// Verify A exists
		if db.Exists("A") {
			fmt.Println("✅ Verified: 'A' persists.")
		}

		// Search
		results, _ := db.Search([]float32{1.0, 0.0}, 5, nil)
		fmt.Printf("🔍 Search found %d items (Expected 2: A and C)\n", len(results))
	}
}
