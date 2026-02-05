package main

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"

	"github.com/hungpdn/nanovec"
)

func main() {
	tmpDir, _ := os.MkdirTemp("", "nanovec_hnsw_*")
	defer os.RemoveAll(tmpDir)
	dbPath := filepath.Join(tmpDir, "hnsw.db")

	// 2. Configure HNSW for high accuracy
	cfg := nanovec.Config{
		Dimension:      64,
		IndexType:      nanovec.IndexTypeHNSW,
		M:              32,  // More connections = Better recall, slower insert
		EfConstruction: 400, // Deeper search during build = Better graph quality
	}

	db, err := nanovec.Open(dbPath, &cfg)
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	// 3. Insert Data
	fmt.Println("🚀 Building HNSW Graph with 5,000 vectors...")
	count := 5000
	ids := make([]string, count)
	vecs := make([][]float32, count)
	metas := make([]map[string]any, count)

	for i := 0; i < count; i++ {
		ids[i] = fmt.Sprintf("node_%d", i)
		vecs[i] = randomVector(64)
	}

	if err := db.InsertBatch(ids, vecs, metas); err != nil {
		log.Fatal(err)
	}
	fmt.Println("✅ Insert complete.")

	// 4. Approximate Nearest Neighbor Search
	query := randomVector(64)
	fmt.Println("🔍 Performing Approximate Search (ANN)...")
	results, _ := db.Search(query, 5, nil)

	for i, res := range results {
		fmt.Printf("   #%d: ID=%s (Score=%.4f)\n", i+1, res.ID, res.Score)
	}
}

func randomVector(dim int) []float32 {
	v := make([]float32, dim)
	for i := 0; i < dim; i++ {
		v[i] = rand.Float32()
	}
	return v
}
