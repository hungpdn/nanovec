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
	// 1. Setup Temp Dir
	tmpDir, err := os.MkdirTemp("", "nanovec_sq8_*")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	dbPath := filepath.Join(tmpDir, "quantized.db")
	fmt.Printf("📂 Database (SQ8 Mode) at: %s\n", dbPath)

	// 2. Enable Quantization
	cfg := nanovec.Config{
		Dimension:    128, // Typical embedding size
		IndexType:    nanovec.IndexTypeFlat,
		Quantization: true, // <--- KEY FEATURE: Reduces RAM usage by 4x
	}

	db, err := nanovec.Open(dbPath, &cfg)
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	// 3. Generate Dummy Data (1,000 vectors)
	count := 1000
	fmt.Printf("⚡ Generating and inserting %d vectors...\n", count)

	ids := make([]string, count)
	vecs := make([][]float32, count)
	metas := make([]map[string]any, count)

	for i := 0; i < count; i++ {
		ids[i] = fmt.Sprintf("vec_%d", i)
		vecs[i] = randomVector(128)
		metas[i] = map[string]any{"idx": i}
	}

	if err := db.InsertBatch(ids, vecs, metas); err != nil {
		log.Fatal(err)
	}

	// 4. Verify Accuracy
	// Search for the first vector. Ideally, it should be the top result.
	// Note: Scores might not be exactly 1.0 due to compression loss, but recall is usually high.
	target := vecs[0]
	fmt.Println("🔍 Verifying accuracy by searching for the first vector...")

	results, _ := db.Search(target, 1, nil)

	if len(results) > 0 {
		top := results[0]
		fmt.Printf("   - Target ID: vec_0\n")
		fmt.Printf("   - Found ID:  %-10s | Score: %.4f\n", top.ID, top.Score)

		if top.ID == "vec_0" {
			fmt.Println("✅ Success: SQ8 found the correct vector despite compression!")
		} else {
			fmt.Println("⚠️ Warning: Top result mismatch (expected with lossy compression)")
		}
	}
}

func randomVector(dim int) []float32 {
	v := make([]float32, dim)
	for i := 0; i < dim; i++ {
		v[i] = rand.Float32()
	}
	return v
}
