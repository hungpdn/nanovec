package main

import (
	"fmt"
	"log"
	"os"

	"github.com/hungpdn/nanovec"
	"github.com/hungpdn/nanovec/pkg/types"
)

func main() {
	// Xóa dữ liệu cũ để chạy sạch từ đầu (Optional)
	os.Remove("./mydata.db.store")
	os.Remove("./mydata.db.idx")

	// 1. Khởi tạo Database
	fmt.Println("--- 1. Initializing Database ---")
	cfg := nanovec.Config{
		Path:      "./mydata.db",
		Dimension: 3, // Vector 3 chiều
	}

	db, err := nanovec.Open(cfg)
	if err != nil {
		log.Fatal("Open DB failed:", err)
	}
	// Đảm bảo đóng DB khi xong để lưu dữ liệu xuống đĩa
	defer db.Close()

	// 2. Insert (Thêm mới)
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

	// 3. Search (Tìm kiếm)
	fmt.Println("\n--- 3. Searching (Before Update) ---")
	// Tìm vector gần doc1 nhất ([0.1, 0.2, 0.35])
	results, err := db.Search([]float32{0.1, 0.2, 0.35}, 2)
	if err != nil {
		log.Fatal(err)
	}
	printResults(results)

	// 4. Update (Cập nhật)
	fmt.Println("\n--- 4. Updating doc1 ---")
	// Kịch bản: doc1 thay đổi nội dung -> Vector thay đổi, Metadata thay đổi
	newVec := []float32{0.15, 0.25, 0.35} // Vector mới (vẫn gần query cũ)
	newMeta := map[string]interface{}{
		"title":    "Hello World v2 (Updated)",
		"category": "greeting_updated",
	}

	// Gọi hàm Update (Bạn cần đảm bảo đã thêm hàm này vào db.go)
	err = db.Update("doc1", newVec, newMeta)
	if err != nil {
		log.Fatal("Update doc1 failed:", err)
	}
	fmt.Println("-> Updated doc1 successfully")

	// Search lại để kiểm chứng
	fmt.Println("\n--- 5. Searching (After Update) ---")
	results, err = db.Search([]float32{0.1, 0.2, 0.35}, 1)
	if err != nil {
		log.Fatal(err)
	}
	// Kết quả phải hiện ra title mới "Hello World v2"
	printResults(results)

	// 5. Delete (Xóa)
	fmt.Println("\n--- 6. Deleting doc1 ---")
	// Gọi hàm Delete (Bạn cần đảm bảo đã thêm hàm này vào db.go)
	err = db.Delete("doc1")
	if err != nil {
		log.Fatal("Delete doc1 failed:", err)
	}
	fmt.Println("-> Deleted doc1")

	// Search lại để kiểm chứng
	fmt.Println("\n--- 7. Searching (After Delete) ---")
	results, err = db.Search([]float32{0.1, 0.2, 0.35}, 5)
	if err != nil {
		log.Fatal(err)
	}

	if len(results) == 0 {
		fmt.Println("-> No results found (Correct!)")
	} else {
		// Doc1 đã xóa, nên kết quả gần nhất bây giờ có thể là doc2 hoặc rỗng (nếu doc2 quá xa)
		printResults(results)
	}
}

// Hàm phụ trợ để in kết quả đẹp hơn
func printResults(results []types.SearchResult) {
	if len(results) == 0 {
		fmt.Println("No results found.")
		return
	}
	for _, res := range results {
		fmt.Printf("   [Found] ID: %-5s | Score: %.4f | Meta: %v\n", res.ID, res.Score, res.Metadata)
	}
}
