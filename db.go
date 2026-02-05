package nanovec

import (
	"fmt"
	"sync"

	"github.com/hungpdn/nanovec/internal"
	"github.com/hungpdn/nanovec/internal/index"
	"github.com/hungpdn/nanovec/internal/storage"
	"github.com/hungpdn/nanovec/pkg/types"
)

// Config configuration for DB
type Config struct {
	Path      string // Path to save data
	Dimension int    // Number of vector dimensions (e.g., 1536 for OpenAI)
}

// DB represents the vector database
type DB struct {
	mu      sync.RWMutex // Ensure thread-safe (Concurrency)
	config  Config
	index   internal.VectorIndex // Search engine (in RAM)
	storage internal.Storage     // Storage engine (on disk)
}

// Open initializes the database
func Open(cfg Config) (*DB, error) {
	// 1. Initialize Storage (REAL BoltDB)
	// Lưu file storage ngay cạnh path config, ví dụ "mydata.db.store"
	storePath := cfg.Path + ".store"
	store, err := storage.NewBoltStorage(storePath)
	if err != nil {
		return nil, err
	}

	// 2. Initialize Index (FlatIndex)
	idx := index.NewFlatIndex(cfg.Dimension)

	// Tự động Load index cũ lên RAM nếu file tồn tại
	// (Giả sử bạn đã implement hàm Load trong FlatIndex như bài trước)
	indexPath := cfg.Path + ".idx"
	_ = idx.Load(indexPath) // Bỏ qua lỗi nếu file chưa tồn tại (lần chạy đầu)

	return &DB{
		config:  cfg,
		index:   idx,
		storage: store, // <--- Sử dụng store thật
	}, nil
}

// Insert adds a new vector
func (db *DB) Insert(id string, vec []float32, meta map[string]interface{}) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if len(vec) != db.config.Dimension {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d", db.config.Dimension, len(vec))
	}

	doc := &types.Document{
		ID:       id,
		Vector:   types.Vector(vec),
		Metadata: meta,
	}

	// 1. Save to disk first (WAL principle)
	if err := db.storage.Put(doc); err != nil {
		return err
	}

	// 2. Update the index in RAM
	if err := db.index.Add(id, types.Vector(vec)); err != nil {
		// !!! ROLLBACK !!!
		// Nếu thêm vào RAM lỗi, ta phải xóa ngay dữ liệu vừa ghi xuống đĩa
		// Để đảm bảo "Một là có cả 2, Hai là không có gì"
		fmt.Printf("Index failed, rolling back storage for ID: %s\n", id)

		// Cố gắng xóa. Nếu xóa fail nốt thì... chịu (Critical Error - Cần log ra file để sysadmin xử lý)
		if rollbackErr := db.storage.Delete(id); rollbackErr != nil {
			return fmt.Errorf("CRITICAL: Index failed (%v) AND Rollback failed (%v)", err, rollbackErr)
		}

		return fmt.Errorf("failed to add to index, rolled back storage: %w", err)
	}

	return nil
}

// Search finds similar vectors
func (db *DB) Search(query []float32, k int) ([]types.SearchResult, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	// 1. Find ID and Score from Index (RAM) - Fast
	ids, scores, err := db.index.Search(types.Vector(query), k)
	if err != nil {
		return nil, err
	}

	// 2. Get Metadata from Storage (Disk) - Slower but doesn't consume RAM
	results := make([]types.SearchResult, 0, len(ids))
	for i, id := range ids {
		doc, err := db.storage.Get(id)
		if err != nil {
			continue // Skip if error
		}
		results = append(results, types.SearchResult{
			ID:       doc.ID,
			Score:    scores[i],
			Metadata: doc.Metadata,
		})
	}

	return results, nil
}

// Update cập nhật document.
// Nếu vector thay đổi, ta phải thực hiện quy trình Delete-then-Insert trên Index.
func (db *DB) Update(id string, newVec []float32, newMeta map[string]interface{}) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	// 1. Kiểm tra xem document có tồn tại không để lấy dữ liệu cũ
	oldDoc, err := db.storage.Get(id)
	if err != nil {
		return fmt.Errorf("document not found: %s", id)
	}

	// 2. Chuẩn bị dữ liệu mới (Merge dữ liệu cũ và mới)
	finalVec := oldDoc.Vector
	vectorChanged := false

	// Nếu user truyền vector mới vào
	if len(newVec) > 0 {
		if len(newVec) != db.config.Dimension {
			return fmt.Errorf("dimension mismatch")
		}
		finalVec = types.Vector(newVec)
		vectorChanged = true
	}

	// Nếu user truyền metadata mới vào
	finalMeta := oldDoc.Metadata
	if newMeta != nil {
		finalMeta = newMeta
	}

	newDoc := &types.Document{
		ID:       id,
		Vector:   finalVec,
		Metadata: finalMeta,
	}

	// 3. Cập nhật Storage (BoltDB tự động ghi đè - Upsert)
	if err := db.storage.Put(newDoc); err != nil {
		return err
	}

	// 4. Cập nhật Index (Phần quan trọng nhất)
	if vectorChanged {
		// A. Xóa vector cũ khỏi Index (Bạn cần thêm hàm Delete cho Index Interface nữa)
		if err := db.index.Delete(id); err != nil {
			return fmt.Errorf("failed to delete old vector from index: %v", err)
		}

		// B. Thêm vector mới vào Index
		if err := db.index.Add(id, finalVec); err != nil {
			return fmt.Errorf("failed to add new vector to index: %v", err)
		}
	}
	// Nếu chỉ sửa metadata, Index không cần làm gì cả

	return nil
}

// Delete xóa document hoàn toàn khỏi database (cả trên đĩa và trong RAM)
func (db *DB) Delete(id string) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	// 0. (Chuẩn bị Rollback) Lấy dữ liệu cũ trước khi xóa
	oldDoc, err := db.storage.Get(id)
	if err != nil {
		return err // Không tìm thấy trong storage thì coi như lỗi luôn
	}

	// 1. Xóa khỏi Storage (BoltDB)
	// Việc này đảm bảo dữ liệu biến mất khỏi đĩa cứng
	if err := db.storage.Delete(id); err != nil {
		return fmt.Errorf("failed to delete from storage: %w", err)
	}

	// 2. Xóa khỏi Index (RAM)
	// Việc này đảm bảo không tìm thấy vector đó nữa trong phiên làm việc hiện tại
	if err := db.index.Delete(id); err != nil {
		// !!! ROLLBACK !!!
		// Index xóa không được -> Phải khôi phục lại dữ liệu trên Storage
		fmt.Printf("Index delete failed, restoring storage for ID: %s\n", id)

		if rollbackErr := db.storage.Put(oldDoc); rollbackErr != nil {
			return fmt.Errorf("CRITICAL: Index delete failed (%v) AND Restore failed (%v)", err, rollbackErr)
		}

		return fmt.Errorf("failed to delete from index, state restored: %w", err)
	}

	return nil
}

// Close closes the connection
// Close: Cần lưu Index xuống đĩa trước khi đóng Storage
func (db *DB) Close() error {
	// 1. Lưu Index (RAM -> Disk)
	indexPath := db.config.Path + ".idx"
	if err := db.index.Save(indexPath); err != nil {
		return fmt.Errorf("failed to save index: %v", err)
	}

	// 2. Đóng Storage (BoltDB)
	return db.storage.Close()
}
