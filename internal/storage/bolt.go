package storage

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/hungpdn/nanovec/pkg/types"
	"go.etcd.io/bbolt"
)

const bucketName = "documents"

// BoltStorage implement interface Storage dùng BoltDB
type BoltStorage struct {
	db *bbolt.DB
}

// NewBoltStorage khởi tạo kết nối tới file database
func NewBoltStorage(path string) (*BoltStorage, error) {
	// Mở file db (nếu chưa có sẽ tự tạo)
	// Timeout 1s để tránh treo nếu file đang bị khóa bởi process khác
	db, err := bbolt.Open(path, 0600, &bbolt.Options{Timeout: 1 * time.Second})
	if err != nil {
		return nil, fmt.Errorf("could not open bolt db: %v", err)
	}

	// Tạo bucket (giống như Table trong SQL) nếu chưa tồn tại
	err = db.Update(func(tx *bbolt.Tx) error {
		_, err := tx.CreateBucketIfNotExists([]byte(bucketName))
		return err
	})
	if err != nil {
		return nil, fmt.Errorf("could not create bucket: %v", err)
	}

	return &BoltStorage{db: db}, nil
}

// Put lưu document xuống đĩa (Upsert)
func (s *BoltStorage) Put(doc *types.Document) error {
	return s.db.Update(func(tx *bbolt.Tx) error {
		b := tx.Bucket([]byte(bucketName))

		// Serialize struct thành JSON để lưu (có thể dùng Gob nếu muốn nhanh hơn)
		data, err := json.Marshal(doc)
		if err != nil {
			return err
		}

		// Key là ID, Value là JSON
		return b.Put([]byte(doc.ID), data)
	})
}

// Get đọc document từ đĩa
func (s *BoltStorage) Get(id string) (*types.Document, error) {
	var doc types.Document

	err := s.db.View(func(tx *bbolt.Tx) error {
		b := tx.Bucket([]byte(bucketName))
		data := b.Get([]byte(id))

		if data == nil {
			return fmt.Errorf("document not found: %s", id)
		}

		return json.Unmarshal(data, &doc)
	})

	if err != nil {
		return nil, err
	}

	return &doc, nil
}

// Thêm vào internal/storage/bolt.go
func (s *BoltStorage) Delete(id string) error {
	return s.db.Update(func(tx *bbolt.Tx) error {
		b := tx.Bucket([]byte(bucketName))
		return b.Delete([]byte(id))
	})
}

// Close đóng kết nối
func (s *BoltStorage) Close() error {
	return s.db.Close()
}
