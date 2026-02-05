package storage

import (
	"bytes"
	"encoding/binary"
	"encoding/gob"
	"fmt"
	"time"

	"github.com/hungpdn/nanovec/pkg/types"
	"go.etcd.io/bbolt"
)

const (
	bucketDocuments = "documents"
	bucketMeta      = "metadata"
	keyDocCount     = "doc_count"
	keyDbVersion    = "db_version"
)

// BoltStorage implement interface Storage use bbolt
type BoltStorage struct {
	db *bbolt.DB
}

// NewBoltStorage opens the database and creates the buckets if not exists
func NewBoltStorage(path string) (*BoltStorage, error) {
	db, err := bbolt.Open(path, 0600, &bbolt.Options{
		Timeout:        1 * time.Second,
		NoFreelistSync: true, // Improve write performance
	})
	if err != nil {
		return nil, fmt.Errorf("could not open bolt db: %v", err)
	}

	err = db.Update(func(tx *bbolt.Tx) error {
		_, _ = tx.CreateBucketIfNotExists([]byte(bucketDocuments))
		_, _ = tx.CreateBucketIfNotExists([]byte(bucketMeta))
		return nil
	})

	if err != nil {
		_ = db.Close()
		return nil, err
	}

	return &BoltStorage{db: db}, nil
}

// GetVersion returns the current modification sequence of the storage
func (s *BoltStorage) GetVersion() (uint64, error) {
	var ver uint64
	err := s.db.View(func(tx *bbolt.Tx) error {
		b := tx.Bucket([]byte(bucketMeta))
		v := b.Get([]byte(keyDbVersion))
		if v != nil {
			ver = binary.LittleEndian.Uint64(v)
		}
		return nil
	})
	return ver, err
}

// incVersion increments the database version
func (s *BoltStorage) incVersion(tx *bbolt.Tx) error {
	b := tx.Bucket([]byte(bucketMeta))
	seq, _ := b.NextSequence()

	buf := make([]byte, 8)
	binary.LittleEndian.PutUint64(buf, seq)
	return b.Put([]byte(keyDbVersion), buf)
}

// Put saves document to disk (Upsert)
func (s *BoltStorage) Put(doc *types.Document) error {
	return s.db.Update(func(tx *bbolt.Tx) error {
		bDocs := tx.Bucket([]byte(bucketDocuments))
		bMeta := tx.Bucket([]byte(bucketMeta))

		// Check if ID exists to maintain accurate count
		if bDocs.Get([]byte(doc.ID)) == nil {
			if err := s.incrementCount(bMeta, 1); err != nil {
				return err
			}
		}
		// Optimize: Use custom binary encoding instead of Gob for speed
		var buf bytes.Buffer
		enc := gob.NewEncoder(&buf)
		if err := enc.Encode(doc); err != nil {
			return err
		}

		if err := bDocs.Put([]byte(doc.ID), buf.Bytes()); err != nil {
			return err
		}

		return s.incVersion(tx)
	})
}

// PutBatch saves multiple documents in a SINGLE transaction
func (s *BoltStorage) PutBatch(docs []*types.Document) error {
	return s.db.Update(func(tx *bbolt.Tx) error {
		bDocs := tx.Bucket([]byte(bucketDocuments))
		bMeta := tx.Bucket([]byte(bucketMeta))

		var newItems uint64
		// Track IDs seen in this batch to prevent double-counting.
		// Even if the input slice has ["A", "A"], we only count "A" as new once.
		visited := make(map[string]bool, len(docs))

		for _, doc := range docs {
			// Only check for "newness" if we haven't processed this ID in this batch yet.
			// (Last Write Wins logic still applies for the Put itself, but we shouldn't increment count twice)
			if !visited[doc.ID] {
				if bDocs.Get([]byte(doc.ID)) == nil {
					newItems++
				}
				visited[doc.ID] = true
			}

			// Use a fresh buffer for each encode to prevent gob stream corruption (EOF errors)
			var buf bytes.Buffer
			enc := gob.NewEncoder(&buf)
			if err := enc.Encode(doc); err != nil {
				return fmt.Errorf("encode %s failed: %w", doc.ID, err)
			}

			if err := bDocs.Put([]byte(doc.ID), buf.Bytes()); err != nil {
				return err
			}
		}

		if err := s.incVersion(tx); err != nil {
			return err
		}

		if newItems > 0 {
			if err := s.incrementCount(bMeta, newItems); err != nil {
				return err
			}
		}
		return nil
	})
}

// Get read document from disk
func (s *BoltStorage) Get(id string) (*types.Document, error) {
	var doc types.Document

	err := s.db.View(func(tx *bbolt.Tx) error {
		b := tx.Bucket([]byte(bucketDocuments))
		data := b.Get([]byte(id))

		if data == nil {
			return fmt.Errorf("document not found: %s", id)
		}

		buf := bytes.NewBuffer(data)
		dec := gob.NewDecoder(buf)
		if err := dec.Decode(&doc); err != nil {
			return fmt.Errorf("failed to decode document: %w", err)
		}
		return nil
	})

	if err != nil {
		return nil, err
	}

	return &doc, nil
}

// Delete remove document
func (s *BoltStorage) Delete(id string) error {
	return s.db.Update(func(tx *bbolt.Tx) error {
		bDocs := tx.Bucket([]byte(bucketDocuments))
		bMeta := tx.Bucket([]byte(bucketMeta))

		if bDocs.Get([]byte(id)) != nil {
			if err := s.decrementCount(bMeta, 1); err != nil {
				return err
			}
			return bDocs.Delete([]byte(id))
		}
		return nil
	})
}

// Scan iterates over all documents
func (s *BoltStorage) Scan(fn func(doc *types.Document) error) error {
	return s.db.View(func(tx *bbolt.Tx) error {
		b := tx.Bucket([]byte(bucketDocuments))
		c := b.Cursor()

		for k, v := c.First(); k != nil; k, v = c.Next() {
			if len(v) == 0 {
				continue
			}

			var doc types.Document
			buf := bytes.NewBuffer(v)
			dec := gob.NewDecoder(buf)
			if err := dec.Decode(&doc); err != nil {
				return fmt.Errorf("corrupt data for id %s: %v", k, err)
			}
			if err := fn(&doc); err != nil {
				return err
			}
		}
		return nil
	})
}

// Has checks if a document exists
func (s *BoltStorage) Has(id string) bool {
	var found bool
	_ = s.db.View(func(tx *bbolt.Tx) error {
		b := tx.Bucket([]byte(bucketDocuments))
		if b != nil && b.Get([]byte(id)) != nil {
			found = true
		}
		return nil
	})
	return found
}

// Count returns the total number of items using the O(1) metadata counter
func (s *BoltStorage) Count() (int, error) {
	var count int
	err := s.db.View(func(tx *bbolt.Tx) error {
		bMeta := tx.Bucket([]byte(bucketMeta))
		if bMeta == nil {
			return nil
		}
		val := bMeta.Get([]byte(keyDocCount))
		if val != nil {
			count = int(binary.LittleEndian.Uint64(val))
		}
		return nil
	})
	return count, err
}

// Helpers for counter
func (s *BoltStorage) incrementCount(b *bbolt.Bucket, delta uint64) error {
	val := b.Get([]byte(keyDocCount))
	var count uint64
	if val != nil {
		count = binary.LittleEndian.Uint64(val)
	}
	count += delta
	buf := make([]byte, 8)
	binary.LittleEndian.PutUint64(buf, count)
	return b.Put([]byte(keyDocCount), buf)
}

func (s *BoltStorage) decrementCount(b *bbolt.Bucket, delta uint64) error {
	val := b.Get([]byte(keyDocCount))
	var count uint64
	if val != nil {
		count = binary.LittleEndian.Uint64(val)
	}
	if delta > count {
		count = 0
	} else {
		count -= delta
	}
	buf := make([]byte, 8)
	binary.LittleEndian.PutUint64(buf, count)
	return b.Put([]byte(keyDocCount), buf)
}

func (s *BoltStorage) Close() error {
	return s.db.Close()
}
