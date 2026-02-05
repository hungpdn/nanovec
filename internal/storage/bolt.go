package storage

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"time"

	"github.com/hungpdn/nanovec/pkg/types"
	"go.etcd.io/bbolt"
)

const bucketName = "documents"

// BoltStorage implement interface Storage use bbolt
type BoltStorage struct {
	db *bbolt.DB
}

// NewBoltStorage opens the database and creates the bucket if not exists
func NewBoltStorage(path string) (*BoltStorage, error) {
	db, err := bbolt.Open(path, 0600, &bbolt.Options{Timeout: 1 * time.Second})
	if err != nil {
		return nil, fmt.Errorf("could not open bolt db: %v", err)
	}

	err = db.Update(func(tx *bbolt.Tx) error {
		_, err := tx.CreateBucketIfNotExists([]byte(bucketName))
		return err
	})
	if err != nil {
		return nil, fmt.Errorf("could not create bucket: %v", err)
	}

	return &BoltStorage{db: db}, nil
}

// Put saves document to disk (Upsert)
func (s *BoltStorage) Put(doc *types.Document) error {
	return s.db.Update(func(tx *bbolt.Tx) error {
		b := tx.Bucket([]byte(bucketName))

		var buf bytes.Buffer
		enc := gob.NewEncoder(&buf)
		if err := enc.Encode(doc); err != nil {
			return fmt.Errorf("failed to encode document: %w", err)
		}

		return b.Put([]byte(doc.ID), buf.Bytes())
	})
}

// PutBatch saves multiple documents in a SINGLE transaction (High Performance)
func (s *BoltStorage) PutBatch(docs []*types.Document) error {
	return s.db.Update(func(tx *bbolt.Tx) error {
		b := tx.Bucket([]byte(bucketName))

		// Reuse buffer to reduce allocations
		var buf bytes.Buffer

		for _, doc := range docs {
			buf.Reset()
			enc := gob.NewEncoder(&buf)
			if err := enc.Encode(doc); err != nil {
				return fmt.Errorf("failed to encode document %s: %w", doc.ID, err)
			}

			if err := b.Put([]byte(doc.ID), buf.Bytes()); err != nil {
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
		b := tx.Bucket([]byte(bucketName))
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
		b := tx.Bucket([]byte(bucketName))
		return b.Delete([]byte(id))
	})
}

// Scan iterates over all documents and executes fn for each.
func (s *BoltStorage) Scan(fn func(doc *types.Document) error) error {
	return s.db.View(func(tx *bbolt.Tx) error {
		b := tx.Bucket([]byte(bucketName))
		c := b.Cursor()

		for k, v := c.First(); k != nil; k, v = c.Next() {
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

// Has checks if a document exists without deserializing it.
func (s *BoltStorage) Has(id string) bool {
	var found bool
	_ = s.db.View(func(tx *bbolt.Tx) error {
		b := tx.Bucket([]byte(bucketName))
		if b == nil {
			return nil
		}
		if v := b.Get([]byte(id)); v != nil {
			found = true
		}
		return nil
	})
	return found
}

// Count returns the total number of items in the bucket
func (s *BoltStorage) Count() (int, error) {
	var count int
	err := s.db.View(func(tx *bbolt.Tx) error {
		b := tx.Bucket([]byte(bucketName))
		if b == nil {
			return nil
		}
		// KeyN is efficient (reads metadata) and doesn't scan the whole tree
		count = b.Stats().KeyN
		return nil
	})
	return count, err
}

// Close closes the database connection
func (s *BoltStorage) Close() error {
	return s.db.Close()
}
