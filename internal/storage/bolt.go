package storage

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
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

// NewBoltStorage opens the database
func NewBoltStorage(path string) (*BoltStorage, error) {
	db, err := bbolt.Open(path, 0600, &bbolt.Options{
		Timeout:        1 * time.Second,
		NoFreelistSync: false, // Set false for reliability (prevent corruption on power loss)
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

// --- High Performance Serialization Helpers ---

// serializeDocument encodes document to binary format efficiently
// Format: [Dim(4b)][Vector(dim*4b)][JsonMetadata(...)]
func serializeDocument(doc *types.Document) ([]byte, error) {
	dim := len(doc.Vector)
	vecSize := 4 + dim*4 // 4 bytes for header + data

	// 1. Pre-allocate buffer for Vector Part
	buf := make([]byte, vecSize)

	// 2. Write Dimension
	binary.LittleEndian.PutUint32(buf[0:4], uint32(dim))

	// 3. Write Vector Data (Fast Loop, No Reflection)
	offset := 4
	for _, v := range doc.Vector {
		bits := math.Float32bits(v)
		binary.LittleEndian.PutUint32(buf[offset:offset+4], bits)
		offset += 4
	}

	// 4. Encode Metadata (Use JSON for compatibility & safety)
	if len(doc.Metadata) > 0 {
		metaBytes, err := json.Marshal(doc.Metadata)
		if err != nil {
			return nil, fmt.Errorf("metadata encode failed: %w", err)
		}
		buf = append(buf, metaBytes...)
	}
	return buf, nil
}

// deserializeDocument decodes binary data back to Document
func deserializeDocument(id string, data []byte) (*types.Document, error) {
	if len(data) < 4 {
		return nil, fmt.Errorf("data too short")
	}

	// 1. Read Dimension
	dim := int(binary.LittleEndian.Uint32(data[0:4]))

	// Safety check
	expectedVecSize := 4 + dim*4
	if len(data) < expectedVecSize {
		return nil, fmt.Errorf("data corrupted or truncated")
	}

	// 2. Read Vector
	vec := make(types.Vector, dim)
	offset := 4
	for i := 0; i < dim; i++ {
		bits := binary.LittleEndian.Uint32(data[offset : offset+4])
		vec[i] = math.Float32frombits(bits)
		offset += 4
	}

	// 3. Read Metadata (if any bytes left)
	var meta map[string]any
	if len(data) > expectedVecSize {
		// Just unmarshal the rest of the slice
		// Note: Numbers will be unmarshaled as float64 by default in JSON
		if err := json.Unmarshal(data[expectedVecSize:], &meta); err != nil {
			return nil, fmt.Errorf("metadata decode failed: %w", err)
		}
	}

	return &types.Document{
		ID:       id, // ID comes from the BoltDB Key
		Vector:   vec,
		Metadata: meta,
	}, nil
}

// --- Storage Implementation ---

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
func (s *BoltStorage) incVersion(tx *bbolt.Tx) (uint64, error) {
	b := tx.Bucket([]byte(bucketMeta))
	seq, _ := b.NextSequence()
	buf := make([]byte, 8)
	binary.LittleEndian.PutUint64(buf, seq)
	return seq, b.Put([]byte(keyDbVersion), buf)
}

// Put saves document to disk (Upsert) AND returns the new Version
func (s *BoltStorage) Put(doc *types.Document) (uint64, error) {
	var newSeq uint64
	err := s.db.Update(func(tx *bbolt.Tx) error {
		bDocs := tx.Bucket([]byte(bucketDocuments))
		bMeta := tx.Bucket([]byte(bucketMeta))

		if bDocs.Get([]byte(doc.ID)) == nil {
			if err := s.incrementCount(bMeta, 1); err != nil {
				return err
			}
		}

		data, err := serializeDocument(doc)
		if err != nil {
			return err
		}

		if err := bDocs.Put([]byte(doc.ID), data); err != nil {
			return err
		}

		newSeq, err = s.incVersion(tx)
		return err
	})

	return newSeq, err
}

// PutBatch saves multiple documents AND returns the new Version
func (s *BoltStorage) PutBatch(docs []*types.Document) (uint64, error) {
	// 1. PRE-SERIALIZATION (Outside Lock) -> CPU bound
	docData := make([][]byte, len(docs))
	for i, doc := range docs {
		data, err := serializeDocument(doc)
		if err != nil {
			return 0, err
		}
		docData[i] = data
	}

	var newSeq uint64
	// 2. WRITE TRANSACTION (Inside Lock) -> IO bound
	err := s.db.Update(func(tx *bbolt.Tx) error {
		bDocs := tx.Bucket([]byte(bucketDocuments))
		bMeta := tx.Bucket([]byte(bucketMeta))

		var newItems uint64
		// Track IDs seen in this batch to prevent double-counting.
		// Even if the input slice has ["A", "A"], we only count "A" as new once.
		visited := make(map[string]bool, len(docs))

		for i, doc := range docs {
			// Only check for "newness" if we haven't processed this ID in this batch yet.
			// (Last Write Wins logic still applies for the Put itself, but we shouldn't increment count twice)
			if !visited[doc.ID] {
				if bDocs.Get([]byte(doc.ID)) == nil {
					newItems++
				}
				visited[doc.ID] = true
			}

			if err := bDocs.Put([]byte(doc.ID), docData[i]); err != nil {
				return err
			}
		}

		if newItems > 0 {
			if err := s.incrementCount(bMeta, newItems); err != nil {
				return err
			}
		}

		var err error
		newSeq, err = s.incVersion(tx)
		return err
	})

	return newSeq, err
}

// Get read document from disk
func (s *BoltStorage) Get(id string) (*types.Document, error) {
	var doc *types.Document

	err := s.db.View(func(tx *bbolt.Tx) error {
		b := tx.Bucket([]byte(bucketDocuments))
		data := b.Get([]byte(id))

		if data == nil {
			return fmt.Errorf("document not found: %s", id)
		}

		var err error
		doc, err = deserializeDocument(id, data)
		if err != nil {
			return err
		}
		return nil
	})

	if err != nil {
		return nil, err
	}
	return doc, nil
}

// Delete remove document
func (s *BoltStorage) Delete(id string) (uint64, error) {
	var newSeq uint64
	err := s.db.Update(func(tx *bbolt.Tx) error {
		bDocs := tx.Bucket([]byte(bucketDocuments))
		bMeta := tx.Bucket([]byte(bucketMeta))

		if bDocs.Get([]byte(id)) == nil {
			val := bMeta.Get([]byte(keyDbVersion))
			if val != nil {
				newSeq = binary.LittleEndian.Uint64(val)
			}
			return nil
		}

		if err := s.decrementCount(bMeta, 1); err != nil {
			return err
		}
		if err := bDocs.Delete([]byte(id)); err != nil {
			return err
		}
		var err error
		newSeq, err = s.incVersion(tx)
		return err
	})

	return newSeq, err
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

			doc, err := deserializeDocument(string(k), v)
			if err != nil {
				return fmt.Errorf("corrupt data for id %s: %v", k, err)
			}

			if err := fn(doc); err != nil {
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
