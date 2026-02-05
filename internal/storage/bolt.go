package storage

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/hungpdn/nanovec/pkg/types"
	"go.etcd.io/bbolt"
)

const (
	bucketDocuments = "documents"
	bucketMeta      = "metadata"
	keyDocCount     = "doc_count"
	keyDbVersion    = "db_version"
	MaxAllowedDim   = 32768 // Limit dimensions to avoid OOM/Corruption
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

	s := &BoltStorage{db: db}
	_ = s.SyncDocCount() // Auto-repair counter on startup
	return s, nil
}

// SyncDocCount ensures the O(1) counter matches the actual B+Tree key count
func (s *BoltStorage) SyncDocCount() error {
	return s.db.Update(func(tx *bbolt.Tx) error {
		bDocs := tx.Bucket([]byte(bucketDocuments))
		bMeta := tx.Bucket([]byte(bucketMeta))

		// Use Bolt's internal bucket stats for high performance counting
		count := uint64(bDocs.Stats().KeyN)

		buf := make([]byte, 8)
		binary.LittleEndian.PutUint64(buf, count)
		return bMeta.Put([]byte(keyDocCount), buf)
	})
}

// GetVersion returns the current database version
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

// Put saves document to disk (Upsert) and returns the new Version
func (s *BoltStorage) Put(doc *types.Document) (uint64, error) {
	var newSeq uint64
	err := s.db.Update(func(tx *bbolt.Tx) error {
		bDocs := tx.Bucket([]byte(bucketDocuments))
		bMeta := tx.Bucket([]byte(bucketMeta))

		isNew := bDocs.Get([]byte(doc.ID)) == nil

		buf := bufferPool.Get().(*bytes.Buffer)
		buf.Reset()
		defer bufferPool.Put(buf)

		if err := serializeDocument(doc, buf); err != nil {
			return err
		}

		if err := bDocs.Put([]byte(doc.ID), buf.Bytes()); err != nil {
			return err
		}

		if isNew {
			if err := incrementCount(bMeta, 1); err != nil {
				return err
			}
		}

		var err error
		newSeq, err = s.incVersion(tx)
		return err
	})

	return newSeq, err
}

// PutBatch saves multiple documents and returns the new Version
func (s *BoltStorage) PutBatch(docs []*types.Document) (uint64, error) {
	type docRange struct {
		id    []byte
		start int
		end   int
	}
	ranges := make([]docRange, len(docs))

	// Get a large buffer from pool to act as our "Arena"
	batchBuf := bufferPool.Get().(*bytes.Buffer)
	batchBuf.Reset()
	defer bufferPool.Put(batchBuf)
	// Pre-serialize outside the transaction to keep DB lock short
	for i, doc := range docs {
		start := batchBuf.Len()
		if err := serializeDocument(doc, batchBuf); err != nil {
			return 0, err
		}
		ranges[i] = docRange{
			id:    []byte(doc.ID),
			start: start,
			end:   batchBuf.Len(),
		}
	}

	var newSeq uint64
	err := s.db.Update(func(tx *bbolt.Tx) error {
		bDocs := tx.Bucket([]byte(bucketDocuments))
		bMeta := tx.Bucket([]byte(bucketMeta))

		allData := batchBuf.Bytes()
		newItemsCount := int64(0)
		seenInBatch := make(map[string]bool)

		for _, r := range ranges {
			idStr := string(r.id)
			if seenInBatch[idStr] {
				if err := bDocs.Put(r.id, allData[r.start:r.end]); err != nil {
					return err
				}
				continue
			}
			seenInBatch[idStr] = true

			if bDocs.Get(r.id) == nil {
				newItemsCount++
			}

			if err := bDocs.Put(r.id, allData[r.start:r.end]); err != nil {
				return err
			}
		}

		if newItemsCount > 0 {
			if err := incrementCount(bMeta, newItemsCount); err != nil {
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
	var currentSeq uint64
	err := s.db.Update(func(tx *bbolt.Tx) error {
		bDocs := tx.Bucket([]byte(bucketDocuments))
		bMeta := tx.Bucket([]byte(bucketMeta))

		idBytes := []byte(id)
		if bDocs.Get(idBytes) == nil {
			v := bMeta.Get([]byte(keyDbVersion))
			if v != nil {
				currentSeq = binary.LittleEndian.Uint64(v)
			}
			return nil
		}

		if err := bDocs.Delete([]byte(id)); err != nil {
			return err
		}

		if err := incrementCount(bMeta, -1); err != nil {
			return err
		}

		var err error
		currentSeq, err = s.incVersion(tx)
		return err
	})

	return currentSeq, err
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

// Close closes the database
func (s *BoltStorage) Close() error {
	return s.db.Close()
}

// bufferPool is used to reduce allocations during serialization
var bufferPool = sync.Pool{
	New: func() any {
		return new(bytes.Buffer)
	},
}

// serializeDocument encodes document to binary format efficiently
// Format: [Dim(4b)][Vector(dim*4b)][JsonMetadata(...)]
func serializeDocument(doc *types.Document, buf *bytes.Buffer) error {
	dim := len(doc.Vector)

	if dim > MaxAllowedDim {
		return fmt.Errorf("vector dimension %d exceeds max allowed %d", dim, MaxAllowedDim)
	}

	var scratch [4]byte
	binary.LittleEndian.PutUint32(scratch[:], uint32(dim))
	buf.Write(scratch[:])

	for _, v := range doc.Vector {
		bits := math.Float32bits(v)
		binary.LittleEndian.PutUint32(scratch[:], bits)
		buf.Write(scratch[:])
	}

	if len(doc.Metadata) > 0 {
		metaBytes, err := json.Marshal(doc.Metadata)
		if err != nil {
			return err
		}
		buf.Write(metaBytes)
	}
	return nil
}

// deserializeDocument decodes binary data back to Document
func deserializeDocument(id string, data []byte) (*types.Document, error) {
	if len(data) < 4 {
		return nil, fmt.Errorf("data too short")
	}

	dim := int(binary.LittleEndian.Uint32(data[0:4]))

	if dim <= 0 || dim > MaxAllowedDim {
		return nil, fmt.Errorf("invalid or corrupted dimension: %d", dim)
	}

	expectedVecSize := 4 + dim*4
	if len(data) < expectedVecSize {
		return nil, fmt.Errorf("data corrupted or truncated")
	}

	vec := make(types.Vector, dim)
	offset := 4
	for i := 0; i < dim; i++ {
		bits := binary.LittleEndian.Uint32(data[offset : offset+4])
		vec[i] = math.Float32frombits(bits)
		offset += 4
	}

	var meta map[string]any
	if len(data) > expectedVecSize {
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

// Helper to safely update the counter
func incrementCount(b *bbolt.Bucket, delta int64) error {
	count := uint64(0)
	if val := b.Get([]byte(keyDocCount)); val != nil {
		count = binary.LittleEndian.Uint64(val)
	}
	if delta < 0 && uint64(-delta) > count {
		count = 0
	} else {
		count = uint64(int64(count) + delta)
	}
	buf := make([]byte, 8)
	binary.LittleEndian.PutUint64(buf, count)
	return b.Put([]byte(keyDocCount), buf)
}
