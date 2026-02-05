package index

import (
	"container/heap"
	"encoding/binary"
	"encoding/gob"
	"os"
	"slices"

	"github.com/hungpdn/nanovec/pkg/errors"
	"github.com/hungpdn/nanovec/pkg/maths"
	"github.com/hungpdn/nanovec/pkg/types"
)

// FlatIndex is a generic implementation for both Float32 and SQ8 (Uint8)
type FlatIndex[T types.Number] struct {
	BaseIndex
	RawVectors []T

	// Injected behavior
	// storeFunc processes a vector and writes it to the destination slice
	// src is the input vector, dst is the slice in RawVectors, buf is a temp buffer (for SQ8)
	storeFunc func(src []float32, dst []T, buf []float32)
	// scoreFunc calculates similarity between query and target vector
	scoreFunc func(query []float32, target []T) float32
}

// NewFlatIndexFloat creates a standard Float32 index
func NewFlatIndexFloat(dim int) *FlatIndex[float32] {
	return &FlatIndex[float32]{
		BaseIndex:  NewBaseIndex(dim),
		RawVectors: make([]float32, 0),
		storeFunc: func(src []float32, dst []float32, _ []float32) {
			copy(dst, src)
			maths.NormalizeInPlace(dst)
		},
		scoreFunc: maths.DotProduct,
	}
}

// NewFlatIndexSQ8 creates a Quantized SQ8 index
func NewFlatIndexSQ8(dim int) *FlatIndex[uint8] {
	return &FlatIndex[uint8]{
		BaseIndex:  NewBaseIndex(dim),
		RawVectors: make([]uint8, 0),
		storeFunc: func(src []float32, dst []uint8, buf []float32) {
			// Copy to temp buffer to normalize before quantizing
			copy(buf, src)
			maths.NormalizeInPlace(buf)
			// Quantize directly to destination
			for i, v := range buf {
				val := (v + 1.0) * 127.5
				if val < 0 {
					val = 0
				} else if val > 255 {
					val = 255
				}
				dst[i] = uint8(val)
			}
		},
		scoreFunc: maths.DotProductSQ8,
	}
}

// Add adds a single vector
func (idx *FlatIndex[T]) Add(id string, vec types.Vector, meta map[string]any) error {
	idx.Lock()
	defer idx.Unlock()

	if len(vec) != idx.dim {
		return errors.ErrDimMismatch
	}
	if err := idx.CheckID(id); err != nil {
		return err
	}

	startPos := len(idx.RawVectors)
	// Append zero-values to grow slice
	idx.RawVectors = append(idx.RawVectors, make([]T, idx.dim)...)

	// Helper buffer for SQ8 (unused for Float32 storeFunc)
	var buf []float32
	if _, ok := any(idx.RawVectors).([]uint8); ok {
		buf = make([]float32, idx.dim)
	}

	idx.storeFunc(vec, idx.RawVectors[startPos:], buf)
	idx.AddMeta(id, meta)
	return nil
}

// AddBatch adds multiple vectors
func (idx *FlatIndex[T]) AddBatch(ids []string, vecs []types.Vector, metas []map[string]any) error {
	idx.Lock()
	defer idx.Unlock()

	if len(ids) != len(vecs) || len(ids) != len(metas) {
		return errors.ErrBatchSizeMismatch
	}

	for _, v := range vecs {
		if len(v) != idx.dim {
			return errors.ErrDimMismatch
		}
	}
	for _, id := range ids {
		if err := idx.CheckID(id); err != nil {
			return err
		}
	}

	currentLen := len(idx.RawVectors)
	addLen := len(ids) * idx.dim
	idx.RawVectors = slices.Grow(idx.RawVectors, addLen)
	idx.RawVectors = idx.RawVectors[:currentLen+addLen]
	idx.IDs = slices.Grow(idx.IDs, len(ids))

	normBuf := make([]float32, idx.dim)

	for i, id := range ids {
		start := currentLen + (i * idx.dim)
		dst := idx.RawVectors[start : start+idx.dim]

		idx.storeFunc(vecs[i], dst, normBuf)
		idx.AddMeta(id, metas[i])
	}

	return nil
}

// Search finds k nearest neighbors.
// Note: 'query' must be already normalized by the caller.
func (idx *FlatIndex[T]) Search(query types.Vector, k int, filter types.FilterFunc) ([]types.SearchResult, error) {
	idx.RLock()
	defer idx.RUnlock()

	normalizedQuery := query

	n := len(idx.IDs)
	h := &ResultHeap{}
	heap.Init(h)

	for i := 0; i < n; i++ {
		id := idx.IDs[i]

		if filter != nil {
			if meta, ok := idx.Metadata[id]; ok {
				if !filter(meta) {
					continue
				}
			} else {
				continue
			}
		}

		start := i * idx.dim
		targetVec := idx.RawVectors[start : start+idx.dim]
		score := idx.scoreFunc(normalizedQuery, targetVec)

		if h.Len() < k {
			heap.Push(h, Item{ID: id, Score: score})
		} else if score > (*h)[0].Score {
			heap.Pop(h)
			heap.Push(h, Item{ID: id, Score: score})
		}
	}

	results := make([]types.SearchResult, h.Len())
	for i := h.Len() - 1; i >= 0; i-- {
		item := heap.Pop(h).(Item)
		results[i] = types.SearchResult{
			ID:       item.ID,
			Score:    item.Score,
			Metadata: idx.Metadata[item.ID],
		}
	}

	return results, nil
}

// Delete removes a vector.
func (idx *FlatIndex[T]) Delete(id string) error {
	idx.Lock()
	defer idx.Unlock()

	pos, lastIndex, exists := idx.PrepareDelete(id)
	if !exists {
		return nil
	}

	if pos < lastIndex {
		destStart := pos * idx.dim
		srcStart := lastIndex * idx.dim
		copy(idx.RawVectors[destStart:destStart+idx.dim], idx.RawVectors[srcStart:srcStart+idx.dim])
	}

	idx.RawVectors = idx.RawVectors[:lastIndex*idx.dim]
	idx.CommitDelete(id, pos, lastIndex)
	return nil
}

// Save persists to disk.
func (idx *FlatIndex[T]) Save(path string) error {
	idx.Lock()
	defer idx.Unlock()

	tmpPath := path + ".tmp"
	f, err := os.Create(tmpPath)
	if err != nil {
		return err
	}
	defer f.Close()

	header := make([]int32, 2)
	header[0] = int32(idx.dim)
	header[1] = int32(len(idx.IDs))
	if err := binary.Write(f, binary.LittleEndian, header); err != nil {
		return err
	}

	enc := gob.NewEncoder(f)
	if err := idx.SaveBase(enc); err != nil {
		return err
	}

	if err := binary.Write(f, binary.LittleEndian, idx.RawVectors); err != nil {
		return err
	}

	if err := f.Sync(); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmpPath, path)
}

// Load reads from disk.
func (idx *FlatIndex[T]) Load(path string) error {
	idx.Lock()
	defer idx.Unlock()

	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	header := make([]int32, 2)
	if err := binary.Read(f, binary.LittleEndian, &header); err != nil {
		return err
	}
	idx.dim = int(header[0])
	count := int(header[1])

	dec := gob.NewDecoder(f)
	if err := idx.LoadBase(dec); err != nil {
		return err
	}

	idx.RawVectors = make([]T, count*idx.dim)
	if err := binary.Read(f, binary.LittleEndian, &idx.RawVectors); err != nil {
		return err
	}
	return nil
}
