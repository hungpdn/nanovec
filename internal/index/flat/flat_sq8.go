package flat

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

// FlatIndexSQ8 stores vectors as uint8 (SQ8) to save 4x RAM.
type FlatIndexSQ8 struct {
	BaseIndex
	// Data in RAM (Compressed)
	RawVectors []uint8
}

func NewFlatIndexSQ8(dim int) *FlatIndexSQ8 {
	return &FlatIndexSQ8{
		BaseIndex:  NewBaseIndex(dim),
		RawVectors: make([]uint8, 0),
	}
}

// Add
func (idx *FlatIndexSQ8) Add(id string, vec types.Vector, meta map[string]any) error {
	idx.Lock()
	defer idx.Unlock()

	if len(vec) != idx.dim {
		return errors.ErrDimMismatch
	}

	if err := idx.CheckID(id); err != nil {
		return err
	}

	// Normalize (Required for SQ8 accuracy)
	// We copy to avoid modifying caller's vector if they didn't normalize
	normVec := make(types.Vector, len(vec))
	copy(normVec, vec)
	maths.NormalizeInPlace(normVec)

	qVec := maths.QuantizeSQ8(normVec)
	idx.RawVectors = append(idx.RawVectors, qVec...)

	idx.AddMeta(id, meta)
	return nil
}

// AddBatch
func (idx *FlatIndexSQ8) AddBatch(ids []string, vecs []types.Vector, metas []map[string]any) error {
	idx.Lock()
	defer idx.Unlock()

	if len(ids) != len(vecs) || len(ids) != len(metas) {
		return errors.ErrBatchSizeMismatch
	}

	needed := 0
	for _, v := range vecs {
		if len(v) != idx.dim {
			return errors.ErrDimMismatch
		}
		needed += len(v)
	}

	for _, id := range ids {
		if err := idx.CheckID(id); err != nil {
			return err
		}
	}

	idx.RawVectors = slices.Grow(idx.RawVectors, needed)

	normBuf := make(types.Vector, idx.dim)
	for i, id := range ids {
		meta := metas[i]

		copy(normBuf, vecs[i])
		maths.NormalizeInPlace(normBuf)

		qVec := maths.QuantizeSQ8(normBuf)
		idx.RawVectors = append(idx.RawVectors, qVec...)

		idx.AddMeta(id, meta)
	}

	return nil
}

// Search
func (idx *FlatIndexSQ8) Search(query types.Vector, k int, filter types.FilterFunc) ([]types.SearchResult, error) {
	idx.RLock()
	defer idx.RUnlock()

	normalizedQuery := make(types.Vector, len(query))
	copy(normalizedQuery, query)
	maths.NormalizeInPlace(normalizedQuery)

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
		end := start + idx.dim
		qVec := idx.RawVectors[start:end]
		dequantBuf := maths.DequantizeSQ8(qVec) // Reusable buffer for dequantization to avoid allocs in loop
		score := maths.DotProduct(normalizedQuery, dequantBuf)

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

// Delete
func (idx *FlatIndexSQ8) Delete(id string) error {
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

// Count
func (idx *FlatIndexSQ8) Count() int {
	idx.RLock()
	defer idx.RUnlock()
	return len(idx.IDs)
}

// Dim
func (idx *FlatIndexSQ8) Dim() int { return idx.dim }

// Save
func (idx *FlatIndexSQ8) Save(path string) error {
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
	if err := enc.Encode(idx.IDs); err != nil {
		return err
	}
	if err := enc.Encode(idx.Metadata); err != nil {
		return err
	}

	// Save Raw uint8 vectors
	if err := enc.Encode(idx.RawVectors); err != nil {
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

func (idx *FlatIndexSQ8) Load(path string) error {
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

	dec := gob.NewDecoder(f)
	if err := dec.Decode(&idx.IDs); err != nil {
		return err
	}

	idx.Metadata = make(map[string]map[string]any)
	if err := dec.Decode(&idx.Metadata); err != nil {
		return err
	}

	idx.idMap = make(map[string]int, len(idx.IDs))
	for i, id := range idx.IDs {
		idx.idMap[id] = i
	}

	// Load Raw uint8 vectors
	if err := dec.Decode(&idx.RawVectors); err != nil {
		return err
	}
	return nil
}
