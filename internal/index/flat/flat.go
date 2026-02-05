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

// FlatIndex save all vectors in Map and iterate through them during the search
type FlatIndex struct {
	BaseIndex
	// Example: There are 2 2D vectors [1,2] and [3,4]
	// RawVectors = [1, 2, 3, 4] -> 100% seamless
	RawVectors []float32
}

func NewFlatIndex(dim int) *FlatIndex {
	return &FlatIndex{
		BaseIndex:  NewBaseIndex(dim),
		RawVectors: make([]float32, 0),
	}
}

// Add
func (idx *FlatIndex) Add(id string, vec types.Vector, meta map[string]any) error {
	idx.Lock()
	defer idx.Unlock()

	if len(vec) != idx.dim {
		return errors.ErrDimMismatch
	}

	if err := idx.CheckID(id); err != nil {
		return err
	}

	startPos := len(idx.RawVectors)
	idx.RawVectors = append(idx.RawVectors, vec...)
	maths.NormalizeInPlace(idx.RawVectors[startPos:])

	idx.AddMeta(id, meta)
	return nil
}

// AddBatch adds multiple vectors in a single lock transaction.
func (idx *FlatIndex) AddBatch(ids []string, vecs []types.Vector, metas []map[string]any) error {
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

	for i, id := range ids {
		vec := vecs[i]
		meta := metas[i]

		startPos := len(idx.RawVectors)
		idx.RawVectors = append(idx.RawVectors, vec...)
		maths.NormalizeInPlace(idx.RawVectors[startPos:])

		idx.AddMeta(id, meta)
	}

	return nil
}

// Search
func (idx *FlatIndex) Search(query types.Vector, k int, filter types.FilterFunc) ([]types.SearchResult, error) {
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

		// Slice bounds check is eliminated by compiler here due to simple ranges
		start := i * idx.dim
		end := start + idx.dim
		targetVec := idx.RawVectors[start:end]
		score := maths.DotProduct(query, targetVec)

		if h.Len() < k {
			heap.Push(h, Item{ID: id, Score: score})
		} else if score > (*h)[0].Score {
			heap.Pop(h)
			heap.Push(h, Item{ID: id, Score: score})
		}
	}

	results := make([]types.SearchResult, h.Len())

	// Pop returns the order from smallest to largest
	for i := h.Len() - 1; i >= 0; i-- {
		item := heap.Pop(h).(Item)
		meta := idx.Metadata[item.ID]
		results[i] = types.SearchResult{
			ID:       item.ID,
			Score:    item.Score,
			Metadata: meta, //
		}
	}

	return results, nil
}

// Delete remove vector by Swap-and-Pop
func (idx *FlatIndex) Delete(id string) error {
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
func (idx *FlatIndex) Count() int {
	idx.RLock()
	defer idx.RUnlock()
	return len(idx.IDs)
}

// Dim
func (idx *FlatIndex) Dim() int { return idx.dim }

// Save save data into disk
func (idx *FlatIndex) Save(path string) error {
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

// Load read data from disk
func (idx *FlatIndex) Load(path string) error {
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
	if err := idx.LoadBase(dec); err != nil {
		return err
	}
	if err := dec.Decode(&idx.RawVectors); err != nil {
		return err
	}

	return nil
}
