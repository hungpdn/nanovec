package index

import (
	"container/heap"
	"encoding/binary"
	"encoding/gob"
	"errors"
	"fmt"
	"os"
	"slices"
	"sync"

	"github.com/hungpdn/nanovec/pkg/maths"
	"github.com/hungpdn/nanovec/pkg/types"
)

// FlatIndex: Save all vectors in Map and iterate through them during the search
type FlatIndex struct {
	mu sync.RWMutex
	// Example: There are 2 2D vectors [1,2] and [3,4]
	// RawVectors = [1, 2, 3, 4] -> 100% seamless
	RawVectors []float32

	// Map the reverse position of the ID
	// IDs[0] correspond to the starting vector at RawVectors[0]
	// IDs[1] correspond to the starting vector at RawVectors[dim]
	IDs      []string
	idMap    map[string]int
	Metadata map[string]map[string]interface{}
	dim      int
}

func NewFlatIndex(dim int) *FlatIndex {
	return &FlatIndex{
		RawVectors: make([]float32, 0),
		IDs:        make([]string, 0),
		idMap:      make(map[string]int),
		Metadata:   make(map[string]map[string]interface{}),
		dim:        dim,
	}
}

func (idx *FlatIndex) Add(id string, vec types.Vector, meta map[string]interface{}) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if len(vec) != idx.dim {
		return types.ErrDimMismatch
	}

	if _, exists := idx.idMap[id]; exists {
		return errors.New("id already exists")
	}

	startPos := len(idx.RawVectors)
	idx.RawVectors = append(idx.RawVectors, vec...)
	maths.NormalizeInPlace(idx.RawVectors[startPos:])

	idx.IDs = append(idx.IDs, id)
	idx.idMap[id] = len(idx.IDs) - 1
	idx.Metadata[id] = meta

	return nil
}

// AddBatch adds multiple vectors in a single lock transaction.
func (idx *FlatIndex) AddBatch(ids []string, vecs []types.Vector, metas []map[string]interface{}) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if len(ids) != len(vecs) || len(ids) != len(metas) {
		return errors.New("batch size mismatch")
	}

	needed := 0
	for _, v := range vecs {
		if len(v) != idx.dim {
			return types.ErrDimMismatch
		}
		needed += len(v)
	}

	for _, id := range ids {
		if _, exists := idx.idMap[id]; exists {
			return fmt.Errorf("id %s already exists", id)
		}
	}

	idx.RawVectors = slices.Grow(idx.RawVectors, needed)

	for i, id := range ids {
		vec := vecs[i]
		meta := metas[i]

		startPos := len(idx.RawVectors)
		idx.RawVectors = append(idx.RawVectors, vec...)
		maths.NormalizeInPlace(idx.RawVectors[startPos:])

		idx.IDs = append(idx.IDs, id)
		idx.idMap[id] = len(idx.IDs) - 1
		idx.Metadata[id] = meta
	}

	return nil
}

// Delete remove vector by Swap-and-Pop
func (idx *FlatIndex) Delete(id string) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	pos, exists := idx.idMap[id]
	if !exists {
		return nil
	}

	lastIndex := len(idx.IDs) - 1
	if pos < lastIndex {
		lastID := idx.IDs[lastIndex]
		idx.IDs[pos] = lastID
		idx.idMap[lastID] = pos

		destStart := pos * idx.dim
		srcStart := lastIndex * idx.dim
		copy(idx.RawVectors[destStart:destStart+idx.dim], idx.RawVectors[srcStart:srcStart+idx.dim])
	}

	delete(idx.idMap, id)
	delete(idx.Metadata, id)
	idx.IDs = idx.IDs[:lastIndex]
	idx.RawVectors = idx.RawVectors[:lastIndex*idx.dim]

	return nil
}

func (idx *FlatIndex) Search(query types.Vector, k int, filter types.FilterFunc) ([]types.SearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	normalizedQuery := make([]float32, len(query))
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

func (idx *FlatIndex) Count() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.IDs)
}

func (idx *FlatIndex) Dim() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.dim
}

func (idx *FlatIndex) Save(path string) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

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

// Load: read data from disk
func (idx *FlatIndex) Load(path string) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

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

	idx.Metadata = make(map[string]map[string]interface{})
	if err := dec.Decode(&idx.Metadata); err != nil {
		return err
	}

	idx.idMap = make(map[string]int, len(idx.IDs))
	for i, id := range idx.IDs {
		idx.idMap[id] = i
	}

	if err := dec.Decode(&idx.RawVectors); err != nil {
		return err
	}

	return nil
}

type Item struct {
	ID    string
	Score float32
}
type ResultHeap []Item

func (h ResultHeap) Len() int            { return len(h) }
func (h ResultHeap) Less(i, j int) bool  { return h[i].Score < h[j].Score } // Min-Heap
func (h ResultHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *ResultHeap) Push(x interface{}) { *h = append(*h, x.(Item)) }
func (h *ResultHeap) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[0 : n-1]
	return item
}
