package index

import (
	"container/heap"
	"encoding/binary"
	"encoding/gob"
	"errors"
	"os"
	"sync"

	"github.com/hungpdn/nanovec/pkg/maths"
	"github.com/hungpdn/nanovec/pkg/types"
)

// FlatIndex: Save all vectors in Map and iterate through them during the search
type FlatIndex struct {
	mu sync.RWMutex
	// Data in RAM (Optimized for SIMD)
	// Example: There are 2 2D vectors [1,2] and [3,4]
	// RawVectors = [1, 2, 3, 4] -> 100% seamless
	RawVectors []float32

	// Map the reverse position of the ID
	// IDs[0] correspond to the starting vector at RawVectors[0]
	// IDs[1] correspond to the starting vector at RawVectors[dim]
	IDs   []string
	idMap map[string]int
	Dim   int
}

func NewFlatIndex(dim int) *FlatIndex {
	return &FlatIndex{
		RawVectors: make([]float32, 0),
		IDs:        make([]string, 0),
		idMap:      make(map[string]int),
		Dim:        dim,
	}
}

func (idx *FlatIndex) Add(id string, vec types.Vector) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if len(vec) != idx.Dim {
		return errors.New("dimension mismatch")
	}

	if _, exists := idx.idMap[id]; exists {
		return errors.New("id already exists")
	}

	startPos := len(idx.RawVectors)
	idx.RawVectors = append(idx.RawVectors, vec...)
	maths.NormalizeInPlace(idx.RawVectors[startPos:])

	idx.IDs = append(idx.IDs, id)
	idx.idMap[id] = len(idx.IDs) - 1

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

	// Swap-and-Pop
	lastIndex := len(idx.IDs) - 1
	if pos < lastIndex {
		lastID := idx.IDs[lastIndex]
		idx.IDs[pos] = lastID
		idx.idMap[lastID] = pos

		destStart := pos * idx.Dim
		srcStart := lastIndex * idx.Dim
		copy(idx.RawVectors[destStart:destStart+idx.Dim], idx.RawVectors[srcStart:srcStart+idx.Dim])
	}

	delete(idx.idMap, id)
	idx.IDs = idx.IDs[:lastIndex]
	idx.RawVectors = idx.RawVectors[:lastIndex*idx.Dim]

	return nil
}

func (idx *FlatIndex) Search(query types.Vector, k int) ([]string, []float32, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	normalizedQuery := make([]float32, len(query))
	copy(normalizedQuery, query)
	maths.NormalizeInPlace(normalizedQuery)

	n := len(idx.IDs)
	h := &ResultHeap{}
	heap.Init(h)

	// Sequential loops on seamless memory
	// CPU Prefetcher will work extremely well here
	for i := 0; i < n; i++ {
		id := idx.IDs[i]
		start := i * idx.Dim
		end := start + idx.Dim

		// Slice bounds check is eliminated by compiler here due to simple ranges
		targetVec := idx.RawVectors[start:end]

		// TODO: call SIMD
		score := maths.DotProduct(query, targetVec)

		if h.Len() < k {
			heap.Push(h, Item{ID: id, Score: score})
		} else if score > (*h)[0].Score {
			heap.Pop(h)
			heap.Push(h, Item{ID: id, Score: score})
		}
	}

	ids := make([]string, h.Len())
	scores := make([]float32, h.Len())

	// Pop returns the order from smallest to largest
	for i := h.Len() - 1; i >= 0; i-- {
		item := heap.Pop(h).(Item)
		ids[i] = item.ID
		scores[i] = item.Score
	}

	return ids, scores, nil
}

func (idx *FlatIndex) Count() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.IDs)
}

func (idx *FlatIndex) Save(path string) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	f, err := os.Create(path)
	if err != nil {
		return err
	}

	defer f.Close()

	header := make([]int32, 2)
	header[0] = int32(idx.Dim)
	header[1] = int32(len(idx.IDs))
	if err := binary.Write(f, binary.LittleEndian, header); err != nil {
		return err
	}

	enc := gob.NewEncoder(f)
	if err := enc.Encode(idx.IDs); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, idx.RawVectors); err != nil {
		return err
	}

	return nil
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
	idx.Dim = int(header[0])
	count := int(header[1])

	dec := gob.NewDecoder(f)
	if err := dec.Decode(&idx.IDs); err != nil {
		return err
	}

	idx.idMap = make(map[string]int, len(idx.IDs))
	for i, id := range idx.IDs {
		idx.idMap[id] = i
	}

	idx.RawVectors = make([]float32, count*idx.Dim)
	if err := binary.Read(f, binary.LittleEndian, &idx.RawVectors); err != nil {
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
