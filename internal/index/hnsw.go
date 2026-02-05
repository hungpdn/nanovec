package index

import (
	"container/heap"
	"encoding/gob"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sync"

	"github.com/hungpdn/nanovec/pkg/errors"
	"github.com/hungpdn/nanovec/pkg/maths"
	"github.com/hungpdn/nanovec/pkg/types"
)

// HNSWIndex is a generic implementation for Float32 and SQ8
type HNSWIndex[T types.Number] struct {
	mu sync.RWMutex

	// Configuration
	dim            int
	M              int
	MMax0          int     // Max connections for layer 0 (2 * M)
	EfConstruction int     // Beam size for construction
	LevelMult      float64 // Normalization factor for level generation

	// Graph State
	nodes      []*Node[T]
	idMap      map[string]int // External ID -> Internal Index
	enterPoint int            // Internal Index of the entry node
	maxLevel   int            // Current max level in the graph

	Metadata map[string]map[string]any
	Version  uint64

	// visitedList stores the "visit token" for each node internal ID.
	visitedList  []uint32
	visitedToken uint32 // Current search generation

	// Injected behavior
	// convertFunc converts input float32 vector to storage type T
	convertFunc func(vec []float32) []T
	// distQueryFunc calculates distance between query (float32) and node (T)
	distQueryFunc func(query []float32, node []T) float32
	// distNodeFunc calculates distance between two nodes (T and T)
	distNodeFunc func(a, b []T) float32
}

// Node represents a point in the graph
type Node[T types.Number] struct {
	ID        string
	Vec       []T
	Level     int
	Neighbors [][]int // [level][neighbor_internal_index]
}

// NewHNSWIndexFloat creates a standard Float32 HNSW index
func NewHNSWIndexFloat(dim, m, efConstruction int) *HNSWIndex[float32] {
	return newHNSWIndex(dim, m, efConstruction,
		func(normVec []float32) []float32 {
			out := make([]float32, len(normVec))
			copy(out, normVec)
			return out
		},
		maths.DotProduct,
		maths.DotProduct,
	)
}

// NewHNSWIndexSQ8 creates a sq8 HNSW index
func NewHNSWIndexSQ8(dim, m, efConstruction int) *HNSWIndex[uint8] {
	return newHNSWIndex(dim, m, efConstruction,
		func(normVec []float32) []uint8 {
			return maths.QuantizeSQ8(normVec)
		},
		maths.DotProductSQ8,
		maths.DotProductUint8,
	)
}

// Helper to construct index
func newHNSWIndex[T types.Number](dim, m, efConstruction int,
	conv func([]float32) []T,
	distQ func([]float32, []T) float32,
	distN func([]T, []T) float32) *HNSWIndex[T] {

	idx := &HNSWIndex[T]{
		dim:            dim,
		M:              m,
		MMax0:          m * 2,
		EfConstruction: efConstruction,
		LevelMult:      1.0 / math.Log(float64(m)),
		nodes:          make([]*Node[T], 0),
		idMap:          make(map[string]int),
		Metadata:       make(map[string]map[string]any),
		enterPoint:     -1,
		maxLevel:       -1,
		convertFunc:    conv,
		distQueryFunc:  distQ,
		distNodeFunc:   distN,
	}
	return idx
}

// Add inserts a vector into the HNSW graph
func (idx *HNSWIndex[T]) Add(id string, vec types.Vector, meta map[string]any) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if _, exists := idx.idMap[id]; exists {
		return fmt.Errorf("id %s already exists", id)
	}
	return idx.internalAdd(id, vec, meta, nil)
}

// AddBatch adds multiple vectors
func (idx *HNSWIndex[T]) AddBatch(ids []string, vecs []types.Vector, metas []map[string]any) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	for _, id := range ids {
		if _, exists := idx.idMap[id]; exists {
			return fmt.Errorf("id %s already exists", id)
		}
	}

	scratch := make([]float32, idx.dim)

	for i, id := range ids {
		if err := idx.internalAdd(id, vecs[i], metas[i], scratch); err != nil {
			return err
		}
	}
	return nil
}

// internalAdd is the core Add logic without Locking (Helper)
func (idx *HNSWIndex[T]) internalAdd(id string, vec types.Vector, meta map[string]any, scratch []float32) error {
	if len(vec) != idx.dim {
		return errors.ErrDimMismatch
	}

	var navQuery []float32
	if scratch != nil {
		navQuery = scratch
	} else {
		navQuery = make([]float32, len(vec))
	}

	copy(navQuery, vec)
	maths.NormalizeInPlace(navQuery)

	storedVec := idx.convertFunc(vec)

	level := randomLevel()
	internalID := len(idx.nodes)

	node := &Node[T]{
		ID:        id,
		Vec:       storedVec,
		Level:     level,
		Neighbors: make([][]int, level+1),
	}

	idx.nodes = append(idx.nodes, node)
	idx.idMap[id] = internalID
	idx.Metadata[id] = meta

	if idx.enterPoint == -1 {
		idx.enterPoint = internalID
		idx.maxLevel = level
		return nil
	}

	currObj := idx.enterPoint
	currDist := idx.distQueryFunc(navQuery, idx.nodes[currObj].Vec)

	for l := idx.maxLevel; l > level; l-- {
		changed := true
		for changed {
			changed = false
			for _, neighborID := range idx.nodes[currObj].Neighbors[l] {
				dist := idx.distQueryFunc(navQuery, idx.nodes[neighborID].Vec)
				if dist > currDist {
					currDist = dist
					currObj = neighborID
					changed = true
				}
			}
		}
	}

	for l := int(math.Min(float64(level), float64(idx.maxLevel))); l >= 0; l-- {
		candidates := idx.searchLayer(navQuery, currObj, idx.EfConstruction, l)
		neighbors := idx.selectNeighbors(candidates, idx.M)
		for _, neighborID := range neighbors {
			idx.addConnection(l, internalID, neighborID)
			idx.addConnection(l, neighborID, internalID)
		}
		if len(candidates) > 0 {
			currObj = candidates[0].id
		}
	}

	if level > idx.maxLevel {
		idx.maxLevel = level
		idx.enterPoint = internalID
	}
	return nil
}

// searchLayer performs BFS with priority queue (Beam Search)
// Returns list of candidates sorted by Score DESC (Closest first)
func (idx *HNSWIndex[T]) searchLayer(query types.Vector, entryPoint, ef, level int) []pqItem {

	idx.visitedToken++
	if idx.visitedToken == 0 {
		idx.visitedToken = 1
		for i := range idx.visitedList {
			idx.visitedList[i] = 0
		}
	}

	if len(idx.visitedList) < len(idx.nodes) {
		newVis := make([]uint32, len(idx.nodes))
		copy(newVis, idx.visitedList)
		idx.visitedList = newVis
	}

	idx.visitedList[entryPoint] = idx.visitedToken

	candidates := &MaxHeap{}
	heap.Init(candidates)
	results := &MinHeap{}
	heap.Init(results)

	dist := idx.distQueryFunc(query, idx.nodes[entryPoint].Vec)
	item := pqItem{id: entryPoint, score: dist}

	heap.Push(candidates, item)
	heap.Push(results, item)

	for candidates.Len() > 0 {
		curr := heap.Pop(candidates).(pqItem)
		furthestResult := (*results)[0]

		if curr.score < furthestResult.score && results.Len() >= ef {
			break
		}

		for _, neighborID := range idx.nodes[curr.id].Neighbors[level] {
			if idx.visitedList[neighborID] != idx.visitedToken {
				idx.visitedList[neighborID] = idx.visitedToken
				dist := idx.distQueryFunc(query, idx.nodes[neighborID].Vec)
				newItem := pqItem{id: neighborID, score: dist}

				if results.Len() < ef || dist > (*results)[0].score {
					heap.Push(candidates, newItem)
					heap.Push(results, newItem)
					if results.Len() > ef {
						heap.Pop(results)
					}
				}
			}
		}
	}

	finalRes := make([]pqItem, results.Len())
	for i := results.Len() - 1; i >= 0; i-- {
		finalRes[i] = heap.Pop(results).(pqItem)
	}
	return finalRes
}

// addConnection connects two nodes at a specific level, pruning if necessary
func (idx *HNSWIndex[T]) addConnection(level, from, to int) {
	node := idx.nodes[from]
	for _, neighbor := range node.Neighbors[level] {
		if neighbor == to {
			return
		}
	}
	node.Neighbors[level] = append(node.Neighbors[level], to)

	maxM := idx.M
	if level == 0 {
		maxM = idx.MMax0
	}

	if len(node.Neighbors[level]) > maxM {
		fromVec := idx.nodes[from].Vec
		h := &MinHeap{}
		heap.Init(h)

		for _, neighborID := range node.Neighbors[level] {
			// Node to Node distance
			score := idx.distNodeFunc(fromVec, idx.nodes[neighborID].Vec)
			heap.Push(h, pqItem{id: neighborID, score: score})
			if h.Len() > maxM {
				heap.Pop(h)
			}
		}

		newNeighbors := make([]int, h.Len())
		for i := 0; i < len(newNeighbors); i++ {
			newNeighbors[i] = (*h)[i].id
		}
		node.Neighbors[level] = newNeighbors
	}
}

// Search performs Approximate Nearest Neighbor search
// Note: 'query' must be already normalized.
func (idx *HNSWIndex[T]) Search(query types.Vector, k int, filter types.FilterFunc) ([]types.SearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if idx.enterPoint == -1 {
		return []types.SearchResult{}, nil
	}

	normalizedQuery := query

	currObj := idx.enterPoint
	currDist := idx.distQueryFunc(normalizedQuery, idx.nodes[currObj].Vec)

	for l := idx.maxLevel; l > 0; l-- {
		changed := true
		for changed {
			changed = false
			for _, neighborID := range idx.nodes[currObj].Neighbors[l] {
				dist := idx.distQueryFunc(normalizedQuery, idx.nodes[neighborID].Vec)
				if dist > currDist {
					currDist = dist
					currObj = neighborID
					changed = true
				}
			}
		}
	}

	efSearch := idx.EfConstruction
	if k > efSearch {
		efSearch = k
	}

	candidates := idx.searchLayer(normalizedQuery, currObj, efSearch, 0)

	results := make([]types.SearchResult, 0, k)
	for _, c := range candidates {
		id := idx.nodes[c.id].ID

		currentInternalID, exists := idx.idMap[id]
		if !exists || currentInternalID != c.id {
			continue
		}

		meta := idx.Metadata[id]
		if filter != nil {
			if meta == nil || !filter(meta) {
				continue
			}
		}

		results = append(results, types.SearchResult{
			ID:       id,
			Score:    c.score,
			Metadata: meta,
		})

		if len(results) >= k {
			break
		}
	}
	return results, nil
}

func (idx *HNSWIndex[T]) Delete(id string) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	if _, ok := idx.idMap[id]; ok {
		delete(idx.idMap, id)
		delete(idx.Metadata, id)
	}
	return nil
}

func (idx *HNSWIndex[T]) Count() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.idMap)
}

func (idx *HNSWIndex[T]) Dim() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.dim
}

func (idx *HNSWIndex[T]) SetVersion(v uint64) {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	idx.Version = v
}

func (idx *HNSWIndex[T]) GetVersion() uint64 {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	return idx.Version
}

// Persistence
// For serialization, we define a DTO to avoid exposing all internal fields if not needed,
// but for simplicity in generics, we can export fields on Node[T] and use it directly.
// In the code above, Node[T] fields are Exported.
func (idx *HNSWIndex[T]) Save(path string) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	tmpPath := path + ".tmp"
	f, err := os.Create(tmpPath)
	if err != nil {
		return err
	}
	defer f.Close()

	enc := gob.NewEncoder(f)
	if err := enc.Encode(idx.Version); err != nil {
		return err
	}
	if err := enc.Encode(idx.dim); err != nil {
		return err
	}
	if err := enc.Encode(idx.enterPoint); err != nil {
		return err
	}
	if err := enc.Encode(idx.maxLevel); err != nil {
		return err
	}

	count := len(idx.nodes)
	if err := enc.Encode(count); err != nil {
		return err
	}

	for _, node := range idx.nodes {
		// Save node structure
		if err := enc.Encode(node); err != nil {
			return err
		}
	}

	if err := enc.Encode(idx.Metadata); err != nil {
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

func (idx *HNSWIndex[T]) Load(path string) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	dec := gob.NewDecoder(f)
	if err := dec.Decode(&idx.Version); err != nil {
		return err
	}
	if err := dec.Decode(&idx.dim); err != nil {
		return err
	}
	if err := dec.Decode(&idx.enterPoint); err != nil {
		return err
	}
	if err := dec.Decode(&idx.maxLevel); err != nil {
		return err
	}

	var count int
	if err := dec.Decode(&count); err != nil {
		return err
	}

	idx.nodes = make([]*Node[T], count)
	idx.idMap = make(map[string]int)

	for i := 0; i < count; i++ {
		node := &Node[T]{}
		if err := dec.Decode(node); err != nil {
			return err
		}
		idx.nodes[i] = node
		idx.idMap[node.ID] = i
	}

	idx.Metadata = make(map[string]map[string]any)
	if err := dec.Decode(&idx.Metadata); err != nil {
		return err
	}

	for id := range idx.idMap {
		if _, isLive := idx.Metadata[id]; !isLive {
			delete(idx.idMap, id)
		}
	}
	return nil
}

// --- Common Heaps/Utils ---

type pqItem struct {
	id    int
	score float32
}

// MinHeap: Keeps the LOWEST score at top
type MinHeap []pqItem

func (h MinHeap) Len() int           { return len(h) }
func (h MinHeap) Less(i, j int) bool { return h[i].score < h[j].score }
func (h MinHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *MinHeap) Push(x any)        { *h = append(*h, x.(pqItem)) }
func (h *MinHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// MaxHeap: Keeps the HIGHEST score at top
type MaxHeap []pqItem

func (h MaxHeap) Len() int           { return len(h) }
func (h MaxHeap) Less(i, j int) bool { return h[i].score > h[j].score }
func (h MaxHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *MaxHeap) Push(x any)        { *h = append(*h, x.(pqItem)) }
func (h *MaxHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// --- Common Logic Functions ---

// randomLevel generates a level for a new node
func randomLevel() int {
	lvl := 0
	for rand.Float64() < 0.5 && lvl < 10 { // Cap level to avoid runaway
		lvl++
	}
	return lvl
}

// selectNeighbors picks the M best candidates to connect
func (idx *HNSWIndex[T]) selectNeighbors(candidates []pqItem, m int) []int {
	count := len(candidates)
	if count > m {
		count = m
	}
	out := make([]int, count)
	for i := 0; i < count; i++ {
		out[i] = candidates[i].id
	}
	return out
}
