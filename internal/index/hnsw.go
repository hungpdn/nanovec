package index

import (
	"container/heap"
	"encoding/binary"
	"encoding/gob"
	"fmt"
	"math"
	"os"
	"slices"
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

	// Concurrency: Pool of search contexts
	searchCtxPool *sync.Pool

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
		searchCtxPool: &sync.Pool{
			New: func() any {
				return newSearchCtx()
			},
		},
	}
	return idx
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

	idx.nodes = slices.Grow(idx.nodes, len(ids))
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

	searchCtx := idx.searchCtxPool.Get().(*searchCtx)
	defer idx.searchCtxPool.Put(searchCtx)

	var navQuery []float32
	if scratch != nil {
		navQuery = scratch
	} else {
		navQuery = make([]float32, len(vec))
	}

	copy(navQuery, vec)
	maths.NormalizeInPlace(navQuery)

	storedVec := idx.convertFunc(navQuery)

	level := searchCtx.randomLevel()
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

	// Greedy search
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
		idx.searchLayer(searchCtx, navQuery, currObj, idx.EfConstruction, l)
		// candidates are in ctx.results (MinHeap, furthest at top?? No, see searchLayer)
		// searchLayer returns nothing, it populates ctx.results.
		// Let's adjust searchLayer to return slice or use heap directly.
		// For efficiency, let's look at modified searchLayer below.
		neighbors := idx.selectNeighborsFromHeap(searchCtx, searchCtx.results, idx.M)

		for _, neighborID := range neighbors {
			idx.addConnection(searchCtx, l, internalID, neighborID)
			idx.addConnection(searchCtx, l, neighborID, internalID)
		}
		// Update currObj to the closest found for next layer
		if searchCtx.results.Len() > 0 {
			// Accessing backing slice directly if possible, or Peek.
			// In MinHeap (results), the TOP is the FURTHEST (worst).
			// We need the CLOSEST. The heap stores 'ef' best candidates.
			// Any element in heap is good, but we want a good start point.
			// Ideally we pick the closest one.
			// Iterating the heap to find min dist is cheap (ef is small).
			bestID := -1
			bestDist := float32(-1)

			for _, item := range *searchCtx.results {
				if bestID == -1 || item.score > bestDist { // Score is similarity (higher better)
					bestDist = item.score
					bestID = item.id
				}
			}
			currObj = bestID
		}
	}

	if level > idx.maxLevel {
		idx.maxLevel = level
		idx.enterPoint = internalID
	}
	return nil
}

// UpdateMetadata updates only the metadata for an ID (O(1))
func (idx *HNSWIndex[T]) UpdateMetadata(id string, meta map[string]any) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if _, exists := idx.idMap[id]; !exists {
		return fmt.Errorf("id %s not found", id)
	}
	idx.Metadata[id] = meta
	return nil
}

// searchLayer performs BFS with priority queue (Beam Search)
// Returns list of candidates sorted by Score DESC (Closest first)
func (idx *HNSWIndex[T]) searchLayer(ctx *searchCtx, query types.Vector, entryPoint, ef, level int) {
	ctx.reset() // Clear heaps

	// 1. Manage Visited List (Thread-Local logic)
	ctx.visitedToken++
	if ctx.visitedToken == 0 {
		ctx.visitedToken = 1
		for i := range ctx.visitedList {
			ctx.visitedList[i] = 0
		}
	}
	// If nodes count grew larger than current visitedList capacity
	currentNodes := len(idx.nodes)
	if len(ctx.visitedList) < currentNodes {
		// Grow by at least 10% or 1024 elements to amortize allocation cost
		grow := currentNodes / 10
		if grow < 1024 {
			grow = 1024
		}
		newCap := currentNodes + grow

		// Create new slice (zero-initialized).
		// NO COPY NEEDED: The logic relies on 'visitedToken'.
		// Since new slice is all 0s, and current token > 0, checks will correctly return "not visited".
		ctx.visitedList = make([]uint32, newCap)
	}

	ctx.visitedList[entryPoint] = ctx.visitedToken

	// 2. Initialize Heaps
	dist := idx.distQueryFunc(query, idx.nodes[entryPoint].Vec)
	item := pqItem{id: entryPoint, score: dist}

	heap.Push(ctx.candidates, item)
	heap.Push(ctx.results, item)

	// 3. Beam Search
	for ctx.candidates.Len() > 0 {
		curr := heap.Pop(ctx.candidates).(pqItem)
		furthestResult := (*ctx.results)[0] // MinHeap Top is the WORST score

		if curr.score < furthestResult.score && ctx.results.Len() >= ef {
			break
		}

		for _, neighborID := range idx.nodes[curr.id].Neighbors[level] {
			if ctx.visitedList[neighborID] != ctx.visitedToken {
				ctx.visitedList[neighborID] = ctx.visitedToken
				dist := idx.distQueryFunc(query, idx.nodes[neighborID].Vec)
				newItem := pqItem{id: neighborID, score: dist}

				if ctx.results.Len() < ef || dist > (*ctx.results)[0].score {
					heap.Push(ctx.candidates, newItem)
					heap.Push(ctx.results, newItem)
					if ctx.results.Len() > ef {
						heap.Pop(ctx.results)
					}
				}
			}
		}
	}
}

// addConnection connects two nodes at a specific level, pruning if necessary
func (idx *HNSWIndex[T]) addConnection(ctx *searchCtx, level, from, to int) {
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

		// REUSE HEAP from context
		// We know ctx.results is a MinHeap. We clear it first just in case.
		// (Though internalAdd flow usually clears it after usage, safety first)
		h := ctx.results
		*h = (*h)[:0] // Reset

		for _, neighborID := range node.Neighbors[level] {
			// Node to Node distance
			score := idx.distNodeFunc(fromVec, idx.nodes[neighborID].Vec)
			heap.Push(h, pqItem{id: neighborID, score: score})

			// Keep only top maxM (Best scores).
			// MinHeap pops the Lowest score.
			// Wait: We want to KEEP the Highest scores (Best neighbors).
			// If we use MinHeap, Pop() removes the WORST candidate.
			// So if Len > maxM, Pop() removes the one with lowest similarity. Correct.
			if h.Len() > maxM {
				heap.Pop(h)
			}
		}

		// Reconstruct neighbors list
		// Reuse buffer for neighbors list?
		// node.Neighbors[level] is the persistent storage, we must overwrite it.
		// We can reuse the existing slice capacity if possible, but simpler to just alloc the small slice here.
		newNeighbors := make([]int, h.Len())
		for i := 0; i < len(newNeighbors); i++ {
			newNeighbors[i] = (*h)[i].id
		}
		node.Neighbors[level] = newNeighbors

		// Clean up heap for next use
		*h = (*h)[:0]
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

	searchCtx := idx.searchCtxPool.Get().(*searchCtx)
	defer idx.searchCtxPool.Put(searchCtx)

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

	idx.searchLayer(searchCtx, normalizedQuery, currObj, efSearch, 0)

	// Heap contents are not sorted, need to pop them
	finalCandidates := make([]pqItem, searchCtx.results.Len())
	for i := searchCtx.results.Len() - 1; i >= 0; i-- {
		finalCandidates[i] = heap.Pop(searchCtx.results).(pqItem)
	}

	results := make([]types.SearchResult, 0, k)
	for _, c := range finalCandidates {
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

// Save
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
	if err := enc.Encode(idx.M); err != nil {
		return err
	}
	if err := enc.Encode(idx.EfConstruction); err != nil {
		return err
	}
	if err := enc.Encode(idx.LevelMult); err != nil {
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

	// Save Nodes
	// ID: [Len (4b)] [Bytes...]
	// Level: [Level (4b)]
	// Neighbors: [NumLayers (4b)] -> Loop: [LayerCount (4b)] -> [NeighborIDs... (4b each)]
	// Vector: [Binary Data]
	neighborBuf := make([]int32, 0, idx.MMax0)
	for _, node := range idx.nodes {

		idBytes := []byte(node.ID)
		if err := binary.Write(f, binary.LittleEndian, int32(len(idBytes))); err != nil {
			return err
		}
		if _, err := f.Write(idBytes); err != nil {
			return err
		}

		if err := binary.Write(f, binary.LittleEndian, int32(node.Level)); err != nil {
			return err
		}

		numLayers := int32(len(node.Neighbors))
		if err := binary.Write(f, binary.LittleEndian, numLayers); err != nil {
			return err
		}

		for _, layer := range node.Neighbors {
			layerCount := int32(len(layer))
			if err := binary.Write(f, binary.LittleEndian, layerCount); err != nil {
				return err
			}

			neighborBuf = neighborBuf[:0]
			for _, nID := range layer {
				neighborBuf = append(neighborBuf, int32(nID))
			}

			if err := binary.Write(f, binary.LittleEndian, neighborBuf); err != nil {
				return err
			}
		}

		if err := binary.Write(f, binary.LittleEndian, node.Vec); err != nil {
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
	if err := dec.Decode(&idx.M); err != nil {
		return err
	}
	idx.MMax0 = idx.M * 2
	if err := dec.Decode(&idx.EfConstruction); err != nil {
		return err
	}
	if err := dec.Decode(&idx.LevelMult); err != nil {
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
		var idLen int32
		if err := binary.Read(f, binary.LittleEndian, &idLen); err != nil {
			return err
		}

		idBytes := make([]byte, idLen)
		if _, err := f.Read(idBytes); err != nil {
			return err
		}
		id := string(idBytes)

		var level int32
		if err := binary.Read(f, binary.LittleEndian, &level); err != nil {
			return err
		}

		var numLayers int32
		if err := binary.Read(f, binary.LittleEndian, &numLayers); err != nil {
			return err
		}

		neighbors := make([][]int, numLayers)
		for l := 0; l < int(numLayers); l++ {
			var layerCount int32
			if err := binary.Read(f, binary.LittleEndian, &layerCount); err != nil {
				return err
			}

			layerInt32 := make([]int32, layerCount)
			if err := binary.Read(f, binary.LittleEndian, &layerInt32); err != nil {
				return err
			}

			// Convert back to int
			layer := make([]int, layerCount)
			for k, v := range layerInt32 {
				layer[k] = int(v)
			}
			neighbors[l] = layer
		}

		vec := make([]T, idx.dim)
		if err := binary.Read(f, binary.LittleEndian, &vec); err != nil {
			return err
		}

		node := &Node[T]{
			ID:        id,
			Vec:       vec,
			Level:     int(level),
			Neighbors: neighbors,
		}
		idx.nodes[i] = node
		idx.idMap[node.ID] = i
	}

	idx.Metadata = make(map[string]map[string]any)
	if err := dec.Decode(&idx.Metadata); err != nil {
		return err
	}

	// Clean dead IDs
	for id := range idx.idMap {
		if _, isLive := idx.Metadata[id]; !isLive {
			delete(idx.idMap, id)
		}
	}
	return nil
}
