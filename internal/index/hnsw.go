package index

import (
	"container/heap"
	"encoding/binary"
	"encoding/gob"
	"fmt"
	"math"
	"os"
	"runtime"
	"slices"
	"sync"

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
	Neighbors [][]int      // [level][neighbor_internal_index]
	mu        sync.RWMutex // Fine-grained lock for this specific node
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
				return newSearchCtx(m * 2)
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

// Add inserts a single vector into the HNSW graph (Thread-safe & Compatible with Parallel AddBatch)
func (idx *HNSWIndex[T]) Add(id string, vec types.Vector, meta map[string]any) error {
	ctx := idx.searchCtxPool.Get().(*searchCtx)
	defer idx.searchCtxPool.Put(ctx)

	// --- PHASE 1: PRE-ALLOCATION (Global Lock) ---
	idx.mu.Lock()

	// Check duplicates
	if _, exists := idx.idMap[id]; exists {
		idx.mu.Unlock()
		return fmt.Errorf("id %s already exists", id)
	}

	navQuery := make([]float32, len(vec))
	copy(navQuery, vec)
	maths.NormalizeInPlace(navQuery)
	storedVec := idx.convertFunc(navQuery)

	level := ctx.randomLevel()
	node := &Node[T]{
		ID:        id,
		Vec:       storedVec,
		Level:     level,
		Neighbors: make([][]int, level+1),
	}

	// Update Global State
	internalID := len(idx.nodes)
	idx.nodes = append(idx.nodes, node)
	idx.idMap[id] = internalID
	idx.Metadata[id] = meta

	// Update entry point if needed
	if idx.enterPoint == -1 {
		idx.enterPoint = internalID
		idx.maxLevel = level
	} else if level > idx.maxLevel {
		idx.maxLevel = level
		idx.enterPoint = internalID
	}

	// Release Global Lock immediately!
	// So that other threads (Search or AddBatch) can continue running
	idx.mu.Unlock()

	// --- PHASE 2: LINKING (Fine-Grained Lock) ---
	idx.parallelLink(ctx, node, navQuery)

	return nil
}

// AddBatch adds multiple vectors
func (idx *HNSWIndex[T]) AddBatch(ids []string, vecs []types.Vector, metas []map[string]any) error {
	// --- PHASE 1: PRE-ALLOCATION (Global Lock) ---
	idx.mu.Lock()

	for _, id := range ids {
		if _, exists := idx.idMap[id]; exists {
			idx.mu.Unlock()
			return fmt.Errorf("id %s already exists", id)
		}
	}

	startIdx := len(idx.nodes)
	count := len(ids)
	idx.nodes = slices.Grow(idx.nodes, count)

	rngCtx := idx.searchCtxPool.Get().(*searchCtx)

	newNodes := make([]*Node[T], count)
	normalizedVecs := make([][]float32, count)

	for i := 0; i < count; i++ {
		// Normalize
		nv := make([]float32, idx.dim)
		copy(nv, vecs[i])
		maths.NormalizeInPlace(nv)
		normalizedVecs[i] = nv

		storedVec := idx.convertFunc(nv)

		level := rngCtx.randomLevel()

		node := &Node[T]{
			ID:        ids[i],
			Vec:       storedVec,
			Level:     level,
			Neighbors: make([][]int, level+1),
		}

		// Update Global State
		internalID := startIdx + i
		idx.nodes = append(idx.nodes, node)
		idx.idMap[ids[i]] = internalID
		idx.Metadata[ids[i]] = metas[i]
		newNodes[i] = node

		// Update maxLevel globally
		if level > idx.maxLevel {
			idx.maxLevel = level
			idx.enterPoint = internalID // Simplification: New highest node becomes EP
		}
	}

	idx.searchCtxPool.Put(rngCtx)

	// Release Global Lock immediately
	// Now other readers can access existing nodes, and we can link new nodes in parallel
	idx.mu.Unlock()

	// --- PHASE 2: PARALLEL LINKING (Fine-Grained) ---
	var wg sync.WaitGroup
	sem := make(chan struct{}, runtime.GOMAXPROCS(0))

	for i := 0; i < count; i++ {
		wg.Add(1)
		sem <- struct{}{}

		go func(i int) {
			defer wg.Done()
			defer func() { <-sem }()

			node := newNodes[i]
			qVec := normalizedVecs[i]

			workerCtx := idx.searchCtxPool.Get().(*searchCtx)
			defer idx.searchCtxPool.Put(workerCtx)

			idx.parallelLink(workerCtx, node, qVec)

		}(i)
	}
	wg.Wait()

	return nil
}

// parallelLink inserts a node into the graph using fine-grained locking
func (idx *HNSWIndex[T]) parallelLink(ctx *searchCtx, node *Node[T], query []float32) {
	// We read enterPoint atomically (or just accept it might change slightly during batch)
	// Since we are in the same batch, using the updated EP from Phase 1 is fine.
	currObj := idx.enterPoint

	// Handle edge case: First node or self-reference
	if currObj == -1 || currObj == idx.idMap[node.ID] {
		return
	}

	currDist := idx.distQueryFunc(query, idx.nodes[currObj].Vec)

	// 1. Greedy Search (Standard)
	for l := idx.maxLevel; l > node.Level; l-- {
		changed := true
		for changed {
			changed = false

			// FINE-GRAINED READ LOCK
			// We must lock the current node to safely read its neighbors
			currNode := idx.nodes[currObj]
			currNode.mu.RLock()

			neighbors := make([]int, len(currNode.Neighbors[l]))
			copy(neighbors, currNode.Neighbors[l])
			currNode.mu.RUnlock()

			for _, neighborID := range neighbors {
				dist := idx.distQueryFunc(query, idx.nodes[neighborID].Vec)
				if dist > currDist {
					currDist = dist
					currObj = neighborID
					changed = true
				}
			}
		}
	}

	// 2. Link at each level
	for l := int(math.Min(float64(node.Level), float64(idx.maxLevel))); l >= 0; l-- {
		// Search layer needs to be updated to use RLock internally
		idx.searchLayerLocked(ctx, query, currObj, idx.EfConstruction, l)

		neighbors := idx.selectNeighborsFromHeap(ctx, ctx.results, idx.M)

		// Bi-directional Connection
		internalID := idx.idMap[node.ID]
		for _, neighborID := range neighbors {
			// Link A -> B
			idx.addConnectionLocked(ctx, l, internalID, neighborID)
			// Link B -> A
			idx.addConnectionLocked(ctx, l, neighborID, internalID)
		}

		// Update currObj for next layer...
		if ctx.results.Len() > 0 {
			// Accessing backing slice directly if possible, or Peek.
			// In MinHeap (results), the TOP is the FURTHEST (worst).
			// We need the CLOSEST. The heap stores 'ef' best candidates.
			// Any element in heap is good, but we want a good start point.
			// Ideally we pick the closest one.
			// Iterating the heap to find min dist is cheap (ef is small).
			bestID := -1
			bestDist := float32(-1)

			for _, item := range *ctx.results {
				if bestID == -1 || item.score > bestDist { // Score is similarity (higher better)
					bestDist = item.score
					bestID = item.id
				}
			}
			currObj = bestID
		}
	}
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

// searchLayerLocked performs BFS with priority queue (Beam Search) but locks nodes when reading neighbors
// Returns list of candidates sorted by Score DESC (Closest first)
func (idx *HNSWIndex[T]) searchLayerLocked(ctx *searchCtx, query types.Vector, entryPoint, ef, level int) {
	ctx.reset()

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

		// Lock node to read neighbors
		node := idx.nodes[curr.id]
		node.mu.RLock()

		// Iterate neighbors directly inside lock is okay if callback is fast
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

		node.mu.RUnlock()
	}
}

// addConnectionLocked connects two nodes safely
func (idx *HNSWIndex[T]) addConnectionLocked(ctx *searchCtx, level, from, to int) {
	node := idx.nodes[from]

	// Lock write: Only lock the 'from' node
	// This prevents deadlock because we never hold 2 node locks at the same time
	node.mu.Lock()
	defer node.mu.Unlock()

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
		// Heuristic Pruning: Keep the best
		fromVec := idx.nodes[from].Vec

		// Reuse Heap in context (save alloc)
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

	idx.searchLayerLocked(searchCtx, normalizedQuery, currObj, efSearch, 0)

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

		switch v := any(node.Vec).(type) {
		case []float32:
			if err := binary.Write(f, binary.LittleEndian, v); err != nil {
				return err
			}
		case []uint8:
			if _, err := f.Write(v); err != nil {
				return err
			}
		default:
			if err := binary.Write(f, binary.LittleEndian, node.Vec); err != nil {
				return err
			}
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

	fi, err := f.Stat()
	if err != nil {
		return err
	}
	fileSize := fi.Size()

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

	if count < 0 {
		return fmt.Errorf("corrupted file: negative node count %d", count)
	}

	if count > 100_000_000 {
		return fmt.Errorf("node count exceeds limit")
	}

	minBytesPerNode := int64(12 + idx.dim)
	minRequiredSize := int64(count) * minBytesPerNode
	if fileSize < minRequiredSize {
		return fmt.Errorf("corrupted file: node count %d requires at least %d bytes, but file is only %d bytes",
			count, minRequiredSize, fileSize)
	}

	idx.nodes = make([]*Node[T], count)
	idx.idMap = make(map[string]int)

	for i := 0; i < count; i++ {
		var idLen int32
		if err := binary.Read(f, binary.LittleEndian, &idLen); err != nil {
			return err
		}

		if idLen < 0 || idLen > types.KB {
			return fmt.Errorf("corrupted file: invalid id length %d at index %d", idLen, i)
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

		if numLayers < 0 || numLayers > 100 {
			return fmt.Errorf("corrupted file: invalid numLayers %d at index %d", numLayers, i)
		}

		neighbors := make([][]int, numLayers)
		for l := 0; l < int(numLayers); l++ {
			var layerCount int32
			if err := binary.Read(f, binary.LittleEndian, &layerCount); err != nil {
				return err
			}

			// Check Layer Count sanity (cannot be much greater than MMax0)
			if layerCount < 0 || layerCount > 10000 {
				return fmt.Errorf("corrupted file: invalid layerCount %d at index %d layer %d", layerCount, i, l)
			}

			layerInt32 := make([]int32, layerCount)
			if err := binary.Read(f, binary.LittleEndian, &layerInt32); err != nil {
				return err
			}

			layer := make([]int, layerCount)
			for k, v := range layerInt32 {
				layer[k] = int(v)
			}
			neighbors[l] = layer
		}

		vec := make([]T, idx.dim)
		switch v := any(vec).(type) {
		case []float32:
			if err := binary.Read(f, binary.LittleEndian, v); err != nil {
				return err
			}
		case []uint8:
			if _, err := f.Read(v); err != nil {
				return err
			}
		default:
			if err := binary.Read(f, binary.LittleEndian, &vec); err != nil {
				return err
			}
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
