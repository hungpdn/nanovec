package index

import (
	"bufio"
	"cmp"
	"container/heap"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"runtime"
	"slices"
	"sync"

	"github.com/hungpdn/nanovec/pkg/bitset"
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
	tombstones *bitset.BitSet // Bitset to mark deleted internal IDs (Ghost Nodes)
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
	distNodeFunc func(a, b *Node[T]) float32
	// distQuantizedFunc calculates distance between quantized query and node (SQ8 only)
	distQuantizedFunc func(query []uint8, node []uint8, qSum, nSum uint32) float32
}

// Node represents a point in the graph
type Node[T types.Number] struct {
	ID        string
	Vec       []T
	VecSum    uint32 // Pre-computed sum for SQ8 (0 for Float32)
	Level     int
	Neighbors [][]int      // [level][neighbor_internal_index]
	mu        sync.RWMutex // Fine-grained lock for this specific node
}

// NewHNSWIndexFloat creates a standard Float32 HNSW index
func NewHNSWIndexFloat(dim, m, efConstruction int) *HNSWIndex[float32] {
	idx := &HNSWIndex[float32]{
		dim:            dim,
		M:              m,
		MMax0:          m * 2,
		EfConstruction: efConstruction,
		LevelMult:      1.0 / math.Log(float64(m)),
		nodes:          make([]*Node[float32], 0),
		idMap:          make(map[string]int),
		tombstones:     bitset.New(0),
		Metadata:       make(map[string]map[string]any),
		enterPoint:     -1,
		maxLevel:       -1,
		convertFunc: func(normVec []float32) []float32 {
			out := make([]float32, len(normVec))
			copy(out, normVec)
			return out
		},
		distQueryFunc: maths.DotProduct,
		distNodeFunc: func(a, b *Node[float32]) float32 {
			return maths.DotProduct(a.Vec, b.Vec)
		},
		searchCtxPool: &sync.Pool{New: func() any { return newSearchCtx(m * 2) }},
	}
	return idx
}

// NewHNSWIndexSQ8 creates a sq8 HNSW index
func NewHNSWIndexSQ8(dim, m, efConstruction int) *HNSWIndex[uint8] {
	idx := &HNSWIndex[uint8]{
		dim:            dim,
		M:              m,
		MMax0:          m * 2,
		EfConstruction: efConstruction,
		LevelMult:      1.0 / math.Log(float64(m)),
		nodes:          make([]*Node[uint8], 0),
		idMap:          make(map[string]int),
		tombstones:     bitset.New(0),
		Metadata:       make(map[string]map[string]any),
		enterPoint:     -1,
		maxLevel:       -1,
		convertFunc: func(normVec []float32) []uint8 {
			return maths.QuantizeSQ8(normVec)
		},
		distQueryFunc: maths.DotProductSQ8,
		distNodeFunc: func(a, b *Node[uint8]) float32 {
			return maths.DotProductUint8Precomputed(a.Vec, b.Vec, a.VecSum, b.VecSum)
		},
		distQuantizedFunc: func(q, n []uint8, qSum, nSum uint32) float32 {
			return maths.DotProductUint8Precomputed(q, n, qSum, nSum)
		},
		searchCtxPool: &sync.Pool{New: func() any { return newSearchCtx(m * 2) }},
	}
	return idx
}

// Add inserts a single vector
func (idx *HNSWIndex[T]) Add(id string, vec types.Vector, meta map[string]any) error {
	ctx := idx.searchCtxPool.Get().(*searchCtx)
	defer idx.searchCtxPool.Put(ctx)

	// Phase 1: pre-allocation (global lock)
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

	var vecSum uint32
	if v, ok := any(storedVec).([]uint8); ok {
		vecSum = maths.CalculateVecSum(v)
	}

	level := ctx.randomLevel()
	node := &Node[T]{
		ID:        id,
		Vec:       storedVec,
		VecSum:    vecSum,
		Level:     level,
		Neighbors: make([][]int, level+1),
	}

	// Update Global State
	internalID := len(idx.nodes)
	idx.tombstones.Grow(internalID + 1)
	idx.nodes = append(idx.nodes, node)
	idx.idMap[id] = internalID
	idx.Metadata[id] = meta

	// Update entry point if needed
	updateEP := false
	if idx.enterPoint == -1 {
		idx.enterPoint = internalID
		idx.maxLevel = level
	} else if level > idx.maxLevel {
		updateEP = true
	}

	// Release Global Lock immediately!
	// So that other threads (Search or AddBatch) can continue running
	idx.mu.Unlock()

	// Phase 2: Parallel Linking (Fine-Grained)
	idx.mu.RLock()
	idx.parallelLink(ctx, node, navQuery)
	idx.mu.RUnlock()

	// Phase 3: Finalize state (global lock)
	if updateEP {
		idx.mu.Lock()
		// Check again in case another thread updated it higher
		if level > idx.maxLevel {
			idx.maxLevel = level
			idx.enterPoint = internalID
		}
		idx.mu.Unlock()
	}
	return nil
}

// AddBatch adds multiple vectors
func (idx *HNSWIndex[T]) AddBatch(ids []string, vecs []types.Vector, metas []map[string]any) error {
	// Phase 1: pre-allocation (global lock)
	idx.mu.Lock()

	for _, id := range ids {
		if _, exists := idx.idMap[id]; exists {
			idx.mu.Unlock()
			return fmt.Errorf("id %s already exists", id)
		}
	}

	startIdx := len(idx.nodes)
	count := len(ids)
	idx.tombstones.Grow(startIdx + count)
	idx.nodes = slices.Grow(idx.nodes, count)

	rngCtx := idx.searchCtxPool.Get().(*searchCtx)

	newNodes := make([]*Node[T], count)
	flatBuf := make([]float32, count*idx.dim)
	normalizedVecs := make([][]float32, count)

	// Local variables for Batch Global State Update
	var batchMaxLevel = -1
	var batchEP = -1

	for i := 0; i < count; i++ {
		start := i * idx.dim
		end := start + idx.dim
		nv := flatBuf[start:end]

		copy(nv, vecs[i])
		maths.NormalizeInPlace(nv)
		normalizedVecs[i] = nv

		storedVec := idx.convertFunc(nv)

		var vecSum uint32
		if v, ok := any(storedVec).([]uint8); ok {
			vecSum = maths.CalculateVecSum(v)
		}

		level := rngCtx.randomLevel()

		node := &Node[T]{
			ID:        ids[i],
			Vec:       storedVec,
			VecSum:    vecSum,
			Level:     level,
			Neighbors: make([][]int, level+1),
		}

		// Update Global State
		internalID := startIdx + i
		idx.nodes = append(idx.nodes, node)
		idx.idMap[ids[i]] = internalID
		idx.Metadata[ids[i]] = metas[i]
		newNodes[i] = node

		// Track max level change LOCALLY
		if level > batchMaxLevel {
			batchMaxLevel = level
			batchEP = internalID
		}

		// Edge case: First node ever -> Set EP immediately
		if idx.enterPoint == -1 {
			idx.enterPoint = internalID
			idx.maxLevel = level
		}
	}

	idx.searchCtxPool.Put(rngCtx)
	idx.mu.Unlock()

	// Phase 2: Parallel Linking (Fine-Grained)
	idx.mu.RLock()
	var wg sync.WaitGroup
	sem := make(chan struct{}, runtime.GOMAXPROCS(0))
	for i := 0; i < count; i++ {
		wg.Add(1)
		sem <- struct{}{} // Block here if full

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
	idx.mu.RUnlock()

	// Phase 3: Finalize Global State
	// Update EP only after connections are established to avoid "Island Nodes"
	idx.mu.Lock()
	if batchMaxLevel > idx.maxLevel {
		idx.maxLevel = batchMaxLevel
		idx.enterPoint = batchEP
	}
	idx.mu.Unlock()
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
		// Fast Copy Neighbors (Critical Section - Very Short)
		// We copy the slice so we can release the lock immediately.
		// node.Neighbors[level] usage inside lock is safe.
		// Allocation here is cheap compared to holding lock during distance calc.
		// To further optimize, we could reuse a buffer from ctx, but a small slice alloc is negligible.
		node.mu.RLock()
		neighbors := make([]int, len(node.Neighbors[level]))
		copy(neighbors, node.Neighbors[level])
		node.mu.RUnlock()

		// Iterate neighbors directly inside lock is okay if callback is fast
		for _, neighborID := range neighbors {
			if ctx.visitedList[neighborID] != ctx.visitedToken {
				ctx.visitedList[neighborID] = ctx.visitedToken

				// Heavy math calculation happens here, fully parallel
				// No other threads are blocked on 'node' while we do this
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

// addConnectionLocked connects two nodes safely
func (idx *HNSWIndex[T]) addConnectionLocked(ctx *searchCtx, level, from, to int) {
	node := idx.nodes[from]
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
		fromNode := idx.nodes[from]

		h := ctx.results
		*h = (*h)[:0] // Reset

		for _, neighborID := range node.Neighbors[level] {
			score := idx.distNodeFunc(fromNode, idx.nodes[neighborID])
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
		*h = (*h)[:0]
	}
}

// Search performs Approximate Nearest Neighbor search
// Note: 'query' must be already normalized
func (idx *HNSWIndex[T]) Search(query types.Vector, k int, filter types.FilterFunc) ([]types.SearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if idx.enterPoint == -1 {
		return []types.SearchResult{}, nil
	}

	normalizedQuery := query

	var q8 []uint8
	var qSum uint32
	useQuantizedSearch := false

	if idx.distQuantizedFunc != nil {
		q8 = maths.QuantizeSQ8(query)
		qSum = maths.CalculateVecSum(q8)
		useQuantizedSearch = true
	}

	distFunc := func(nodeIdx int) float32 {
		node := idx.nodes[nodeIdx]
		if useQuantizedSearch {
			n := any(node.Vec).([]uint8)
			return idx.distQuantizedFunc(q8, n, qSum, node.VecSum)
		}
		return idx.distQueryFunc(normalizedQuery, node.Vec)
	}

	searchCtx := idx.searchCtxPool.Get().(*searchCtx)
	defer idx.searchCtxPool.Put(searchCtx)

	currObj := idx.enterPoint
	currDist := distFunc(currObj)

	for l := idx.maxLevel; l > 0; l-- {
		changed := true
		for changed {
			changed = false

			// We must RLock the node to safely read its Neighbors slice,
			// because Add() might be modifying it concurrently.
			currNode := idx.nodes[currObj]
			currNode.mu.RLock()
			neighbors := make([]int, len(currNode.Neighbors[l]))
			copy(neighbors, currNode.Neighbors[l])
			currNode.mu.RUnlock()

			for _, neighborID := range neighbors {
				dist := distFunc(neighborID)
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

	candidates := *searchCtx.results
	slices.SortFunc(candidates, func(a, b pqItem) int {
		return cmp.Compare(b.score, a.score)
	})

	results := make([]types.SearchResult, 0, k)
	for _, c := range candidates {

		if idx.tombstones.IsSet(c.id) {
			continue
		}

		id := idx.nodes[c.id].ID

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
	internalID, exists := idx.idMap[id]
	if !exists {
		return nil
	}
	idx.tombstones.Set(internalID) // Soft Delete
	delete(idx.idMap, id)
	delete(idx.Metadata, id)
	return nil
}

// Save stores the index to disk using Buffered I/O + Binary + JSON
func (idx *HNSWIndex[T]) Save(path string) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	tmpPath := path + ".tmp"
	f, err := os.Create(tmpPath)
	if err != nil {
		return err
	}
	defer f.Close()

	w := bufio.NewWriterSize(f, 64*types.KB)

	// 1. Header Params (Write to 'w' instead of 'f')
	if err = binary.Write(w, binary.LittleEndian, idx.Version); err != nil {
		return err
	}
	if err = binary.Write(w, binary.LittleEndian, int32(idx.dim)); err != nil {
		return err
	}
	if err = binary.Write(w, binary.LittleEndian, int32(idx.M)); err != nil {
		return err
	}
	if err = binary.Write(w, binary.LittleEndian, int32(idx.EfConstruction)); err != nil {
		return err
	}
	if err = binary.Write(w, binary.LittleEndian, idx.LevelMult); err != nil {
		return err
	}
	if err = binary.Write(w, binary.LittleEndian, int32(idx.enterPoint)); err != nil {
		return err
	}
	if err = binary.Write(w, binary.LittleEndian, int32(idx.maxLevel)); err != nil {
		return err
	}

	count := len(idx.nodes)
	if err = binary.Write(w, binary.LittleEndian, int32(count)); err != nil {
		return err
	}

	// 2. Nodes (Write to 'w')
	neighborBuf := make([]int32, 0, idx.MMax0)
	for _, node := range idx.nodes {

		idBytes := []byte(node.ID)
		if err = binary.Write(w, binary.LittleEndian, int32(len(idBytes))); err != nil {
			return err
		}
		if _, err = w.Write(idBytes); err != nil {
			return err
		}
		if err = binary.Write(w, binary.LittleEndian, uint32(node.VecSum)); err != nil {
			return err
		}
		if err = binary.Write(w, binary.LittleEndian, int32(node.Level)); err != nil {
			return err
		}
		numLayers := int32(len(node.Neighbors))
		if err = binary.Write(w, binary.LittleEndian, numLayers); err != nil {
			return err
		}

		for _, layer := range node.Neighbors {
			layerCount := int32(len(layer))
			if err = binary.Write(w, binary.LittleEndian, layerCount); err != nil {
				return err
			}

			neighborBuf = neighborBuf[:0]
			for _, nID := range layer {
				neighborBuf = append(neighborBuf, int32(nID))
			}
			if err = binary.Write(w, binary.LittleEndian, neighborBuf); err != nil {
				return err
			}
		}

		switch v := any(node.Vec).(type) {
		case []float32:
			if err = binary.Write(w, binary.LittleEndian, v); err != nil {
				return err
			}
		case []uint8:
			if _, err = w.Write(v); err != nil {
				return err
			}
		default:
			if err = binary.Write(w, binary.LittleEndian, node.Vec); err != nil {
				return err
			}
		}
	}

	// 3. Metadata (JSON) to 'w'
	metaBytes, err := json.Marshal(idx.Metadata)
	if err != nil {
		return err
	}
	if err = binary.Write(w, binary.LittleEndian, int32(len(metaBytes))); err != nil {
		return err
	}
	if _, err = w.Write(metaBytes); err != nil {
		return err
	}

	// 4. Tombstones (Binary) to 'w'
	if err = idx.tombstones.Save(w); err != nil {
		return err
	}

	if err = w.Flush(); err != nil {
		return err
	}

	if err = f.Sync(); err != nil {
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}
	return os.Rename(tmpPath, path)
}

// Load loads the index from disk using Buffered I/O + Binary + JSON
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

	r := bufio.NewReaderSize(f, 64*types.KB)

	// 1. Header (Read from 'r' instead of 'f')
	if err := binary.Read(r, binary.LittleEndian, &idx.Version); err != nil {
		return err
	}

	var dim32, m32, ef32, ep32, maxLvl32, count32 int32
	if err := binary.Read(r, binary.LittleEndian, &dim32); err != nil {
		return err
	}
	idx.dim = int(dim32)

	if err := binary.Read(r, binary.LittleEndian, &m32); err != nil {
		return err
	}
	idx.M = int(m32)
	idx.MMax0 = idx.M * 2

	if err := binary.Read(r, binary.LittleEndian, &ef32); err != nil {
		return err
	}
	idx.EfConstruction = int(ef32)

	if err := binary.Read(r, binary.LittleEndian, &idx.LevelMult); err != nil {
		return err
	}

	if err := binary.Read(r, binary.LittleEndian, &ep32); err != nil {
		return err
	}
	idx.enterPoint = int(ep32)

	if err := binary.Read(r, binary.LittleEndian, &maxLvl32); err != nil {
		return err
	}
	idx.maxLevel = int(maxLvl32)

	if err := binary.Read(r, binary.LittleEndian, &count32); err != nil {
		return err
	}

	count := int(count32)
	if count < 0 {
		return fmt.Errorf("corrupted file: negative node count %d", count)
	}

	minBytesPerNode := int64(12 + idx.dim)
	minRequiredSize := int64(count) * minBytesPerNode
	if fileSize < minRequiredSize {
		return fmt.Errorf("corrupted file: too small")
	}

	idx.nodes = make([]*Node[T], count)
	idx.idMap = make(map[string]int)

	// 2. Nodes (Read from 'r')
	nodePool := make([]Node[T], count)
	for i := 0; i < count; i++ {
		var idLen int32
		if err := binary.Read(r, binary.LittleEndian, &idLen); err != nil {
			return err
		}
		if idLen < 0 || idLen > 4096 {
			return fmt.Errorf("invalid id len")
		}

		idBytes := make([]byte, idLen)
		if _, err := io.ReadFull(r, idBytes); err != nil {
			return err
		}
		id := string(idBytes)

		var vecSum uint32
		if err := binary.Read(r, binary.LittleEndian, &vecSum); err != nil {
			return err
		}

		var level int32
		if err := binary.Read(r, binary.LittleEndian, &level); err != nil {
			return err
		}

		var numLayers int32
		if err := binary.Read(r, binary.LittleEndian, &numLayers); err != nil {
			return err
		}

		neighbors := make([][]int, numLayers)
		for l := 0; l < int(numLayers); l++ {
			var layerCount int32
			if err := binary.Read(r, binary.LittleEndian, &layerCount); err != nil {
				return err
			}

			layerInt32 := make([]int32, layerCount)
			if err := binary.Read(r, binary.LittleEndian, &layerInt32); err != nil {
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
			if err := binary.Read(r, binary.LittleEndian, v); err != nil {
				return err
			}
		case []uint8:
			if _, err := io.ReadFull(r, v); err != nil {
				return err
			}
		default:
			if err := binary.Read(r, binary.LittleEndian, &vec); err != nil {
				return err
			}
		}

		node := &nodePool[i]
		node.ID = id
		node.Vec = vec
		node.VecSum = vecSum
		node.Level = int(level)
		node.Neighbors = neighbors

		idx.nodes[i] = node
		idx.idMap[node.ID] = i
	}

	// 3. Metadata (JSON) from 'r'
	var metaLen int32
	if err := binary.Read(r, binary.LittleEndian, &metaLen); err != nil {
		return err
	}
	if metaLen < 0 {
		return fmt.Errorf("invalid meta len")
	}
	metaBytes := make([]byte, metaLen)
	if _, err := io.ReadFull(r, metaBytes); err != nil {
		return err
	}
	idx.Metadata = make(map[string]map[string]any)
	if err := json.Unmarshal(metaBytes, &idx.Metadata); err != nil {
		return err
	}

	// 4. Tombstones (Binary) from 'r'
	idx.tombstones = bitset.New(0)
	if err := idx.tombstones.Load(r); err != nil {
		return fmt.Errorf("failed to load tombstones: %w", err)
	}

	// Clean ghosts
	for id := range idx.idMap {
		if _, ok := idx.Metadata[id]; !ok {
			delete(idx.idMap, id)
		}
	}

	return nil
}
