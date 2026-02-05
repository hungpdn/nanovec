package index

import (
	"container/heap"
	"encoding/gob"
	"fmt"
	"math"
	"os"
	"sync"

	"github.com/hungpdn/nanovec/pkg/errors"
	"github.com/hungpdn/nanovec/pkg/maths"
	"github.com/hungpdn/nanovec/pkg/types"
)

// HNSWIndex implements Hierarchical Navigable Small World graph.
type HNSWIndex struct {
	mu sync.RWMutex

	// Configuration
	dim            int
	M              int
	MMax0          int     // Max connections for layer 0 (2 * M)
	EfConstruction int     // Beam size for construction
	LevelMult      float64 // Normalization factor for level generation

	// Graph State
	nodes      []*hnswNode
	idMap      map[string]int // External ID -> Internal Index
	enterPoint int            // Internal Index of the entry node
	maxLevel   int            // Current max level in the graph

	// Optimization: Reuse visited maps to reduce GC pressure
	visitedPool sync.Pool

	// Metadata storage (Same as FlatIndex)
	Metadata map[string]map[string]any
}

// hnswNode represents a point in the graph
type hnswNode struct {
	id        string
	vec       types.Vector
	level     int
	neighbors [][]int // [level][neighbor_internal_index]
}

func NewHNSWIndex(dim, m, efConstruction int) *HNSWIndex {
	idx := &HNSWIndex{
		dim:            dim,
		M:              m,
		MMax0:          m * 2,
		EfConstruction: efConstruction,
		LevelMult:      1.0 / math.Log(float64(m)),
		nodes:          make([]*hnswNode, 0),
		idMap:          make(map[string]int),
		Metadata:       make(map[string]map[string]any),
		enterPoint:     -1,
		maxLevel:       -1,
	}

	idx.visitedPool.New = func() any {
		// Pre-allocate map with estimated capacity to avoid resize during traversal
		// Using EfConstruction as a heuristic for visited set size
		return make(map[int]bool, efConstruction)
	}

	return idx
}

// Add inserts a vector into the HNSW graph
func (idx *HNSWIndex) Add(id string, vec types.Vector, meta map[string]any) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if _, exists := idx.idMap[id]; exists {
		return fmt.Errorf("id %s already exists", id)
	}

	return idx.internalAdd(id, vec, meta)
}

// searchLayer performs BFS with priority queue (Beam Search)
// Returns list of candidates sorted by Score DESC (Closest first)
func (idx *HNSWIndex) searchLayer(query types.Vector, entryPoint, ef, level int) []pqItem {
	visited := idx.visitedPool.Get().(map[int]bool)
	defer func() {
		clear(visited)
		idx.visitedPool.Put(visited)
	}()

	visited[entryPoint] = true

	// Candidates queue (Min-Heap by definition, but we store items to Explore)
	// We want to explore Best nodes first. So we need a MaxHeap logic.
	// However, Go's heap is MinHeap. So we can negate score or implement MaxHeap.
	// Let's use `candidates` as a set of nodes to explore (MaxHeap: explore closest to query first).
	candidates := &MaxHeap{}
	heap.Init(candidates)

	// Results queue (Keep Top `ef` best nodes found so far).
	// We want to keep High Scores. If size > ef, we remove the Lowest Score.
	// So `results` should be a MinHeap (peek/pop lowest score).
	results := &MinHeap{}
	heap.Init(results)

	dist := maths.DotProduct(query, idx.nodes[entryPoint].vec)
	item := pqItem{id: entryPoint, score: dist}

	heap.Push(candidates, item)
	heap.Push(results, item)

	for candidates.Len() > 0 {
		// Get best candidate to explore
		curr := heap.Pop(candidates).(pqItem)
		furthestResult := (*results)[0] // Peek Min (Worst in our Top-ef)

		// If current candidate is worse than the worst result we already have,
		// and we have filled our buffer `ef`, we can stop exploring this branch.
		if curr.score < furthestResult.score && results.Len() >= ef {
			break
		}

		for _, neighborID := range idx.nodes[curr.id].neighbors[level] {
			if !visited[neighborID] {
				visited[neighborID] = true
				dist := maths.DotProduct(query, idx.nodes[neighborID].vec)
				newItem := pqItem{id: neighborID, score: dist}

				// If neighbor is better than worst result OR we have space
				if results.Len() < ef || dist > (*results)[0].score {
					heap.Push(candidates, newItem)
					heap.Push(results, newItem)

					if results.Len() > ef {
						heap.Pop(results) // Remove worst
					}
				}
			}
		}
	}

	// Convert heap to slice and sort DESC (Best first)
	finalRes := make([]pqItem, results.Len())
	for i := results.Len() - 1; i >= 0; i-- {
		finalRes[i] = heap.Pop(results).(pqItem)
	}
	return finalRes
}

// addConnection connects two nodes at a specific level, pruning if necessary
func (idx *HNSWIndex) addConnection(level, from, to int) {
	node := idx.nodes[from]
	// Check if already connected (linear scan is fast for small M)
	for _, neighbor := range node.neighbors[level] {
		if neighbor == to {
			return
		}
	}
	node.neighbors[level] = append(node.neighbors[level], to)

	// Pruning
	maxM := idx.M
	if level == 0 {
		maxM = idx.MMax0
	}

	if len(node.neighbors[level]) > maxM {
		// Re-evaluate neighbors to keep only the best M
		// Get all neighbor vectors and sort by distance to `from`
		fromVec := idx.nodes[from].vec

		// Use a MinHeap to keep Top M best neighbors
		h := &MinHeap{}
		heap.Init(h)

		for _, neighborID := range node.neighbors[level] {
			score := maths.DotProduct(fromVec, idx.nodes[neighborID].vec)
			heap.Push(h, pqItem{id: neighborID, score: score})
			if h.Len() > maxM {
				heap.Pop(h) // Drop worst (smallest score)
			}
		}

		// Rebuild slice
		newNeighbors := make([]int, h.Len())
		// Heap pops worst-to-best if we pop until empty, but we just need IDs
		// Order doesn't strictly matter for storage, but let's keep it clean
		for i := 0; i < len(newNeighbors); i++ {
			newNeighbors[i] = (*h)[i].id
		}
		node.neighbors[level] = newNeighbors
	}
}

// Search performs Approximate Nearest Neighbor search
func (idx *HNSWIndex) Search(query types.Vector, k int, filter types.FilterFunc) ([]types.SearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if idx.enterPoint == -1 {
		return []types.SearchResult{}, nil
	}

	normalizedQuery := make([]float32, len(query))
	copy(normalizedQuery, query)
	maths.NormalizeInPlace(normalizedQuery)

	currObj := idx.enterPoint
	currDist := maths.DotProduct(normalizedQuery, idx.nodes[currObj].vec)

	// 1. Zoom in: Go from Top level to Layer 1
	for l := idx.maxLevel; l > 0; l-- {
		changed := true
		for changed {
			changed = false
			for _, neighborID := range idx.nodes[currObj].neighbors[l] {
				dist := maths.DotProduct(normalizedQuery, idx.nodes[neighborID].vec)
				if dist > currDist {
					currDist = dist
					currObj = neighborID
					changed = true
				}
			}
		}
	}

	// 2. Layer 0 Search (Beam Search with k buffer)
	// We use searchLayer with ef = max(EfConstruction, k) usually,
	// but for Search API typically uses a dedicated EfSearch param.
	// For now let's use EfConstruction or k*2 as heuristic.
	efSearch := idx.EfConstruction
	if k > efSearch {
		efSearch = k
	}

	candidates := idx.searchLayer(normalizedQuery, currObj, efSearch, 0)

	// 3. Post-Process (Filter and Format)
	results := make([]types.SearchResult, 0, k)
	for _, c := range candidates {
		id := idx.nodes[c.id].id

		// Check for ghost nodes (Soft Deleted)
		// This filters out both "Deleted" items and "Stale" items (from updates)
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

// AddBatch adds multiple items
func (idx *HNSWIndex) AddBatch(ids []string, vecs []types.Vector, metas []map[string]any) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	for _, id := range ids {
		if _, exists := idx.idMap[id]; exists {
			return fmt.Errorf("id %s already exists", id)
		}
	}

	for i, id := range ids {
		if err := idx.internalAdd(id, vecs[i], metas[i]); err != nil {
			return err
		}
	}
	return nil
}

// internalAdd is the core Add logic without Locking (Helper)
func (idx *HNSWIndex) internalAdd(id string, vec types.Vector, meta map[string]any) error {
	if len(vec) != idx.dim {
		return errors.ErrDimMismatch
	}

	level := randomLevel()
	internalID := len(idx.nodes)

	node := &hnswNode{
		id:        id,
		vec:       vec,
		level:     level,
		neighbors: make([][]int, level+1),
	}
	maths.NormalizeInPlace(node.vec)

	idx.nodes = append(idx.nodes, node)
	idx.idMap[id] = internalID
	idx.Metadata[id] = meta

	if idx.enterPoint == -1 {
		idx.enterPoint = internalID
		idx.maxLevel = level
		return nil
	}

	currObj := idx.enterPoint
	currDist := maths.DotProduct(vec, idx.nodes[currObj].vec)

	for l := idx.maxLevel; l > level; l-- {
		changed := true
		for changed {
			changed = false
			for _, neighborID := range idx.nodes[currObj].neighbors[l] {
				dist := maths.DotProduct(vec, idx.nodes[neighborID].vec)
				if dist > currDist {
					currDist = dist
					currObj = neighborID
					changed = true
				}
			}
		}
	}

	for l := int(math.Min(float64(level), float64(idx.maxLevel))); l >= 0; l-- {
		candidates := idx.searchLayer(vec, currObj, idx.EfConstruction, l)
		neighbors := selectNeighbors(candidates, idx.M)
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

func (idx *HNSWIndex) Delete(id string) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	// Soft delete: remove from idMap/Metadata but keep node in graph to maintain connectivity
	if _, ok := idx.idMap[id]; ok {
		delete(idx.idMap, id)
		delete(idx.Metadata, id)
	}
	return nil
}

func (idx *HNSWIndex) Count() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.idMap)
}

func (idx *HNSWIndex) Dim() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.dim
}

// DTO for serialization since hnswNode has unexported fields
type hnswNodeDTO struct {
	ID        string
	Vec       types.Vector
	Level     int
	Neighbors [][]int
}

// Atomic Save for HNSW
func (idx *HNSWIndex) Save(path string) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	tmpPath := path + ".tmp"
	f, err := os.Create(tmpPath)
	if err != nil {
		return err
	}
	defer f.Close()

	enc := gob.NewEncoder(f)
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
		nodeDTO := hnswNodeDTO{
			ID:        node.id,
			Vec:       node.vec,
			Level:     node.level,
			Neighbors: node.neighbors,
		}
		if err := enc.Encode(nodeDTO); err != nil {
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

// Load
func (idx *HNSWIndex) Load(path string) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	dec := gob.NewDecoder(f)
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

	idx.nodes = make([]*hnswNode, count)
	idx.idMap = make(map[string]int)

	for i := 0; i < count; i++ {
		var dto hnswNodeDTO
		if err := dec.Decode(&dto); err != nil {
			return err
		}
		node := &hnswNode{
			id:        dto.ID,
			vec:       dto.Vec,
			level:     dto.Level,
			neighbors: dto.Neighbors,
		}
		idx.nodes[i] = node
		idx.idMap[node.id] = i
	}

	idx.Metadata = make(map[string]map[string]any)
	if err := dec.Decode(&idx.Metadata); err != nil {
		return err
	}

	// Prune Ghost Nodes
	// If an ID is in nodes/idMap but NOT in Metadata, it was soft-deleted.
	for id := range idx.idMap {
		if _, isLive := idx.Metadata[id]; !isLive {
			delete(idx.idMap, id)
		}
	}

	return nil
}
