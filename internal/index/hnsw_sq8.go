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

// HNSWIndexSQ8 implements HNSW with SQ8 Quantization
type HNSWIndexSQ8 struct {
	mu sync.RWMutex

	dim            int
	M              int
	MMax0          int
	EfConstruction int
	LevelMult      float64

	nodes      []*hnswNodeSQ8
	idMap      map[string]int
	enterPoint int
	maxLevel   int

	visitedPool sync.Pool
	Metadata    map[string]map[string]any
}

type hnswNodeSQ8 struct {
	id        string
	vec       types.Vector8 // Quantized Vector
	level     int
	neighbors [][]int
}

func NewHNSWIndexSQ8(dim, m, efConstruction int) *HNSWIndexSQ8 {
	idx := &HNSWIndexSQ8{
		dim:            dim,
		M:              m,
		MMax0:          m * 2,
		EfConstruction: efConstruction,
		LevelMult:      1.0 / math.Log(float64(m)),
		nodes:          make([]*hnswNodeSQ8, 0),
		idMap:          make(map[string]int),
		Metadata:       make(map[string]map[string]any),
		enterPoint:     -1,
		maxLevel:       -1,
	}

	idx.visitedPool.New = func() any {
		return make(map[int]bool, efConstruction)
	}

	return idx
}

func (idx *HNSWIndexSQ8) Add(id string, vec types.Vector, meta map[string]any) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if _, exists := idx.idMap[id]; exists {
		return fmt.Errorf("id %s already exists", id)
	}
	return idx.internalAdd(id, vec, meta, nil)
}

func (idx *HNSWIndexSQ8) AddBatch(ids []string, vecs []types.Vector, metas []map[string]any) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	for _, id := range ids {
		if _, exists := idx.idMap[id]; exists {
			return fmt.Errorf("id %s already exists", id)
		}
	}

	normBuf := make(types.Vector, idx.dim)

	for i, id := range ids {
		if err := idx.internalAdd(id, vecs[i], metas[i], normBuf); err != nil {
			return err
		}
	}
	return nil
}

func (idx *HNSWIndexSQ8) internalAdd(id string, vec types.Vector, meta map[string]any, reusableBuf types.Vector) error {
	if len(vec) != idx.dim {
		return errors.ErrDimMismatch
	}

	// Quantization Pipeline: Copy -> Normalize -> Quantize
	var normVec types.Vector
	if reusableBuf != nil {
		normVec = reusableBuf
	} else {
		normVec = make(types.Vector, len(vec))
	}

	copy(normVec, vec)
	maths.NormalizeInPlace(normVec)
	qVec := maths.QuantizeSQ8(normVec)

	level := idx.randomLevel()
	internalID := len(idx.nodes)

	node := &hnswNodeSQ8{
		id:        id,
		vec:       qVec,
		level:     level,
		neighbors: make([][]int, level+1),
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
	// Use DotProductSQ8 for Distance(QueryFloat, NodeUint8)
	currDist := maths.DotProductSQ8(normVec, idx.nodes[currObj].vec)

	for l := idx.maxLevel; l > level; l-- {
		changed := true
		for changed {
			changed = false
			for _, neighborID := range idx.nodes[currObj].neighbors[l] {
				dist := maths.DotProductSQ8(normVec, idx.nodes[neighborID].vec)
				if dist > currDist {
					currDist = dist
					currObj = neighborID
					changed = true
				}
			}
		}
	}

	for l := int(math.Min(float64(level), float64(idx.maxLevel))); l >= 0; l-- {
		// Pass normVec (Float) for searching candidates
		candidates := idx.searchLayer(normVec, currObj, idx.EfConstruction, l)
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

func (idx *HNSWIndexSQ8) searchLayer(query types.Vector, entryPoint, ef, level int) []pqItem {
	visited := idx.visitedPool.Get().(map[int]bool)
	defer func() {
		clear(visited)
		idx.visitedPool.Put(visited)
	}()

	visited[entryPoint] = true

	candidates := &MaxHeap{}
	heap.Init(candidates)
	results := &MinHeap{}
	heap.Init(results)

	dist := maths.DotProductSQ8(query, idx.nodes[entryPoint].vec)
	item := pqItem{id: entryPoint, score: dist}

	heap.Push(candidates, item)
	heap.Push(results, item)

	for candidates.Len() > 0 {
		curr := heap.Pop(candidates).(pqItem)
		furthestResult := (*results)[0]

		if curr.score < furthestResult.score && results.Len() >= ef {
			break
		}

		for _, neighborID := range idx.nodes[curr.id].neighbors[level] {
			if !visited[neighborID] {
				visited[neighborID] = true
				dist := maths.DotProductSQ8(query, idx.nodes[neighborID].vec)
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

func (idx *HNSWIndexSQ8) addConnection(level, from, to int) {
	node := idx.nodes[from]
	for _, neighbor := range node.neighbors[level] {
		if neighbor == to {
			return
		}
	}
	node.neighbors[level] = append(node.neighbors[level], to)

	maxM := idx.M
	if level == 0 {
		maxM = idx.MMax0
	}

	if len(node.neighbors[level]) > maxM {
		fromVec := idx.nodes[from].vec
		h := &MinHeap{}
		heap.Init(h)

		for _, neighborID := range node.neighbors[level] {
			// Compare two Uint8 vectors
			score := maths.DotProductUint8(fromVec, idx.nodes[neighborID].vec)
			heap.Push(h, pqItem{id: neighborID, score: score})
			if h.Len() > maxM {
				heap.Pop(h)
			}
		}

		newNeighbors := make([]int, h.Len())
		for i := 0; i < len(newNeighbors); i++ {
			newNeighbors[i] = (*h)[i].id
		}
		node.neighbors[level] = newNeighbors
	}
}

func (idx *HNSWIndexSQ8) Search(query types.Vector, k int, filter types.FilterFunc) ([]types.SearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if idx.enterPoint == -1 {
		return []types.SearchResult{}, nil
	}

	normalizedQuery := make(types.Vector, len(query))
	copy(normalizedQuery, query)
	maths.NormalizeInPlace(normalizedQuery)

	currObj := idx.enterPoint
	currDist := maths.DotProductSQ8(normalizedQuery, idx.nodes[currObj].vec)

	for l := idx.maxLevel; l > 0; l-- {
		changed := true
		for changed {
			changed = false
			for _, neighborID := range idx.nodes[currObj].neighbors[l] {
				dist := maths.DotProductSQ8(normalizedQuery, idx.nodes[neighborID].vec)
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
		id := idx.nodes[c.id].id

		// Stale/Zombie check
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

func (idx *HNSWIndexSQ8) Delete(id string) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	if _, ok := idx.idMap[id]; ok {
		delete(idx.idMap, id)
		delete(idx.Metadata, id)
	}
	return nil
}

func (idx *HNSWIndexSQ8) Count() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.idMap)
}

func (idx *HNSWIndexSQ8) Dim() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.dim
}

func (idx *HNSWIndexSQ8) randomLevel() int {
	lvl := 0
	for rand.Float64() < 0.5 && lvl < 10 {
		lvl++
	}
	return lvl
}

func (idx *HNSWIndexSQ8) selectNeighbors(candidates []pqItem, m int) []int {
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

// Persistence

type hnswNodeSQ8DTO struct {
	ID        string
	Vec       types.Vector8
	Level     int
	Neighbors [][]int
}

func (idx *HNSWIndexSQ8) Save(path string) error {
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
		dto := hnswNodeSQ8DTO{
			ID:        node.id,
			Vec:       node.vec,
			Level:     node.level,
			Neighbors: node.neighbors,
		}
		if err := enc.Encode(dto); err != nil {
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

func (idx *HNSWIndexSQ8) Load(path string) error {
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

	idx.nodes = make([]*hnswNodeSQ8, count)
	idx.idMap = make(map[string]int)

	for i := 0; i < count; i++ {
		var dto hnswNodeSQ8DTO
		if err := dec.Decode(&dto); err != nil {
			return err
		}
		node := &hnswNodeSQ8{
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

	// Prune ghost nodes
	for id := range idx.idMap {
		if _, isLive := idx.Metadata[id]; !isLive {
			delete(idx.idMap, id)
		}
	}

	return nil
}
