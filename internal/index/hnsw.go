package index

import (
	"encoding/gob"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sync"

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

	// Metadata storage (Same as FlatIndex)
	Metadata map[string]map[string]interface{}
}

// hnswNode represents a point in the graph
type hnswNode struct {
	id        string
	vec       types.Vector
	level     int
	neighbors [][]int // [level][neighbor_internal_index]
}

func NewHNSWIndex(dim, m, efConstruction int) *HNSWIndex {
	return &HNSWIndex{
		dim:            dim,
		M:              m,
		MMax0:          m * 2,
		EfConstruction: efConstruction,
		LevelMult:      1.0 / math.Log(float64(m)),
		nodes:          make([]*hnswNode, 0),
		idMap:          make(map[string]int),
		Metadata:       make(map[string]map[string]interface{}),
		enterPoint:     -1,
		maxLevel:       -1,
	}
}

// Add inserts a vector into the HNSW graph
func (idx *HNSWIndex) Add(id string, vec types.Vector, meta map[string]interface{}) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if len(vec) != idx.dim {
		return types.ErrDimMismatch
	}
	if _, exists := idx.idMap[id]; exists {
		return fmt.Errorf("id %s already exists", id)
	}

	// 1. Create Node
	// Randomly determine the level for this node
	level := idx.randomLevel()
	internalID := len(idx.nodes)

	node := &hnswNode{
		id:        id,
		vec:       vec,
		level:     level,
		neighbors: make([][]int, level+1),
	}

	// Normalize for Cosine Similarity (HNSW typically uses Euclidean, but we normalize for Dot Product)
	maths.NormalizeInPlace(node.vec)

	idx.nodes = append(idx.nodes, node)
	idx.idMap[id] = internalID
	idx.Metadata[id] = meta

	// 2. Insert into Graph
	if idx.enterPoint == -1 {
		// First node becomes entry point
		idx.enterPoint = internalID
		idx.maxLevel = level
		return nil
	}

	currObj := idx.enterPoint
	currDist := maths.DotProduct(vec, idx.nodes[currObj].vec)

	// A. Search from Top Level down to node's level
	// Greedy search to find the closest entry point at the node's level
	for l := idx.maxLevel; l > level; l-- {
		changed := true
		for changed {
			changed = false
			// Scan neighbors of current node
			for _, neighborID := range idx.nodes[currObj].neighbors[l] {
				dist := maths.DotProduct(vec, idx.nodes[neighborID].vec)
				if dist > currDist { // Dot Product: Higher is closer
					currDist = dist
					currObj = neighborID
					changed = true
				}
			}
		}
	}

	// B. Insert connections from node's level down to 0
	// For each level, we find 'M' nearest neighbors and link them
	// Note: This is a simplified logic. Robust implementations use a priority queue (efConstruction).
	for l := int(math.Min(float64(level), float64(idx.maxLevel))); l >= 0; l-- {
		// 1. Find nearest neighbors at this level using greedy BFS
		// (For skeleton simplicity, we connect to currObj.
		// Real implementation does a Beam Search here.)

		// Bidirectional connection
		idx.addConnection(l, internalID, currObj)
		idx.addConnection(l, currObj, internalID)

		// Move greedy pointer for next level
		// ... (Greedy walk logic similar to above) ...
	}

	// C. Update Entry Point if new node is higher
	if level > idx.maxLevel {
		idx.maxLevel = level
		idx.enterPoint = internalID
	}

	return nil
}

// randomLevel generates a level for a new node
func (idx *HNSWIndex) randomLevel() int {
	lvl := 0
	for rand.Float64() < 0.5 { // Simplified probability
		lvl++
	}
	return lvl
}

// addConnection connects two nodes at a specific level, pruning if necessary
func (idx *HNSWIndex) addConnection(level, from, to int) {
	node := idx.nodes[from]
	node.neighbors[level] = append(node.neighbors[level], to)

	// Pruning (Heuristic selection)
	maxM := idx.M
	if level == 0 {
		maxM = idx.MMax0
	}

	if len(node.neighbors[level]) > maxM {
		// Simple pruning: Keep most recent (Stack).
		// Real implementation: Keep nearest by distance.
		node.neighbors[level] = node.neighbors[level][:maxM]
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

	// 1. Zoom in: Go from Top level to Layer 0
	currObj := idx.enterPoint
	currDist := maths.DotProduct(normalizedQuery, idx.nodes[currObj].vec)

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

	// 2. Layer 0 Search (Beam Search / EF Search)
	// We start at 'currObj' found from above.
	// For skeleton, we do a simple greedy search or a small BFS.
	// We collect candidates in a Max-Heap (keeping Top-K).

	// ... (Priority Queue Logic would go here) ...

	// Stub return for skeleton to compile and run:
	// Just return the entry point as a result
	meta := idx.Metadata[idx.nodes[currObj].id]
	return []types.SearchResult{{
		ID:       idx.nodes[currObj].id,
		Score:    currDist,
		Metadata: meta,
	}}, nil
}

// AddBatch adds multiple items (For HNSW, simply loop Add)
func (idx *HNSWIndex) AddBatch(ids []string, vecs []types.Vector, metas []map[string]interface{}) error {
	// HNSW insertion is complex to parallelize safely without fine-grained locking.
	// For v1.1, we lock globally and loop.
	idx.mu.Lock()
	defer idx.mu.Unlock()

	for i, id := range ids {
		// Internal Add logic (without locking again)
		// NOTE: Copy the logic from Add() here or refactor Add() to separate InternalAdd
		// For skeleton brevity, we just call the logic concept:
		_ = i // prevent unused error
		_ = id
	}
	return nil
}

// Delete is hard in HNSW. Strategy: Soft Delete or Rebuild.
func (idx *HNSWIndex) Delete(id string) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	// Mark as deleted in a bitmap, filter during Search
	delete(idx.idMap, id)
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

// Save serializes the graph
func (idx *HNSWIndex) Save(path string) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Similar to FlatIndex: Write Header, ID Map, Nodes, Neighbors
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	enc := gob.NewEncoder(f)
	// Encode basic metadata
	if err := enc.Encode(idx.dim); err != nil {
		return err
	}
	if err := enc.Encode(idx.enterPoint); err != nil {
		return err
	}
	if err := enc.Encode(idx.nodes); err != nil {
		return err
	} // Nodes structure might need simplification for Gob

	return nil
}

// Load deserializes the graph
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
	// Decode nodes...
	return nil
}
