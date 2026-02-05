package index

import (
	"math/rand/v2"
)

// searchCtx holds temporary data for a single search request
// allowing thread-safe concurrent searches without allocation
type searchCtx struct {
	// visitedList stores the "visit token" for each node internal ID
	visitedList  []uint32
	visitedToken uint32   // Current search generation
	candidates   *maxHeap // maxHeap for candidates
	results      *minHeap // minHeap for results: Pop lowest score (worst in top-k) to replace
	// Scratch buffer for neighbor selection to avoid allocations
	neighborBuf []int
	// Scratch buffer for reading neighbors safely inside RLock without alloc
	scratchNeighbors []int
	// Thread-local RNG to avoid global lock contention
	rng *rand.Rand
}

func newSearchCtx(neighborCap int) *searchCtx {
	c := make(maxHeap, 0, 256)
	r := make(minHeap, 0, 256)

	// Initialize a thread-local RNG
	rSource := rand.NewPCG(rand.Uint64(), rand.Uint64())

	return &searchCtx{
		visitedList:      make([]uint32, 0),
		visitedToken:     0,
		candidates:       &c,
		results:          &r,
		neighborBuf:      make([]int, 0, neighborCap),
		scratchNeighbors: make([]int, 0, neighborCap),
		rng:              rand.New(rSource),
	}
}

// reset clears the search context for reuse
func (ctx *searchCtx) reset() {
	*ctx.candidates = (*ctx.candidates)[:0]
	*ctx.results = (*ctx.results)[:0]
}

// randomLevel generates a level for a new node using the THREAD-LOCAL RNG.
// This avoids contention on the global math/rand mutex.
func (ctx *searchCtx) randomLevel() int {
	lvl := 0
	for ctx.rng.Float64() < 0.5 && lvl < 10 {
		lvl++
	}
	return lvl
}

type pqItem struct {
	id    int
	score float32
}

// minHeap: Keeps the LOWEST score at top (root).
// Used for "Results" list (Top-K). We want to easily remove the worst item (min score) if we find a better one.
type minHeap []pqItem

func (h minHeap) Len() int { return len(h) }

func (h *minHeap) Push(x pqItem) {
	*h = append(*h, x)
	h.up(len(*h) - 1)
}

func (h *minHeap) Pop() pqItem {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[0 : n-1]
	if n > 1 {
		root := old[0]
		(*h)[0] = item
		h.down(0, len(*h))
		return root
	}
	return item
}

// Peek returns the smallest element (root) without removing it
func (h minHeap) Peek() pqItem {
	if len(h) == 0 {
		return pqItem{}
	}
	return h[0]
}

func (h *minHeap) up(j int) {
	for {
		i := (j - 1) / 2 // parent
		if i == j || !((*h)[j].score < (*h)[i].score) {
			break
		}
		(*h)[j], (*h)[i] = (*h)[i], (*h)[j]
		j = i
	}
}

func (h *minHeap) down(i0, n int) {
	i := i0
	for {
		j1 := 2*i + 1
		if j1 >= n || j1 < 0 {
			break
		}
		j := j1
		j2 := j1 + 1
		if j2 < n && (*h)[j2].score < (*h)[j1].score {
			j = j2
		}
		if !((*h)[j].score < (*h)[i].score) {
			break
		}
		(*h)[i], (*h)[j] = (*h)[j], (*h)[i]
		i = j
	}
}

// maxHeap: Keeps the HIGHEST score at top (root).
// Used for "Candidates" (Beam Search). We want to process the best (closest) node first.
type maxHeap []pqItem

func (h maxHeap) Len() int { return len(h) }

func (h *maxHeap) Push(x pqItem) {
	*h = append(*h, x)
	h.up(len(*h) - 1)
}

func (h *maxHeap) Pop() pqItem {
	old := *h
	n := len(old)
	if n == 0 {
		return pqItem{}
	}
	root := old[0]
	last := old[n-1]
	*h = old[0 : n-1]

	if n > 1 {
		(*h)[0] = last
		h.down(0, len(*h))
	}
	return root
}

func (h maxHeap) Peek() pqItem {
	if len(h) == 0 {
		return pqItem{}
	}
	return h[0]
}

func (h *maxHeap) up(j int) {
	for {
		i := (j - 1) / 2
		if i == j || !((*h)[j].score > (*h)[i].score) { // Greater than
			break
		}
		(*h)[j], (*h)[i] = (*h)[i], (*h)[j]
		j = i
	}
}

func (h *maxHeap) down(i0, n int) {
	i := i0
	for {
		j1 := 2*i + 1
		if j1 >= n || j1 < 0 {
			break
		}
		j := j1
		j2 := j1 + 1
		if j2 < n && (*h)[j2].score > (*h)[j1].score { // Greater than
			j = j2
		}
		if !((*h)[j].score > (*h)[i].score) { // Greater than
			break
		}
		(*h)[i], (*h)[j] = (*h)[j], (*h)[i]
		i = j
	}
}
