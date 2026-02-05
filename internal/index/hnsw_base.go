package index

import (
	"container/heap"
	"math/rand/v2"
)

// searchCtx holds temporary data for a single search request
// allowing thread-safe concurrent searches without allocation
type searchCtx struct {
	// visitedList stores the "visit token" for each node internal ID
	visitedList  []uint32
	visitedToken uint32 // Current search generation
	candidates   *MaxHeap
	results      *MinHeap
	// Scratch buffer for neighbor selection to avoid allocations
	neighborBuf []int
	// Thread-local RNG to avoid global lock contention
	rng *rand.Rand
}

func newSearchCtx(neighborCap int) *searchCtx {
	c := &MaxHeap{}
	r := &MinHeap{}
	heap.Init(c)
	heap.Init(r)

	// Initialize a thread-local RNG
	rSource := rand.NewPCG(rand.Uint64(), rand.Uint64())

	return &searchCtx{
		visitedList:  make([]uint32, 0),
		visitedToken: 0,
		candidates:   c,
		results:      r,
		neighborBuf:  make([]int, 0, neighborCap),
		rng:          rand.New(rSource),
	}
}

func (ctx *searchCtx) reset() {
	*ctx.candidates = (*ctx.candidates)[:0]
	*ctx.results = (*ctx.results)[:0]
}

// randomLevel generates a level for a new node using the THREAD-LOCAL RNG.
// This avoids contention on the global math/rand mutex.
func (ctx *searchCtx) randomLevel() int {
	lvl := 0
	// ctx.rng.Float64() is strictly local to this goroutine/context -> No Lock needed.
	for ctx.rng.Float64() < 0.5 && lvl < 10 {
		lvl++
	}
	return lvl
}

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

// Helper for selecting neighbors from heap
func (idx *HNSWIndex[T]) selectNeighborsFromHeap(ctx *searchCtx, h *MinHeap, m int) []int {
	count := h.Len()

	// 1. Reset buffer length
	ctx.neighborBuf = ctx.neighborBuf[:0]

	// 2. Grow capacity if needed (rarely happens if initialized with > M)
	if cap(ctx.neighborBuf) < count {
		ctx.neighborBuf = make([]int, 0, count)
	}

	// 3. Fill buffer by popping from heap
	// Note: We consume the heap, but it's fine since ctx is reset per layer
	for i := 0; i < count; i++ {
		ctx.neighborBuf = append(ctx.neighborBuf, heap.Pop(h).(pqItem).id)
	}

	// 4. Select best candidates (last M items because MinHeap pop order is Worst -> Best)
	start := 0
	if count > m {
		start = count - m
	}

	// Return a slice backing the reusable buffer
	return ctx.neighborBuf[start:]
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
