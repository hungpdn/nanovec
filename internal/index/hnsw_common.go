package index

import "math/rand"

// --- Priority Queues Helpers ---

type pqItem struct {
	id    int
	score float32
}

// MinHeap: Keeps the LOWEST score at top.
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

// MaxHeap: Keeps the HIGHEST score at top.
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
func selectNeighbors(candidates []pqItem, m int) []int {
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
