package index

import (
	"encoding/binary"
	"encoding/json"
	"io"
	"sync"

	"github.com/hungpdn/nanovec/pkg/errors"
)

// BaseIndex contains common components for all Index Flat types
type BaseIndex struct {
	mu sync.RWMutex

	// Map the reverse position of the ID
	// IDs[0] correspond to the starting vector at RawVectors[0]
	// IDs[1] correspond to the starting vector at RawVectors[dim]
	dim      int
	idMap    map[string]int
	IDs      []string
	Metadata map[string]map[string]any
	Version  uint64
}

func NewBaseIndex(dim int) BaseIndex {
	return BaseIndex{
		dim:      dim,
		idMap:    make(map[string]int),
		IDs:      make([]string, 0),
		Metadata: make(map[string]map[string]any),
	}
}

// Helper lock
func (b *BaseIndex) Lock()    { b.mu.Lock() }
func (b *BaseIndex) Unlock()  { b.mu.Unlock() }
func (b *BaseIndex) RLock()   { b.mu.RLock() }
func (b *BaseIndex) RUnlock() { b.mu.RUnlock() }

// CheckID check ID exists
func (b *BaseIndex) CheckID(id string) error {
	if _, exists := b.idMap[id]; exists {
		return errors.ErrIDAlreadyExists
	}
	return nil
}

// AddMeta save ID and Metadata (called after the vector has been successfully appended)
func (b *BaseIndex) AddMeta(id string, meta map[string]any) {
	b.IDs = append(b.IDs, id)
	b.idMap[id] = len(b.IDs) - 1
	b.Metadata[id] = meta
}

// PrepareDelete return the position to be deleted and the last index to perform Swap-and-Pop
// Return: (pos, lastIndex, exists)
func (b *BaseIndex) PrepareDelete(id string) (int, int, bool) {
	pos, exists := b.idMap[id]
	if !exists {
		return -1, -1, false
	}
	lastIndex := len(b.IDs) - 1
	return pos, lastIndex, true
}

// CommitDelete perform metadata deletion and map updates after vector swapping
func (b *BaseIndex) CommitDelete(id string, pos, lastIndex int) {
	// Update the map for the last element that was just swapped
	if pos < lastIndex {
		lastID := b.IDs[lastIndex]
		b.IDs[pos] = lastID
		b.idMap[lastID] = pos
	}

	delete(b.idMap, id)
	delete(b.Metadata, id)
	b.IDs = b.IDs[:lastIndex]
}

// Dim
func (idx *BaseIndex) Dim() int { return idx.dim }

// Count
func (idx *BaseIndex) Count() int {
	idx.RLock()
	defer idx.RUnlock()
	return len(idx.IDs)
}

func (idx *BaseIndex) SetVersion(v uint64) {
	idx.Lock()
	defer idx.Unlock()
	idx.Version = v
}

func (idx *BaseIndex) GetVersion() uint64 {
	idx.Lock()
	defer idx.Unlock()
	return idx.Version
}

// SaveBase: save common sections (IDs, Metadata) using BINARY + JSON (No Gob)
func (b *BaseIndex) SaveBase(w io.Writer) error {

	count := int32(len(b.IDs))
	if err := binary.Write(w, binary.LittleEndian, count); err != nil {
		return err
	}
	for _, id := range b.IDs {
		idBytes := []byte(id)
		if err := binary.Write(w, binary.LittleEndian, int32(len(idBytes))); err != nil {
			return err
		}
		if _, err := w.Write(idBytes); err != nil {
			return err
		}
	}

	if err := binary.Write(w, binary.LittleEndian, b.Version); err != nil {
		return err
	}

	metaJSON, err := json.Marshal(b.Metadata)
	if err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, int32(len(metaJSON))); err != nil {
		return err
	}
	if _, err := w.Write(metaJSON); err != nil {
		return err
	}

	return nil
}

// LoadBase read common sections (IDs, Metadata) and rebuild idMap
func (b *BaseIndex) LoadBase(r io.Reader) error {
	var count int32
	if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
		return err
	}

	b.IDs = make([]string, count)
	b.idMap = make(map[string]int, count)

	for i := 0; i < int(count); i++ {
		var idLen int32
		if err := binary.Read(r, binary.LittleEndian, &idLen); err != nil {
			return err
		}
		if idLen < 0 || idLen > 4096 {
			return errors.ErrDimMismatch
		}

		idBytes := make([]byte, idLen)
		if _, err := io.ReadFull(r, idBytes); err != nil {
			return err
		}
		id := string(idBytes)
		b.IDs[i] = id
		b.idMap[id] = i
	}

	if err := binary.Read(r, binary.LittleEndian, &b.Version); err != nil {
		return err
	}

	var metaLen int32
	if err := binary.Read(r, binary.LittleEndian, &metaLen); err != nil {
		return err
	}
	if metaLen < 0 {
		return errors.ErrDimMismatch
	}

	metaJSON := make([]byte, metaLen)
	if _, err := io.ReadFull(r, metaJSON); err != nil {
		return err
	}

	if err := json.Unmarshal(metaJSON, &b.Metadata); err != nil {
		return err
	}

	return nil
}

// --- Result Heap (Common) ---
type Item struct {
	ID    string
	Score float32
}
type ResultHeap []Item

func (h ResultHeap) Len() int           { return len(h) }
func (h ResultHeap) Less(i, j int) bool { return h[i].Score < h[j].Score } // Min-Heap
func (h ResultHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *ResultHeap) Push(x any)        { *h = append(*h, x.(Item)) }
func (h *ResultHeap) Pop() any {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[0 : n-1]
	return item
}
