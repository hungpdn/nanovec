package index

import (
	"encoding/binary"
	"encoding/json"
	"io"
	"sync"
	"unsafe"

	"github.com/hungpdn/nanovec/pkg/errors"
)

// BaseIndex contains common components for all Index Flat types
type BaseIndex struct {
	mu sync.RWMutex

	// Map the reverse position of the ID
	// IDs[0] correspond to the starting vector at RawVectors[0]
	// IDs[1] correspond to the starting vector at RawVectors[dim]
	dim       int
	idMap     map[string]int
	IDs       []string
	Metadatas []map[string]any
	Version   uint64
}

func NewBaseIndex(dim int) BaseIndex {
	return BaseIndex{
		dim:       dim,
		idMap:     make(map[string]int),
		IDs:       make([]string, 0),
		Metadatas: make([]map[string]any, 0),
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
	b.Metadatas = append(b.Metadatas, meta)
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

// CommitDelete performs Swap-and-Pop on both IDs and Metadatas
func (b *BaseIndex) CommitDelete(id string, pos, lastIndex int) {
	// Swap data from lastIndex to pos
	if pos < lastIndex {
		lastID := b.IDs[lastIndex]
		lastMeta := b.Metadatas[lastIndex]

		b.IDs[pos] = lastID
		b.Metadatas[pos] = lastMeta

		b.idMap[lastID] = pos
	}

	// Remove last element
	delete(b.idMap, id)
	// Zero out pointers to avoid memory leaks
	b.IDs[lastIndex] = ""
	b.Metadatas[lastIndex] = nil

	b.IDs = b.IDs[:lastIndex]
	b.Metadatas = b.Metadatas[:lastIndex]
}

// Dim
func (b *BaseIndex) Dim() int { return b.dim }

// Count
func (b *BaseIndex) Count() int {
	b.RLock()
	defer b.RUnlock()
	return len(b.IDs)
}

func (b *BaseIndex) SetVersion(v uint64) {
	b.Lock()
	defer b.Unlock()
	b.Version = v
}

func (b *BaseIndex) GetVersion() uint64 {
	b.Lock()
	defer b.Unlock()
	return b.Version
}

// SaveBase
// Format: [Count] -> [IDs...] -> [Version] -> [Meta Items...]
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

	for _, meta := range b.Metadatas {
		var metaBytes []byte
		var err error
		if meta != nil {
			metaBytes, err = json.Marshal(meta)
			if err != nil {
				return err
			}
		} else {
			metaBytes = []byte("null")
		}

		if err := binary.Write(w, binary.LittleEndian, int32(len(metaBytes))); err != nil {
			return err
		}
		if _, err := w.Write(metaBytes); err != nil {
			return err
		}
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

	b.Metadatas = make([]map[string]any, count)
	for i := 0; i < int(count); i++ {
		var metaLen int32
		if err := binary.Read(r, binary.LittleEndian, &metaLen); err != nil {
			return err
		}
		if metaLen < 0 {
			return errors.ErrDimMismatch
		}

		metaBytes := make([]byte, metaLen)
		if _, err := io.ReadFull(r, metaBytes); err != nil {
			return err
		}

		var meta map[string]any
		if err := json.Unmarshal(metaBytes, &meta); err != nil {
			return err
		}
		b.Metadatas[i] = meta
	}

	return nil
}

// unsafeCast converts []byte to []T without copying using Go 1.17+ unsafe.Slice
// The unsafeCast technique assumes that the byte order (Endianness) of the file on
// disk matches that of the CPU (usually Little Endian). If you copy a DB file from
// an x86 machine (Little Endian) to an s390x mainframe (Big Endian), the float data will be incorrect.
func unsafeCast[T any](data []byte) []T {
	if len(data) == 0 {
		return nil
	}
	var zero T
	sizeOfT := int(unsafe.Sizeof(zero))
	count := len(data) / sizeOfT

	ptr := unsafe.Pointer(&data[0])
	return unsafe.Slice((*T)(ptr), count)
}

// bytesFromSlice casts []T to []byte (Zero-Copy)
func bytesFromSlice[T any](s []T) []byte {
	if len(s) == 0 {
		return nil
	}
	var zero T
	sizeOfT := int(unsafe.Sizeof(zero))
	length := len(s) * sizeOfT
	return unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), length)
}

// --- SPECIALIZED HEAP IMPLEMENTATION (Zero-Alloc) ---
type Item struct {
	ID    string
	Score float32
}

// ResultHeap is a Min-Heap of Items
type ResultHeap []Item

func (h ResultHeap) Len() int { return len(h) }

// PushItem adds an item to the heap
func (h *ResultHeap) PushItem(x Item) {
	*h = append(*h, x)
	h.up(len(*h) - 1)
}

// PopItem removes the minimum item (root)
func (h *ResultHeap) PopItem() Item {
	n := len(*h) - 1
	(*h)[0], (*h)[n] = (*h)[n], (*h)[0]
	h.down(0, n)
	item := (*h)[n]
	*h = (*h)[0:n]
	return item
}

func (h *ResultHeap) up(j int) {
	for {
		i := (j - 1) / 2 // parent
		if i == j || !((*h)[j].Score < (*h)[i].Score) {
			break
		}
		(*h)[j], (*h)[i] = (*h)[i], (*h)[j]
		j = i
	}
}

func (h *ResultHeap) down(i0, n int) {
	i := i0
	for {
		j1 := 2*i + 1
		if j1 >= n || j1 < 0 { // j1 < 0 after int overflow
			break
		}
		j := j1 // left child
		j2 := j1 + 1
		if j2 < n && (*h)[j2].Score < (*h)[j1].Score {
			j = j2 // = 2*i + 2  // right child
		}
		if !((*h)[j].Score < (*h)[i].Score) {
			break
		}
		(*h)[i], (*h)[j] = (*h)[j], (*h)[i]
		i = j
	}
}

// Fix establishes the heap invariant after the element at index i has changed its value.
func (h *ResultHeap) Fix(i int) {
	h.down(i, len(*h))
	h.up(i)
}
