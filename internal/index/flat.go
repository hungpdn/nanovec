package index

import (
	"bufio"
	"bytes"
	"container/heap"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"runtime"
	"slices"
	"sync"
	"unsafe"

	"github.com/hungpdn/nanovec/pkg/errors"
	"github.com/hungpdn/nanovec/pkg/maths"
	"github.com/hungpdn/nanovec/pkg/types"
	"golang.org/x/sys/unix"
)

// ChunkSize defines how many vectors per chunk
// 4096 * 128 dim * 4 bytes = ~2MB per chunk
// This size is L2 Cache friendly and avoids massive allocations
const ChunkSize = 4096

// ResultHeap Pool to reduce GC pressure
var heapPool = sync.Pool{
	New: func() any {
		h := make(ResultHeap, 0, 20)
		return &h
	},
}

// FlatIndex is a generic implementation for both Float32 and SQ8 (Uint8)
type FlatIndex[T types.Number] struct {
	BaseIndex
	chunks   [][]T  // Chunking to prevent OOM
	readOnly bool   // Config
	mmapData []byte // Mmap state

	// Injected behavior
	// storeFunc processes a vector and writes it to the destination slice
	// src is the input vector, dst is the slice in RawVectors, buf is a temp buffer (for SQ8)
	storeFunc func(src []float32, dst []T, buf []float32)
	// scoreFunc calculates similarity between query and target vector
	scoreFunc func(query []float32, target []T) float32
}

// NewFlatIndexFloat creates a standard Float32 index
func NewFlatIndexFloat(dim int) *FlatIndex[float32] {
	return &FlatIndex[float32]{
		BaseIndex: NewBaseIndex(dim),
		chunks:    make([][]float32, 0),
		storeFunc: func(src []float32, dst []float32, _ []float32) {
			copy(dst, src)
			maths.NormalizeInPlace(dst)
		},
		scoreFunc: maths.DotProduct,
	}
}

// NewFlatIndexSQ8 creates a Quantized SQ8 index
func NewFlatIndexSQ8(dim int) *FlatIndex[uint8] {
	return &FlatIndex[uint8]{
		BaseIndex: NewBaseIndex(dim),
		chunks:    make([][]uint8, 0),
		storeFunc: func(src []float32, dst []uint8, buf []float32) {
			copy(buf, src)
			maths.NormalizeInPlace(buf)
			maths.QuantizeSQ8Into(buf, dst)
		},
		scoreFunc: maths.DotProductSQ8,
	}
}

// SetReadOnly enables mmap mode
func (idx *FlatIndex[T]) SetReadOnly(ro bool) {
	idx.readOnly = ro
}

// getVectorSlice returns the slice backing the vector at global index i
func (idx *FlatIndex[T]) getVectorSlice(i int) []T {
	chunkIdx := i / ChunkSize
	offset := (i % ChunkSize) * idx.dim
	return idx.chunks[chunkIdx][offset : offset+idx.dim]
}

// ensureSpace ensures there is space for a new vector, allocating a new chunk if needed
// Returns the destination slice for the new vector
func (idx *FlatIndex[T]) ensureSpace() []T {
	totalCount := len(idx.IDs)
	chunkIdx := totalCount / ChunkSize
	offset := (totalCount % ChunkSize) * idx.dim

	// If we need a new chunk
	if chunkIdx >= len(idx.chunks) {
		newChunk := make([]T, ChunkSize*idx.dim)
		idx.chunks = append(idx.chunks, newChunk)
	}

	return idx.chunks[chunkIdx][offset : offset+idx.dim]
}

// Add adds a single vector
func (idx *FlatIndex[T]) Add(id string, vec types.Vector, meta map[string]any) error {
	if idx.readOnly {
		return fmt.Errorf("cannot insert in read-only mode")
	}

	idx.Lock()
	defer idx.Unlock()

	if len(vec) != idx.dim {
		return errors.ErrDimMismatch
	}
	if err := idx.CheckID(id); err != nil {
		return err
	}

	dst := idx.ensureSpace()

	// Helper buffer for SQ8 (unused for Float32 storeFunc)
	var buf []float32
	if _, ok := any(idx.chunks).([][]uint8); ok {
		buf = make([]float32, idx.dim)
	}

	idx.storeFunc(vec, dst, buf)
	idx.AddMeta(id, meta)
	return nil
}

// AddBatch adds multiple vectors
func (idx *FlatIndex[T]) AddBatch(ids []string, vecs []types.Vector, metas []map[string]any) error {
	if idx.readOnly {
		return fmt.Errorf("cannot insert in read-only mode")
	}

	if len(ids) != len(vecs) || len(ids) != len(metas) {
		return errors.ErrBatchSizeMismatch
	}

	for _, v := range vecs {
		if len(v) != idx.dim {
			return errors.ErrDimMismatch
		}
	}

	processedData := make([]T, len(ids)*idx.dim)
	normBuf := make([]float32, idx.dim)

	for i := range ids {
		dst := processedData[i*idx.dim : (i+1)*idx.dim]
		idx.storeFunc(vecs[i], dst, normBuf)
	}

	idx.Lock()
	defer idx.Unlock()

	for _, id := range ids {
		if err := idx.CheckID(id); err != nil {
			return err
		}
	}

	idx.IDs = slices.Grow(idx.IDs, len(ids))
	idx.Metadatas = slices.Grow(idx.Metadatas, len(metas))

	for i, id := range ids {
		dst := idx.ensureSpace()
		src := processedData[i*idx.dim : (i+1)*idx.dim]
		copy(dst, src)
		idx.AddMeta(id, metas[i])
	}

	return nil
}

func (idx *FlatIndex[T]) UpdateMetadata(id string, meta map[string]any) error {
	idx.Lock()
	defer idx.Unlock()

	pos, exists := idx.idMap[id]
	if !exists {
		return errors.ErrIDAlreadyExists
	}
	idx.Metadatas[pos] = meta
	return nil
}

// Search finds k nearest neighbors.
// Note: 'query' must be already normalized by the caller.
func (idx *FlatIndex[T]) Search(query types.Vector, k int, filter types.FilterFunc) ([]types.SearchResult, error) {
	idx.RLock()
	defer idx.RUnlock()

	totalIDs := len(idx.IDs)
	if totalIDs == 0 {
		return []types.SearchResult{}, nil
	}

	normalizedQuery := query

	numWorkers := runtime.GOMAXPROCS(0)
	if totalIDs < ChunkSize*2 {
		numWorkers = 1
	}

	workerHeaps := make([]*ResultHeap, numWorkers)
	var wg sync.WaitGroup

	totalChunks := len(idx.chunks)
	chunksPerWorker := (totalChunks + numWorkers - 1) / numWorkers

	worker := func(workerID, startChunk, endChunk int) {
		defer wg.Done()

		hPtr := heapPool.Get().(*ResultHeap)
		*hPtr = (*hPtr)[:0]
		h := hPtr

		localGlobalIdx := startChunk * ChunkSize

		for cIdx := startChunk; cIdx < endChunk; cIdx++ {
			chunk := idx.chunks[cIdx]

			// Tight Loop
			for i := 0; i < len(chunk); i += idx.dim {
				if localGlobalIdx >= totalIDs {
					break
				}

				if filter != nil {
					// Direct slice access (fastest)
					meta := idx.Metadatas[localGlobalIdx]
					if meta == nil || !filter(meta) {
						localGlobalIdx++
						continue
					}
				}

				targetVec := chunk[i : i+idx.dim]
				score := idx.scoreFunc(normalizedQuery, targetVec)

				if h.Len() < k {
					heap.Push(h, Item{ID: idx.IDs[localGlobalIdx], Score: score})
				} else if score > (*h)[0].Score {
					heap.Pop(h)
					heap.Push(h, Item{ID: idx.IDs[localGlobalIdx], Score: score})
				}
				localGlobalIdx++
			}
		}
		// Store result in the allocated slot directly
		workerHeaps[workerID] = h
	}

	activeWorkers := 0
	for i := 0; i < numWorkers; i++ {
		start := i * chunksPerWorker
		end := start + chunksPerWorker
		if start >= totalChunks {
			break
		}
		if end > totalChunks {
			end = totalChunks
		}
		wg.Add(1)
		go worker(activeWorkers, start, end)
		activeWorkers++
	}

	wg.Wait()

	finalHeap := &ResultHeap{}
	heap.Init(finalHeap)

	for i := 0; i < activeWorkers; i++ {
		h := workerHeaps[i]
		if h == nil {
			continue
		}
		for _, item := range *h {
			if finalHeap.Len() < k {
				heap.Push(finalHeap, item)
			} else if item.Score > (*finalHeap)[0].Score {
				heap.Pop(finalHeap)
				heap.Push(finalHeap, item)
			}
		}
		heapPool.Put(h)
	}

	results := make([]types.SearchResult, finalHeap.Len())
	for i := finalHeap.Len() - 1; i >= 0; i-- {
		item := heap.Pop(finalHeap).(Item)
		metaIdx := idx.idMap[item.ID]

		results[i] = types.SearchResult{
			ID:       item.ID,
			Score:    item.Score,
			Metadata: idx.Metadatas[metaIdx],
		}
	}

	return results, nil
}

// Delete removes a vector.
func (idx *FlatIndex[T]) Delete(id string) error {
	if idx.readOnly {
		return fmt.Errorf("cannot delete in read-only mode")
	}

	idx.Lock()
	defer idx.Unlock()

	pos, lastIndex, exists := idx.PrepareDelete(id)
	if !exists {
		return nil
	}

	if pos < lastIndex {
		destSlice := idx.getVectorSlice(pos)
		srcSlice := idx.getVectorSlice(lastIndex)
		copy(destSlice, srcSlice)
	}

	idx.CommitDelete(id, pos, lastIndex)
	return nil
}

// DeletedCount returns wasted space (Unused capacity in chunks)
func (idx *FlatIndex[T]) DeletedCount() int {
	idx.RLock()
	defer idx.RUnlock()
	if idx.dim == 0 {
		return 0
	}
	totalCapacity := len(idx.chunks) * ChunkSize
	usedCount := len(idx.IDs)
	return totalCapacity - usedCount
}

// Close cleans up resources (unmap if needed)
func (idx *FlatIndex[T]) Close() error {
	idx.Lock()
	defer idx.Unlock()
	// Clear chunks first to prevent race conditions
	idx.chunks = nil
	if idx.mmapData != nil {
		err := unix.Munmap(idx.mmapData)
		idx.mmapData = nil
		return err
	}
	return nil
}

// Save persists to disk.
func (idx *FlatIndex[T]) Save(path string) error {
	idx.Lock()
	defer idx.Unlock()

	tmpPath := path + ".tmp"
	f, err := os.Create(tmpPath)
	if err != nil {
		return err
	}
	defer f.Close()

	w := bufio.NewWriterSize(f, 64*types.KB)

	header := make([]int32, 2)
	header[0] = int32(idx.dim)
	header[1] = int32(len(idx.IDs))

	// Write Header to Buffer
	if err := binary.Write(w, binary.LittleEndian, header); err != nil {
		return err
	}

	// Write Base (IDs + Metadata) to Buffer
	// SaveBase accepts io.Writer, so wrapping it in bufio works perfectly.
	if err := idx.SaveBase(w); err != nil {
		return err
	}

	// --- MEMORY ALIGNMENT ---
	// We MUST ensure the vectors start at a memory-aligned address (4 bytes).
	// Since SaveBase writes variable length data (Strings, JSON), the current
	// file position might be unaligned (e.g., 101 bytes).
	if err := w.Flush(); err != nil {
		return err
	}
	currentPos, err := f.Seek(0, 1)
	if err != nil {
		return err
	}

	align := 4 /// float32 needs 4-byte alignment
	if remainder := int(currentPos % int64(align)); remainder != 0 {
		padding := align - remainder
		// Write zero padding directly to file
		padBytes := make([]byte, padding)
		if _, err := w.Write(padBytes); err != nil {
			return err
		}
	}
	// ---------------------------

	// Write vectors chunk by chunk
	vectorsWritten := 0
	totalVectors := len(idx.IDs)

	for _, chunk := range idx.chunks {
		countInChunk := ChunkSize

		if vectorsWritten+countInChunk > totalVectors {
			countInChunk = totalVectors - vectorsWritten
		}

		if countInChunk > 0 {
			chunkSlice := chunk[:countInChunk*idx.dim]
			rawBytes := bytesFromSlice(chunkSlice)
			if _, err := w.Write(rawBytes); err != nil {
				return err
			}
			vectorsWritten += countInChunk
		}

		if vectorsWritten == totalVectors {
			break
		}
	}

	// Flush buffer to disk before Sync
	if err := w.Flush(); err != nil {
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

// Load reads from disk.
func (idx *FlatIndex[T]) Load(path string) error {
	idx.Lock()
	defer idx.Unlock()

	//  Cleanup old data
	if idx.mmapData != nil {
		_ = unix.Munmap(idx.mmapData)
		idx.mmapData = nil
	}
	idx.chunks = make([][]T, 0)

	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	stat, err := f.Stat()
	if err != nil {
		return err
	}
	fileSize := stat.Size()

	// --- SMART LOADING STRATEGY ---
	// 1. ReadOnly Mode: Always use Mmap (Shared).
	// 2. Write Mode (Linux/Mac): Use Mmap (Private/COW) for fast startup.
	// 3. Write Mode (Windows): Fallback to RAM Copy (to avoid file locking issues on Save).
	useMmap := idx.readOnly || runtime.GOOS != "windows"
	if useMmap {
		prot := unix.PROT_READ
		flags := unix.MAP_SHARED
		if !idx.readOnly {
			// Write Mode: Need Write Permission + Private Copy (COW)
			prot = unix.PROT_READ | unix.PROT_WRITE
			flags = unix.MAP_PRIVATE
		}

		data, err := unix.Mmap(int(f.Fd()), 0, int(fileSize), prot, flags)
		if err != nil {
			goto FallbackRAM
		}

		idx.mmapData = data

		r := bytes.NewReader(data)

		header := make([]int32, 2)
		if err := binary.Read(r, binary.LittleEndian, &header); err != nil {
			return err
		}
		idx.dim = int(header[0])
		count := int(header[1])

		if err := idx.LoadBase(r); err != nil {
			return err
		}

		// --- HANDLE ALIGNMENT PADDING ---
		// We need to skip padding bytes to align to 4 bytes
		bytesRead := r.Size() - int64(r.Len())
		align := 4
		if remainder := int(bytesRead) % align; remainder != 0 {
			padding := align - remainder
			if _, err := r.Seek(int64(padding), io.SeekCurrent); err != nil {
				return err
			}
		}

		// Calculate where the vector data starts
		// r.Size() = Total Size
		// r.Len() = Remaining bytes (which is the vector data)
		vectorDataOffset := r.Size() - int64(r.Len())

		// Slice the mmap data to get the vector section
		rawVectorBytes := idx.mmapData[vectorDataOffset:]

		var zero T
		expectedBytes := count * idx.dim * int(unsafe.Sizeof(zero))
		if len(rawVectorBytes) < expectedBytes {
			return fmt.Errorf("corrupted file: expected %d bytes for vectors, got %d", expectedBytes, len(rawVectorBytes))
		}

		// Zero-Copy Cast
		fullVectorSlice := unsafeCast[T](rawVectorBytes)

		// Slice into Chunks
		numChunks := (count + ChunkSize - 1) / ChunkSize
		idx.chunks = make([][]T, 0, numChunks)

		for i := 0; i < numChunks; i++ {
			start := i * ChunkSize * idx.dim
			end := start + (ChunkSize * idx.dim)

			if end > len(fullVectorSlice) {
				end = len(fullVectorSlice)
			}
			idx.chunks = append(idx.chunks, fullVectorSlice[start:end])
		}
		return nil
	}

FallbackRAM:
	// ---  Standard ram load (Windows or Fallback) ---
	r := bufio.NewReaderSize(f, 64*types.KB)

	// Read Header from Buffer
	header := make([]int32, 2)
	if err := binary.Read(r, binary.LittleEndian, &header); err != nil {
		return err
	}
	idx.dim = int(header[0])
	count := int(header[1])

	// Read Base from Buffer
	if err := idx.LoadBase(r); err != nil {
		return err
	}

	// --- HANDLE ALIGNMENT PADDING (RAM Path) ---
	// To find padding, we need the logical read position.
	// f.Seek(0, 1) returns physical file position.
	// r.Buffered() returns bytes buffered in RAM but not yet read.
	physPos, err := f.Seek(0, 1)
	if err != nil {
		return err
	}
	logicalPos := physPos - int64(r.Buffered())

	align := 4
	if remainder := int(logicalPos) % align; remainder != 0 {
		padding := align - remainder
		if _, err := r.Discard(padding); err != nil {
			return err
		}
	}

	// Rebuild Chunks
	numChunks := (count + ChunkSize - 1) / ChunkSize
	idx.chunks = make([][]T, 0, numChunks)

	vectorsRead := 0
	for i := 0; i < numChunks; i++ {
		toReadCount := ChunkSize
		if vectorsRead+toReadCount > count {
			toReadCount = count - vectorsRead
		}

		newChunk := make([]T, ChunkSize*idx.dim)
		targetByteSlice := bytesFromSlice(newChunk[:toReadCount*idx.dim])

		if _, err := io.ReadFull(r, targetByteSlice); err != nil {
			return err
		}
		idx.chunks = append(idx.chunks, newChunk)
		vectorsRead += toReadCount
	}
	return nil
}
