package index

import (
	"bufio"
	"bytes"
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
const ChunkSize = 4096

// DefaultHeapCap defines the initial capacity for the result heap.
const DefaultHeapCap = 1024

// ResultHeap Pool to reduce GC pressure
var heapPool = sync.Pool{
	New: func() any {
		h := make(ResultHeap, 0, DefaultHeapCap)
		return &h
	},
}

// FlatIndex is a generic implementation for both Float32 and SQ8 (Uint8)
type FlatIndex[T types.Number] struct {
	BaseIndex
	chunks   [][]T      // Chunking to prevent allocation failure on massive datasets
	vecSums  [][]uint32 // [SQ8 Only] Precomputed sums for integer arithmetic
	readOnly bool       // Config
	mmapData []byte     // Reference to mmap handle for cleanup

	// Injected behavior
	storeFunc    func(src []float32, dst []T, buf []float32)
	prepareQuery func(q []float32) ([]T, uint32)
	scoreFunc    func(q []T, t []T, qSum, tSum uint32) float32
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
		prepareQuery: func(q []float32) ([]float32, uint32) {
			return q, 0
		},
		scoreFunc: func(q, t []float32, _, _ uint32) float32 {
			return maths.DotProduct(q, t)
		},
	}
}

// NewFlatIndexSQ8 creates a Quantized SQ8 index
func NewFlatIndexSQ8(dim int) *FlatIndex[uint8] {
	return &FlatIndex[uint8]{
		BaseIndex: NewBaseIndex(dim),
		chunks:    make([][]uint8, 0),
		vecSums:   make([][]uint32, 0),
		storeFunc: func(src []float32, dst []uint8, buf []float32) {
			copy(buf, src)
			maths.NormalizeInPlace(buf)
			maths.QuantizeSQ8Into(buf, dst)
		},
		prepareQuery: func(q []float32) ([]uint8, uint32) {
			q8 := maths.QuantizeSQ8(q)
			qSum := maths.CalculateVecSum(q8)
			return q8, qSum
		},
		scoreFunc: maths.DotProductUint8Precomputed,
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
func (idx *FlatIndex[T]) ensureSpace() []T {
	totalCount := len(idx.IDs)
	chunkIdx := totalCount / ChunkSize
	offset := (totalCount % ChunkSize) * idx.dim

	if chunkIdx >= len(idx.chunks) {
		newChunk := make([]T, ChunkSize*idx.dim)
		idx.chunks = append(idx.chunks, newChunk)

		if idx.vecSums != nil {
			newSums := make([]uint32, ChunkSize)
			idx.vecSums = append(idx.vecSums, newSums)
		}
	}

	return idx.chunks[chunkIdx][offset : offset+idx.dim]
}

// Add adds a single vector
func (idx *FlatIndex[T]) Add(id string, vec types.Vector, meta map[string]any) error {
	if idx.readOnly {
		return errors.ErrReadOnly
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

	var buf []float32
	if idx.vecSums != nil {
		buf = make([]float32, idx.dim)
	} else if _, ok := any(idx.chunks).([][]uint8); ok {
		buf = make([]float32, idx.dim)
	}

	idx.storeFunc(vec, dst, buf)

	if idx.vecSums != nil {
		if v8, ok := any(dst).([]uint8); ok {
			sum := maths.CalculateVecSum(v8)
			chunkIdx := (len(idx.IDs)) / ChunkSize
			idxInChunk := (len(idx.IDs)) % ChunkSize
			idx.vecSums[chunkIdx][idxInChunk] = sum
		}
	}

	idx.AddMeta(id, meta)
	return nil
}

// AddBatch adds multiple vectors
func (idx *FlatIndex[T]) AddBatch(ids []string, vecs []types.Vector, metas []map[string]any) error {
	if idx.readOnly {
		return errors.ErrReadOnly
	}

	if len(ids) != len(vecs) || len(ids) != len(metas) {
		return errors.ErrBatchSizeMismatch
	}

	for _, v := range vecs {
		if len(v) != idx.dim {
			return errors.ErrDimMismatch
		}
	}

	// Phase 1: Heavy computation OUTSIDE the lock
	processedData := make([]T, len(ids)*idx.dim)
	normBuf := make([]float32, idx.dim)

	var batchSums []uint32
	if idx.vecSums != nil {
		batchSums = make([]uint32, len(ids))
	}

	for i := range ids {
		dst := processedData[i*idx.dim : (i+1)*idx.dim]
		idx.storeFunc(vecs[i], dst, normBuf)
		if idx.vecSums != nil {
			if v8, ok := any(dst).([]uint8); ok {
				batchSums[i] = maths.CalculateVecSum(v8)
			}
		}
	}

	// Phase 2: Critical Section
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

		if idx.vecSums != nil {
			globalIdx := len(idx.IDs)
			chunkIdx := globalIdx / ChunkSize
			idxInChunk := globalIdx % ChunkSize
			idx.vecSums[chunkIdx][idxInChunk] = batchSums[i]
		}

		idx.AddMeta(id, metas[i])
	}

	return nil
}

func (idx *FlatIndex[T]) UpdateMetadata(id string, meta map[string]any) error {
	idx.Lock()
	defer idx.Unlock()
	pos, exists := idx.idMap[id]
	if !exists {
		return fmt.Errorf("id %s not found", id)
	}
	idx.Metadatas[pos] = meta
	return nil
}

// Search finds k nearest neighbors
func (idx *FlatIndex[T]) Search(query types.Vector, k int, filter types.FilterFunc) ([]types.SearchResult, error) {
	idx.RLock()
	defer idx.RUnlock()

	totalIDs := len(idx.IDs)
	if totalIDs == 0 {
		return []types.SearchResult{}, nil
	}

	qProcessed, qSum := idx.prepareQuery(query)

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
		*hPtr = (*hPtr)[:0] // Reset slice length, keep capacity
		h := hPtr

		localGlobalIdx := startChunk * ChunkSize
		var minScore float32 = -3.4e38

		for cIdx := startChunk; cIdx < endChunk; cIdx++ {
			chunk := idx.chunks[cIdx]
			chunkLen := len(chunk)

			// Optimized Branching
			if idx.vecSums != nil {
				// SQ8 PATH
				if cIdx >= len(idx.vecSums) {
					break
				}
				chunkSums := idx.vecSums[cIdx]

				vecIdx := 0
				for i := 0; i < chunkLen; i += idx.dim {
					if localGlobalIdx >= totalIDs {
						break
					}
					if filter != nil {
						meta := idx.Metadatas[localGlobalIdx]
						if meta == nil || !filter(meta) {
							localGlobalIdx++
							vecIdx++
							continue
						}
					}

					targetVec := chunk[i : i+idx.dim]
					tSum := chunkSums[vecIdx] // Fast lookup

					score := idx.scoreFunc(qProcessed, targetVec, qSum, tSum)

					if h.Len() < k {
						h.PushItem(Item{ID: idx.IDs[localGlobalIdx], Score: score})
						if h.Len() == k {
							minScore = (*h)[0].Score
						}
					} else if score > minScore {
						(*h)[0] = Item{ID: idx.IDs[localGlobalIdx], Score: score}
						h.Fix(0)
						minScore = (*h)[0].Score
					}
					localGlobalIdx++
					vecIdx++
				}
			} else {
				// FLOAT32 PATH
				for i := 0; i < chunkLen; i += idx.dim {
					if localGlobalIdx >= totalIDs {
						break
					}
					if filter != nil {
						meta := idx.Metadatas[localGlobalIdx]
						if meta == nil || !filter(meta) {
							localGlobalIdx++
							continue
						}
					}

					targetVec := chunk[i : i+idx.dim]
					score := idx.scoreFunc(qProcessed, targetVec, 0, 0)

					if h.Len() < k {
						h.PushItem(Item{ID: idx.IDs[localGlobalIdx], Score: score})
						if h.Len() == k {
							minScore = (*h)[0].Score
						}
					} else if score > minScore {
						(*h)[0] = Item{ID: idx.IDs[localGlobalIdx], Score: score}
						h.Fix(0)
						minScore = (*h)[0].Score
					}
					localGlobalIdx++
				}
			}
		}
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
	for i := 0; i < activeWorkers; i++ {
		h := workerHeaps[i]
		if h == nil {
			continue
		}
		for _, item := range *h {
			if finalHeap.Len() < k {
				finalHeap.PushItem(item)
			} else if item.Score > (*finalHeap)[0].Score {
				finalHeap.PopItem()
				finalHeap.PushItem(item)
			}
		}
		heapPool.Put(h)
	}

	results := make([]types.SearchResult, finalHeap.Len())
	for i := finalHeap.Len() - 1; i >= 0; i-- {
		item := finalHeap.PopItem()
		metaIdx := idx.idMap[item.ID]
		results[i] = types.SearchResult{
			ID:       item.ID,
			Score:    item.Score,
			Metadata: idx.Metadatas[metaIdx],
		}
	}
	return results, nil
}

// Delete removes a vector
func (idx *FlatIndex[T]) Delete(id string) error {
	if idx.readOnly {
		return errors.ErrReadOnly
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

		if idx.vecSums != nil {
			cDst := pos / ChunkSize
			iDst := pos % ChunkSize
			cSrc := lastIndex / ChunkSize
			iSrc := lastIndex % ChunkSize
			idx.vecSums[cDst][iDst] = idx.vecSums[cSrc][iSrc]
		}
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
	return (len(idx.chunks) * ChunkSize) - len(idx.IDs)
}

// Close cleans up resources (unmap if needed)
func (idx *FlatIndex[T]) Close() error {
	idx.Lock()
	defer idx.Unlock()
	idx.chunks = nil
	if idx.mmapData != nil {
		err := unix.Munmap(idx.mmapData)
		idx.mmapData = nil
		return err
	}
	return nil
}

// Save persists to disk
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

	header := []int32{int32(idx.dim), int32(len(idx.IDs))}
	if err = binary.Write(w, binary.LittleEndian, header); err != nil {
		return err
	}

	if err = idx.SaveBase(w); err != nil {
		return err
	}

	if err = w.Flush(); err != nil {
		return err
	}
	currentPos, err := f.Seek(0, 1)
	if err != nil {
		return err
	}

	align := 4
	if remainder := int(currentPos % int64(align)); remainder != 0 {
		pad := make([]byte, align-remainder)
		if _, err := w.Write(pad); err != nil {
			return err
		}
	}

	vectorsWritten := 0
	totalVectors := len(idx.IDs)

	for _, chunk := range idx.chunks {
		count := ChunkSize
		if vectorsWritten+count > totalVectors {
			count = totalVectors - vectorsWritten
		}
		if count > 0 {
			chunkSlice := chunk[:count*idx.dim]
			rawBytes := bytesFromSlice(chunkSlice)
			if _, err := w.Write(rawBytes); err != nil {
				return err
			}
			vectorsWritten += count
		}
		if vectorsWritten == totalVectors {
			break
		}
	}

	hasSums := int32(0)
	if idx.vecSums != nil {
		hasSums = 1
	}
	if err := binary.Write(w, binary.LittleEndian, hasSums); err != nil {
		return err
	}

	if hasSums == 1 {
		sumsWritten := 0
		for _, chunk := range idx.vecSums {
			count := ChunkSize
			if sumsWritten+count > totalVectors {
				count = totalVectors - sumsWritten
			}
			if count > 0 {
				chunkSlice := chunk[:count]
				rawBytes := bytesFromSlice(chunkSlice)
				if _, err := w.Write(rawBytes); err != nil {
					return err
				}
				sumsWritten += count
			}
			if sumsWritten == totalVectors {
				break
			}
		}
	}

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

func (idx *FlatIndex[T]) Load(path string) error {
	idx.Lock()
	defer idx.Unlock()

	if idx.mmapData != nil {
		_ = unix.Munmap(idx.mmapData)
		idx.mmapData = nil
	}
	idx.chunks = make([][]T, 0)
	if idx.vecSums != nil {
		idx.vecSums = make([][]uint32, 0)
	}

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

	useMmap := idx.readOnly

	if useMmap && runtime.GOOS != "windows" {
		prot := unix.PROT_READ
		flags := unix.MAP_SHARED

		var data []byte
		data, err = unix.Mmap(int(f.Fd()), 0, int(fileSize), prot, flags)
		if err == nil {
			idx.mmapData = data
			r := bytes.NewReader(data)

			header := make([]int32, 2)
			if err = binary.Read(r, binary.LittleEndian, &header); err != nil {
				return err
			}
			idx.dim = int(header[0])
			count := int(header[1])

			if err = idx.LoadBase(r); err != nil {
				return err
			}

			bytesRead := r.Size() - int64(r.Len())
			align := 4
			if rem := int(bytesRead) % align; rem != 0 {
				if _, err = r.Seek(int64(align-rem), io.SeekCurrent); err != nil {
					return err
				}
			}

			vectorOffset := r.Size() - int64(r.Len())
			if vectorOffset > int64(len(idx.mmapData)) {
				return fmt.Errorf("file truncated")
			}

			rawVectors := idx.mmapData[vectorOffset:]
			var zero T
			expectedBytes := count * idx.dim * int(unsafe.Sizeof(zero))
			if len(rawVectors) < expectedBytes {
				if count > 0 {
					return fmt.Errorf("corrupted vector data")
				}
			}

			fullSlice := unsafeCast[T](rawVectors)
			numChunks := (count + ChunkSize - 1) / ChunkSize
			idx.chunks = make([][]T, 0, numChunks)
			for i := 0; i < numChunks; i++ {
				start := i * ChunkSize * idx.dim
				end := start + (ChunkSize * idx.dim)
				if end > len(fullSlice) {
					end = len(fullSlice)
				}
				idx.chunks = append(idx.chunks, fullSlice[start:end])
			}

			_, _ = r.Seek(int64(count*idx.dim*int(unsafe.Sizeof(zero))), io.SeekCurrent)

			hasSumsData := false
			if r.Len() >= 4 {
				var hasSums int32
				_ = binary.Read(r, binary.LittleEndian, &hasSums)
				if hasSums == 1 {
					hasSumsData = true
				}
			}

			if idx.vecSums != nil {
				if hasSumsData {
					sumOffset := r.Size() - int64(r.Len())
					sumBytesLen := count * 4
					if sumOffset+int64(sumBytesLen) > int64(len(idx.mmapData)) {
						return fmt.Errorf("file truncated (sums)")
					}
					rawSums := idx.mmapData[sumOffset : sumOffset+int64(sumBytesLen)]
					fullSums := unsafeCast[uint32](rawSums)
					idx.vecSums = make([][]uint32, 0, numChunks)
					for i := 0; i < numChunks; i++ {
						start := i * ChunkSize
						end := start + ChunkSize
						if end > len(fullSums) {
							end = len(fullSums)
						}
						idx.vecSums = append(idx.vecSums, fullSums[start:end])
					}
				} else {
					// Self-Healing
					idx.vecSums = make([][]uint32, 0, numChunks)
					for _, chunk := range idx.chunks {
						if v8, ok := any(chunk).([]uint8); ok {
							chunkLen := len(v8) / idx.dim
							sums := make([]uint32, ChunkSize)
							for i := 0; i < chunkLen; i++ {
								vec := v8[i*idx.dim : (i+1)*idx.dim]
								sums[i] = maths.CalculateVecSum(vec)
							}
							idx.vecSums = append(idx.vecSums, sums)
						}
					}
				}
			}
			return nil
		}
	}

	// Fallback RAM
	r := bufio.NewReaderSize(f, 64*types.KB)
	header := make([]int32, 2)
	if err = binary.Read(r, binary.LittleEndian, &header); err != nil {
		return err
	}
	idx.dim = int(header[0])
	count := int(header[1])

	if err = idx.LoadBase(r); err != nil {
		return err
	}

	physPos, _ := f.Seek(0, 1)
	logicalPos := physPos - int64(r.Buffered())
	align := 4
	if rem := int(logicalPos) % align; rem != 0 {
		if _, err := r.Discard(align - rem); err != nil {
			return err
		}
	}

	numChunks := (count + ChunkSize - 1) / ChunkSize
	idx.chunks = make([][]T, 0, numChunks)
	vectorsRead := 0
	for i := 0; i < numChunks; i++ {
		toRead := ChunkSize
		if vectorsRead+toRead > count {
			toRead = count - vectorsRead
		}
		newChunk := make([]T, ChunkSize*idx.dim)
		byteSlice := bytesFromSlice(newChunk[:toRead*idx.dim])
		if _, err := io.ReadFull(r, byteSlice); err != nil {
			return err
		}
		idx.chunks = append(idx.chunks, newChunk)
		vectorsRead += toRead
	}

	var hasSums int32
	hasSumsData := false
	if err := binary.Read(r, binary.LittleEndian, &hasSums); err == nil {
		if hasSums == 1 {
			hasSumsData = true
		}
	}

	if idx.vecSums != nil {
		idx.vecSums = make([][]uint32, 0, numChunks)
		if hasSumsData {
			sumsRead := 0
			for i := 0; i < numChunks; i++ {
				toRead := ChunkSize
				if sumsRead+toRead > count {
					toRead = count - sumsRead
				}
				newSumChunk := make([]uint32, ChunkSize)
				targetBytes := bytesFromSlice(newSumChunk[:toRead])
				if _, err := io.ReadFull(r, targetBytes); err != nil {
					return err
				}
				idx.vecSums = append(idx.vecSums, newSumChunk)
				sumsRead += toRead
			}
		} else {
			// Self-Healing
			for _, chunk := range idx.chunks {
				if v8, ok := any(chunk).([]uint8); ok {
					chunkLen := len(v8) / idx.dim
					sums := make([]uint32, ChunkSize)
					for i := 0; i < chunkLen; i++ {
						vec := v8[i*idx.dim : (i+1)*idx.dim]
						sums[i] = maths.CalculateVecSum(vec)
					}
					idx.vecSums = append(idx.vecSums, sums)
				}
			}
		}
	}
	return nil
}
