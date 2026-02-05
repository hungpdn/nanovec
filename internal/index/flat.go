package index

import (
	"container/heap"
	"encoding/binary"
	"encoding/gob"
	"errors"
	"os"
	"sync"

	"github.com/hungpdn/nanovec/pkg/maths"
	"github.com/hungpdn/nanovec/pkg/types"
)

// FlatIndex: Lưu toàn bộ vector trong Map và duyệt hết khi tìm kiếm
type FlatIndex struct {
	sync.RWMutex

	// Dữ liệu trên RAM (Tối ưu cho SIMD)
	// Ví dụ: có 2 vector 2 chiều [1,2] và [3,4]
	// RawVectors = [1, 2, 3, 4] -> Liền mạch 100%
	RawVectors []float32

	// Map vị trí ngược lại ID
	// IDs[0] ứng với vector bắt đầu tại RawVectors[0]
	// IDs[1] ứng với vector bắt đầu tại RawVectors[dim]
	IDs []string

	Dim int
}

func NewFlatIndex(dim int) *FlatIndex {
	return &FlatIndex{
		RawVectors: make([]float32, 0),
		IDs:        make([]string, 0),
		Dim:        dim,
	}
}

func (idx *FlatIndex) Add(id string, vec types.Vector) error {
	idx.Lock()
	defer idx.Unlock()

	if len(vec) != idx.Dim {
		return errors.New("dimension mismatch")
	}

	// 1. Normalize
	normVec := maths.Normalize(vec)

	// 2. Append vào mảng liền mạch
	idx.RawVectors = append(idx.RawVectors, normVec...)
	idx.IDs = append(idx.IDs, id)

	return nil
}

// Thêm vào internal/index/flat.go

// Delete xóa vector khỏi index bằng kỹ thuật Swap-and-Pop
func (idx *FlatIndex) Delete(id string) error {
	idx.Lock()
	defer idx.Unlock()

	// 1. Tìm vị trí (Index) của ID trong mảng IDs
	// (Lưu ý: Để tối ưu hơn, bạn có thể thêm map[string]int vào struct FlatIndex để tra cứu O(1))
	pos := -1
	for i, existingID := range idx.IDs {
		if existingID == id {
			pos = i
			break
		}
	}

	if pos == -1 {
		// ID không tồn tại, coi như thành công hoặc trả lỗi tùy logic
		return nil
	}

	// 2. Thực hiện Swap-and-Pop
	lastIndex := len(idx.IDs) - 1

	// Nếu phần tử cần xóa không phải là phần tử cuối cùng
	if pos < lastIndex {
		// A. Chuyển ID cuối cùng vào vị trí pos
		idx.IDs[pos] = idx.IDs[lastIndex]

		// B. Chuyển Vector cuối cùng vào vị trí pos trong RawVectors
		// Tính toán offset
		destStart := pos * idx.Dim
		srcStart := lastIndex * idx.Dim

		// Copy đè vector cuối lên vector cần xóa
		copy(idx.RawVectors[destStart:destStart+idx.Dim], idx.RawVectors[srcStart:srcStart+idx.Dim])
	}

	// 3. Cắt bỏ phần tử cuối cùng (Pop)
	idx.IDs = idx.IDs[:lastIndex]
	idx.RawVectors = idx.RawVectors[:lastIndex*idx.Dim]

	return nil
}

func (idx *FlatIndex) Search(query types.Vector, k int) ([]string, []float32, error) {
	idx.RLock()
	defer idx.RUnlock()
	// 1. Normalize query vector
	query = maths.Normalize(query)
	n := len(idx.IDs)

	// 2. Sử dụng Min-Heap để giữ Top-K kết quả tốt nhất
	// (Tránh sort toàn bộ mảng kết quả - O(N log N) -> O(N log K))
	h := &ResultHeap{}
	heap.Init(h)

	// QUAN TRỌNG: Loop tuần tự trên bộ nhớ liền mạch
	// CPU Prefetcher sẽ hoạt động cực tốt ở đây
	for i := 0; i < n; i++ {
		// Cắt slice (Zero copy trong Go)
		id := idx.IDs[i]
		start := i * idx.Dim
		end := start + idx.Dim
		targetVec := idx.RawVectors[start:end]

		// Gọi hàm SIMD (đã viết ở bài trước)
		// Tính điểm nhanh bằng DotProduct
		score := maths.DotProduct(query, targetVec)

		if h.Len() < k {
			heap.Push(h, Item{ID: id, Score: score})
		} else if score > (*h)[0].Score {
			// Nếu điểm mới cao hơn điểm thấp nhất trong Top K
			heap.Pop(h)
			heap.Push(h, Item{ID: id, Score: score})
		}
	}

	// 3. Trả về kết quả (đang lộn xộn trong heap, cần sort lại nếu muốn đẹp)
	// Để đơn giản, ta pop ra từ heap
	ids := make([]string, h.Len())
	scores := make([]float32, h.Len())

	// Pop trả về từ nhỏ nhất đến lớn nhất, nên ta đi ngược
	for i := h.Len() - 1; i >= 0; i-- {
		item := heap.Pop(h).(Item)
		ids[i] = item.ID
		scores[i] = item.Score
	}

	return ids, scores, nil
}

func (idx *FlatIndex) Save(path string) error {
	idx.RLock()
	defer idx.RUnlock()

	f, err := os.Create(path)
	if err != nil {
		return err
	}

	defer f.Close()

	// 1. Ghi số chiều (Dim) và số lượng vector (Count) vào Header
	header := make([]int32, 2)
	header[0] = int32(idx.Dim)
	header[1] = int32(len(idx.IDs))
	if err := binary.Write(f, binary.LittleEndian, header); err != nil {
		return err
	}

	// 2. Ghi IDs (Sử dụng Gob cho tiện)
	enc := gob.NewEncoder(f)
	if err := enc.Encode(idx.IDs); err != nil {
		return err
	}
	// 3. Ghi mảng RawVectors (Dump memory)
	// Đây là phần nặng nhất, binary.Write ghi cả cục sẽ rất nhanh
	if err := binary.Write(f, binary.LittleEndian, idx.RawVectors); err != nil {
		return err
	}

	return nil
}

// Load: Đọc dữ liệu từ đĩa
func (idx *FlatIndex) Load(path string) error {
	idx.Lock()
	defer idx.Unlock()

	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	// 1. Đọc Header
	header := make([]int32, 2)
	if err := binary.Read(f, binary.LittleEndian, &header); err != nil {
		return err
	}
	idx.Dim = int(header[0])
	count := int(header[1])

	// 2. Đọc IDs
	dec := gob.NewDecoder(f)
	if err := dec.Decode(&idx.IDs); err != nil {
		return err
	}

	// 3. Đọc RawVectors
	idx.RawVectors = make([]float32, count*idx.Dim)
	if err := binary.Read(f, binary.LittleEndian, &idx.RawVectors); err != nil {
		return err
	}

	return nil
}

// --- Heap Implementation ---
type Item struct {
	ID    string
	Score float32
}
type ResultHeap []Item

func (h ResultHeap) Len() int            { return len(h) }
func (h ResultHeap) Less(i, j int) bool  { return h[i].Score < h[j].Score } // Min-Heap
func (h ResultHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *ResultHeap) Push(x interface{}) { *h = append(*h, x.(Item)) }
func (h *ResultHeap) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[0 : n-1]
	return item
}
