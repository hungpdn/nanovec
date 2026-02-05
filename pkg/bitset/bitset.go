package bitset

// Simple BitSet implementation
type BitSet struct {
	Data []uint64
	Size int
}

func New(size int) *BitSet {
	// Requires (size + 63) / 64 elements uint64
	count := (size + 63) / 64
	return &BitSet{
		Data: make([]uint64, count),
		Size: size,
	}
}

// Grow resize bitset if needed
func (b *BitSet) Grow(newSize int) {
	if newSize <= b.Size {
		return
	}
	newCount := (newSize + 63) / 64
	if newCount > len(b.Data) {
		newData := make([]uint64, newCount)
		copy(newData, b.Data)
		b.Data = newData
	}
	b.Size = newSize
}

func (b *BitSet) Set(i int) {
	if i >= b.Size {
		return // Should grow first
	}
	b.Data[i/64] |= (1 << (i % 64))
}

func (b *BitSet) Unset(i int) {
	if i >= b.Size {
		return
	}
	b.Data[i/64] &= ^(1 << (i % 64))
}

func (b *BitSet) IsSet(i int) bool {
	if i >= b.Size {
		return false
	}
	return (b.Data[i/64] & (1 << (i % 64))) != 0
}
