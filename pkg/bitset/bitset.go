package bitset

import (
	"encoding/binary"
	"io"
	"math/bits"
)

// BitSet implementation
type BitSet struct {
	data []uint64
	size int
}

func New(size int) *BitSet {
	// Requires (size + 63) / 64 elements uint64
	count := (size + 63) / 64
	return &BitSet{
		data: make([]uint64, count),
		size: size,
	}
}

// Grow resize bitset if needed
func (b *BitSet) Grow(newSize int) {
	if newSize <= b.size {
		return
	}
	newCount := (newSize + 63) / 64
	if newCount > len(b.data) {
		newData := make([]uint64, newCount)
		copy(newData, b.data)
		b.data = newData
	}
	b.size = newSize
}

func (b *BitSet) Set(i int) {
	if i >= b.size {
		b.Grow(i + 1)
	}
	b.data[i>>6] |= (1 << (i & 63))
}

func (b *BitSet) Unset(i int) {
	if i >= b.size {
		return
	}
	b.data[i>>6] &= ^(1 << (i & 63))
}

func (b *BitSet) IsSet(i int) bool {
	if i >= b.size {
		return false
	}
	return (b.data[i>>6] & (1 << (i & 63))) != 0
}

// Count returns the number of set bits (population count)
func (b *BitSet) CountSetBits() int {
	count := 0
	for _, v := range b.data {
		count += bits.OnesCount64(v)
	}
	return count
}

// Save serializes the bitset to an io.Writer
// Format: [Size(int32)][DataLen(int32)][Data([]uint64)]
func (b *BitSet) Save(w io.Writer) error {
	if err := binary.Write(w, binary.LittleEndian, int32(b.size)); err != nil {
		return err
	}

	if err := binary.Write(w, binary.LittleEndian, int32(len(b.data))); err != nil {
		return err
	}

	return binary.Write(w, binary.LittleEndian, b.data)
}

// Load deserializes the bitset from an io.Reader
func (b *BitSet) Load(r io.Reader) error {
	var size int32
	if err := binary.Read(r, binary.LittleEndian, &size); err != nil {
		return err
	}
	b.size = int(size)

	var dataLen int32
	if err := binary.Read(r, binary.LittleEndian, &dataLen); err != nil {
		return err
	}

	b.data = make([]uint64, dataLen)
	byteSize := int(dataLen) * 8
	rawBytes := make([]byte, byteSize)

	if _, err := io.ReadFull(r, rawBytes); err != nil {
		return err
	}

	for i := 0; i < int(dataLen); i++ {
		offset := i * 8
		b.data[i] = binary.LittleEndian.Uint64(rawBytes[offset : offset+8])
	}

	return nil
}
