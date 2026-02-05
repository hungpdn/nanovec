package bitset

import (
	"fmt"
	"testing"
)

// TestNew verifies the initialization logic and underlying slice allocation
func TestNew(t *testing.T) {
	tests := []struct {
		size          int
		expectedCount int
	}{
		{1, 1},
		{63, 1},
		{64, 1},
		{65, 2},
		{128, 2},
		{129, 3},
	}

	for _, tt := range tests {
		bs := New(tt.size)
		if len(bs.Data) != tt.expectedCount {
			t.Errorf("New(%d): expected Data len %d, got %d", tt.size, tt.expectedCount, len(bs.Data))
		}
		if bs.Size != tt.size {
			t.Errorf("New(%d): expected Size %d, got %d", tt.size, tt.size, bs.Size)
		}
	}
}

// TestSetAndIsSet verifies basic setting and checking of bits
func TestSetAndIsSet(t *testing.T) {
	bs := New(100)

	indicesToSet := []int{0, 1, 63, 64, 99}

	// 1. Set bits
	for _, i := range indicesToSet {
		bs.Set(i)
	}

	// 2. Verify they are set
	for _, i := range indicesToSet {
		if !bs.IsSet(i) {
			t.Errorf("Index %d should be set", i)
		}
	}

	// 3. Verify others are NOT set
	if bs.IsSet(2) {
		t.Error("Index 2 should not be set")
	}
	if bs.IsSet(65) {
		t.Error("Index 65 should not be set")
	}
}

// TestUnset verifies the unset logic
func TestUnset(t *testing.T) {
	bs := New(64)

	// Set bit 10
	bs.Set(10)
	if !bs.IsSet(10) {
		t.Fatal("Failed to set bit 10 initially")
	}

	// Unset bit 10
	bs.Unset(10)
	if bs.IsSet(10) {
		t.Error("Bit 10 should be unset")
	}

	// Unset a bit that was never set (idempotency)
	bs.Unset(20)
	if bs.IsSet(20) {
		t.Error("Bit 20 should be unset")
	}
}

// TestGrow verifies resizing logic and data persistence
func TestGrow(t *testing.T) {
	bs := New(10)
	bs.Set(5)

	// Grow to a size that requires a new uint64 in the slice
	// 10 fits in 1 uint64. 100 requires 2 uint64s.
	bs.Grow(100)

	if bs.Size != 100 {
		t.Errorf("Expected size 100, got %d", bs.Size)
	}

	// 1. Check if old data persisted
	if !bs.IsSet(5) {
		t.Error("Bit 5 should still be set after growing")
	}

	// 2. Check if we can set data in the new range
	bs.Set(90)
	if !bs.IsSet(90) {
		t.Error("Bit 90 should be set")
	}

	// 3. Grow to smaller size (should do nothing based on your implementation)
	oldLen := len(bs.Data)
	bs.Grow(50)
	if bs.Size != 100 {
		t.Error("Grow should not shrink the size property if newSize <= oldSize")
	}
	if len(bs.Data) != oldLen {
		t.Error("Grow should not change underlying data if newSize <= oldSize")
	}
}

// TestOutOfBounds ensures nothing panics and logic holds for invalid indices
func TestOutOfBounds(t *testing.T) {
	bs := New(10)

	// Try to set beyond size (should be ignored based on your code)
	bs.Set(20)
	if bs.IsSet(20) {
		t.Error("IsSet(20) should be false because Size is 10")
	}

	// Try to check beyond size
	if bs.IsSet(999) {
		t.Error("IsSet(999) should be false")
	}

	// Try to unset beyond size (should not panic)
	bs.Unset(20)
}

// ExampleBitSet demonstrates how to use the package in documentation
func ExampleBitSet() {
	// Create a bitset to hold 70 bits
	bs := New(70)

	// Set some bits
	bs.Set(0)
	bs.Set(63) // Last bit of first uint64
	bs.Set(64) // First bit of second uint64

	fmt.Printf("Bit 0 is set: %v\n", bs.IsSet(0))
	fmt.Printf("Bit 5 is set: %v\n", bs.IsSet(5))

	// Resize to accommodate more bits
	bs.Grow(150)
	bs.Set(100)

	fmt.Printf("Bit 100 is set: %v\n", bs.IsSet(100))
	fmt.Printf("Total Size: %d\n", bs.Size)

	// Output:
	// Bit 0 is set: true
	// Bit 5 is set: false
	// Bit 100 is set: true
	// Total Size: 150
}
