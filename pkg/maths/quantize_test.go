package maths

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/hungpdn/nanovec/pkg/types"
)

// TestQuantizeSQ8 tests the quantization logic including boundaries and clamping
func TestQuantizeSQ8(t *testing.T) {
	tests := []struct {
		name     string
		input    types.Vector  // []float32
		expected types.Vector8 // []uint8
	}{
		{
			name:     "Lower Bound (-1.0)",
			input:    types.Vector{-1.0},
			expected: types.Vector8{0},
		},
		{
			name:     "Upper Bound (1.0)",
			input:    types.Vector{1.0},
			expected: types.Vector8{255},
		},
		{
			name: "Zero Point (0.0)",
			// Logic: (0.0 + 1.0) * 127.5 = 127.5 -> uint8 truncates to 128
			input:    types.Vector{0.0},
			expected: types.Vector8{128},
		},
		{
			name:     "Clamping Underflow (<-1.0)",
			input:    types.Vector{-2.0, -50.0},
			expected: types.Vector8{0, 0},
		},
		{
			name:     "Clamping Overflow (>1.0)",
			input:    types.Vector{1.1, 100.0},
			expected: types.Vector8{255, 255},
		},
		{
			name:  "Mixed Values",
			input: types.Vector{-0.5, 0.5},
			// -0.5 -> 0.5 * 127.5 = 63.75 -> 64
			//  0.5 -> 1.5 * 127.5 = 191.25 -> 191
			expected: types.Vector8{64, 191},
		},
		{
			name:     "Empty Vector",
			input:    types.Vector{},
			expected: types.Vector8{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := QuantizeSQ8(tt.input)

			// Check for nil vs empty slice consistency if needed,
			// but DeepEqual handles strict comparison well.
			if len(got) == 0 && len(tt.expected) == 0 {
				return
			}

			if !reflect.DeepEqual(got, tt.expected) {
				t.Errorf("QuantizeSQ8() = %v, want %v", got, tt.expected)
			}
		})
	}
}

// ExampleQuantizeSQ8 demonstrates how to convert a standard normalized vector
// into a quantized 8-bit vector for storage optimization.
func ExampleQuantizeSQ8() {
	// A standard normalized vector (values between -1 and 1)
	vec := types.Vector{-1.0, 0.0, 0.5, 1.0}

	// Quantize to uint8 [0, 255]
	quantized := QuantizeSQ8(vec)

	fmt.Printf("Original:  %.1f\n", vec)
	fmt.Printf("Quantized: %d\n", quantized)

	// Output:
	// Original:  [-1.0 0.0 0.5 1.0]
	// Quantized: [0 128 191 255]
}
