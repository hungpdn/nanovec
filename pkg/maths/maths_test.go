package maths

import (
	"fmt"
	"math"
	"testing"
)

const epsilon = 1e-4

// floatEquals checks if two floats are within a small margin of error
func floatEquals(a, b float32) bool {
	return math.Abs(float64(a-b)) < epsilon
}

func TestNormalizeInPlace(t *testing.T) {
	tests := []struct {
		name     string
		input    []float32
		expected []float32 // Normalized expected
	}{
		{
			name:     "Standard Vector 3-4",
			input:    []float32{3, 4},
			expected: []float32{0.6, 0.8},
		},
		{
			name:     "Unit Vector",
			input:    []float32{1, 0},
			expected: []float32{1, 0},
		},
		{
			name:     "Zero Vector",
			input:    []float32{0, 0, 0},
			expected: []float32{0, 0, 0},
		},
		{
			name:     "Negative Values",
			input:    []float32{-3, 0, 4},
			expected: []float32{-0.6, 0, 0.8},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Copy input because NormalizeInPlace modifies it
			vec := make([]float32, len(tt.input))
			copy(vec, tt.input)

			NormalizeInPlace(vec)

			for i := range vec {
				if !floatEquals(vec[i], tt.expected[i]) {
					t.Errorf("Index %d: expected %f, got %f", i, tt.expected[i], vec[i])
				}
			}
		})
	}
}

func TestNormalize(t *testing.T) {
	input := []float32{3, 4}
	output := Normalize(input)

	// Check if output is correct
	if !floatEquals(output[0], 0.6) || !floatEquals(output[1], 0.8) {
		t.Errorf("Expected [0.6, 0.8], got %v", output)
	}

	// Check if input remains unchanged (immutability check)
	if input[0] != 3 || input[1] != 4 {
		t.Error("Normalize should not modify the input slice")
	}
}

func TestDotProduct(t *testing.T) {
	tests := []struct {
		name     string
		a, b     []float32
		expected float32
	}{
		{
			name:     "Simple",
			a:        []float32{1, 2},
			b:        []float32{3, 4},
			expected: 11.0, // 1*3 + 2*4
		},
		{
			name:     "Orthogonal",
			a:        []float32{1, 0},
			b:        []float32{0, 1},
			expected: 0.0,
		},
		{
			name:     "Loop Unrolling Edge Case (Len 5)",
			a:        []float32{1, 1, 1, 1, 1},
			b:        []float32{2, 2, 2, 2, 2},
			expected: 10.0, // Trigger cleanup loop
		},
		{
			name:     "Empty",
			a:        []float32{},
			b:        []float32{},
			expected: 0.0,
		},
		{
			name:     "Mismatch Length",
			a:        []float32{1, 2},
			b:        []float32{1},
			expected: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			res := DotProduct(tt.a, tt.b)
			if !floatEquals(res, tt.expected) {
				t.Errorf("Expected %f, got %f", tt.expected, res)
			}
		})
	}
}

// Helper to calculate expected SQ8 result logically:
// dec(u) = (u - 127.5) / 127.5
// dot = sum(q * dec(u))
func calculateExpectedSQ8(q []float32, u []uint8) float32 {
	var sum float32
	for i := range q {
		dequantized := (float32(u[i]) - Scale) * InvScale
		sum += q[i] * dequantized
	}
	return sum
}

func TestDotProductSQ8(t *testing.T) {
	tests := []struct {
		name  string
		query []float32
		qVec  []uint8
	}{
		{
			name:  "Max Values",
			query: []float32{1.0, 1.0},
			qVec:  []uint8{255, 255},
			// 255 -> 1.0. Dot should be 2.0
		},
		{
			name:  "Min Values",
			query: []float32{1.0, 1.0},
			qVec:  []uint8{0, 0},
			// 0 -> -1.0. Dot should be -2.0
		},
		{
			name:  "Mixed Length (Len 5)",
			query: []float32{0.5, 0.5, 0.5, 0.5, 0.5},
			qVec:  []uint8{255, 0, 255, 0, 127}, // 127 is roughly 0
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			expected := calculateExpectedSQ8(tt.query, tt.qVec)
			res := DotProductSQ8(tt.query, tt.qVec)

			if !floatEquals(res, expected) {
				t.Errorf("Expected %f, got %f", expected, res)
			}
		})
	}
}

// Helper for Uint8-Uint8 logical expectation
// dec(x) = (x - 127.5) / 127.5
func calculateExpectedUint8(a, b []uint8) float32 {
	var sum float32
	for i := range a {
		da := (float32(a[i]) - Scale) * InvScale
		db := (float32(b[i]) - Scale) * InvScale
		sum += da * db
	}
	return sum
}

func TestDotProductUint8(t *testing.T) {
	tests := []struct {
		name string
		a, b []uint8
	}{
		{
			name: "Perfect Match (1.0 * 1.0)",
			a:    []uint8{255, 255},
			b:    []uint8{255, 255},
		},
		{
			name: "Inverse Match (1.0 * -1.0)",
			a:    []uint8{255, 255},
			b:    []uint8{0, 0},
		},
		{
			name: "Zero Vectors (Approx)",
			a:    []uint8{127, 128},
			b:    []uint8{127, 128},
		},
		{
			name: "Odd Length (5)",
			a:    []uint8{255, 0, 255, 128, 64},
			b:    []uint8{255, 255, 0, 128, 64},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			expected := calculateExpectedUint8(tt.a, tt.b)

			// Test Standard
			res := DotProductUint8(tt.a, tt.b)
			if !floatEquals(res, expected) {
				t.Errorf("Standard: Expected %f, got %f", expected, res)
			}

			// Test Precomputed
			sumA := CalculateVecSum(tt.a)
			sumB := CalculateVecSum(tt.b)
			resPre := DotProductUint8Precomputed(tt.a, tt.b, sumA, sumB)
			if !floatEquals(resPre, expected) {
				t.Errorf("Precomputed: Expected %f, got %f", expected, resPre)
			}
		})
	}
}

// --- EXAMPLES ---

func ExampleNormalize() {
	// A vector with magnitude 5 (3-4-5 triangle)
	vec := []float32{3, 0, 4}

	normalized := Normalize(vec)

	fmt.Printf("Original: %v\n", vec)
	fmt.Printf("Normalized: %.1f\n", normalized)

	// Output:
	// Original: [3 0 4]
	// Normalized: [0.6 0.0 0.8]
}

func ExampleDotProductSQ8() {
	// Query vector (float32)
	query := []float32{1.0, -1.0}

	// Quantized Database vector (uint8)
	// 255 represents +1.0
	// 0 represents -1.0
	dbVec := []uint8{255, 0}

	// Calculation:
	// Index 0: 1.0 * 1.0 = 1.0
	// Index 1: -1.0 * -1.0 = 1.0
	// Sum = 2.0
	score := DotProductSQ8(query, dbVec)

	fmt.Printf("Score: %.1f\n", score)

	// Output:
	// Score: 2.0
}

func ExampleDotProductUint8Precomputed() {
	// Two quantized vectors
	// 255 maps to +1.0
	vecA := []uint8{255, 255, 255, 255}
	vecB := []uint8{255, 255, 255, 255}

	// Pre-calculate sums (usually done when indexing the vector)
	sumA := CalculateVecSum(vecA)
	sumB := CalculateVecSum(vecB)

	// Perform optimized dot product
	score := DotProductUint8Precomputed(vecA, vecB, sumA, sumB)

	fmt.Printf("Similarity: %.1f\n", score)

	// Output:
	// Similarity: 4.0
}
