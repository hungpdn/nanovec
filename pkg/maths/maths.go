package maths

import (
	"math"
)

// NormalizeInPlace normalizes the vector directly in the provided slice.
// This allows Zero-Allocation if the caller owns the memory.
func NormalizeInPlace(vec []float32) {
	var sum float32
	for _, v := range vec {
		sum += v * v
	}
	magnitude := float32(math.Sqrt(float64(sum)))

	// Avoid dividing by zero
	if magnitude < 1e-9 {
		return
	}

	// Multiply by reciprocal is faster than division
	invMag := 1.0 / magnitude
	for i := range vec {
		vec[i] *= invMag
	}
}

// Normalize transforms the vector to a length equal to 1.
// It returns a new slice, leaving the original unchanged.
func Normalize(vec []float32) []float32 {
	// Create a copy to avoid modifying the input
	out := make([]float32, len(vec))
	copy(out, vec)
	NormalizeInPlace(out)
	return out
}

// DotProduct calculates the dot product of two vectors.
// This is the most frequently called function (Hot path).
func DotProduct(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}
	if len(a) == 0 {
		return 0
	}

	// OPTIMIZATION: Bounds Check Elimination
	// Check the last element to prove to the compiler that the slice is long enough.
	// This prevents bounds checks inside the loop.
	_ = a[len(a)-1]
	_ = b[len(a)-1]

	var sum float32
	for i := 0; i < len(a); i++ {
		sum += a[i] * b[i]
	}

	return sum
}
