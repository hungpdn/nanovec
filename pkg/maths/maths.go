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

	if magnitude < 1e-9 {
		return
	}

	invMag := 1.0 / magnitude
	for i := range vec {
		vec[i] *= invMag
	}
}

// Normalize transforms the vector to a length equal to 1.
func Normalize(vec []float32) []float32 {
	out := make([]float32, len(vec))
	copy(out, vec)
	NormalizeInPlace(out)
	return out
}

// DotProduct calculates the dot product of two vectors.
// Optimized with Loop Unrolling (4x) for SIMD auto-vectorization.
func DotProduct(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	// Bounds Check Elimination
	_ = a[len(a)-1]
	_ = b[len(a)-1]

	var sum float32
	i := 0

	// Unroll loop 4x to help compiler generate vector instructions
	for ; i <= len(a)-4; i += 4 {
		sum += a[i]*b[i] +
			a[i+1]*b[i+1] +
			a[i+2]*b[i+2] +
			a[i+3]*b[i+3]
	}

	// Handle remaining elements
	for ; i < len(a); i++ {
		sum += a[i] * b[i]
	}

	return sum
}
