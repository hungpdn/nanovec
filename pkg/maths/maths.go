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

// DotProductSQ8 calculates dot product between Query(Float32) and Node(Uint8).
// Used during Search.
func DotProductSQ8(query []float32, qVec []uint8) float32 {
	if len(query) != len(qVec) || len(query) == 0 {
		return 0
	}

	// Bounds Check Elimination
	_ = query[len(query)-1]
	_ = qVec[len(qVec)-1]

	var sum float32
	i := 0

	// Unrolled Loop (4x)
	for ; i <= len(query)-4; i += 4 {
		v0 := (float32(qVec[i]) * InvScale) - 1.0
		v1 := (float32(qVec[i+1]) * InvScale) - 1.0
		v2 := (float32(qVec[i+2]) * InvScale) - 1.0
		v3 := (float32(qVec[i+3]) * InvScale) - 1.0

		sum += query[i]*v0 +
			query[i+1]*v1 +
			query[i+2]*v2 +
			query[i+3]*v3
	}

	// Handle remaining
	for ; i < len(query); i++ {
		val := (float32(qVec[i]) * InvScale) - 1.0
		sum += query[i] * val
	}

	return sum
}

// DotProductUint8 calculates dot product between Node(Uint8) and Node(Uint8).
// Used during HNSW Graph Construction (adding connections).
func DotProductUint8(a, b []uint8) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	_ = a[len(a)-1]
	_ = b[len(a)-1]

	var sum float32
	i := 0

	for ; i <= len(a)-4; i += 4 {
		vA0 := (float32(a[i]) * InvScale) - 1.0
		vB0 := (float32(b[i]) * InvScale) - 1.0

		vA1 := (float32(a[i+1]) * InvScale) - 1.0
		vB1 := (float32(b[i+1]) * InvScale) - 1.0

		vA2 := (float32(a[i+2]) * InvScale) - 1.0
		vB2 := (float32(b[i+2]) * InvScale) - 1.0

		vA3 := (float32(a[i+3]) * InvScale) - 1.0
		vB3 := (float32(b[i+3]) * InvScale) - 1.0

		sum += vA0*vB0 + vA1*vB1 + vA2*vB2 + vA3*vB3
	}

	for ; i < len(a); i++ {
		vA := (float32(a[i]) * InvScale) - 1.0
		vB := (float32(b[i]) * InvScale) - 1.0
		sum += vA * vB
	}

	return sum
}
