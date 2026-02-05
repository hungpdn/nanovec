package maths

import (
	"math"
)

// Constants
const (
	Scale    = 127.5
	InvScale = 1.0 / Scale
)

// NormalizeInPlace normalizes the vector directly
// This allows Zero-Allocation if the caller owns the memory
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

// Normalize transforms the vector to a length equal to 1
func Normalize(vec []float32) []float32 {
	out := make([]float32, len(vec))
	copy(out, vec)
	NormalizeInPlace(out)
	return out
}

// DotProduct calculates dot product of two float32 vectors
// Optimized with Loop Unrolling (4x) for SIMD auto-vectorization
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

// DotProductSQ8 calculates dot product optimized using Distributive Property
// Speedup: ~1.5x - 2x compared to the naive safe version.
func DotProductSQ8(query []float32, qVec []uint8) float32 {
	if len(query) != len(qVec) || len(query) == 0 {
		return 0
	}

	// 1. Calculate sum of query vector (Overhead is negligible)
	var querySum float32
	for _, v := range query {
		querySum += v
	}

	var sumProd float32
	i := 0
	n := len(query)

	_ = query[n-1]
	_ = qVec[n-1]

	// 2. Accumulate Cross-Product (q * u)
	// We accumulate in float32, so NO integer overflow risk.
	for ; i <= n-4; i += 4 {
		sumProd += query[i]*float32(qVec[i]) +
			query[i+1]*float32(qVec[i+1]) +
			query[i+2]*float32(qVec[i+2]) +
			query[i+3]*float32(qVec[i+3])
	}

	for ; i < n; i++ {
		sumProd += query[i] * float32(qVec[i])
	}

	// 3. Apply formula: S * sum(q*u) - sum(q)
	return (sumProd * InvScale) - querySum
}

// DotProductUint8 calculates dot product between Node(Uint8) and Node(Uint8).
// OPTIMIZATION: Uses pure Integer Arithmetic (SIMD-friendly) inside the loop.
// Formula: S^2 * sum(u*v) - S * (sum(u) + sum(v)) + Dim
func DotProductUint8(a, b []uint8) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var sumProd uint32 // Accumulate product (max ~10^8, fits in uint32)
	var sumA uint32    // Accumulate sum of A
	var sumB uint32    // Accumulate sum of B

	i := 0
	n := len(a)

	// Bounds Check Elimination
	_ = a[n-1]
	_ = b[n-1]

	// 4x Loop Unrolling
	for ; i <= n-4; i += 4 {
		// Load bytes as uint32 to prevent overflow during multiplication
		vA0, vA1, vA2, vA3 := uint32(a[i]), uint32(a[i+1]), uint32(a[i+2]), uint32(a[i+3])
		vB0, vB1, vB2, vB3 := uint32(b[i]), uint32(b[i+1]), uint32(b[i+2]), uint32(b[i+3])

		// Integer Multiply & Add
		sumProd += vA0*vB0 + vA1*vB1 + vA2*vB2 + vA3*vB3

		// Integer Sums
		sumA += vA0 + vA1 + vA2 + vA3
		sumB += vB0 + vB1 + vB2 + vB3
	}

	// Handle remaining elements
	for ; i < n; i++ {
		vA, vB := uint32(a[i]), uint32(b[i])
		sumProd += vA * vB
		sumA += vA
		sumB += vB
	}

	// Apply final formula (Float conversion happens only ONCE here)
	// S^2 * sumProd - S * (sumA + sumB) + Dim
	const InvScaleSq = InvScale * InvScale

	return float32(sumProd)*InvScaleSq - float32(sumA+sumB)*InvScale + float32(n)
}

// CalculateVecSum calculates the sum of elements for SQ8 pre-computation
func CalculateVecSum(vec []uint8) uint32 {
	var sum uint32
	for _, v := range vec {
		sum += uint32(v)
	}
	return sum
}

// DotProductUint8Precomputed uses pre-calculated sums to avoid summing inside the loop.
// Formula: S^2 * sum(u*v) - S * (sum(u) + sum(v)) + Dim
func DotProductUint8Precomputed(a, b []uint8, sumA, sumB uint32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var sumProd uint32
	i := 0
	n := len(a)

	_ = a[n-1]
	_ = b[n-1]

	// 4x Loop Unrolling for SIMD
	for ; i <= n-4; i += 4 {
		vA0, vA1, vA2, vA3 := uint32(a[i]), uint32(a[i+1]), uint32(a[i+2]), uint32(a[i+3])
		vB0, vB1, vB2, vB3 := uint32(b[i]), uint32(b[i+1]), uint32(b[i+2]), uint32(b[i+3])
		sumProd += vA0*vB0 + vA1*vB1 + vA2*vB2 + vA3*vB3
	}

	for ; i < n; i++ {
		sumProd += uint32(a[i]) * uint32(b[i])
	}

	const InvScaleSq = InvScale * InvScale
	return float32(sumProd)*InvScaleSq - float32(sumA+sumB)*InvScale + float32(n)
}
