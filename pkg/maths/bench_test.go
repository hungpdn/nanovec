package maths

import (
	"math/rand"
	"testing"
)

func randomFloat32(n int) []float32 {
	v := make([]float32, n)
	for i := 0; i < n; i++ {
		v[i] = rand.Float32()
	}
	return v
}

func randomUint8(n int) []uint8 {
	v := make([]uint8, n)
	for i := 0; i < n; i++ {
		v[i] = uint8(rand.Intn(256))
	}
	return v
}

// 1. Measure performance using standard Dot Product (Float32)
func BenchmarkDotProduct_Float32_1536(b *testing.B) {
	v1 := randomFloat32(1536)
	v2 := randomFloat32(1536)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		DotProduct(v1, v2)
	}
}

// 2. Dot Product SQ8 (Uint8) Performance Test
func BenchmarkDotProduct_SQ8_1536(b *testing.B) {
	q := randomFloat32(1536)
	u := randomUint8(1536)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		DotProductSQ8(q, u)
	}
}

// 3. Quantization Overhead Measurement
func BenchmarkQuantizeSQ8_1536(b *testing.B) {
	v := randomFloat32(1536)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		QuantizeSQ8(v)
	}
}

func BenchmarkDotProduct_Uint8_1536(b *testing.B) {
	v1 := randomUint8(1536)
	v2 := randomUint8(1536)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		DotProductUint8(v1, v2)
	}
}
