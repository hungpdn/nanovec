package maths

import "github.com/hungpdn/nanovec/pkg/types"

// QuantizeSQ8 converts a float32 vector to uint8 (0-255).
// It assumes the vector is normalized or falls roughly within [-1, 1].
// This maps [-1.0, 1.0] to [0, 255].
func QuantizeSQ8(vec types.Vector) types.Vector8 {
	out := make(types.Vector8, len(vec))
	for i, v := range vec {
		// Shift range from [-1, 1] to [0, 2] -> multiply by 127.5 -> [0, 255]
		val := (v + 1.0) * 127.5

		// Clamp values to avoid overflow/underflow
		if val < 0 {
			val = 0
		}
		if val > 255 {
			val = 255
		}
		out[i] = uint8(val)
	}
	return out
}

// DequantizeSQ8 (Optional) converts uint8 back to float32 approx.
// Useful if you need to reconstruct the vector for re-ranking.
func DequantizeSQ8(vec types.Vector8) types.Vector {
	out := make(types.Vector, len(vec))
	const invScale = 1.0 / 127.5
	for i, v := range vec {
		out[i] = (float32(v) * invScale) - 1.0
	}
	return out
}
