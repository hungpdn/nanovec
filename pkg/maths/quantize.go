package maths

import "github.com/hungpdn/nanovec/pkg/types"

// QuantizeSQ8 converts a float32 vector to uint8 (0-255).
// It assumes the vector is normalized or falls roughly within [-1, 1].
// This maps [-1.0, 1.0] to [0, 255].
func QuantizeSQ8(vec types.Vector) types.Vector8 {
	out := make(types.Vector8, len(vec))
	for i, v := range vec {
		// Shift range from [-1, 1] to [0, 2] -> multiply by 127.5 -> [0, 255]
		val := (v + 1.0) * Scale

		// Clamp values to avoid overflow/underflow
		if val < 0 {
			val = 0
		}
		if val > 255 {
			val = 255
		}

		// Add 0.5 for nearest neighbor rounding before truncation
		// 127.9 + 0.5 = 128.4 -> uint8(128.4) = 128 (Correct)
		// 127.2 + 0.5 = 127.7 -> uint8(127.7) = 127 (Correct)
		out[i] = uint8(val + 0.5)
	}
	return out
}
