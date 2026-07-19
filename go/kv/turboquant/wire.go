// SPDX-Licence-Identifier: EUPL-1.2

package turboquant

import "math"

// putFloat32LE writes v into out[0:4] as little-endian IEEE-754 bits. out
// must have at least 4 bytes; a shorter slice panics on the out-of-range
// index — a caller under-allocating its wire buffer is a bug at the call
// site, not a runtime condition to recover from.
//
//	buf := make([]byte, 4)
//	putFloat32LE(buf, 1.5)
func putFloat32LE(out []byte, v float32) {
	bits := math.Float32bits(v)
	out[0] = byte(bits)
	out[1] = byte(bits >> 8)
	out[2] = byte(bits >> 16)
	out[3] = byte(bits >> 24)
}

// getFloat32LE reads a little-endian IEEE-754 float32 from data[0:4]. data
// must have at least 4 bytes.
//
//	getFloat32LE([]byte{0, 0, 0xC0, 0x3F}) // 1.5
func getFloat32LE(data []byte) float32 {
	bits := uint32(data[0]) | uint32(data[1])<<8 | uint32(data[2])<<16 | uint32(data[3])<<24
	return math.Float32frombits(bits)
}

// putUint32LE writes v into out[0:4] little-endian. Same length contract as
// putFloat32LE.
//
//	buf := make([]byte, 4)
//	putUint32LE(buf, 258)
func putUint32LE(out []byte, v uint32) {
	out[0] = byte(v)
	out[1] = byte(v >> 8)
	out[2] = byte(v >> 16)
	out[3] = byte(v >> 24)
}

// getUint32LE reads a little-endian uint32 from data[0:4]. Same length
// contract as getFloat32LE.
//
//	getUint32LE([]byte{2, 1, 0, 0}) // 258
func getUint32LE(data []byte) uint32 {
	return uint32(data[0]) | uint32(data[1])<<8 | uint32(data[2])<<16 | uint32(data[3])<<24
}
