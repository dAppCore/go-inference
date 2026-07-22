// SPDX-Licence-Identifier: EUPL-1.2

package hip

import "math"

// hipBFloat16ToFloat32 widens one bf16 value (raw uint16 bits) to float32.
// It lives UNTAGGED — the portable unified-vision/encoder loaders consume it
// on every platform (the "Mac untagged vet 0" contract) — while the tagged
// projection-reference file uses it on the device lane; one definition serves
// both sides of the build fence.
func hipBFloat16ToFloat32(value uint16) float32 {
	return math.Float32frombits(uint32(value) << 16)
}

// hipFloat32ToBFloat16 narrows one float32 to bf16 bits with round-to-nearest-
// even — the inverse of hipBFloat16ToFloat32, untagged for the same portable
// consumers.
func hipFloat32ToBFloat16(value float32) uint16 {
	bits := math.Float32bits(value)
	bits += 0x7fff + ((bits >> 16) & 1)
	return uint16(bits >> 16)
}
