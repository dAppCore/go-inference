// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"encoding/binary"
	"math"
)

func appendUint16LE(out []byte, value uint16) []byte {
	var buf [2]byte
	binary.LittleEndian.PutUint16(buf[:], value)
	return append(out, buf[:]...)
}

func float32ToFloat16(value float32) uint16 {
	bits := math.Float32bits(value)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits >> 23) & 0xff)
	frac := bits & 0x7fffff
	if exp == 255 {
		if frac == 0 {
			return sign | 0x7c00
		}
		return sign | 0x7e00
	}
	exp = exp - 127 + 15
	if exp >= 31 {
		return sign | 0x7c00
	}
	if exp <= 0 {
		if exp < -10 {
			return sign
		}
		frac |= 0x800000
		shift := uint32(14 - exp)
		half := uint16(frac >> shift)
		if (frac>>(shift-1))&1 != 0 {
			half++
		}
		return sign | half
	}
	half := sign | uint16(exp<<10) | uint16(frac>>13)
	if frac&0x00001000 != 0 {
		half++
	}
	return half
}

// cvtF32 builds a contiguous [seqLen*headDim] head tensor whose value at index
// i is i, so a slice over a [start,end) row range is trivially predictable.
func cvtF32(seqLen, headDim int) []float32 {
	out := make([]float32, seqLen*headDim)
	for i := range out {
		out[i] = float32(i)
	}
	return out
}

// cvtRawF16 encodes a [seqLen*headDim] head tensor as little-endian float16
// bytes — the raw payload shape the raw-tensor slicers expect.
func cvtRawF16(seqLen, headDim int) []byte {
	out := make([]byte, 0, seqLen*headDim*2)
	for i := range seqLen * headDim {
		out = appendUint16LE(out, float32ToFloat16(float32(i)))
	}
	return out
}

func testSnapshot() *Snapshot {
	return &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1, 2},
		Generated:     []int32{2},
		TokenOffset:   2,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        2,
		HeadDim:       2,
		NumQueryHeads: 8,
		LogitShape:    []int32{1, 1, 3},
		Logits:        []float32{0.1, 0.2, 0.7},
		Layers: []LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []HeadSnapshot{{
				Key:   []float32{1, 0, 0, 1},
				Value: []float32{0, 1, 1, 0},
			}},
		}},
	}
}
