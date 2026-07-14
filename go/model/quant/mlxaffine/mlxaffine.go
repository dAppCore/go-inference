// SPDX-Licence-Identifier: EUPL-1.2

// Package mlxaffine writes the MLX group-affine weight-quantisation format —
// the packed-uint32 weight tensor plus its bf16 `.scales` / `.biases` siblings
// that `mlx_lm.convert` produces and this engine already READS (model.QuantConfig
// declares the block; engine/metal's affine_qmv and the safetensors quantised-tensor
// path consume the bytes). It is the one missing brick: go-inference has the whole
// quantise-a-model pipeline (gguf.QuantizeModelPack, autoround) and the reader, but
// nothing WROTE the native format the engine loads. This package closes that loop.
//
// The layout, reverse-engineered from and byte-verified against mlx-community's own
// snapshots (see oracle_test.go):
//
//   - a weight logically (outDim × inDim) is quantised group-affine along inDim:
//     every `groupSize` consecutive elements of a row share one scale + one bias;
//   - the packed weight is a uint32 tensor shaped [outDim, inDim·bits/32], each word
//     holding 32/bits little-endian codes (LSB-first: code p occupies bits
//     [p·bits, p·bits+bits));
//   - `.scales` and `.biases` are bf16 tensors shaped [outDim, inDim/groupSize];
//   - dequantisation is w ≈ scale·q + bias (q the unsigned code in [0, 2^bits−1]).
//
// The MLX affine derivation anchors the group's larger-magnitude edge to an exact
// code and signs the scale accordingly (QuantizeTensor's affineGroup) — the detail
// that makes the bytes match mlx rather than merely dequantise close. All arithmetic
// is float32, matching MLX's Metal kernel; Go's native float32 reproduces it exactly.
//
//	packed, scales, biases, err := mlxaffine.QuantizeTensor(w, outDim, inDim, 4, 64)
//	w2, err := mlxaffine.DequantizeTensor(packed, scales, biases, outDim, inDim, 4, 64)
package mlxaffine

import (
	"encoding/binary"
	"math"

	core "dappco.re/go"
)

// Mode is the mlx quantization block's `mode` value this package writes. Only the
// affine (group-affine) mode is implemented; mxfp4/mxfp8/nvfp4 are distinct formats.
const Mode = "affine"

// affineEps is the scale floor MLX clamps a group's scale to before deriving the
// edge code — guards a constant (zero-range) group from a divide-by-zero. float32
// to match the Metal kernel's precision.
const affineEps float32 = 1e-7

// SupportedBits reports whether bits is a width this package packs byte-exactly.
// MLX's clean LSB-first uint32 packing needs 32 % bits == 0 (1, 2, 4, 8); the 3/5/6-bit
// MLX layouts split codes across word boundaries and are NOT reproduced here (no
// reference snapshot to verify them against — refusing beats emitting wrong bytes).
// 1-bit codes are {0,1}: the composed quant lane reads Bonsai's b1 packs through
// DequantizeTensor and bridges them to the stock b_2 device kernels with RepackB1ToB2
// (repack.go), an exact code-widening (w = scale·q + bias is unchanged).
func SupportedBits(bits int) bool { return bits == 1 || bits == 2 || bits == 4 || bits == 8 }

// EligibleShape reports whether a tensor of the given shape is quantised by the MLX
// convention: a 2-D matrix whose inner (inDim) dimension is a whole number of groups.
// 1-D tensors (norms), and matrices whose inDim is not a multiple of groupSize, are
// passed through wide — the exact rule mlx_lm.convert's default predicate applies
// (verified against the 12B snapshot: 332 quantised, 292 passed, zero disagreement).
func EligibleShape(shape []uint64, groupSize int) bool {
	return groupSize > 0 && len(shape) == 2 && shape[1] > 0 && int(shape[1])%groupSize == 0
}

// PackedWords returns the number of uint32 words a row of inDim codes packs into.
func PackedWords(inDim, bits int) int { return inDim * bits / 32 }

// bitsMaxCode is the largest unsigned code a bits-wide field holds: 2^bits − 1.
func bitsMaxCode(bits int) int { return (1 << uint(bits)) - 1 }

// QuantizeTensor group-affine quantises a row-major (outDim × inDim) float32 weight
// into the MLX native format: the packed uint32 weight bytes, the bf16 scales, and
// the bf16 biases — byte-for-byte as mlx_lm.convert writes them.
//
// values must hold exactly outDim·inDim elements (row-major). bits ∈ {2,4,8};
// groupSize must divide inDim, and inDim·bits must be a whole number of 32-bit words.
// The returned slices are little-endian and ready to place in a safetensors tensor of
// dtype U32 (packed, shape [outDim, inDim·bits/32]) and BF16 (scales/biases, shape
// [outDim, inDim/groupSize]).
func QuantizeTensor(values []float32, outDim, inDim, bits, groupSize int) (packed, scales, biases []byte, err error) {
	if !SupportedBits(bits) {
		return nil, nil, nil, core.Errorf("mlxaffine: unsupported bits %d (want 2, 4, or 8)", bits)
	}
	if outDim <= 0 || inDim <= 0 {
		return nil, nil, nil, core.NewError("mlxaffine: outDim and inDim must be positive")
	}
	if groupSize <= 0 || inDim%groupSize != 0 {
		return nil, nil, nil, core.Errorf("mlxaffine: groupSize %d must be positive and divide inDim %d", groupSize, inDim)
	}
	elemsPerWord := 32 / bits
	if groupSize%elemsPerWord != 0 {
		return nil, nil, nil, core.Errorf("mlxaffine: groupSize %d must be a multiple of %d codes-per-word for %d-bit", groupSize, elemsPerWord, bits)
	}
	if len(values) != outDim*inDim {
		return nil, nil, nil, core.Errorf("mlxaffine: values length %d != outDim·inDim %d", len(values), outDim*inDim)
	}

	wordsPerRow := PackedWords(inDim, bits)
	wordsPerGroup := groupSize / elemsPerWord
	groupsPerRow := inDim / groupSize
	nBins := float32(bitsMaxCode(bits))

	packed = make([]byte, outDim*wordsPerRow*4)
	scales = make([]byte, outDim*groupsPerRow*2)
	biases = make([]byte, outDim*groupsPerRow*2)

	for r := 0; r < outDim; r++ {
		rowBase := r * inDim
		for g := 0; g < groupsPerRow; g++ {
			grp := values[rowBase+g*groupSize : rowBase+(g+1)*groupSize]
			scale, bias := affineGroup(grp, nBins)

			sbi := (r*groupsPerRow + g) * 2
			binary.LittleEndian.PutUint16(scales[sbi:], float32ToBFloat16(scale))
			binary.LittleEndian.PutUint16(biases[sbi:], float32ToBFloat16(bias))

			for wg := 0; wg < wordsPerGroup; wg++ {
				var word uint32
				for p := 0; p < elemsPerWord; p++ {
					code := quantiseCode(grp[wg*elemsPerWord+p], bias, scale, nBins)
					word |= code << (uint(p) * uint(bits))
				}
				wordIdx := r*wordsPerRow + g*wordsPerGroup + wg
				binary.LittleEndian.PutUint32(packed[wordIdx*4:], word)
			}
		}
	}
	return packed, scales, biases, nil
}

// affineGroup derives one group's (scale, bias) by MLX's edge-anchoring rule. The
// group's dominant edge (the min or max of larger magnitude) is pinned to an exact
// integer code q0, the scale is signed so that edge maps to q0, and the scale is then
// re-derived as edge/q0 so the edge dequantises exactly. bias = edge (or 0 when the
// edge already rounds to code 0). All float32 — the precision MLX's Metal kernel uses.
func affineGroup(g []float32, nBins float32) (scale, bias float32) {
	wmin, wmax := g[0], g[0]
	for _, v := range g[1:] {
		if v < wmin {
			wmin = v
		}
		if v > wmax {
			wmax = v
		}
	}
	scale = (wmax - wmin) / nBins
	if scale < affineEps {
		scale = affineEps
	}
	edge := wmax
	if abs32(wmin) > abs32(wmax) {
		edge = wmin // min-edge dominates → keep scale positive
	} else {
		scale = -scale // max-edge dominates → sign the scale negative
	}
	q0 := roundHalfAway(edge / scale)
	if q0 == 0 {
		return scale, 0
	}
	return edge / q0, edge
}

// quantiseCode maps one weight to its unsigned code: round((x−bias)/scale) clamped
// to [0, nBins]. Round-half-away-from-zero matches MLX's Metal round().
func quantiseCode(x, bias, scale, nBins float32) uint32 {
	q := roundHalfAway((x - bias) / scale)
	if q < 0 {
		q = 0
	} else if q > nBins {
		q = nBins
	}
	return uint32(q)
}

// DequantizeTensor is the inverse of QuantizeTensor: it unpacks the uint32 codes and
// applies w = scale·q + bias per group, returning outDim·inDim float32 values. Used by
// the round-trip tests and as a CPU dequant utility (the engine dequantises on-device).
func DequantizeTensor(packed, scales, biases []byte, outDim, inDim, bits, groupSize int) ([]float32, error) {
	if !SupportedBits(bits) {
		return nil, core.Errorf("mlxaffine: unsupported bits %d (want 2, 4, or 8)", bits)
	}
	if outDim <= 0 || inDim <= 0 || groupSize <= 0 || inDim%groupSize != 0 {
		return nil, core.NewError("mlxaffine: invalid dimensions")
	}
	elemsPerWord := 32 / bits
	wordsPerRow := PackedWords(inDim, bits)
	groupsPerRow := inDim / groupSize
	if len(packed) != outDim*wordsPerRow*4 {
		return nil, core.Errorf("mlxaffine: packed length %d != %d", len(packed), outDim*wordsPerRow*4)
	}
	if len(scales) != outDim*groupsPerRow*2 || len(biases) != outDim*groupsPerRow*2 {
		return nil, core.NewError("mlxaffine: scales/biases length does not match [outDim, inDim/groupSize]")
	}
	mask := uint32((1 << bits) - 1)
	out := make([]float32, outDim*inDim)
	for r := 0; r < outDim; r++ {
		for j := 0; j < inDim; j++ {
			wordIdx := r*wordsPerRow + j/elemsPerWord
			shift := uint(j%elemsPerWord) * uint(bits)
			code := (binary.LittleEndian.Uint32(packed[wordIdx*4:]) >> shift) & mask
			gi := (r*groupsPerRow + j/groupSize) * 2
			scale := bfloat16ToFloat32(binary.LittleEndian.Uint16(scales[gi:]))
			bias := bfloat16ToFloat32(binary.LittleEndian.Uint16(biases[gi:]))
			out[r*inDim+j] = scale*float32(code) + bias
		}
	}
	return out, nil
}

// roundHalfAway rounds v to the nearest integer, ties away from zero — the rule
// Metal's round() applies. math.Round widens v exactly (float32→float64) and rounds
// ties away from zero, so the result is bit-identical to the on-device kernel.
func roundHalfAway(v float32) float32 { return float32(math.Round(float64(v))) }

// abs32 is the float32 magnitude, computed without a float64 round-trip so the
// edge-dominance comparison stays in the kernel's precision.
func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

// float32ToBFloat16 rounds a float32 to bfloat16 (its high 16 bits) with
// round-to-nearest-even — the conversion MLX's static_cast<bfloat16_t> performs when
// it stores scales/biases. NaN is passed through as a quiet NaN; finite weights (the
// only inputs here) never hit that path. Verified bit-exact against every scale/bias
// in the reference snapshot (oracle_test.go).
func float32ToBFloat16(f float32) uint16 {
	u := math.Float32bits(f)
	if u&0x7fffffff > 0x7f800000 { // NaN → quiet NaN, preserve sign + payload high bits
		return uint16(u>>16) | 0x0040
	}
	u += 0x7fff + ((u >> 16) & 1) // round-to-nearest-even before truncating the low 16 bits
	return uint16(u >> 16)
}

// bfloat16ToFloat32 widens a bfloat16 bit pattern to float32 — the exact left-shift
// (bf16 is the high 16 bits of a float32). Mirrors safetensors.BFloat16ToFloat32; kept
// local so the quantiser has no dependency edge into the loader for one shift.
func bfloat16ToFloat32(bits uint16) float32 { return math.Float32frombits(uint32(bits) << 16) }
