// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import core "dappco.re/go"

// DiffusionSDPA computes the block-diffusion canvas attention core: q is
// [nHeads,qLen,headDim], k/v are [nKVHeads,keyLen,headDim], and mask is an
// optional additive [qLen,keyLen] fp32 mask using 0 for attend and -Inf for
// blocked positions.
func DiffusionSDPA(q, k, v []byte, qLen, keyLen, nHeads, nKVHeads, headDim int, scale float32, mask []float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if qLen < 0 || keyLen < 0 || nHeads <= 0 || nKVHeads <= 0 || headDim <= 0 {
		return nil, core.NewError("native.DiffusionSDPA: invalid dimensions")
	}
	if nHeads%nKVHeads != 0 {
		return nil, core.NewError("native.DiffusionSDPA: nHeads must be a multiple of nKVHeads")
	}
	if len(mask) != 0 && len(mask) != qLen*keyLen {
		return nil, core.NewError("native.DiffusionSDPA: mask must be qLen*keyLen")
	}
	if len(q) != nHeads*qLen*headDim*bf16Size {
		return nil, core.NewError("native.DiffusionSDPA: len(q) must equal nHeads*qLen*headDim*2 bytes")
	}
	if len(k) != nKVHeads*keyLen*headDim*bf16Size || len(v) != len(k) {
		return nil, core.NewError("native.DiffusionSDPA: len(k)/len(v) must equal nKVHeads*keyLen*headDim*2 bytes")
	}
	if qLen == 0 {
		return []byte{}, nil
	}
	if keyLen == 0 {
		return nil, core.NewError("native.DiffusionSDPA: keyLen must be positive when qLen is non-zero")
	}

	grp := nHeads / nKVHeads
	out := make([]byte, nHeads*qLen*headDim*bf16Size)
	for h := range nHeads {
		kvh := h / grp
		qh := bf16HeadF32(q, h, qLen, headDim)
		kh := bf16HeadF32(k, kvh, keyLen, headDim)
		vh := bf16HeadF32(v, kvh, keyLen, headDim)

		scores, err := matRowsF32(kh, qh, qLen, keyLen, headDim)
		if err != nil {
			return nil, err
		}
		for i := range scores {
			scores[i] *= scale
		}
		if len(mask) > 0 {
			for i := range scores {
				scores[i] += mask[i]
			}
		}
		probs, err := SoftmaxF32(scores, keyLen)
		if err != nil {
			return nil, err
		}
		oh, err := matRowsF32(transposeF32(vh, keyLen, headDim), probs, qLen, headDim, keyLen)
		if err != nil {
			return nil, err
		}
		base := h * qLen * headDim * bf16Size
		for i, val := range oh {
			b := f32ToBF16(val)
			out[base+i*bf16Size], out[base+i*bf16Size+1] = byte(b), byte(b>>8)
		}
	}
	return out, nil
}
