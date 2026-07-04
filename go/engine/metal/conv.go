// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
)

// Conv2dBF16 is a byte-parity NHWC 2-D convolution: out[n,oh,ow,oc] = Σ_{kh,kw,ic}
// in[n, oh·strideH-padH+kh, ow·strideW-padW+kw, ic]·weight[oc,kh,kw,ic], fp32 accumulation rounded
// to bf16 — matching metal.Conv2d (groups 1, dilation 1). Out-of-bounds (padding) taps contribute
// zero. The gemma4 audio subsampler runs two of these (3×3, stride 2, pad 1). in is [N,H,W,inC] bf16,
// weight is [outC,kh,kw,inC] bf16; returns [N,outH,outW,outC] bf16 with outH=(H+2padH-kh)/strideH+1.
// (The depthwise conv1d in AudioLightConv proved the host fp32-accum conv is byte-identical to MLX's;
// this is the 2-D sibling — verified the same way.)
func Conv2dBF16(in, weight []byte, N, H, W, inC, outC, kh, kw, strideH, strideW, padH, padW int) ([]byte, error) {
	if len(in) != N*H*W*inC*bf16Size {
		return nil, core.NewError("native.Conv2dBF16: len(in) must equal N*H*W*inC*2 bytes")
	}
	if len(weight) != outC*kh*kw*inC*bf16Size {
		return nil, core.NewError("native.Conv2dBF16: len(weight) must equal outC*kh*kw*inC*2 bytes")
	}
	outH := (H+2*padH-kh)/strideH + 1
	outW := (W+2*padW-kw)/strideW + 1
	inF, wF := bf16ToF32Slice(in), bf16ToF32Slice(weight)
	out := make([]byte, N*outH*outW*outC*bf16Size)
	idx := func(dims ...int) int { // row-major flatten over the trailing dims given as (i,size) pairs
		o := 0
		for j := 0; j < len(dims); j += 2 {
			o = o*dims[j+1] + dims[j]
		}
		return o
	}
	for n := 0; n < N; n++ {
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				for oc := 0; oc < outC; oc++ {
					var acc float32
					for r := 0; r < kh; r++ {
						ih := oh*strideH - padH + r
						if ih < 0 || ih >= H {
							continue
						}
						for c := 0; c < kw; c++ {
							iw := ow*strideW - padW + c
							if iw < 0 || iw >= W {
								continue
							}
							for ic := 0; ic < inC; ic++ {
								acc += inF[idx(n, N, ih, H, iw, W, ic, inC)] * wF[idx(oc, outC, r, kh, c, kw, ic, inC)]
							}
						}
					}
					o := idx(n, N, oh, outH, ow, outW, oc, outC)
					h := f32ToBF16(acc)
					out[o*bf16Size], out[o*bf16Size+1] = byte(h), byte(h>>8)
				}
			}
		}
	}
	return out, nil
}

// Conv2dF32 is the fp32 NHWC convolution, BYTE-IDENTICAL to metal.Conv2d(f32) (the subsampler's
// second conv runs fp32). metal implements Conv2d as im2col (unfold) + a steel GEMM, so a direct
// triple-loop sum diverges ~1 ULP from the GEMM's accumulation order; this replicates it: unfold the
// receptive fields into [outH·outW, kh·kw·inC] (K order kh,kw,inC), then MatMulF32NT against the
// weight [outC, kh·kw·inC] (the steel GEMM). in is [N,H,W,inC], weight [outC,kh,kw,inC].
func Conv2dF32(in, weight []float32, N, H, W, inC, outC, kh, kw, strideH, strideW, padH, padW int) ([]float32, error) {
	if len(in) != N*H*W*inC {
		return nil, core.NewError("native.Conv2dF32: len(in) must equal N*H*W*inC")
	}
	if len(weight) != outC*kh*kw*inC {
		return nil, core.NewError("native.Conv2dF32: len(weight) must equal outC*kh*kw*inC")
	}
	outH := (H+2*padH-kh)/strideH + 1
	outW := (W+2*padW-kw)/strideW + 1
	K := kh * kw * inC
	out := make([]float32, N*outH*outW*outC)
	for n := 0; n < N; n++ {
		// unfold: [outH·outW, kh·kw·inC], K index = (r·kw + c)·inC + ic.
		unfolded := make([]float32, outH*outW*K)
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				m := oh*outW + ow
				for r := 0; r < kh; r++ {
					ih := oh*strideH - padH + r
					if ih < 0 || ih >= H {
						continue
					}
					for c := 0; c < kw; c++ {
						iw := ow*strideW - padW + c
						if iw < 0 || iw >= W {
							continue
						}
						inBase := ((n*H+ih)*W + iw) * inC
						kBase := (r*kw + c) * inC
						copy(unfolded[m*K+kBase:m*K+kBase+inC], in[inBase:inBase+inC])
					}
				}
			}
		}
		// out[m, oc] = Σ_K unfolded[m,K]·weight[oc,K] — the nt steel GEMM metal dispatches.
		o, err := MatMulF32NT(unfolded, weight, outH*outW, K, outC)
		if err != nil {
			return nil, err
		}
		copy(out[n*outH*outW*outC:(n+1)*outH*outW*outC], o)
	}
	return out, nil
}
