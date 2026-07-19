// SPDX-Licence-Identifier: EUPL-1.2

package deepseekvl2

import (
	"math"

	core "dappco.re/go"
)

// vision_sam.go is deepencoder.ImageEncoderViT ported host-side: PatchEmbed (conv, stride==
// kernel) → + the fixed absolute position table (never interpolated — the checkpoint's ONE
// supported input, samImgSize×samImgSize, always lands on samGridSize×samGridSize exactly, see
// weights_sam.go's doc comment) → samDepth pre-norm ViTDet blocks (windowed OR global attention
// with a decomposed relative-position bias — deepencoder.add_decomposed_rel_pos) → the neck (two
// convs) → net_2/net_3 (two stride-2 convs), the tower's own spatial "optical compression" from a
// 64×64 patch grid down to 16×16. All tensors are kept row-major [H*W,C] channel-last (a flat
// token sequence) throughout, converting to/from PyTorch's NCHW only inside conv2D's own indexing
// — the same convention model/composed/vision.go's Qwen-VL tower uses, so a conv here is just
// "a token's neighbourhood" rather than a distinct tensor layout to track.

// conv2D runs one Conv2d(inC,outC,kernel,stride,padding) over input[H,W,inC] (channel-last),
// weight [outC,inC,kernel,kernel] row-major (the PyTorch/safetensors layout), bias nil ⇒ no bias.
// Returns output[Hout,Wout,outC] channel-last and the output spatial dims. Output ROWS (fixed oy)
// are independent — parallelFor (mathops.go's file doc comment); a real checkpoint's neck/
// downsample convs are large enough (e.g. 64x64x256x256x3x3) that a single core is impractically
// slow for a gate.
func conv2D(input []float32, h, w, inC int, weight, bias []float32, outC, kernel, stride, padding int) (out []float32, outH, outW int) {
	outH = (h+2*padding-kernel)/stride + 1
	outW = (w+2*padding-kernel)/stride + 1
	out = make([]float32, outH*outW*outC)
	parallelFor(outH, func(oy int) {
		iy0 := oy*stride - padding
		for ox := range outW {
			ix0 := ox*stride - padding
			dst := out[(oy*outW+ox)*outC : (oy*outW+ox)*outC+outC]
			for oc := range outC {
				var acc float64
				wBase := oc * inC * kernel * kernel
				for ky := range kernel {
					iy := iy0 + ky
					if iy < 0 || iy >= h {
						continue
					}
					for kx := range kernel {
						ix := ix0 + kx
						if ix < 0 || ix >= w {
							continue
						}
						srcRow := input[(iy*w+ix)*inC : (iy*w+ix)*inC+inC]
						// weight is [outC,inC,KH,KW] row-major — inC is NOT innermost (KW is), so each
						// input channel's weight lives kernel*kernel elements apart; indexed explicitly
						// rather than sliced (a transposed-weight fast path is a future perf lever, not
						// this correctness-first pass — see the file doc comment).
						kOff := ky*kernel + kx
						for ic := range inC {
							acc += float64(srcRow[ic]) * float64(weight[wBase+ic*kernel*kernel+kOff])
						}
					}
				}
				if bias != nil {
					acc += float64(bias[oc])
				}
				dst[oc] = float32(acc)
			}
		}
	})
	return out, outH, outW
}

// layerNorm2D applies LayerNorm2d (per-pixel, over the CHANNEL axis only — deepencoder.LayerNorm2d:
// mean/var computed across channels at each spatial position, unlike layerNormBias's
// last-flat-dimension convention) to x[H*W,C] channel-last — the same maths, since x is already
// channel-last flat, so this is literally layerNormBias with D=C.
func layerNorm2D(x []float32, w, b []float32, c int, eps float32) []float32 {
	return layerNormBias(x, w, b, c, eps)
}

// samPatchEmbed runs PatchEmbed: Conv2d(3,768,kernel=16,stride=16) over pixels[1024,1024,3]
// channel-last normalised to [-1,1] (BasicImageTransform's mean=std=0.5 — see vision.go's
// preprocessing), returning the [64,64,768] flat patch grid.
func samPatchEmbed(pixels []float32, w SAMWeights) []float32 {
	out, _, _ := conv2D(pixels, samImgSize, samImgSize, 3, w.PatchEmbedW, w.PatchEmbedB, samEmbedDim, samPatchSize, samPatchSize, 0)
	return out
}

// windowPartition splits x[H,W,C] channel-last into non-overlapping winSize×winSize windows,
// zero-padding H/W up to the next multiple of winSize first (deepencoder.window_partition).
// Returns the windows flat [numWinH*numWinW*winSize*winSize,C] (window-major, then row-major
// within each window) plus the padded dims and window grid shape.
func windowPartition(x []float32, h, w, c, winSize int) (windows []float32, hp, wp, numWinH, numWinW int) {
	padH := (winSize - h%winSize) % winSize
	padW := (winSize - w%winSize) % winSize
	hp, wp = h+padH, w+padW
	numWinH, numWinW = hp/winSize, wp/winSize
	windows = make([]float32, numWinH*numWinW*winSize*winSize*c)
	idx := 0
	for wy := range numWinH {
		for wx := range numWinW {
			for py := range winSize {
				srcY := wy*winSize + py
				for px := range winSize {
					srcX := wx*winSize + px
					if srcY < h && srcX < w {
						copy(windows[idx*c:idx*c+c], x[(srcY*w+srcX)*c:(srcY*w+srcX)*c+c])
					}
					idx++
				}
			}
		}
	}
	return windows, hp, wp, numWinH, numWinW
}

// windowUnpartition reverses windowPartition: windows[numWinH*numWinW*winSize*winSize,C] back to
// x[h,w,C] channel-last, dropping the padding (deepencoder.window_unpartition).
func windowUnpartition(windows []float32, winSize, hp, wp, numWinH, numWinW, h, w, c int) []float32 {
	out := make([]float32, h*w*c)
	idx := 0
	for wy := range numWinH {
		for wx := range numWinW {
			for py := range winSize {
				srcY := wy*winSize + py
				for px := range winSize {
					srcX := wx*winSize + px
					if srcY < h && srcX < w {
						copy(out[(srcY*w+srcX)*c:(srcY*w+srcX)*c+c], windows[idx*c:idx*c+c])
					}
					idx++
				}
			}
		}
	}
	return out
}

// samRelPosBias returns, for a query at grid row/col (qr,qc) with head vector qVec[headDim], the
// additive bias against every key position in a gridSize×gridSize grid — deepencoder's
// add_decomposed_rel_pos, specialised to q_size==k_size (always true here: a windowed block's
// query/key grid is winSize×winSize, a global block's is samGridSize×samGridSize — see
// weights_sam.go's RelPosLen doc comment for why this means get_rel_pos's interpolation branch
// never triggers, so relative_coords[qi,kj] = qi-kj+(size-1) is a DIRECT table index, no resample).
func samRelPosBias(qVec []float32, qr, qc, gridSize, headDim int, relPosH, relPosW []float32) (relH, relW []float64) {
	relH = make([]float64, gridSize)
	relW = make([]float64, gridSize)
	for kr := range gridSize {
		row := relPosH[(qr-kr+gridSize-1)*headDim : (qr-kr+gridSize-1)*headDim+headDim]
		var acc float64
		for d := range headDim {
			acc += float64(qVec[d]) * float64(row[d])
		}
		relH[kr] = acc
	}
	for kc := range gridSize {
		row := relPosW[(qc-kc+gridSize-1)*headDim : (qc-kc+gridSize-1)*headDim+headDim]
		var acc float64
		for d := range headDim {
			acc += float64(qVec[d]) * float64(row[d])
		}
		relW[kc] = acc
	}
	return relH, relW
}

// samAttentionForward runs one block's multi-head self-attention with decomposed relative
// position bias over a gridSize×gridSize token grid x[gridSize*gridSize,dim] channel-last (a
// single window's tokens for a windowed block, or the whole grid for a global block —
// samBlockForward resolves which before calling this). No causal mask (SAM's tower attends
// bidirectionally, like every vision tower in this codebase). dim/headDim are derived from the
// weights themselves (never a package constant) so this same function serves both the real
// samEmbedDim/samNumHeads=768/12 tower AND vision_sam_test.go's toy-scale goldens.
func samAttentionForward(x []float32, gridSize, numHeads int, attn SAMAttnWeights) []float32 {
	dim := len(attn.ProjBias)
	headDim := dim / numHeads
	l := gridSize * gridSize
	qkv := linear(x, attn.QKVWeight, dim, 3*dim, attn.QKVBias) // [l, 3*dim]
	q := make([]float32, l*dim)
	k := make([]float32, l*dim)
	v := make([]float32, l*dim)
	for t := range l {
		row := qkv[t*3*dim : (t+1)*3*dim]
		copy(q[t*dim:(t+1)*dim], row[0:dim])
		copy(k[t*dim:(t+1)*dim], row[dim:2*dim])
		copy(v[t*dim:(t+1)*dim], row[2*dim:3*dim])
	}

	scale := 1.0 / math.Sqrt(float64(headDim))
	out := make([]float32, l*dim)
	// Parallelise over the flattened (head, query-token) space — each unit computes its own
	// softmax row into a disjoint out[] slice; `scores` MUST be allocated per-unit (never shared
	// across goroutines — mathops.go's parallelFor doc comment) since each unit's causal-free
	// softmax pass writes and re-reads it repeatedly. This is SAM's dominant cost (global blocks
	// attend over the full 4096-token grid) — confirmed impractically slow single-threaded.
	parallelFor(numHeads*l, func(unit int) {
		h, qt := unit/l, unit%l
		off := h * headDim
		scores := make([]float64, l)
		qr, qc := qt/gridSize, qt%gridSize
		qVec := q[qt*dim+off : qt*dim+off+headDim]
		relH, relW := samRelPosBias(qVec, qr, qc, gridSize, headDim, attn.RelPosH, attn.RelPosW)
		var maxScore = math.Inf(-1)
		for kt := range l {
			kr, kc := kt/gridSize, kt%gridSize
			kVec := k[kt*dim+off : kt*dim+off+headDim]
			var dot float64
			for d := range headDim {
				dot += float64(qVec[d]) * float64(kVec[d])
			}
			sc := dot*scale + relH[kr] + relW[kc]
			scores[kt] = sc
			if sc > maxScore {
				maxScore = sc
			}
		}
		var sum float64
		for kt := range l {
			e := math.Exp(scores[kt] - maxScore)
			scores[kt] = e
			sum += e
		}
		oVec := out[qt*dim+off : qt*dim+off+headDim]
		for d := range headDim {
			var acc float64
			for kt := range l {
				acc += scores[kt] * float64(v[kt*dim+off+d])
			}
			oVec[d] = float32(acc / sum)
		}
	})
	return linear(out, attn.ProjWeight, dim, dim, attn.ProjBias)
}

// samBlockForward runs one pre-norm ViTDet block over the FULL gridH×gridW grid
// x[gridH*gridW,dim]: window-partition (if b.WindowSize>0) or run global — attention —
// unpartition, residual, LayerNorm, GELU MLP, residual (deepencoder.Block.forward). dim/mlpHidden
// are derived from the weights themselves, gridH/gridW/numHeads are explicit parameters — see
// samAttentionForward's doc comment for why (real tower dims vs toy-golden dims share this code).
func samBlockForward(x []float32, gridH, gridW, numHeads int, b SAMBlockWeights) []float32 {
	dim := len(b.Norm1W)
	mlpHidden := len(b.MLPLin1B)
	normed := layerNormBias(x, b.Norm1W, b.Norm1B, dim, 1e-6) // SAM's norm_layer default eps
	var attnOut []float32
	if b.WindowSize > 0 {
		windows, hp, wp, numWinH, numWinW := windowPartition(normed, gridH, gridW, dim, b.WindowSize)
		outWindows := make([]float32, len(windows))
		winTokens := b.WindowSize * b.WindowSize
		for wi := range numWinH * numWinW {
			win := windows[wi*winTokens*dim : (wi+1)*winTokens*dim]
			copy(outWindows[wi*winTokens*dim:(wi+1)*winTokens*dim], samAttentionForward(win, b.WindowSize, numHeads, b.Attn))
		}
		attnOut = windowUnpartition(outWindows, b.WindowSize, hp, wp, numWinH, numWinW, gridH, gridW, dim)
	} else {
		// samAttentionForward's global (non-windowed) path takes one square gridSize — true for
		// every global block this checkpoint (and vision_sam_test.go's toy golden) ever runs.
		attnOut = samAttentionForward(normed, gridH, numHeads, b.Attn)
	}
	hidden := addRows(x, attnOut)
	normed2 := layerNormBias(hidden, b.Norm2W, b.Norm2B, dim, 1e-6)
	mlp := linear(geluRow(linear(normed2, b.MLPLin1W, dim, mlpHidden, b.MLPLin1B)), b.MLPLin2W, mlpHidden, dim, b.MLPLin2B)
	return addRows(hidden, mlp)
}

// SAMForward runs the whole SAM ViT-B tower over pixels[1024,1024,3] channel-last (normalised to
// [-1,1]): patch embed + position → samDepth blocks → neck → net_2 → net_3, returning the final
// [16,16,1024] flat token grid (samGridSize/4 per axis — two stride-2 downsamples) channel-last —
// this IS `sam_out.flatten(2).permute(0,2,1)` from the reference (vision.go's doc comment), ready
// both as CLIP's external patch_embeds input and for the projector concat.
func SAMForward(pixels []float32, w SAMWeights) ([]float32, error) {
	if len(pixels) != samImgSize*samImgSize*3 {
		return nil, core.NewError(core.Sprintf("deepseekvl2.SAMForward: pixel buffer len %d, want %d (%dx%dx3)", len(pixels), samImgSize*samImgSize*3, samImgSize, samImgSize))
	}
	hidden := samPatchEmbed(pixels, w) // [64*64,768]
	hidden = addRows(hidden, w.PosEmbed)
	for _, b := range w.Blocks {
		hidden = samBlockForward(hidden, samGridSize, samGridSize, samNumHeads, b)
	}

	// neck: Conv1x1(768->256) -> LayerNorm2d -> Conv3x3 pad1 (256->256) -> LayerNorm2d, all at the
	// full 64x64 grid (deepencoder.py's nn.Sequential neck, run on x.permute(0,3,1,2) in the
	// reference — our channel-last convention needs no such permute, conv2D already reads NCHW-
	// equivalent weights over a channel-last activation).
	neck1, gh, gw := conv2D(hidden, samGridSize, samGridSize, samEmbedDim, w.NeckConv1W, nil, samOutChans, 1, 1, 0)
	neck1 = layerNorm2D(neck1, w.NeckLN1W, w.NeckLN1B, samOutChans, 1e-6)
	neck2, gh, gw := conv2D(neck1, gh, gw, samOutChans, w.NeckConv2W, nil, samOutChans, 3, 1, 1)
	neck2 = layerNorm2D(neck2, w.NeckLN2W, w.NeckLN2B, samOutChans, 1e-6)

	net2, gh, gw := conv2D(neck2, gh, gw, samOutChans, w.Net2W, nil, samNeckOut2, 3, 2, 1)
	net3, gh, gw := conv2D(net2, gh, gw, samNeckOut2, w.Net3W, nil, samNeckOut3, 3, 2, 1)
	if gh != samGridSize/4 || gw != samGridSize/4 {
		return nil, core.NewError(core.Sprintf("deepseekvl2.SAMForward: final grid %dx%d, want %dx%d", gh, gw, samGridSize/4, samGridSize/4))
	}
	return net3, nil
}
