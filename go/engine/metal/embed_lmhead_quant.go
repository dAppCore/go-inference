// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"

	core "dappco.re/go"
)

// The quant siblings of the decode bookends: in a 4-bit gemma4 checkpoint the embedding
// table is itself quantised (mlx quantises nn.Embedding, and gemma ties the LM head to it),
// so the input embedding must dequantise a gathered row and the LM head is a quantised
// projection. bf16 []byte throughout (the seam's lingua franca).

// EmbedTokensQuant is the gemma4 input embedding for a 4-bit affine-quantised table: it
// gathers each token's row and dequantises it on the HOST (value = scale·code + bias per
// group, 4-bit codes unpacked from the packed bytes), then applies `scale` (= √hidden,
// metal's EmbeddingScale). Only the gathered rows are dequantised — not the whole table — so a
// 4-bit embedding stays 4-bit in memory. packed is the [vocab × dModel] affine-packed weight
// (dModel·bits/8 bytes per row), scales/biases the per-group bf16 (dModel/groupSize per row).
// Pure host (no device): byte-for-byte equal to metal.Dequantize on the gathered rows (gated).
func EmbedTokensQuant(packed, scales, biases []byte, tokenIDs []int32, vocab, dModel, groupSize, bits int, scale float32) ([][]byte, error) {
	groups, rowPacked, rowSB, err := quantEmbedShape(packed, scales, biases, vocab, dModel, groupSize, bits)
	if err != nil {
		return nil, err
	}
	out := make([][]byte, len(tokenIDs))
	for i, tok := range tokenIDs {
		if tok < 0 || int(tok) >= vocab {
			return nil, core.NewError("native.EmbedTokensQuant: token id out of range")
		}
		pRow := packed[int(tok)*rowPacked : (int(tok)+1)*rowPacked]
		sRow := scales[int(tok)*rowSB : (int(tok)+1)*rowSB]
		bRow := biases[int(tok)*rowSB : (int(tok)+1)*rowSB]
		emb := make([]byte, dModel*bf16Size)
		embedTokenQuantRowInto(emb, pRow, sRow, bRow, dModel, groupSize, bits, groups, scale)
		out[i] = emb
	}
	return out, nil
}

func quantEmbedShape(packed, scales, biases []byte, vocab, dModel, groupSize, bits int) (groups, rowPacked, rowSB int, err error) {
	if bits <= 0 || bits > 8 {
		return 0, 0, 0, core.NewError("native.EmbedTokensQuant: bits must be in 1..8")
	}
	if groupSize <= 0 || dModel%groupSize != 0 {
		return 0, 0, 0, core.NewError("native.EmbedTokensQuant: groupSize must be > 0 and divide dModel")
	}
	groups = dModel / groupSize
	rowPacked = dModel * bits / 8 // packed bytes per row (dModel/2 for 4-bit)
	rowSB = groups * bf16Size     // scales (or biases) bytes per row
	if len(packed) != vocab*rowPacked {
		return 0, 0, 0, core.NewError("native.EmbedTokensQuant: packed size != vocab·dModel·bits/8")
	}
	if len(scales) != vocab*rowSB || len(biases) != vocab*rowSB {
		return 0, 0, 0, core.NewError("native.EmbedTokensQuant: scales/biases size != vocab·(dModel/groupSize) bf16")
	}
	return groups, rowPacked, rowSB, nil
}

func embedTokenQuant(packed, scales, biases []byte, tok int32, vocab, dModel, groupSize, bits int, scale float32) ([]byte, error) {
	emb := make([]byte, dModel*bf16Size)
	return embedTokenQuantInto(emb, packed, scales, biases, tok, vocab, dModel, groupSize, bits, scale)
}

func embedTokenQuantInto(dst, packed, scales, biases []byte, tok int32, vocab, dModel, groupSize, bits int, scale float32) ([]byte, error) {
	groups, rowPacked, rowSB, err := quantEmbedShape(packed, scales, biases, vocab, dModel, groupSize, bits)
	if err != nil {
		return nil, err
	}
	if len(dst) != dModel*bf16Size {
		return nil, core.NewError("native.EmbedTokensQuant: dst must be dModel bf16 bytes")
	}
	if tok < 0 || int(tok) >= vocab {
		return nil, core.NewError("native.EmbedTokensQuant: token id out of range")
	}
	pRow := packed[int(tok)*rowPacked : (int(tok)+1)*rowPacked]
	sRow := scales[int(tok)*rowSB : (int(tok)+1)*rowSB]
	bRow := biases[int(tok)*rowSB : (int(tok)+1)*rowSB]
	embedTokenQuantRowInto(dst, pRow, sRow, bRow, dModel, groupSize, bits, groups, scale)
	return dst, nil
}

func embedTokenQuantRow(pRow, sRow, bRow []byte, dModel, groupSize, bits, groups int, scale float32) []byte {
	emb := make([]byte, dModel*bf16Size)
	embedTokenQuantRowInto(emb, pRow, sRow, bRow, dModel, groupSize, bits, groups, scale)
	return emb
}

func embedTokenQuantRowInto(emb, pRow, sRow, bRow []byte, dModel, groupSize, bits, groups int, scale float32) {
	if bits == 4 {
		// 4-bit fast path: nibbles are byte-aligned (no bit-spanning), and the affine params are
		// per-group — hoist their bf16ToF32 out of the inner loop (they change per group, not per
		// element). Byte-identical to the general path: same code value, same (s·code+b)·scale order.
		for g := 0; g < groups; g++ {
			s := bf16ToF32(sRow[g*bf16Size], sRow[g*bf16Size+1])
			b := bf16ToF32(bRow[g*bf16Size], bRow[g*bf16Size+1])
			base := g * groupSize
			for j := 0; j < groupSize; j++ {
				c := base + j
				var code float32
				if c&1 == 0 {
					code = float32(pRow[c>>1] & 0x0F) // low nibble for even c
				} else {
					code = float32(pRow[c>>1] >> 4) // high nibble for odd c
				}
				h := f32ToBF16((s*code + b) * scale)
				emb[c*bf16Size] = byte(h)
				emb[c*bf16Size+1] = byte(h >> 8)
			}
		}
		return
	}
	for c := 0; c < dModel; c++ {
		// affine codes are bit-packed LSB-first contiguous, spanning byte boundaries for 5/6-bit.
		code := extractAffineCode(pRow, c*bits, bits)
		g := c / groupSize
		s := bf16ToF32(sRow[g*bf16Size], sRow[g*bf16Size+1])
		b := bf16ToF32(bRow[g*bf16Size], bRow[g*bf16Size+1])
		h := f32ToBF16((s*float32(code) + b) * scale)
		emb[c*bf16Size] = byte(h)
		emb[c*bf16Size+1] = byte(h >> 8)
	}
}

// extractAffineCode reads the bits-wide affine code at bit offset bitOff from a packed row,
// LSB-first contiguous — MLX's affine packing (the 4-bit nibble-low-first layout generalised),
// spanning byte boundaries for non-byte-aligned widths (5/6-bit). For 4-bit it reduces to the
// nibble read, for 8-bit to the byte read.
func extractAffineCode(p []byte, bitOff, bits int) uint32 {
	var v uint32
	for got := 0; got < bits; {
		bi := (bitOff + got) / 8
		off := (bitOff + got) % 8
		take := 8 - off
		if take > bits-got {
			take = bits - got
		}
		chunk := (uint32(p[bi]) >> uint(off)) & ((1 << uint(take)) - 1)
		v |= chunk << uint(got)
		got += take
	}
	return v
}

// LMHeadQuant is the gemma4 output head when the LM projection is 4-bit quantised (the tied
// embedding of a 4-bit checkpoint): final RMSNorm, the quantised output projection (QMVBF16
// over the packed embedding), then the optional final-logit soft-cap (monotonic, preserves the
// argmax). hidden/finalNormW are dModel bf16; packed/scales/biases are the [vocab × dModel]
// affine-quant embedding; returns vocab bf16 logits. Norm + projection run on-device (QMVBF16
// is byte-parity-gated vs metal.QuantizedMatmul), the soft-cap is a host pass. softCap <= 0
// skips the cap.
func LMHeadQuant(hidden, finalNormW, packed, scales, biases []byte, dModel, vocab, groupSize, bits int, eps, softCap float32) ([]byte, error) {
	return LMHeadQuantInto(nil, hidden, finalNormW, packed, scales, biases, dModel, vocab, groupSize, bits, eps, softCap)
}

// LMHeadQuantInto is LMHeadQuant writing into caller-owned logits storage when
// cap(out) >= vocab*2. The quantised projection binds the result slice directly
// where possible, so the no-cap decode bookend avoids a scratch-to-result copy.
func LMHeadQuantInto(out []byte, hidden, finalNormW, packed, scales, biases []byte, dModel, vocab, groupSize, bits int, eps, softCap float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(hidden) != dModel*bf16Size {
		return nil, core.NewError("native.LMHeadQuant: hidden must be dModel bf16 bytes")
	}
	if len(finalNormW) != dModel*bf16Size {
		return nil, core.NewError("native.LMHeadQuant: finalNormW must be dModel bf16 bytes")
	}
	if groupSize <= 0 || dModel%groupSize != 0 {
		return nil, core.NewError("native.LMHeadQuant: groupSize must be > 0 and divide dModel")
	}
	wantPacked := vocab * dModel * bits / 8
	wantSB := vocab * (dModel / groupSize) * bf16Size
	if len(packed) != wantPacked || len(scales) != wantSB || len(biases) != wantSB {
		return nil, core.NewError("native.LMHeadQuant: packed/scales/biases size mismatch vs vocab·dModel")
	}
	outLen := vocab * bf16Size
	callerOut := cap(out) >= outLen
	if callerOut {
		out = out[:outLen]
	} else {
		out = make([]byte, outLen)
	}
	if dModel == 0 || vocab == 0 {
		return out, nil
	}
	var encErr error
	withAutoreleasePool(func() {
		ioScratch, err := getQMVBF16Scratch(vocab, dModel)
		if err != nil {
			encErr = err
			return
		}
		defer putQMVBF16Scratch(ioScratch)
		hiddenBuf, logitsBuf, err := ioScratch.buffers(hidden)
		if err != nil {
			encErr = err
			return
		}
		directOut := false
		if callerOut {
			if tmp, ok := ioScratch.outputView(out); ok {
				logitsBuf = tmp
				directOut = true
			}
		}
		finalNormBuf := residentBytes(finalNormW)
		packedBuf, scalesBuf, biasesBuf := residentBytes(packed), residentBytes(scales), residentBytes(biases)
		normed := scratchBF16(dModel)

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		if encErr = encRMSNormBF16(enc, hiddenBuf, finalNormBuf, normed, 0, dModel, eps); encErr != nil {
			endEncodingFast(enc)
			return
		}
		if encErr = encQMVBF16(enc, packedBuf, scalesBuf, biasesBuf, normed, logitsBuf, 0, 0, 0, 0, vocab, dModel, groupSize, bits); encErr != nil {
			endEncodingFast(enc)
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if !directOut {
			copy(out, ioScratch.out.bytes[:outLen])
		}
	})
	if encErr != nil {
		return nil, encErr
	}
	if softCap > 0 {
		for i := 0; i < vocab; i++ {
			v := bf16ToF32(out[i*bf16Size], out[i*bf16Size+1])
			h := f32ToBF16(softCap * float32(math.Tanh(float64(v/softCap))))
			out[i*bf16Size] = byte(h)
			out[i*bf16Size+1] = byte(h >> 8)
		}
	}
	return out, nil
}
