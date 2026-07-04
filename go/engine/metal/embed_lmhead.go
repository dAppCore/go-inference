// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"

	core "dappco.re/go"
)

// The decode bookends: the input embedding (token ids → scaled hidden vectors that feed
// the decode) and the LM head (a hidden state → vocab logits). Together with a backend's
// DecodeForward they are the whole token → logits path; sampling + the tokenizer sit on
// top. bf16 []byte throughout (the seam's lingua franca).

// EmbedTokensBF16 is the gemma4 input embedding: each token's row of the embedding table
// scaled by `scale` (= sqrt(hidden) — metal's EmbeddingScale). table is [vocab × dModel]
// row-major bf16; returns one dModel bf16 vector per token id. The gather + scalar scale
// is pure data movement (no kernel); the scale is applied in f32 then rounded to bf16,
// matching metal's MulScalar(embed, sqrt(hidden)).
func EmbedTokensBF16(table []byte, tokenIDs []int32, vocab, dModel int, scale float32) ([][]byte, error) {
	if len(table) != vocab*dModel*bf16Size {
		return nil, core.NewError("native.EmbedTokensBF16: table must be vocab*dModel bf16 bytes")
	}
	rowBytes := dModel * bf16Size
	out := make([][]byte, len(tokenIDs))
	for i, tok := range tokenIDs {
		emb := make([]byte, rowBytes)
		if _, err := embedTokenBF16Into(emb, table, tok, vocab, dModel, scale); err != nil {
			return nil, err
		}
		out[i] = emb
	}
	return out, nil
}

func embedTokenBF16(table []byte, tok int32, vocab, dModel int, scale float32) ([]byte, error) {
	emb := make([]byte, dModel*bf16Size)
	return embedTokenBF16Into(emb, table, tok, vocab, dModel, scale)
}

func embedTokenBF16Into(dst, table []byte, tok int32, vocab, dModel int, scale float32) ([]byte, error) {
	if len(table) != vocab*dModel*bf16Size {
		return nil, core.NewError("native.EmbedTokensBF16: table must be vocab*dModel bf16 bytes")
	}
	rowBytes := dModel * bf16Size
	if len(dst) != rowBytes {
		return nil, core.NewError("native.EmbedTokensBF16: dst must be dModel bf16 bytes")
	}
	if tok < 0 || int(tok) >= vocab {
		return nil, core.NewError("native.EmbedTokensBF16: token id out of range")
	}
	row := table[int(tok)*rowBytes : (int(tok)+1)*rowBytes]
	for j := 0; j < dModel; j++ {
		v := bf16ToF32(row[j*bf16Size], row[j*bf16Size+1]) * scale
		h := f32ToBF16(v)
		dst[j*bf16Size] = byte(h)
		dst[j*bf16Size+1] = byte(h >> 8)
	}
	return dst, nil
}

// LMHeadBF16 is the gemma4 output head on a single hidden state: final RMSNorm, the
// output projection (dModel → vocab), then the optional final-logit soft-cap
// (softCap·tanh(logit/softCap), which is monotonic so it preserves the argmax). hidden
// and finalNormW are dModel bf16, outWeight is [vocab × dModel] row-major bf16 (the tied
// embedding or a separate head); returns vocab bf16 logits. The norm + projection run
// on-device in one command buffer with resident fixed weights; the soft-cap is a host
// elementwise pass. softCap <= 0 skips the cap.
func LMHeadBF16(hidden, finalNormW, outWeight []byte, dModel, vocab int, eps, softCap float32) ([]byte, error) {
	return LMHeadBF16Into(nil, hidden, finalNormW, outWeight, dModel, vocab, eps, softCap)
}

// LMHeadBF16Into is LMHeadBF16 writing into caller-owned logits storage when
// cap(out) >= vocab*2. The Metal projection binds the result slice directly
// where possible, so the no-cap decode bookend avoids a scratch-to-result copy.
func LMHeadBF16Into(out []byte, hidden, finalNormW, outWeight []byte, dModel, vocab int, eps, softCap float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(hidden) != dModel*bf16Size {
		return nil, core.NewError("native.LMHeadBF16: hidden must be dModel bf16 bytes")
	}
	if len(finalNormW) != dModel*bf16Size {
		return nil, core.NewError("native.LMHeadBF16: finalNormW must be dModel bf16 bytes")
	}
	if len(outWeight) != vocab*dModel*bf16Size {
		return nil, core.NewError("native.LMHeadBF16: outWeight must be vocab*dModel bf16 bytes")
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
		outWeightBuf := residentBytes(outWeight)
		normedBuf := scratchBF16(dModel)

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		if encErr = encRMSNormBF16(enc, hiddenBuf, finalNormBuf, normedBuf, 0, dModel, eps); encErr != nil {
			endEncodingFast(enc)
			return
		}
		if encErr = encGemvBF16(enc, outWeightBuf, normedBuf, logitsBuf, vocab, dModel); encErr != nil {
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
			capped := softCap * float32(math.Tanh(float64(v/softCap)))
			h := f32ToBF16(capped)
			out[i*bf16Size] = byte(h)
			out[i*bf16Size+1] = byte(h >> 8)
		}
	}
	return out, nil
}
