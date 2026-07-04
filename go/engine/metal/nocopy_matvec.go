// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import core "dappco.re/go"

// nocopy_matvec.go is the resident-weight sibling of MatVecBF16: the matrix is supplied as a bufView — a
// no-copy view into a resident shard buffer (from shardBuffers.bufFor) — instead of host bytes, so a
// per-token op over a FIXED weight binds the weight at its shard offset rather than re-uploading the whole
// matrix to a fresh Metal buffer every call. This is the gemma4 PLE projection's hot path: PerLayerInputs
// runs once per token over a model-level [numLayers·pliDim, dModel] projection that never changes, and the
// host-bytes MatVecBF16 was the single biggest per-token tail cost — the NewBufferWithBytes balloon the
// CPU profile flagged (52% cum). Byte-identical to MatVecBF16(mat, vec) when matView backs the same bytes.

// pleResidentDisabled forces the PLE projection back onto the host-bytes MatVecBF16 path (a test hook for
// the resident-vs-host byte-identity check; always false in production).
var pleResidentDisabled bool

// icbDisabledForTest forces Generate onto the per-op stepToken path instead of replaying the recorded ICB
// (a test hook for the ICB-on/off cross-load reproducibility A/B; always false in production).
var icbDisabledForTest bool

// resetResidentBufsForTest clears the address-keyed resident-weight cache. residentBytes assumes ONE model
// per process (keys by &weight[0] in the stable mmap); a test that loads several models reuses freed mmap
// addresses after Close → munmap, so a stale cache hit returns a prior model's buffer. Tests that load more
// than one model must reset between loads. Never called in production (a served process loads one model).
func resetResidentBufsForTest() {
	residentBufMu.Lock()
	for _, r := range residentBufs {
		closeResidentBuf(r)
	}
	residentBufs = map[uintptr]residentBuf{}
	residentBufMu.Unlock()
}

// MatVecBF16Buf computes out[outDim] = mat[outDim,inDim] @ vec[inDim] in bf16, with the matrix bound from a
// resident no-copy buffer view (matView) at its offset; vec/out stay per-call (small). A nil matView.buf is
// an error — the caller falls back to MatVecBF16 when no shard buffer is available.
func MatVecBF16Buf(matView bufView, vec []byte, outDim, inDim int) ([]byte, error) {
	return MatVecBF16BufInto(nil, matView, vec, outDim, inDim)
}

func MatVecBF16BufInto(out []byte, matView bufView, vec []byte, outDim, inDim int) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if matView.buf == nil {
		return nil, core.NewError("native.MatVecBF16Buf: nil resident matrix buffer")
	}
	if len(vec) != inDim*bf16Size {
		return nil, core.NewError("native.MatVecBF16Buf: len(vec) must equal inDim*2 bytes")
	}
	if outDim == 0 || inDim == 0 {
		outLen := outDim * bf16Size
		if cap(out) < outLen {
			return make([]byte, outLen), nil
		}
		return out[:outLen], nil
	}
	bm, bn, sm, sn, tm, tn := gemvTiles(inDim, outDim)
	name := gemvKernelName("bfloat16", bm, bn, sm, sn, tm, tn)
	pso, err := pipelineFor(name)
	if err != nil {
		return nil, err
	}
	outLen := outDim * bf16Size
	callerOut := cap(out) >= outLen
	if !callerOut {
		out = make([]byte, outLen)
	} else {
		out = out[:outLen]
	}
	var encErr error
	withAutoreleasePool(func() {
		scratch, err := getQMVBF16Scratch(outDim, inDim)
		if err != nil {
			encErr = err
			return
		}
		defer putQMVBF16Scratch(scratch)
		vecBuf, outBuf, err := scratch.buffers(vec)
		if err != nil {
			encErr = err
			return
		}
		directOut := false
		if callerOut {
			if tmp, ok := scratch.outputView(out); ok {
				outBuf = tmp
				directOut = true
			}
		}

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		emitGemv(encSink{enc}, pso, matView.buf, matView.off, vecBuf, outBuf, 0, inDim, outDim, bm, bn, sm, tm)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)

		if !directOut {
			copy(out, scratch.out.bytes[:outLen])
		}
	})
	if encErr != nil {
		return nil, encErr
	}
	return out, nil
}
