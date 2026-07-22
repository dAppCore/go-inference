// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"

	core "dappco.re/go"
)

// sdpa_sinks.go is the attention-sinks SDPA surface (gpt_oss): the public single/two-pass GPU
// drivers over the has_sinks(25) sdpa_vector pipeline variants, and the HOST reference (the CPU
// oracle) the byte gates compare against.
//
// The semantics, verified against both lineage references, fetched from source (not recalled):
//
//	https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gpt_oss/modeling_gpt_oss.py
//	  eager_attention_forward: "combined_logits = torch.cat([attn_weights, sinks], dim=-1)"
//	  → softmax → "scores = probs[..., :-1]" — the sink is ONE extra softmax column per head,
//	  dropped from the output weights (it contributes denominator mass, never value mass).
//	  "self.sinks = nn.Parameter(torch.empty(config.num_attention_heads))" — shape [heads].
//	https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/gpt_oss.py
//	  "scaled_dot_product_attention(q, k, v, cache, self.sm_scale, mask=mask, sinks=self.sinks)"
//	  — the InferenceIllusionist MLX-4bit checkpoint's own lineage drives MLX fast attention's
//	  sinks parameter, which is EXACTLY the kernel lane these pipelines enable: MLX v0.32.0
//	  mlx/backend/metal/kernels/sdpa_vector.h (the pinned source of the shipped metallib) seeds
//	  the online softmax with "max_score = sinks[head]; sum_exp_score = 1" in one simdgroup
//	  (single-pass) / block 0 (2-pass pass 1) — algebraically identical to the concat-and-drop
//	  reference above.

// softmaxWithSinkF32 computes softmax over [logits ; sink] and returns ONLY the logits' weights
// (the sink's column is dropped — its exp stays in the denominator). This is the reference formula
// in its numerically-stable fused form: m = max(max(logits), sink); w_i = exp(l_i-m) / (Σ_j
// exp(l_j-m) + exp(sink-m)). The returned weights therefore sum to LESS than 1 by exactly the
// sink's softmax share — the probability mass the head "dumps on nothing".
func softmaxWithSinkF32(logits []float32, sink float32) ([]float32, error) {
	if len(logits) == 0 {
		return nil, core.NewError("native.softmaxWithSinkF32: empty logits")
	}
	m := float64(sink)
	for _, l := range logits {
		if float64(l) > m {
			m = float64(l)
		}
	}
	denom := math.Exp(float64(sink) - m)
	w := make([]float32, len(logits))
	for i, l := range logits {
		e := math.Exp(float64(l) - m)
		w[i] = float32(e)
		denom += e
	}
	if denom == 0 { // all -Inf: no mass anywhere — surface it rather than divide by zero
		return nil, core.NewError("native.softmaxWithSinkF32: zero softmax denominator")
	}
	inv := 1.0 / denom
	for i := range w {
		w[i] = float32(float64(w[i]) * inv)
	}
	return w, nil
}

// sdpaHostRefWithSinks is the CPU oracle for single-query attention WITH sinks: the same raw-bf16
// byte ABI as SDPA (q (b,nHeads,1,headDim), k/v (b,nKVHeads,kvLen,headDim) → out (b,nHeads,1,
// headDim)) computed on the host in float64/float32 — per head: logits_j = scale·(q·k_j), weights =
// softmaxWithSinkF32(logits, sinks[head % nHeads]), out = Σ_j weights_j·v_j. sinks is bf16 [nHeads]
// (the per-layer checkpoint tensor). The GPU parity gates compare SDPAWithSinks/SDPA2PassWithSinks
// against this, and the arch host-reference tests compute expected attention through it.
func sdpaHostRefWithSinks(qb, kb, vb, sinks []byte, b, nHeads, nKVHeads, headDim, kvLen int, scale float32) ([]byte, error) {
	if nKVHeads <= 0 || nHeads%nKVHeads != 0 {
		return nil, core.NewError("native.sdpaHostRefWithSinks: nHeads must be a multiple of nKVHeads")
	}
	if len(qb) != b*nHeads*headDim*bf16Size || len(kb) != b*nKVHeads*kvLen*headDim*bf16Size || len(vb) != len(kb) {
		return nil, core.NewError("native.sdpaHostRefWithSinks: q/k/v byte size mismatch")
	}
	if len(sinks) != nHeads*bf16Size {
		return nil, core.NewError("native.sdpaHostRefWithSinks: sinks must be nHeads bf16 values")
	}
	q := bf16ToF32Slice(qb)
	k := bf16ToF32Slice(kb)
	v := bf16ToF32Slice(vb)
	s := bf16ToF32Slice(sinks)
	gqa := nHeads / nKVHeads
	out := make([]float32, b*nHeads*headDim)
	logits := make([]float32, kvLen)
	for bi := 0; bi < b; bi++ {
		for h := 0; h < nHeads; h++ {
			qOff := (bi*nHeads + h) * headDim
			kvHead := h / gqa
			kvOff := (bi*nKVHeads + kvHead) * kvLen * headDim
			for j := 0; j < kvLen; j++ {
				var dot float64
				for d := 0; d < headDim; d++ {
					dot += float64(q[qOff+d]) * float64(k[kvOff+j*headDim+d])
				}
				logits[j] = float32(float64(scale) * dot)
			}
			w, err := softmaxWithSinkF32(logits, s[h])
			if err != nil {
				return nil, err
			}
			for d := 0; d < headDim; d++ {
				var acc float64
				for j := 0; j < kvLen; j++ {
					acc += float64(w[j]) * float64(v[kvOff+j*headDim+d])
				}
				out[qOff+d] = float32(acc)
			}
		}
	}
	return f32ToBf16Slice(out), nil
}

// SDPAWithSinks is SDPA (single-query decode attention over a contiguous bf16 KV cache) through
// the has_sinks(25) sdpa_vector variant: sinks (bf16 [nHeads] learned per-head logits) joins each
// head's softmax denominator as one extra key contributing no value mass. Same byte ABI as SDPA
// plus the sinks vector; nHeads is also the kernel's num_q_heads (buffer 17) so a b>1 dispatch
// indexes the same per-head table. Gated against sdpaHostRefWithSinks in sdpa_sinks_test.go.
func SDPAWithSinks(qb, kb, vb, sinks []byte, b, nHeads, nKVHeads, headDim, kvLen int, scale float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if nKVHeads == 0 || nHeads%nKVHeads != 0 {
		return nil, core.NewError("native.SDPAWithSinks: nHeads must be a multiple of nKVHeads")
	}
	if len(sinks) != nHeads*bf16Size {
		return nil, core.NewError("native.SDPAWithSinks: sinks must be nHeads bf16 values")
	}
	pso, err := sdpaVectorSinksPipelineForHeadDim(headDim)
	if err != nil {
		return nil, err
	}

	outLen := b * nHeads * headDim * bf16Size
	out := make([]byte, outLen)
	var encErr error
	withAutoreleasePool(func() {
		scratch, err := getSDPABF16Scratch(len(qb), len(kb), len(vb), outLen)
		if err != nil {
			encErr = err
			return
		}
		defer putSDPABF16Scratch(scratch)
		qBuf, kBuf, vBuf, outBuf, err := scratch.buffers(qb, kb, vb)
		if err != nil {
			encErr = err
			return
		}
		sinksBuf := residentBytes(sinks)
		if sinksBuf == nil {
			encErr = core.NewError("native.SDPAWithSinks: sinks buffer upload failed")
			return
		}

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		emitSDPAAtSinks(encSink{enc}, pso, qBuf, 0, kBuf, vBuf, outBuf, 0, 0, nil, b*nHeads, b*nKVHeads, kvLen, int64(kvLen*headDim), int64(headDim), int64(kvLen*headDim), int64(headDim), scale, sinksBuf, 0, nHeads)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)

		copy(out, scratch.out.bytes[:outLen])
	})
	if encErr != nil {
		return nil, encErr
	}
	return out, nil
}

// SDPA2PassWithSinks is SDPA2Pass through the has_sinks pass-1 variant (sinks at buffer 18, seeded
// only in block 0 — counted exactly once in the merged softmax); pass 2 is the unchanged
// sink-agnostic merge. Token-identical in intent to SDPAWithSinks past the 2-pass knee; gated
// against the same host oracle.
func SDPA2PassWithSinks(qb, kb, vb, sinks []byte, b, nHeads, nKVHeads, headDim, kvLen int, scale float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if nKVHeads == 0 || nHeads%nKVHeads != 0 {
		return nil, core.NewError("native.SDPA2PassWithSinks: nHeads must be a multiple of nKVHeads")
	}
	if len(sinks) != nHeads*bf16Size {
		return nil, core.NewError("native.SDPA2PassWithSinks: sinks must be nHeads bf16 values")
	}
	blocks := sdpa2PassBlocks(kvLen, nKVHeads)
	pso1, err := sdpaVector2Pass1SinksPipelineForHeadDim(headDim, blocks)
	if err != nil {
		return nil, err
	}
	pso2, err := sdpaVector2Pass2PipelineForHeadDim(headDim)
	if err != nil {
		return nil, err
	}

	outLen := b * nHeads * headDim * bf16Size
	out := make([]byte, outLen)
	var encErr error
	withAutoreleasePool(func() {
		scratch, err := getSDPABF16Scratch(len(qb), len(kb), len(vb), outLen)
		if err != nil {
			encErr = err
			return
		}
		defer putSDPABF16Scratch(scratch)
		qBuf, kBuf, vBuf, outBuf, err := scratch.buffers(qb, kb, vb)
		if err != nil {
			encErr = err
			return
		}
		sinksBuf := residentBytes(sinks)
		if sinksBuf == nil {
			encErr = core.NewError("native.SDPA2PassWithSinks: sinks buffer upload failed")
			return
		}
		nbh := b * nHeads
		partials, sums, maxs, err := scratch.twoPassBuffers(nbh, blocks, headDim)
		if err != nil {
			encErr = err
			return
		}

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		sink := encSink{enc}
		emitSDPA2Pass1NAtSinks(sink, pso1, qBuf, 0, kBuf, vBuf, partials, sums, maxs, 0, nil, b, nHeads, nKVHeads, kvLen, int(blocks), int64(kvLen*headDim), int64(headDim), int64(kvLen*headDim), int64(headDim), scale, sinksBuf, 0)
		emitSDPA2Pass2(sink, pso2, partials, sums, maxs, outBuf, b, nHeads, int(blocks))
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)

		copy(out, scratch.out.bytes[:outLen])
	})
	if encErr != nil {
		return nil, encErr
	}
	return out, nil
}
