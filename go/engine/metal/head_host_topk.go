// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// Host-side top-k candidate selection for the sampled decode. Measured on the
// real e2b head (262k vocab, M3 Ultra, real logits): every GPU top-k SELECTION
// kernel is value-dependent and slow — fused Q4 tiles 22.6ms, qmv+topk tiles
// 17.7ms, logits-sample@k 33ms — while the argmax tiles run 0.58ms and the
// full-logits readback costs 0.56ms. A one-pass K-array select over the host
// logits costs well under 1ms, so for TopK>1 at real vocab sizes the fastest
// sampler is: GPU logits matvec -> host select -> model.SampleCandidates.
// (Synthetic-hidden microbenches flattered the GPU kernels ~7-10x; only real
// logits show the selection churn.)

// hostTopKSampleVocabMin is the vocab size above which the host top-k lane is
// preferred over the GPU selection kernels. Fixture-sized vocabs stay on the
// GPU lanes (their chained-sampled parity tests exercise those paths, and at
// tiny vocab the kernels are fine); every real gemma4 vocab (262k) crosses it.
const hostTopKSampleVocabMin = 4096

// hostTopKSamplePreferred reports whether TopK sampling for this vocab should
// select candidates on the host instead of the GPU selection kernels.
// TopK==1 stays greedy-shaped and TopK==0 stays on the (fast) full-vocab
// logits-sample lane; only the 1 < K <= headSampleTopKMaxK window pays the
// slow selection kernels.
func hostTopKSamplePreferred(params model.SampleParams, vocab int) bool {
	return params.TopK > 1 && params.TopK <= headSampleTopKMaxK && vocab >= hostTopKSampleVocabMin
}

// hostTopKCandidatesBF16 selects the k largest logits in one pass, skipping
// suppressed ids, and returns them as raw bf16 bytes + aligned ids for
// model.SampleCandidates. The candidate values are the ORIGINAL bf16 bit
// patterns (no convert round-trip). Selection keeps a k-slot array with a
// tracked minimum: the common case is one convert + one compare per vocab
// entry; replacements (~k·ln(vocab/k)) rescan the k slots. Order of the
// returned candidates is unspecified — SampleCandidates ranks internally.
// vals/ids are caller scratch reused across tokens; k is capped by
// headSampleTopKMaxK so [headSampleTopKMaxK] stack arrays fit every call.
func hostTopKCandidatesBF16(logits []byte, vocab, k int, suppress []int32, vals []byte, ids []int32) ([]byte, []int32) {
	if k > vocab {
		k = vocab
	}
	if k <= 0 || len(logits) < vocab*bf16Size {
		return vals[:0], ids[:0]
	}
	var slotF [headSampleTopKMaxK]float32
	n := 0
	minVal := float32(0)
	minIdx := 0
	for i := range vocab {
		v := bf16ToF32(logits[i*bf16Size], logits[i*bf16Size+1])
		if n == k && v <= minVal {
			continue
		}
		if tokenSuppressed(i, suppress) {
			continue
		}
		if n < k {
			slotF[n] = v
			ids[n] = int32(i)
			n++
			if n == k { // full: establish the tracked minimum
				minVal, minIdx = slotF[0], 0
				for j := 1; j < k; j++ {
					if slotF[j] < minVal {
						minVal, minIdx = slotF[j], j
					}
				}
			}
			continue
		}
		slotF[minIdx] = v
		ids[minIdx] = int32(i)
		minVal, minIdx = slotF[0], 0
		for j := 1; j < k; j++ {
			if slotF[j] < minVal {
				minVal, minIdx = slotF[j], j
			}
		}
	}
	for j := 0; j < n; j++ {
		src := int(ids[j]) * bf16Size
		vals[j*bf16Size] = logits[src]
		vals[j*bf16Size+1] = logits[src+1]
	}
	return vals[:n*bf16Size], ids[:n]
}

// sampleHostTopKBF16 is the host TopK>1 sampler: one-pass candidate selection
// then the shared candidate sampler (temperature, top-p, min-p all apply
// within the selected set — mathematically identical to full-vocab softmax
// restricted to its top-k, since softmax ratios survive subsetting).
func sampleHostTopKBF16(logits []byte, vocab int, sampler *model.Sampler, params model.SampleParams) (int32, error) {
	var valScratch [headSampleTopKMaxK * bf16Size]byte
	var idScratch [headSampleTopKMaxK]int32
	vals, ids := hostTopKCandidatesBF16(logits, vocab, params.TopK, params.SuppressTokens, valScratch[:], idScratch[:])
	if len(ids) == 0 {
		return 0, core.NewError("native.sampleHostTopKBF16: no unsuppressed candidates")
	}
	return sampler.SampleCandidates(vals, ids, params)
}
