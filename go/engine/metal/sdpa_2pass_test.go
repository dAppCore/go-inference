// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"math"
	"testing"
	"unsafe"

	"github.com/tmc/apple/metal"
)

// TestSDPA2PassMatchesReference validates the two-pass long-context SDPA path
// (sdpa_vector_2pass_1 → sdpa_vector_2pass_2) against a host float reference at a
// kvLen well past the single-pass degradation knee (2048). Pass 1 splits the cache
// across `blocks` threadgroups emitting per-block online-softmax partials; pass 2
// merges them. A pass proves the split-and-merge is token-identical to a straight
// softmax — the long-context KV lever ("improving the KV improves toks, more so as
// context grows"). It also cross-checks 2-pass against the proven single-pass SDPA
// at the same inputs: the two MLX kernels must agree.
func TestSDPA2PassMatchesReference(t *testing.T) {
	requireNativeRuntime(t)

	const b, nHeads, nKV, headDim, kvLen = 1, 4, 2, 64, 2048
	gqa := nHeads / nKV
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	qb := toBF16Bytes(syntheticFloat32(b*nHeads*headDim, 3))
	kb := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 5))
	vb := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 7))

	// --- host float reference: straight online softmax over bf16-rounded inputs ---
	rb := func(s []byte, i int) float32 { return bf16ToF32(s[i*2], s[i*2+1]) }
	ref := make([]byte, b*nHeads*headDim*2)
	for h := 0; h < nHeads; h++ {
		kvh := h / gqa
		m := float32(-3e38)
		for j := 0; j < kvLen; j++ {
			var dot float32
			for d := 0; d < headDim; d++ {
				dot += rb(qb, h*headDim+d) * rb(kb, (kvh*kvLen+j)*headDim+d)
			}
			if dot*scale > m {
				m = dot * scale
			}
		}
		var denom float32
		acc := make([]float32, headDim)
		for j := 0; j < kvLen; j++ {
			var dot float32
			for d := 0; d < headDim; d++ {
				dot += rb(qb, h*headDim+d) * rb(kb, (kvh*kvLen+j)*headDim+d)
			}
			p := float32(math.Exp(float64(dot*scale - m)))
			denom += p
			for d := 0; d < headDim; d++ {
				acc[d] += p * rb(vb, (kvh*kvLen+j)*headDim+d)
			}
		}
		for d := 0; d < headDim; d++ {
			o := f32ToBF16(acc[d] / denom)
			ref[(h*headDim+d)*2], ref[(h*headDim+d)*2+1] = byte(o), byte(o>>8)
		}
	}

	got, err := SDPA2Pass(qb, kb, vb, b, nHeads, nKV, headDim, kvLen, scale)
	if err != nil {
		t.Fatalf("SDPA2Pass: %v", err)
	}
	if cos := cosineBF16(got, ref); cos < 0.999 {
		t.Fatalf("2-pass SDPA cosine=%.6f vs host reference — block split/merge broken", cos)
	} else {
		t.Logf("2-pass SDPA (kvLen=%d, blocks=%d): cosine=%.6f vs host reference — the cache reduction fans over %d threadgroups, token-identical", kvLen, sdpa2PassBlocks(kvLen), cos, sdpa2PassBlocks(kvLen))
	}

	// cross-check against the proven single-pass kernel at the same inputs.
	sp, err := SDPA(qb, kb, vb, b, nHeads, nKV, headDim, kvLen, scale)
	if err != nil {
		t.Fatalf("SDPA (single-pass cross-check): %v", err)
	}
	if cos := cosineBF16(got, sp); cos < 0.999 {
		t.Fatalf("2-pass vs single-pass SDPA cosine=%.6f — the two MLX kernels disagree", cos)
	} else {
		t.Logf("2-pass vs single-pass SDPA: cosine=%.6f — agree", cos)
	}
}

func TestSDPA2PassAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const batch, nHeads, nKV, headDim, kvLen = 1, 4, 2, 64, 2048
	q := toBF16Bytes(syntheticFloat32(batch*nHeads*headDim, 3))
	k := toBF16Bytes(syntheticFloat32(batch*nKV*kvLen*headDim, 5))
	v := toBF16Bytes(syntheticFloat32(batch*nKV*kvLen*headDim, 7))
	if _, err := SDPA2Pass(q, k, v, batch, nHeads, nKV, headDim, kvLen, 0.125); err != nil {
		t.Fatalf("SDPA2Pass warmup: %v", err)
	}

	var sdpaErr error
	allocs := testing.AllocsPerRun(5, func() {
		_, sdpaErr = SDPA2Pass(q, k, v, batch, nHeads, nKV, headDim, kvLen, 0.125)
	})
	if sdpaErr != nil {
		t.Fatalf("SDPA2Pass: %v", sdpaErr)
	}
	if allocs > 10 {
		t.Fatalf("SDPA2Pass allocations = %.0f, want <= 10", allocs)
	}
}

func TestSDPA2PassIntoUsesCallerBacking(t *testing.T) {
	requireNativeRuntime(t)

	const batch, nHeads, nKV, headDim, kvLen = 1, 4, 2, 64, 2048
	q := toBF16Bytes(syntheticFloat32(batch*nHeads*headDim, 3))
	k := toBF16Bytes(syntheticFloat32(batch*nKV*kvLen*headDim, 5))
	v := toBF16Bytes(syntheticFloat32(batch*nKV*kvLen*headDim, 7))
	out := make([]byte, batch*nHeads*headDim*bf16Size)
	for i := range out {
		out[i] = 0xA5
	}

	got, err := SDPA2PassInto(out, q, k, v, batch, nHeads, nKV, headDim, kvLen, 0.125)
	if err != nil {
		t.Fatalf("SDPA2PassInto: %v", err)
	}
	if len(got) != len(out) {
		t.Fatalf("SDPA2PassInto len = %d, want %d", len(got), len(out))
	}
	if unsafe.Pointer(&got[0]) != unsafe.Pointer(&out[0]) {
		t.Fatal("SDPA2PassInto did not return caller-owned output backing")
	}
	want, err := SDPA2Pass(q, k, v, batch, nHeads, nKV, headDim, kvLen, 0.125)
	if err != nil {
		t.Fatalf("SDPA2Pass reference: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("SDPA2PassInto output differs from allocating wrapper")
	}
}

// TestEncSDPA2PassSeqMajorMatchesSinglePass validates the LIVE decode wiring: the
// encoder-level encSDPA2PassStrided against encSDPAStrided with the exact SEQ-MAJOR
// cache layout the decode path passes ([seq, nKVHeads, headDim] ⇒ kHeadStride=headDim,
// kSeqStride=kvDim) at a window past the single-pass knee. The standalone SDPA2Pass
// gate used head-major strides; this proves the encoder binding + seq-major strides +
// once-allocated intermediates (the encSDPADecode hot path) are token-identical to the
// proven single-pass kernel — the only untested seam in the live-path routing.
func TestEncSDPA2PassSeqMajorMatchesSinglePass(t *testing.T) {
	requireNativeRuntime(t)

	const nHeads, nKV, headDim, n = 8, 1, 128, 2048 // MQA global-layer shape (gemma4 big models)
	qDim, kvDim := nHeads*headDim, nKV*headDim
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	qb := toBF16Bytes(syntheticFloat32(qDim, 3))
	kb := toBF16Bytes(syntheticFloat32(n*kvDim, 5)) // seq-major [n, nKV, headDim]
	vb := toBF16Bytes(syntheticFloat32(n*kvDim, 7))

	out1 := make([]byte, qDim*2)
	out2 := make([]byte, qDim*2)
	withAutoreleasePool(func() {
		qBuf, kBuf, vBuf := sharedBytes(qb), sharedBytes(kb), sharedBytes(vb)
		o1 := device.NewBufferWithLengthOptions(uint(qDim*2), metal.MTLResourceStorageModeShared)
		o2 := device.NewBufferWithLengthOptions(uint(qDim*2), metal.MTLResourceStorageModeShared)
		blocks := int(sdpa2PassBlocks(n))
		partials := scratchBF16(blocks * qDim)
		sums, maxs := scratchF32(blocks*nHeads), scratchF32(blocks*nHeads)
		khs, kss := int64(headDim), int64(kvDim) // SEQ-MAJOR strides (the live-path convention)

		cb1 := queue.CommandBuffer()
		enc1 := cb1.ComputeCommandEncoder()
		if err := encSDPAStrided(enc1, qBuf, kBuf, vBuf, o1, nHeads, nKV, headDim, n, khs, kss, khs, kss, scale, 0); err != nil {
			t.Fatalf("encSDPAStrided: %v", err)
		}
		enc1.EndEncoding()
		cb1.Commit()
		cb1.WaitUntilCompleted()

		cb2 := queue.CommandBuffer()
		enc2 := cb2.ComputeCommandEncoder()
		if err := encSDPA2PassStrided(enc2, qBuf, kBuf, vBuf, o2, partials, sums, maxs, nHeads, nKV, headDim, n, khs, kss, khs, kss, scale, 0); err != nil {
			t.Fatalf("encSDPA2PassStrided: %v", err)
		}
		enc2.EndEncoding()
		cb2.Commit()
		cb2.WaitUntilCompleted()

		copy(out1, unsafe.Slice((*byte)(o1.Contents()), qDim*2))
		copy(out2, unsafe.Slice((*byte)(o2.Contents()), qDim*2))
	})

	if cos := cosineBF16(out2, out1); cos < 0.999 {
		t.Fatalf("encoder 2-pass vs single-pass (seq-major, n=%d) cosine=%.6f — live-path wiring broken", n, cos)
	} else {
		t.Logf("encoder 2-pass vs single-pass (seq-major MQA, nHeads=%d, headDim=%d, n=%d, blocks=%d): cosine=%.6f — live-path routing token-identical", nHeads, headDim, n, sdpa2PassBlocks(n), cos)
	}
}
