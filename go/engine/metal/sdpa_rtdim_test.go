// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"
	"unsafe"

	"dappco.re/go/inference/model"
	"github.com/tmc/apple/metal"
)

// sdpa_rtdim_test.go gates kernels/lthn_sdpa_rtdim.metal + sdpa_rtdim.go — the runtime-head-dim
// decode SDPA fallback (#28). Three layers: the kernel itself against the host float reference and
// against the FIXED kernel it stands in for (byte-band idiom, matching sdpa_sinks_test.go /
// sdpa_vector_q8_test.go), the dispatch wrapper's decisions, and the end-to-end #28 repro — a real
// ArchSession forward at head_dim 32, the shape train_trainer_layers_test.go's eligibleFixtureModel
// and train_trainer_mask_test.go's maskTrainerFixture moved to head_dim 64 to dodge.

func readBF16Buffer(nElems int, buf metal.MTLBuffer) []byte {
	return append([]byte(nil), unsafe.Slice((*byte)(buf.Contents()), nElems*bf16Size)...)
}

// TestSdpaVectorRTDimValidHeadDim_Good pins the accepted domain: positive multiples of the
// simdgroup width, up to and including the register-array cap.
func TestSdpaVectorRTDimValidHeadDim_Good(t *testing.T) {
	for _, hd := range []int{32, 64, 96, 128, 160, 224, 256} {
		if !sdpaVectorRTDimValidHeadDim(hd) {
			t.Fatalf("head_dim %d should be valid for the runtime-dim fallback (positive multiple of %d, <= %d)", hd, sdpaRTDimBD, sdpaVectorRTDimMaxHeadDim)
		}
	}
}

// TestSdpaVectorRTDimValidHeadDim_Bad pins the refused domain: non-positive, not a multiple of the
// simdgroup width, or past the register-array cap — never silently accepted.
func TestSdpaVectorRTDimValidHeadDim_Bad(t *testing.T) {
	for _, hd := range []int{0, -32, 33, 48, 257, 288, 512} {
		if sdpaVectorRTDimValidHeadDim(hd) {
			t.Fatalf("head_dim %d should be REJECTED by the runtime-dim fallback guard", hd)
		}
	}
}

// TestSdpaRTDim_SDPA32_Good gates the runtime-dim fallback at head_dim=32 (the #28 shape) against
// the independent host float reference (sdpaBF16Reference, sdpa_test.go) across several sequence
// lengths spanning the kernel's BN=32 sweep boundary: 1 (trivial softmax collapse), 17 (< one
// sweep), 32 (exactly one sweep), 65 (just past two sweeps), 257 (many sweeps + remainder) —
// ">threadgroup-size" for this kernel family means past one 32-wide simdgroup sweep, not the
// 1024-thread dispatch width (which is fixed regardless of kvLen).
func TestSdpaRTDim_SDPA32_Good(t *testing.T) {
	requireNativeRuntime(t)

	const b, nHeads, nKV, headDim = 1, 2, 1, 32
	scale := float32(0.125)
	for _, kvLen := range []int{1, 17, 32, 65, 257} {
		q := toBF16Bytes(syntheticFloat32(b*nHeads*headDim, 3))
		k := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 5))
		v := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 7))

		got, err := SDPA(q, k, v, b, nHeads, nKV, headDim, kvLen, scale)
		if err != nil {
			t.Fatalf("kvLen=%d: SDPA at head_dim=32 (no fixed pipeline — must route to the runtime-dim fallback, #28): %v", kvLen, err)
		}
		want := sdpaBF16Reference(q, k, v, nHeads, nKV, headDim, kvLen, scale)
		cos := cosineBF16(got, want)
		if cos < 0.999 {
			t.Fatalf("kvLen=%d: head_dim=32 rtdim SDPA cosine=%.6f vs host reference, want ~1", kvLen, cos)
		}
		gotF, wantF := bf16Floats(got), bf16Floats(want)
		var worst float64
		for i := range wantF {
			d := math.Abs(float64(gotF[i] - wantF[i]))
			if d > worst {
				worst = d
			}
			if d > 0.02 {
				t.Fatalf("kvLen=%d: dim %d: rtdim %v vs host reference %v (|d|=%g)", kvLen, i, gotF[i], wantF[i], d)
			}
		}
		t.Logf("kvLen=%d: head_dim=32 rtdim vs host reference: cosine=%.6f worst|d|=%.5g", kvLen, cos, worst)
	}
}

// TestSdpaRTDim_SDPA32_Bad proves an absent width that is ALSO ineligible for the fallback (not a
// multiple of the simdgroup width) refuses loudly rather than silently corrupting output.
func TestSdpaRTDim_SDPA32_Bad(t *testing.T) {
	requireNativeRuntime(t)

	x := toBF16Bytes(syntheticFloat32(33, 3))
	if _, err := SDPA(x, x, x, 1, 1, 1, 33, 1, 1); err == nil {
		t.Fatal("expected SDPA at an absent, non-fallback-eligible head_dim (33) to refuse, not silently proceed")
	}
}

// TestSdpaRTDim_SDPA32_Ugly proves real GQA attention correctness at head_dim=32 (nHeads>nKVHeads,
// kvLen crossing the BN=32 sweep boundary) against the independent host reference — the same
// shape TestSdpa_SDPA_Ugly proves for the fixed hd=64 kernel.
func TestSdpaRTDim_SDPA32_Ugly(t *testing.T) {
	requireNativeRuntime(t)

	const b, nHeads, nKV, headDim, kvLen = 1, 8, 2, 32, 48
	scale := float32(0.125)
	q := toBF16Bytes(syntheticFloat32(b*nHeads*headDim, 3))
	k := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 5))
	v := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 7))

	got, err := SDPA(q, k, v, b, nHeads, nKV, headDim, kvLen, scale)
	if err != nil {
		t.Fatalf("SDPA (GQA, head_dim=32): %v", err)
	}
	want := sdpaBF16Reference(q, k, v, nHeads, nKV, headDim, kvLen, scale)
	if cos := cosineBF16(got, want); cos < 0.999 {
		t.Fatalf("GQA head_dim=32 rtdim SDPA cosine=%.6f vs host reference, want ~1", cos)
	}
}

// TestSdpaRTDim_MatchesFixedAtHeadDim64_Good is the byte-band idiom (sdpa_sinks_test.go /
// sdpa_vector_q8_test.go's TestSDPAVectorQ8Parity) applied to the runtime-dim kernel: at
// head_dim=64 — a width the fixed metallib DOES instantiate — the runtime-dim kernel is invoked
// DIRECTLY (bypassing sdpaVectorDispatchForHeadDim's fixed-first routing, which would otherwise
// always pick the fixed pipeline at this width) and compared against SDPA's proven fixed-kernel
// output over IDENTICAL inputs. Same sequence-length spread as TestSdpaRTDim_SDPA32_Good.
func TestSdpaRTDim_MatchesFixedAtHeadDim64_Good(t *testing.T) {
	requireNativeRuntime(t)

	const nHeads, nKV, headDim = 4, 2, 64
	scale := float32(0.125)
	for _, kvLen := range []int{1, 17, 32, 65, 257} {
		q := toBF16Bytes(syntheticFloat32(nHeads*headDim, 3))
		k := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 5))
		v := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 7))

		want, err := SDPA(q, k, v, 1, nHeads, nKV, headDim, kvLen, scale) // the FIXED pipeline (head_dim=64 is instantiated)
		if err != nil {
			t.Fatalf("kvLen=%d: fixed SDPA: %v", kvLen, err)
		}

		var got []byte
		var encErr error
		withAutoreleasePool(func() {
			pso, perr := sdpaVectorRTDimPipeline() // FORCE the fallback kernel, bypassing the fixed-pipeline dispatch
			if perr != nil {
				encErr = perr
				return
			}
			qBuf, kBuf, vBuf := sharedBytes(q), sharedBytes(k), sharedBytes(v)
			outBuf := scratchBF16(nHeads * headDim)
			cb := commandBufferFast(queue)
			enc := computeCommandEncoderFast(cb)
			emitSDPARTDim(encSink{enc}, pso, qBuf, kBuf, vBuf, outBuf, 0, nil, nHeads, nKV, headDim, kvLen, int64(kvLen*headDim), int64(headDim), int64(kvLen*headDim), int64(headDim), scale)
			endEncodingFast(enc)
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
			got = readBF16Buffer(nHeads*headDim, outBuf)
		})
		if encErr != nil {
			t.Fatalf("kvLen=%d: rtdim SDPA: %v", kvLen, encErr)
		}
		cos := cosineBF16(got, want)
		if cos < 0.999 {
			t.Fatalf("kvLen=%d: rtdim kernel vs FIXED head_dim=64 kernel cosine=%.6f, want ~1", kvLen, cos)
		}
		// tight per-element band — both kernels run the IDENTICAL bf16 inputs through the SAME
		// algorithm, only D is runtime vs compile-time, so they should track closely.
		gotF, wantF := bf16Floats(got), bf16Floats(want)
		var worst float64
		for i := range wantF {
			d := math.Abs(float64(gotF[i] - wantF[i]))
			if d > worst {
				worst = d
			}
			if d > 0.02 {
				t.Fatalf("kvLen=%d: dim %d: rtdim %v vs fixed %v (|d|=%g)", kvLen, i, gotF[i], wantF[i], d)
			}
		}
		t.Logf("kvLen=%d: head_dim=64 rtdim vs FIXED kernel: cosine=%.6f worst|d|=%.5g", kvLen, cos, worst)
	}
}

// TestSdpaRTDim_EmitSDPA2Pass1RTDimAt_Good gates the (not-yet-wired, see sdpa_rtdim.go) pass-1
// runtime-dim kernel in isolation at head_dim=64, past the 2-pass knee: forces both the fixed
// pass-1 kernel and the runtime-dim pass-1 kernel over IDENTICAL inputs and compares their raw
// partials/sums/maxs intermediates — the exact bookkeeping pass 2 would consume — proving the
// runtime-dim port is correct even though nothing wires it into the merge yet.
func TestSdpaRTDim_EmitSDPA2Pass1RTDimAt_Good(t *testing.T) {
	requireNativeRuntime(t)

	const nHeads, nKV, headDim, kvLen = 4, 2, 64, 2000
	scale := float32(0.125)
	blocks := int(sdpa2PassBlocks(kvLen, nKV))
	q := toBF16Bytes(syntheticFloat32(nHeads*headDim, 3))
	k := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 5))
	v := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 7))
	khs, kss := int64(kvLen*headDim), int64(headDim)

	var fixedPartials, rtPartials []byte
	var fixedSums, fixedMaxs, rtSums, rtMaxs []float32
	var encErr error
	withAutoreleasePool(func() {
		qBuf, kBuf, vBuf := sharedBytes(q), sharedBytes(k), sharedBytes(v)

		fixedPSO, err := sdpaVector2Pass1PipelineForHeadDim(headDim, int32(blocks))
		if err != nil {
			encErr = err
			return
		}
		rtPSO, err := sdpaVector2Pass1RTDimPipeline()
		if err != nil {
			encErr = err
			return
		}

		fPart, fSums, fMaxs := scratchBF16(nHeads*blocks*headDim), scratchF32(nHeads*blocks), scratchF32(nHeads*blocks)
		rPart, rSums, rMaxs := scratchBF16(nHeads*blocks*headDim), scratchF32(nHeads*blocks), scratchF32(nHeads*blocks)

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		sink := encSink{enc}
		emitSDPA2Pass1At(sink, fixedPSO, qBuf, 0, kBuf, vBuf, fPart, fSums, fMaxs, 0, 1, nHeads, nKV, kvLen, blocks, khs, kss, khs, kss, scale)
		if err := emitSDPA2Pass1RTDimAt(sink, rtPSO, qBuf, 0, kBuf, vBuf, rPart, rSums, rMaxs, 0, nil, 1, nHeads, nKV, headDim, kvLen, blocks, khs, kss, khs, kss, scale); err != nil {
			encErr = err
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)

		fixedPartials = readBF16Buffer(nHeads*blocks*headDim, fPart)
		rtPartials = readBF16Buffer(nHeads*blocks*headDim, rPart)
		fixedSums = append([]float32(nil), unsafe.Slice((*float32)(fSums.Contents()), nHeads*blocks)...)
		fixedMaxs = append([]float32(nil), unsafe.Slice((*float32)(fMaxs.Contents()), nHeads*blocks)...)
		rtSums = append([]float32(nil), unsafe.Slice((*float32)(rSums.Contents()), nHeads*blocks)...)
		rtMaxs = append([]float32(nil), unsafe.Slice((*float32)(rMaxs.Contents()), nHeads*blocks)...)
	})
	if encErr != nil {
		t.Fatalf("pass-1 fixed-vs-rtdim dispatch: %v", encErr)
	}
	if cos := cosineBF16(rtPartials, fixedPartials); cos < 0.999 {
		t.Fatalf("pass-1 partials: rtdim vs fixed cosine=%.6f, want ~1", cos)
	}
	for i := range fixedSums {
		if d := math.Abs(float64(rtSums[i] - fixedSums[i])); d > 0.02 {
			t.Fatalf("pass-1 sums[%d]: rtdim %v vs fixed %v (|d|=%g)", i, rtSums[i], fixedSums[i], d)
		}
		if d := math.Abs(float64(rtMaxs[i] - fixedMaxs[i])); d > 0.02 {
			t.Fatalf("pass-1 maxs[%d]: rtdim %v vs fixed %v (|d|=%g)", i, rtMaxs[i], fixedMaxs[i], d)
		}
	}
}

// TestSdpaRTDim_EmitSDPA2Pass1RTDimAt_Bad proves the batch=1 constraint refuses rather than
// silently mis-addressing a batch>1 dispatch (kernels/lthn_sdpa_rtdim.metal's 2-pass kernel has no
// batch grid axis — see the doc comment above emitSDPA2Pass1RTDimAt).
func TestSdpaRTDim_EmitSDPA2Pass1RTDimAt_Bad(t *testing.T) {
	rec := &recordingDispatchSink{}
	err := emitSDPA2Pass1RTDimAt(rec, nil, nil, 0, nil, nil, nil, nil, nil, 0, nil, 2, 4, 2, 64, 2000, 64, 0, 0, 0, 0, 0)
	if err == nil {
		t.Fatal("expected emitSDPA2Pass1RTDimAt to refuse a batch != 1 dispatch")
	}
}

// TestEncSDPADecodeAt_HeadDim32PastTheKnee_Good proves the #28 fix has no cliff at the 2-pass
// knee: decode SDPA at an absent head_dim (32) still succeeds once the attended window crosses
// sdpa2PassMinKV, where the FIXED 2-pass kernel pair is unavailable (pass 2 has no runtime-dim
// port — sdpa_rtdim.go) — encSDPADecodeAt falls all the way through to the single-pass runtime-dim
// fallback instead of erroring. Mirrors TestEncSDPADecodeAt_Ugly's proof shape (hd=64, fixed pair)
// at hd=32.
func TestEncSDPADecodeAt_HeadDim32PastTheKnee_Good(t *testing.T) {
	requireNativeRuntime(t)

	const headDim = 32
	const n = sdpa2PassMinKV
	rowBytes := headDim * bf16Size
	q := sharedBytes(make([]byte, rowBytes))
	k := sharedBytes(make([]byte, n*rowBytes))
	want := toBF16Bytes(syntheticFloat32(headDim, 29))
	vBytes := make([]byte, n*rowBytes)
	for row := range n {
		copy(vBytes[row*rowBytes:(row+1)*rowBytes], want)
	}
	v := sharedBytes(vBytes)
	out := scratchBF16(headDim)
	sc := newAttnScratch(headDim, headDim, headDim, 1, n)
	if sc.p2Partials == nil || sc.p2Sums == nil || sc.p2Maxs == nil {
		t.Fatal("newAttnScratch did not provision two-pass intermediates at the SDPA knee")
	}
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	if err := encSDPADecodeAt(enc, sc, q, 0, k, v, out, 0, 1, 1, headDim, n, headDim, headDim, headDim, headDim, 1, 0); err != nil {
		endEncodingFast(enc)
		t.Fatalf("encSDPADecodeAt at head_dim=32 past the 2-pass knee (n=%d): %v — the fallback must never cliff past the knee (#28)", n, err)
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	// Every value row is identical, so the softmax weighting and its reduction order cannot
	// change the expected output.
	eqBytes(t, "encSDPADecodeAt head_dim=32 past the knee output", unsafe.Slice((*byte)(out.Contents()), rowBytes), want)
}

// hd32FixtureModel is eligibleFixtureModel (train_trainer_layers_test.go) with the workaround
// undone: head_dim 32 was the ORIGINAL shape those fixtures used before #28 forced a move to 64
// ("hd32 has no sdpa_vector pipeline in the metallib" — both fixtures' own comments). Same
// geometry otherwise (dModel=headDim, nHeads=1, dFF=2·dModel).
func hd32FixtureModel(nL int) (*BF16Model, model.Arch) {
	const dModel, nHeads, nKV, headDim, dFF, vocab = 32, 1, 1, 32, 64, 32
	layers := make([]DecodeLayerWeights, nL)
	types := make([]string, nL)
	for li := range layers {
		layers[li] = stableLayerWeights(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
		types[li] = "full_attention"
	}
	specs := model.DeriveLayers(types, 0)
	for i := range specs {
		specs[i].HeadDim, specs[i].KVHeads = headDim, nKV
	}
	embed := toBF16Bytes(scaleSlice(syntheticFloat32(vocab*dModel, 21), 0.1))
	g := &BF16Model{Layers: layers, Embed: embed, FinalNorm: toBF16Bytes(syntheticFloat32(dModel, 22)), LMHead: embed, Tied: true}
	arch := model.Arch{
		Hidden: dModel, Heads: nHeads, KVHeads: nKV, HeadDim: headDim, FF: dFF, Vocab: vocab,
		GlobalHeadDim: headDim, GlobalKVHeads: nKV,
		Eps: 1e-5, AttnScale: 0.176776695, RopeBase: 10000, RopeScale: 1, RopeLocalBase: 10000,
		RotaryDim: headDim, RotaryDimLocal: headDim, Layer: specs,
	}
	return g, arch
}

// TestSdpaRTDim_TrainerSessionAtHeadDim32_Good is the #28 repro: a synthetic session at head_dim
// 32 — the exact shape train_trainer_layers_test.go's eligibleFixtureModel and
// train_trainer_mask_test.go's maskTrainerFixture moved to head_dim 64 to dodge — now runs the
// decode SDPA without error. ForwardCaptureHiddens drives the forward ONE TOKEN AT A TIME via
// StepWithID — the same per-token decode chokepoint (encSDPADecodeAt) every real generate() call
// and the trainer's own B=0 parity anchor (TestNewLoRATrainerPerLayer_Good) uses — so this proves
// the fix through the real forward path, not just the raw SDPA entry point.
//
// LTHN_NATIVE_TRACE=1 is set deliberately: it flips archDecodeState.trace, which makes
// (*ArchSession).icbEligible() false, so the session takes the per-op re-encode (stepToken) path
// this fix covers rather than attempting to RECORD the arch ICB. A DISCOVERED, SEPARATE, OUT OF
// SCOPE boundary sits behind the ICB recorder: icb.go's sdpaVectorPipelineICB /
// decode_forward_arch_icb.go's per-layer PSO build (sdpaPSOByHd) resolve the SAME fixed-only
// 64/96/128/256-class kernel names with no runtime-dim fallback, so OpenSession() at head_dim=32
// still fails without LTHN_NATIVE_TRACE — confirmed empirically (both with and without
// LTHN_DECODE_ICB=0, which only gates ICB USE, not the unconditional build-if-eligible at session
// construction). Wiring the ICB recorder is materially larger (record-once/replay semantics
// spread across 4+ call sites in icb.go/icb_layer.go/decode_forward_icb.go/
// decode_forward_arch_icb.go) and is named here as the honest follow-up, not attempted blind.
func TestSdpaRTDim_TrainerSessionAtHeadDim32_Good(t *testing.T) {
	requireNativeRuntime(t)
	t.Setenv("LTHN_NATIVE_TRACE", "1")

	g, arch := hd32FixtureModel(2)
	tm, err := NewBF16TokenModel(g, arch, 16)
	if err != nil {
		t.Fatalf("NewBF16TokenModel (head_dim=32): %v", err)
	}
	sess, err := tm.OpenSession()
	if err != nil {
		t.Fatalf("OpenSession (head_dim=32): %v", err)
	}
	as, ok := sess.(*ArchSession)
	if !ok {
		t.Fatalf("OpenSession returned %T, want *ArchSession", sess)
	}

	ids := []int32{1, 2, 3, 4, 5, 6}
	embeds, perLayer, err := as.ForwardCaptureHiddens(ids)
	if err != nil {
		t.Fatalf("ForwardCaptureHiddens at head_dim=32 (the #28 repro — this shape used to be dodged, not fixed): %v", err)
	}
	if len(embeds) != len(ids) {
		t.Fatalf("embeds length = %d, want %d", len(embeds), len(ids))
	}
	if len(perLayer) != 2 {
		t.Fatalf("perLayer length = %d, want 2 layers", len(perLayer))
	}
	rowBytes := arch.Hidden * bf16Size
	for li, out := range perLayer {
		if len(out) != len(ids)*rowBytes {
			t.Fatalf("layer %d output length = %d, want %d", li, len(out), len(ids)*rowBytes)
		}
	}
}
