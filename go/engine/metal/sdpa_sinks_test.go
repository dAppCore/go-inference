// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"

	"github.com/tmc/apple/metal"
)

// TestSdpaSinks_SoftmaxWithSinkF32_Good byte-gates the fused host formula against the LITERAL
// reference computation (transformers modeling_gpt_oss.py eager_attention_forward, fetched
// 2026-07-19): softmax over the CONCATENATED [logits ; sink] vector, then drop the sink's column.
// The two forms must agree to float64 round-off on every element.
func TestSdpaSinks_SoftmaxWithSinkF32_Good(t *testing.T) {
	logits := []float32{0.5, -1.25, 3.0, 0.0, 2.25}
	const sink = float32(1.5)

	got, err := softmaxWithSinkF32(logits, sink)
	if err != nil {
		t.Fatalf("softmaxWithSinkF32: %v", err)
	}

	// literal reference: softmax([logits ; sink]) then drop the last column.
	cat := append(append([]float64(nil), 0.5, -1.25, 3.0, 0.0, 2.25), float64(sink))
	m := math.Inf(-1)
	for _, l := range cat {
		m = math.Max(m, l)
	}
	var denom float64
	for _, l := range cat {
		denom += math.Exp(l - m)
	}
	var sum float64
	for i, l := range cat[:len(cat)-1] {
		want := math.Exp(l-m) / denom
		if diff := math.Abs(float64(got[i]) - want); diff > 1e-7 {
			t.Fatalf("weight[%d] = %v, literal concat-softmax-drop reference %v (diff %g)", i, got[i], want, diff)
		}
		sum += float64(got[i])
	}
	// the sink's share is exactly the missing mass — the weights must NOT renormalise to 1.
	sinkShare := math.Exp(float64(sink)-m) / denom
	if diff := math.Abs(sum + sinkShare - 1); diff > 1e-6 {
		t.Fatalf("weights sum %v + sink share %v != 1 (diff %g)", sum, sinkShare, diff)
	}
}

// TestSdpaSinks_SoftmaxWithSinkF32_Bad proves the empty-logits guard.
func TestSdpaSinks_SoftmaxWithSinkF32_Bad(t *testing.T) {
	if _, err := softmaxWithSinkF32(nil, 0); err == nil {
		t.Fatal("expected softmaxWithSinkF32 to reject empty logits")
	}
}

// TestSdpaSinks_SoftmaxWithSinkF32_Ugly pins the two boundary behaviours: a hugely dominant sink
// drains (almost) all probability mass from the keys, and a -Inf sink reduces the formula to the
// PLAIN softmax exactly (the sink column contributes exp(-Inf)=0 to the denominator).
func TestSdpaSinks_SoftmaxWithSinkF32_Ugly(t *testing.T) {
	logits := []float32{1, 2, 3}

	dominated, err := softmaxWithSinkF32(logits, 60)
	if err != nil {
		t.Fatalf("softmaxWithSinkF32(dominant sink): %v", err)
	}
	for i, w := range dominated {
		if float64(w) > 1e-20 {
			t.Fatalf("weight[%d] = %v under a +60 sink, want ~0 (the sink drains the mass)", i, w)
		}
	}

	negInf := float32(math.Inf(-1))
	plainWithSink, err := softmaxWithSinkF32(logits, negInf)
	if err != nil {
		t.Fatalf("softmaxWithSinkF32(-Inf sink): %v", err)
	}
	var denom float64
	for _, l := range logits {
		denom += math.Exp(float64(l) - 3)
	}
	for i, l := range logits {
		want := math.Exp(float64(l)-3) / denom
		if diff := math.Abs(float64(plainWithSink[i]) - want); diff > 1e-7 {
			t.Fatalf("-Inf sink weight[%d] = %v, plain softmax %v (diff %g)", i, plainWithSink[i], want, diff)
		}
	}
}

// TestSdpaSinks_SDPAHostRefWithSinks_Good hand-computes a tiny single-head case end to end: one
// head, two keys, headDim 2 — logits, sink softmax and the weighted-V sum are all verifiable on
// paper. bf16 quantisation of the inputs is applied to the expectation the same way.
func TestSdpaSinks_SDPAHostRefWithSinks_Good(t *testing.T) {
	// q = [1, 0]; k0 = [1, 0], k1 = [0, 1]; v0 = [1, 2], v1 = [3, 4]; sink = 0; scale = 1.
	q := f32ToBf16Slice([]float32{1, 0})
	k := f32ToBf16Slice([]float32{1, 0, 0, 1})
	v := f32ToBf16Slice([]float32{1, 2, 3, 4})
	sinks := f32ToBf16Slice([]float32{0})

	got, err := sdpaHostRefWithSinks(q, k, v, sinks, 1, 1, 1, 2, 2, 1)
	if err != nil {
		t.Fatalf("sdpaHostRefWithSinks: %v", err)
	}
	// logits = [1, 0], sink = 0 → denom = e^1 + e^0 + e^0; w = [e/D, 1/D], D = e + 2.
	d := math.E + 2
	w0, w1 := math.E/d, 1/d
	want := []float32{float32(w0*1 + w1*3), float32(w0*2 + w1*4)}
	gotF := bf16ToF32Slice(got)
	for i := range want {
		if diff := math.Abs(float64(gotF[i]) - float64(want[i])); diff > 0.01 { // bf16 output rounding
			t.Fatalf("out[%d] = %v, hand-computed %v (diff %g)", i, gotF[i], want[i], diff)
		}
	}
}

// TestSdpaSinks_SDPAHostRefWithSinks_Bad proves the shape guards: wrong sinks length and a
// non-multiple GQA factor both refuse.
func TestSdpaSinks_SDPAHostRefWithSinks_Bad(t *testing.T) {
	x := f32ToBf16Slice(make([]float32, 4))
	if _, err := sdpaHostRefWithSinks(x, x, x, f32ToBf16Slice([]float32{0, 0}), 1, 1, 1, 4, 1, 1); err == nil {
		t.Fatal("expected sinks-length mismatch to refuse")
	}
	if _, err := sdpaHostRefWithSinks(x, x, x, f32ToBf16Slice([]float32{0, 0, 0}), 1, 3, 2, 4, 1, 1); err == nil {
		t.Fatal("expected nHeads%nKVHeads!=0 to refuse")
	}
}

// recordingDispatchSink is a host-side dispatchSink fake: it records every binding index so the
// emit-layer tests can assert the sinks ABI (and its absence on the plain emitters) WITHOUT a GPU
// or metallib — the byte-gate for "existing arches never bind the constant's buffers".
type recordingDispatchSink struct {
	bufIdx []uint
	i32    map[uint]int32
}

func (r *recordingDispatchSink) setPSO(metal.MTLComputePipelineState) {}
func (r *recordingDispatchSink) setBuf(_ metal.MTLBuffer, _, idx uint) {
	r.bufIdx = append(r.bufIdx, idx)
}
func (r *recordingDispatchSink) setI32(v int32, idx uint) {
	if r.i32 == nil {
		r.i32 = map[uint]int32{}
	}
	r.i32[idx] = v
}
func (r *recordingDispatchSink) setI64(int64, uint)                     {}
func (r *recordingDispatchSink) setF32(float32, uint)                   {}
func (r *recordingDispatchSink) dispatchThreads(_, _ metal.MTLSize)     {}
func (r *recordingDispatchSink) dispatchThreadgroups(_, _ metal.MTLSize) {}

func (r *recordingDispatchSink) boundBuf(idx uint) bool {
	for _, i := range r.bufIdx {
		if i == idx {
			return true
		}
	}
	return false
}

// TestSdpaSinks_EmitSDPAAt_Good is the non-sinks REGRESSION gate at the emit layer: the plain
// emitSDPAAt must bind nothing at the has_sinks lane's indices (16/17) — proof the existing
// arches' dispatch stream is untouched by the sinks addition.
func TestSdpaSinks_EmitSDPAAt_Good(t *testing.T) {
	rec := &recordingDispatchSink{}
	emitSDPAAt(rec, nil, nil, 0, nil, nil, nil, 0, 0, nil, 8, 4, 16, 1024, 64, 1024, 64, 0.125)
	if rec.boundBuf(16) {
		t.Fatal("plain emitSDPAAt bound a buffer at index 16 (the sinks lane) — non-sinks ABI changed")
	}
	if _, ok := rec.i32[17]; ok {
		t.Fatal("plain emitSDPAAt set num_q_heads at index 17 (the sinks lane) — non-sinks ABI changed")
	}
}

// TestSdpaSinks_EmitSDPAAtSinks_Good asserts the single-pass sinks ABI: sinks bound at buffer(16)
// and num_q_heads at 17, alongside the unchanged 0..10 core ABI (gqa at 4, N at 5).
func TestSdpaSinks_EmitSDPAAtSinks_Good(t *testing.T) {
	rec := &recordingDispatchSink{}
	emitSDPAAtSinks(rec, nil, nil, 0, nil, nil, nil, 0, 0, nil, 8, 4, 16, 1024, 64, 1024, 64, 0.125, nil, 0, 8)
	if !rec.boundBuf(16) {
		t.Fatal("emitSDPAAtSinks did not bind sinks at buffer(16)")
	}
	if got := rec.i32[17]; got != 8 {
		t.Fatalf("emitSDPAAtSinks num_q_heads(17) = %d, want 8", got)
	}
	if got := rec.i32[4]; got != 2 {
		t.Fatalf("emitSDPAAtSinks gqa_factor(4) = %d, want 2", got)
	}
	if got := rec.i32[5]; got != 16 {
		t.Fatalf("emitSDPAAtSinks N(5) = %d, want 16", got)
	}
}

// TestSdpaSinks_EmitSDPA2Pass1NAtSinks_Good asserts the 2-pass pass-1 sinks ABI: sinks bound at
// buffer(18) — and NOT at the single-pass lane's 16 — beside the unchanged 0..12 core ABI.
func TestSdpaSinks_EmitSDPA2Pass1NAtSinks_Good(t *testing.T) {
	rec := &recordingDispatchSink{}
	emitSDPA2Pass1NAtSinks(rec, nil, nil, 0, nil, nil, nil, nil, nil, 0, nil, 1, 8, 4, 2048, 64, 2048*64, 64, 2048*64, 64, 0.125, nil, 0)
	if !rec.boundBuf(18) {
		t.Fatal("emitSDPA2Pass1NAtSinks did not bind sinks at buffer(18)")
	}
	if rec.boundBuf(16) {
		t.Fatal("emitSDPA2Pass1NAtSinks bound index 16 — that is the single-pass sinks index, not pass 1's")
	}
	if got := rec.i32[7]; got != 2048 {
		t.Fatalf("emitSDPA2Pass1NAtSinks N(7) = %d, want 2048", got)
	}
}

// TestSdpaSinks_SDPAWithSinks_Good is the GPU byte-gate: the has_sinks sdpa_vector kernel against
// the host oracle across every head, plus the -Inf-free property that the sink visibly REDUCES
// the plain SDPA's weights (the drained mass). Skips without MLX_METALLIB_PATH; the orchestrator
// runs it at merge.
func TestSdpaSinks_SDPAWithSinks_Good(t *testing.T) {
	requireNativeRuntime(t)

	const b, nHeads, nKV, headDim, kvLen = 1, 8, 4, 64, 48
	const scale = float32(0.125)
	q := toBF16Bytes(syntheticFloat32(b*nHeads*headDim, 3))
	k := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 5))
	v := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 7))
	sinks := toBF16Bytes(syntheticFloat32(nHeads, 11)) // nonzero per-head sinks

	got, err := SDPAWithSinks(q, k, v, sinks, b, nHeads, nKV, headDim, kvLen, scale)
	if err != nil {
		t.Fatalf("SDPAWithSinks: %v", err)
	}
	want, err := sdpaHostRefWithSinks(q, k, v, sinks, b, nHeads, nKV, headDim, kvLen, scale)
	if err != nil {
		t.Fatalf("sdpaHostRefWithSinks: %v", err)
	}
	gotF, wantF := bf16ToF32Slice(got), bf16ToF32Slice(want)
	for i := range wantF {
		if diff := math.Abs(float64(gotF[i]) - float64(wantF[i])); diff > 0.02 { // bf16 kernel vs f64 host
			t.Fatalf("out[%d] = %v, host oracle %v (diff %g)", i, gotF[i], wantF[i], diff)
		}
	}

	// the sink must CHANGE the output vs plain SDPA (drained probability mass): identical outputs
	// would mean the constant/binding silently didn't engage.
	plain, err := SDPA(q, k, v, b, nHeads, nKV, headDim, kvLen, scale)
	if err != nil {
		t.Fatalf("SDPA: %v", err)
	}
	plainF := bf16ToF32Slice(plain)
	var maxDiff float64
	for i := range plainF {
		maxDiff = math.Max(maxDiff, math.Abs(float64(gotF[i])-float64(plainF[i])))
	}
	if maxDiff < 1e-4 {
		t.Fatalf("sinks output identical to plain SDPA (max diff %g) — has_sinks lane did not engage", maxDiff)
	}
}

// TestSdpaSinks_SDPAWithSinks_Bad proves the driver guards without touching the GPU result path:
// wrong sinks length and bad GQA both refuse.
func TestSdpaSinks_SDPAWithSinks_Bad(t *testing.T) {
	requireNativeRuntime(t)

	x := toBF16Bytes(syntheticFloat32(2*64, 3))
	if _, err := SDPAWithSinks(x, x, x, toBF16Bytes(syntheticFloat32(3, 5)), 1, 2, 1, 64, 1, 1); err == nil {
		t.Fatal("expected SDPAWithSinks to reject a sinks length != nHeads")
	}
	if _, err := SDPAWithSinks(x, x, x, toBF16Bytes(syntheticFloat32(3, 5)), 1, 3, 2, 64, 1, 1); err == nil {
		t.Fatal("expected SDPAWithSinks to reject nHeads not divisible by nKVHeads")
	}
}

// TestSdpaSinks_SDPA2PassWithSinks_Good gates the long-context pass against the host oracle past
// the 2-pass knee (the block-0 seeding must count the sink exactly once across the merged blocks).
func TestSdpaSinks_SDPA2PassWithSinks_Good(t *testing.T) {
	requireNativeRuntime(t)

	const b, nHeads, nKV, headDim, kvLen = 1, 8, 4, 64, 2048
	const scale = float32(0.125)
	q := toBF16Bytes(syntheticFloat32(b*nHeads*headDim, 3))
	k := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 5))
	v := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 7))
	sinks := toBF16Bytes(syntheticFloat32(nHeads, 11))

	got, err := SDPA2PassWithSinks(q, k, v, sinks, b, nHeads, nKV, headDim, kvLen, scale)
	if err != nil {
		t.Fatalf("SDPA2PassWithSinks: %v", err)
	}
	want, err := sdpaHostRefWithSinks(q, k, v, sinks, b, nHeads, nKV, headDim, kvLen, scale)
	if err != nil {
		t.Fatalf("sdpaHostRefWithSinks: %v", err)
	}
	gotF, wantF := bf16ToF32Slice(got), bf16ToF32Slice(want)
	for i := range wantF {
		if diff := math.Abs(float64(gotF[i]) - float64(wantF[i])); diff > 0.02 {
			t.Fatalf("out[%d] = %v, host oracle %v (diff %g)", i, gotF[i], wantF[i], diff)
		}
	}
}
