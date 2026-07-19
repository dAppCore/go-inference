// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"testing"
)

func TestProjectorHasVReflectsOptionalWeight(t *testing.T) {
	if (bf16Projector{}).hasV() {
		t.Fatal("bf16Projector without wV reported hasV")
	}
	if (qmvProjector{}).hasV() {
		t.Fatal("qmvProjector without V weight reported hasV")
	}
	requireNativeRuntime(t)
	if !(bf16Projector{wV: copyView(toBF16Bytes([]float32{1}))}).hasV() {
		t.Fatal("bf16Projector with wV did not report hasV")
	}
	qw := quantWeightFixture(t, 64, 64, 64, 4, 3)
	qv := qmvWeight{wq: copyView(qw.Packed), scales: copyView(qw.Scales), biases: copyView(qw.Biases)}
	if !(qmvProjector{v: qv}).hasV() {
		t.Fatal("qmvProjector with V weight did not report hasV")
	}
}

func TestProjectorRejectsBadProjectionIndex(t *testing.T) {
	if err := (bf16Projector{}).project(nil, nil, nil, 0, projIndex(99)); err == nil {
		t.Fatal("expected bf16Projector to reject bad projection index")
	}
	if err := (qmvProjector{}).project(nil, nil, nil, 0, projIndex(99)); err == nil {
		t.Fatal("expected qmvProjector to reject bad projection index")
	}
}

// TestProjector_BiasViewProjO_Good pins the o_proj bias seam (gpt_oss attention_bias=true): both
// projectors return the bound bO for projO — and the ZERO bufView regression for every other arch
// (nil bO ⇒ encProjBias no-ops, so the pre-BO dispatch stream is unchanged). Pure host, no GPU.
func TestProjector_BiasViewProjO_Good(t *testing.T) {
	requireNativeRuntime(t) // bufView construction touches resident buffers
	bo := copyView(toBF16Bytes([]float32{1, 2}))
	if got := (bf16Projector{bO: bo}).biasView(projO); got.buf == nil {
		t.Fatal("bf16Projector.biasView(projO) lost the bound BO")
	}
	if got := (qmvProjector{bO: bo}).biasView(projO); got.buf == nil {
		t.Fatal("qmvProjector.biasView(projO) lost the bound BO")
	}
	if got := (bf16Projector{}).biasView(projO); got.buf != nil {
		t.Fatal("bf16Projector without BO returned a non-zero projO bias view — bias-free arches must stay biasless")
	}
	if got := (qmvProjector{}).biasView(projO); got.buf != nil {
		t.Fatal("qmvProjector without BO returned a non-zero projO bias view — bias-free arches must stay biasless")
	}
	// the MLP projections never carry a bias in any supported arch — the seam must stay closed.
	for _, p := range []projIndex{projGate, projUp, projDown} {
		if got := (qmvProjector{bO: bo}).biasView(p); got.buf != nil {
			t.Fatalf("qmvProjector.biasView(%v) returned a bias — only q/k/v/o may carry one", p)
		}
	}
}

// TestProjector_ProjectOBias_Good is the GPU gate for the o_proj bias add: qmvProjector.project(projO)
// with a nonzero BO must equal the composed reference (the bias-free projection + one AddBF16 of BO)
// BYTE for byte — the add is exactly the bias's contribution — and the nil-BO path must equal the
// bias-free projection bytes exactly (the regression half).
func TestProjector_ProjectOBias_Good(t *testing.T) {
	requireNativeRuntime(t)
	const qDim, dModel, gs, bits = 64, 64, 32, 4
	w := quantWeightFixture(t, dModel, qDim, gs, bits, 7)
	attn := toBF16Bytes(syntheticFloat32(qDim, 3))
	bo := toBF16Bytes(syntheticFloat32(dModel, 9))

	base := qmvProjector{
		o:      qmvWeight{wq: copyView(w.Packed), scales: copyView(w.Scales), biases: copyView(w.Biases)},
		dModel: dModel, qDim: qDim, groupSize: gs, bits: bits,
	}
	run := func(p qmvProjector) []byte {
		t.Helper()
		var got []byte
		withAutoreleasePool(func() {
			in := residentBytes(attn)
			out, err := newPinnedNoCopyBytes(dModel * bf16Size)
			if err != nil {
				t.Fatalf("newPinnedNoCopyBytes: %v", err)
			}
			defer out.Close()
			cb := commandBufferFast(queue)
			enc := computeCommandEncoderFast(cb)
			if err := p.project(enc, in, out.buf, 0, projO); err != nil {
				endEncodingFast(enc)
				t.Fatalf("project(projO): %v", err)
			}
			endEncodingFast(enc)
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
			got = append([]byte(nil), out.bytes...)
		})
		return got
	}

	plain := run(base)
	withBO := base
	withBO.bO = copyView(bo)
	biased := run(withBO)

	// composed reference: plain + BO through the same add kernel.
	ref, err := AddBF16(plain, bo)
	if err != nil {
		t.Fatalf("AddBF16: %v", err)
	}
	if !bytes.Equal(biased, ref) {
		t.Fatal("project(projO) with BO != bias-free projection + AddBF16(BO) — the o_proj bias is not exactly the bias's contribution")
	}
	if bytes.Equal(biased, plain) {
		t.Fatal("nonzero BO left the projection output unchanged — the bias add did not engage")
	}
	// regression: a projector without BO must reproduce the bias-free bytes exactly.
	if again := run(base); !bytes.Equal(again, plain) {
		t.Fatal("nil-BO projection bytes changed across runs — the no-bias path is not stable/identical")
	}
}
