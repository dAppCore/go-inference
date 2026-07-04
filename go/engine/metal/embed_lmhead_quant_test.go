// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"testing"
)

// TestEmbedLMHeadQuant (gates the quant decode bookends against metal as the oracle) lives in
// embed_lmhead_quant_metal_test.go — it needs the real cgo metal package, so it's gated behind
// metal_runtime. The tests below are hermetic: they only need quantWeightFixture (pure Go).

func TestLMHeadQuantAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, vocab, groupSize, bits = 64, 128, 32, 4
	hidden := toBF16Bytes(syntheticFloat32(dModel, 31))
	finalNormW := toBF16Bytes(syntheticFloat32(dModel, 7))
	qw := quantWeightFixture(t, vocab, dModel, groupSize, bits, 53)
	if _, err := LMHeadQuant(hidden, finalNormW, qw.Packed, qw.Scales, qw.Biases, dModel, vocab, groupSize, bits, 1e-6, 0); err != nil {
		t.Fatalf("LMHeadQuant warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := LMHeadQuant(hidden, finalNormW, qw.Packed, qw.Scales, qw.Biases, dModel, vocab, groupSize, bits, 1e-6, 0); err != nil {
			t.Fatalf("LMHeadQuant: %v", err)
		}
	})
	if allocs > 35 {
		t.Fatalf("LMHeadQuant allocations = %.0f, want <= 35", allocs)
	}
}

func TestLMHeadQuantIntoReusesOutputBackingAndBypassesScratchOutput(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const dModel, vocab, groupSize, bits = 64, 128, 32, 4
	hidden := toBF16Bytes(syntheticFloat32(dModel, 31))
	finalNormW := toBF16Bytes(syntheticFloat32(dModel, 7))
	qw := quantWeightFixture(t, vocab, dModel, groupSize, bits, 53)
	want, err := LMHeadQuant(hidden, finalNormW, qw.Packed, qw.Scales, qw.Biases, dModel, vocab, groupSize, bits, 1e-6, 0)
	if err != nil {
		t.Fatalf("LMHeadQuant reference: %v", err)
	}
	out := bytes.Repeat([]byte{0xa5}, vocab*bf16Size)

	scratch, err := getQMVBF16Scratch(vocab, dModel)
	if err != nil {
		t.Fatalf("getQMVBF16Scratch: %v", err)
	}
	sentinel := bytes.Repeat([]byte{0x6a}, len(scratch.out.bytes))
	copy(scratch.out.bytes, sentinel)
	putQMVBF16Scratch(scratch)

	got, err := LMHeadQuantInto(out, hidden, finalNormW, qw.Packed, qw.Scales, qw.Biases, dModel, vocab, groupSize, bits, 1e-6, 0)
	if err != nil {
		t.Fatalf("LMHeadQuantInto: %v", err)
	}
	if len(got) != len(want) || &got[0] != &out[0] {
		t.Fatal("LMHeadQuantInto did not reuse caller-owned output backing")
	}
	eqBytes(t, "LMHeadQuantInto", got, want)

	scratch, err = getQMVBF16Scratch(vocab, dModel)
	if err != nil {
		t.Fatalf("getQMVBF16Scratch after call: %v", err)
	}
	defer putQMVBF16Scratch(scratch)
	if !bytes.Equal(scratch.out.bytes, sentinel) {
		t.Fatal("LMHeadQuantInto wrote through pooled scratch output instead of caller output")
	}
}
