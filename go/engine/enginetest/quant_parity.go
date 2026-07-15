// SPDX-Licence-Identifier: EUPL-1.2

package enginetest

import (
	"bytes"
	"math"
	"testing"

	"dappco.re/go/inference/model"
)

// affineParityTol bounds the per-element float32 gap QuantParity accepts when a
// backend's MatVec is not byte-identical to ReferenceAffineMatVec. Two correct
// implementations of the same group-affine dot product can still land on
// different bf16 output bytes: a GPU kernel reduces inDim terms with a
// parallel/tree accumulation order, this reference sums serially in float64 —
// the two orders round differently at the final bf16 store even when every
// intermediate is exact. The tolerance is documented, not tuned to pass: it is
// a small multiple of one bf16 ULP (~2^-7 relative) at the fixture's O(1)
// output magnitude, not a value picked after the fact to make a divergent
// implementation pass.
const affineParityTol = 0.05

// quantAffineFixture builds a small, fully deterministic group-affine
// fixture — every value is a formula, never randomness, so a divergence
// between two runs (or two backends) is always a real behavioural
// difference, never fixture noise. outDim=8, inDim=128, groupSize=64, bits=4
// mirrors the (groupSize, bits) pairing engine/metal's own real-dispatch
// quant tests already exercise (qgemv_test.go, arch_session_bench_test.go) —
// a combination the metallib's compiled kernel templates are known to
// instantiate — while staying a unit-scale, no-accelerator-required check.
func quantAffineFixture() (x, packed, scales, biases []byte, outDim, inDim, groupSize, bits int) {
	outDim, inDim, groupSize, bits = 8, 128, 64, 4
	groupsPerRow := inDim / groupSize
	rowPacked := inDim * bits / 8
	rowSB := groupsPerRow * 2
	maxCode := uint32(1)<<uint(bits) - 1

	packed = make([]byte, outDim*rowPacked)
	scales = make([]byte, outDim*rowSB)
	biases = make([]byte, outDim*rowSB)
	for r := 0; r < outDim; r++ {
		pRow := packed[r*rowPacked : (r+1)*rowPacked]
		sRow := scales[r*rowSB : (r+1)*rowSB]
		bRow := biases[r*rowSB : (r+1)*rowSB]
		for g := 0; g < groupsPerRow; g++ {
			scale := 0.25 + 0.125*float32(g+r)
			bias := -1 + 0.5*float32((g+r)%3)
			sh, bh := bf16Encode(scale), bf16Encode(bias)
			sRow[g*2], sRow[g*2+1] = byte(sh), byte(sh>>8)
			bRow[g*2], bRow[g*2+1] = byte(bh), byte(bh>>8)
			for j := 0; j < groupSize; j++ {
				c := g*groupSize + j
				code := uint32(c*5+r*3+1) % (maxCode + 1)
				affineSetCode(pRow, c*bits, bits, code)
			}
		}
	}
	x = make([]byte, inDim*2)
	for i := 0; i < inDim; i++ {
		v := 0.1 * float32((i%7)-3)
		h := bf16Encode(v)
		x[i*2], x[i*2+1] = byte(h), byte(h>>8)
	}
	return x, packed, scales, biases, outDim, inDim, groupSize, bits
}

// QuantParity validates a backend's registered "affine" quant compute
// (model.BackendQuant(backend, "affine"), model/quant.go) against the
// pure-Go ReferenceAffineMatVec on a small deterministic fixture: it checks
// arithmetic correctness, not throughput, so it runs in CI with no
// accelerator present. A backend that has not registered the "affine" kind
// is reported and skipped — present ⇒ exercised, absent ⇒ skipped and
// reported, the same optional-capability shape SessionHandle/TextModel use
// for their own probed capabilities (session.go, textmodel.go).
func QuantParity(t *testing.T, backend string) {
	t.Helper()
	q, ok := model.BackendQuant(backend, "affine")
	if !ok {
		t.Skipf("no backend quant registered for (%q, %q) — nothing to check", backend, "affine")
	}

	x, packed, scales, biases, outDim, inDim, groupSize, bits := quantAffineFixture()

	got, err := q.MatVec(x, packed, scales, biases, outDim, inDim, groupSize, bits)
	if err != nil {
		t.Fatalf("%s/affine MatVec: %v", backend, err)
	}
	want, err := ReferenceAffineMatVec(x, packed, scales, biases, outDim, inDim, groupSize, bits)
	if err != nil {
		t.Fatalf("ReferenceAffineMatVec (harness bug): %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("%s/affine MatVec output length = %d, want %d", backend, len(got), len(want))
	}
	if bytes.Equal(got, want) {
		return // byte-identical — the strongest receipt
	}
	for i := 0; i < outDim; i++ {
		gv := float64(bf16Decode(got[i*2], got[i*2+1]))
		wv := float64(bf16Decode(want[i*2], want[i*2+1]))
		if d := math.Abs(gv - wv); d > affineParityTol {
			t.Fatalf("%s/affine MatVec[%d] = %v, reference = %v (diff %v > tol %v)",
				backend, i, gv, wv, d, affineParityTol)
		}
	}
}
