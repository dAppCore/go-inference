// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"
)

// TestDiffusionAttention_DiffusionSDPA_Good proves the block-diffusion canvas attention core
// against a host softmax-attention reference (diffusionSDPAReference, diffusion_test.go) with a
// GQA head ratio and a real additive mask — the same reference diffusion_test.go's
// TestDiffusionSDPAWithMaskMatchesReference_Good uses, named here for AX-7's file-aware
// convention (DiffusionSDPA is declared in diffusion_attention.go, so its triplet lives here).
func TestDiffusionAttention_DiffusionSDPA_Good(t *testing.T) {
	requireNativeRuntime(t)

	const (
		qLen     = 3
		keyLen   = 5
		nHeads   = 4
		nKVHeads = 2
		headDim  = 8
	)
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	q := toBF16Bytes(bf16Round(syntheticFloat32(nHeads*qLen*headDim, 31)))
	k := toBF16Bytes(bf16Round(syntheticFloat32(nKVHeads*keyLen*headDim, 37)))
	v := toBF16Bytes(bf16Round(syntheticFloat32(nKVHeads*keyLen*headDim, 41)))
	mask := make([]float32, qLen*keyLen)
	negInf := float32(math.Inf(-1))
	mask[0*keyLen+0] = negInf
	mask[1*keyLen+1] = negInf
	mask[2*keyLen+0] = negInf
	mask[2*keyLen+1] = negInf

	got, err := DiffusionSDPA(q, k, v, qLen, keyLen, nHeads, nKVHeads, headDim, scale, mask)
	if err != nil {
		t.Fatalf("DiffusionSDPA: %v", err)
	}
	want := diffusionSDPAReference(q, k, v, qLen, keyLen, nHeads, nKVHeads, headDim, scale, mask)
	relL2, cos := relL2Cos(bf16Floats(got), bf16Floats(want))
	if relL2 > 1e-2 || cos < 0.999 {
		t.Fatalf("DiffusionSDPA rel-L2/cos = %.3e/%.6f, want masked attention reference", relL2, cos)
	}
}

// TestDiffusionAttention_DiffusionSDPA_Bad exercises every dimension/length guard
// DiffusionSDPA validates before it touches the GPU.
func TestDiffusionAttention_DiffusionSDPA_Bad(t *testing.T) {
	requireNativeRuntime(t)

	const qLen, keyLen, nHeads, nKVHeads, headDim = 2, 3, 4, 2, 8
	scale := float32(0.125)
	validQ := toBF16Bytes(syntheticFloat32(nHeads*qLen*headDim, 3))
	validK := toBF16Bytes(syntheticFloat32(nKVHeads*keyLen*headDim, 5))
	validV := toBF16Bytes(syntheticFloat32(nKVHeads*keyLen*headDim, 7))

	cases := []struct {
		name                      string
		q, k, v                   []byte
		qLen, keyLen, nH, nKV, hd int
		mask                      []float32
	}{
		{"gqa not a multiple", validQ, validK, validV, qLen, keyLen, 4, 3, headDim, nil},
		{"q length mismatch", []byte{0, 0}, validK, validV, qLen, keyLen, nHeads, nKVHeads, headDim, nil},
		{"k length mismatch", validQ, []byte{0, 0}, validV, qLen, keyLen, nHeads, nKVHeads, headDim, nil},
		{"v length mismatch", validQ, validK, []byte{0, 0}, qLen, keyLen, nHeads, nKVHeads, headDim, nil},
		{"mask wrong length", validQ, validK, validV, qLen, keyLen, nHeads, nKVHeads, headDim, make([]float32, qLen*keyLen+1)},
		{"zero keyLen with positive qLen", validQ, validK, validV, qLen, 0, nHeads, nKVHeads, headDim, nil},
		{"negative qLen", validQ, validK, validV, -1, keyLen, nHeads, nKVHeads, headDim, nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := DiffusionSDPA(c.q, c.k, c.v, c.qLen, c.keyLen, c.nH, c.nKV, c.hd, scale, c.mask); err == nil {
				t.Fatalf("DiffusionSDPA(%s): expected an error, got none", c.name)
			}
		})
	}
}

// TestDiffusionAttention_DiffusionSDPA_Ugly exercises the qLen==0 boundary: DiffusionSDPA
// returns an empty (non-nil error) slice rather than erroring or dereferencing an empty cache,
// the diffusion canvas's "nothing new to denoise this step" case.
func TestDiffusionAttention_DiffusionSDPA_Ugly(t *testing.T) {
	requireNativeRuntime(t)

	const keyLen, nHeads, nKVHeads, headDim = 5, 2, 1, 8
	k := toBF16Bytes(syntheticFloat32(nKVHeads*keyLen*headDim, 5))
	v := toBF16Bytes(syntheticFloat32(nKVHeads*keyLen*headDim, 7))

	got, err := DiffusionSDPA(nil, k, v, 0, keyLen, nHeads, nKVHeads, headDim, 0.125, nil)
	if err != nil {
		t.Fatalf("DiffusionSDPA with qLen=0: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("DiffusionSDPA with qLen=0 = %d bytes, want 0", len(got))
	}
}
