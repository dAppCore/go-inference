// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"math"
	"testing"
	"unsafe"
)

// mtp_attn_test.go covers SDPACausalBF16/SDPACausalBF16Into hermetically: scratch-pool
// residency, allocation budgets, output-buffer reuse, causal-masking correctness against a
// host softmax reference, and the input-guard error paths. (An earlier revision of this file
// pointed at a "mtp_attn_metal_test.go" byte-identical cgo oracle test; pkg/metal was retired —
// engine/metal is now the sovereign implementation, so there is no cgo oracle left to gate
// behind metal_runtime. The host reference below is the correctness gate instead.)

func sdpaScale(D int) float32 { return float32(1.0 / math.Sqrt(float64(D))) }

func TestBF16ToF32Into_Converts(t *testing.T) {
	in := toBF16Bytes([]float32{-1.5, 0, 2.25})
	out := make([]float32, 3)
	bf16ToF32Into(out, in)
	if out[0] != -1.5 || out[1] != 0 || out[2] != 2.25 {
		t.Fatalf("bf16ToF32Into = %v", out)
	}
}

// sdpaCausalBF16Reference computes causal scaled-dot-product attention on bf16 q/k/v in the
// same head-major [H,L,D] layout SDPACausalBF16 expects, entirely in float64 host maths — an
// independent implementation from the package's own f32 GPU composition (matMulF32NT +
// softmaxF32 + matMulF32), so it proves the causal masking and GQA head mapping rather than
// re-testing the matmul/softmax kernels themselves.
func sdpaCausalBF16Reference(q, k, v []byte, H, Hkv, qL, kL, D int, scale float32) []byte {
	gqa := H / Hkv
	out := make([]byte, H*qL*D*bf16Size)
	bf16At := func(b []byte, i int) float64 { return float64(bf16ToF32(b[i*2], b[i*2+1])) }
	for h := range H {
		hk := h / gqa
		for i := range qL {
			lim := kL - qL + i
			scores := make([]float64, lim+1)
			maxS := math.Inf(-1)
			for j := 0; j <= lim; j++ {
				var dot float64
				for d := range D {
					dot += bf16At(q, (h*qL+i)*D+d) * bf16At(k, (hk*kL+j)*D+d)
				}
				scores[j] = dot * float64(scale)
				if scores[j] > maxS {
					maxS = scores[j]
				}
			}
			var denom float64
			for j := range scores {
				scores[j] = math.Exp(scores[j] - maxS)
				denom += scores[j]
			}
			for d := range D {
				var acc float64
				for j := 0; j <= lim; j++ {
					acc += scores[j] / denom * bf16At(v, (hk*kL+j)*D+d)
				}
				b := f32ToBF16(float32(acc))
				base := (h*qL+i)*D + d
				out[base*bf16Size], out[base*bf16Size+1] = byte(b), byte(b>>8)
			}
		}
	}
	return out
}

// TestMtpAttn_SDPACausalBF16_Good proves the composed f32 GQA-causal attention (matMulF32NT +
// causal mask + softmaxF32 + matMulF32) matches an independent host reference.
func TestMtpAttn_SDPACausalBF16_Good(t *testing.T) {
	requireNativeRuntime(t)

	const H, Hkv, qL, kL, D = 4, 2, 3, 5, 16
	scale := sdpaScale(D)
	q := toBF16Bytes(syntheticFloat32(H*qL*D, 3))
	k := toBF16Bytes(syntheticFloat32(Hkv*kL*D, 5))
	v := toBF16Bytes(syntheticFloat32(Hkv*kL*D, 7))

	got, err := SDPACausalBF16(q, k, v, H, Hkv, qL, kL, D, scale)
	if err != nil {
		t.Fatalf("SDPACausalBF16: %v", err)
	}
	want := sdpaCausalBF16Reference(q, k, v, H, Hkv, qL, kL, D, scale)
	if cos := cosineBF16(got, want); cos < 0.999 {
		t.Fatalf("SDPACausalBF16 cosine=%.6f vs host causal-softmax reference, want ~1", cos)
	}
}

// TestMtpAttn_SDPACausalBF16_Bad exercises the dimension/GQA guards SDPACausalBF16Into
// validates before touching the scratch pool.
func TestMtpAttn_SDPACausalBF16_Bad(t *testing.T) {
	const H, Hkv, qL, kL, D = 2, 1, 2, 3, 8
	scale := sdpaScale(D)
	validQ := toBF16Bytes(syntheticFloat32(H*qL*D, 3))
	validK := toBF16Bytes(syntheticFloat32(Hkv*kL*D, 5))
	validV := toBF16Bytes(syntheticFloat32(Hkv*kL*D, 7))

	cases := []struct {
		name    string
		q, k, v []byte
		H, Hkv  int
	}{
		{"H not a multiple of Hkv", validQ, validK, validV, 3, 2},
		{"q length mismatch", []byte{0, 0}, validK, validV, H, Hkv},
		{"k length mismatch", validQ, []byte{0, 0}, validV, H, Hkv},
		{"v length mismatch", validQ, validK, []byte{0, 0}, H, Hkv},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := SDPACausalBF16(c.q, c.k, c.v, c.H, c.Hkv, qL, kL, D, scale); err == nil {
				t.Fatalf("SDPACausalBF16(%s): expected an error, got none", c.name)
			}
		})
	}
}

// TestMtpAttn_SDPACausalBF16_Ugly proves the causal boundary itself: perturbing a key strictly
// AFTER a query's causal limit (kL-qL+i) must not change that query's output at all.
func TestMtpAttn_SDPACausalBF16_Ugly(t *testing.T) {
	requireNativeRuntime(t)

	const H, Hkv, qL, kL, D = 1, 1, 2, 4, 8
	scale := sdpaScale(D)
	q := toBF16Bytes(syntheticFloat32(H*qL*D, 3))
	k := toBF16Bytes(syntheticFloat32(Hkv*kL*D, 5))
	v := toBF16Bytes(syntheticFloat32(Hkv*kL*D, 7))

	got, err := SDPACausalBF16(q, k, v, H, Hkv, qL, kL, D, scale)
	if err != nil {
		t.Fatalf("SDPACausalBF16: %v", err)
	}

	// query i=0's causal limit is kL-qL+0 = 2, so key row 3 (index kL-1) is strictly future for
	// it; perturbing it must leave query 0's output untouched even though it DOES affect query
	// i=1 (limit kL-qL+1 = 3, which includes row 3).
	kPerturbed := append([]byte(nil), k...)
	base := (0*kL + kL - 1) * D * bf16Size
	for i := base; i < base+D*bf16Size; i += bf16Size {
		h := f32ToBF16(1000)
		kPerturbed[i], kPerturbed[i+1] = byte(h), byte(h>>8)
	}
	got2, err := SDPACausalBF16(q, kPerturbed, v, H, Hkv, qL, kL, D, scale)
	if err != nil {
		t.Fatalf("SDPACausalBF16 (perturbed): %v", err)
	}

	row0Len := D * bf16Size
	if !bytes.Equal(got[:row0Len], got2[:row0Len]) {
		t.Fatalf("query 0's output changed when a future-only key was perturbed — causal mask leaked:\n got  %v\n got2 %v", bf16Floats(got[:row0Len]), bf16Floats(got2[:row0Len]))
	}
	if bytes.Equal(got[row0Len:], got2[row0Len:]) {
		t.Fatal("query 1's output did NOT change when an in-window key was perturbed — the fixture is not exercising the causal boundary")
	}
}

func TestSDPACausalBF16ScratchPoolKeepsShapesResident(t *testing.T) {
	small := getSDPACausalBF16Scratch(2, 1, 4, 4, 64)
	putSDPACausalBF16Scratch(small)
	large := getSDPACausalBF16Scratch(4, 2, 8, 8, 64)
	putSDPACausalBF16Scratch(large)
	forceNativeGC()
	forceNativeGC()

	gotSmall := getSDPACausalBF16Scratch(2, 1, 4, 4, 64)
	defer putSDPACausalBF16Scratch(gotSmall)
	if gotSmall != small {
		t.Fatal("SDPA causal BF16 scratch pool evicted the small shape after using a larger shape")
	}

	gotLarge := getSDPACausalBF16Scratch(4, 2, 8, 8, 64)
	defer putSDPACausalBF16Scratch(gotLarge)
	if gotLarge != large {
		t.Fatal("SDPA causal BF16 scratch pool evicted the large shape after reusing the small shape")
	}
}

func TestSDPACausalBF16AllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const H, Hkv, qL, kL, D = 2, 1, 4, 4, 64
	scale := sdpaScale(D)
	q := toBF16Bytes(syntheticFloat32(H*qL*D, 3))
	k := toBF16Bytes(syntheticFloat32(Hkv*kL*D, 5))
	v := toBF16Bytes(syntheticFloat32(Hkv*kL*D, 7))
	if _, err := SDPACausalBF16(q, k, v, H, Hkv, qL, kL, D, scale); err != nil {
		t.Fatalf("SDPACausalBF16 warmup: %v", err)
	}

	var attnErr error
	allocs := testing.AllocsPerRun(3, func() {
		_, attnErr = SDPACausalBF16(q, k, v, H, Hkv, qL, kL, D, scale)
	})
	if attnErr != nil {
		t.Fatalf("SDPACausalBF16: %v", attnErr)
	}
	if allocs > 390 {
		t.Fatalf("SDPACausalBF16 allocations = %.0f, want <= 390", allocs)
	}
}

// TestMtpAttn_SDPACausalBF16Into_Good proves SDPACausalBF16Into reuses caller-owned output
// backing and matches the allocating wrapper byte-for-byte.
func TestMtpAttn_SDPACausalBF16Into_Good(t *testing.T) {
	requireNativeRuntime(t)

	const H, Hkv, qL, kL, D = 2, 1, 4, 4, 64
	scale := sdpaScale(D)
	q := toBF16Bytes(syntheticFloat32(H*qL*D, 3))
	k := toBF16Bytes(syntheticFloat32(Hkv*kL*D, 5))
	v := toBF16Bytes(syntheticFloat32(Hkv*kL*D, 7))
	want, err := SDPACausalBF16(q, k, v, H, Hkv, qL, kL, D, scale)
	if err != nil {
		t.Fatalf("SDPACausalBF16 reference: %v", err)
	}
	out := bytes.Repeat([]byte{0xa5}, len(want))
	outPtr := unsafe.Pointer(&out[0])

	got, err := SDPACausalBF16Into(out, q, k, v, H, Hkv, qL, kL, D, scale)
	if err != nil {
		t.Fatalf("SDPACausalBF16Into: %v", err)
	}
	if len(got) != len(want) || unsafe.Pointer(&got[0]) != outPtr {
		t.Fatal("SDPACausalBF16Into did not reuse caller-owned output backing")
	}
	eqBytes(t, "SDPACausalBF16Into", got, want)
}

// TestMtpAttn_SDPACausalBF16Into_Bad mirrors SDPACausalBF16's dimension guards through the
// caller-output entry point.
func TestMtpAttn_SDPACausalBF16Into_Bad(t *testing.T) {
	const H, Hkv, qL, kL, D = 2, 1, 2, 3, 8
	scale := sdpaScale(D)
	validQ := toBF16Bytes(syntheticFloat32(H*qL*D, 3))
	validK := toBF16Bytes(syntheticFloat32(Hkv*kL*D, 5))
	validV := toBF16Bytes(syntheticFloat32(Hkv*kL*D, 7))
	out := make([]byte, H*qL*D*bf16Size)

	if _, err := SDPACausalBF16Into(out, validQ, validK, validV, 3, 2, qL, kL, D, scale); err == nil {
		t.Fatal("expected SDPACausalBF16Into to reject H not a multiple of Hkv")
	}
	if _, err := SDPACausalBF16Into(out, []byte{0, 0}, validK, validV, H, Hkv, qL, kL, D, scale); err == nil {
		t.Fatal("expected SDPACausalBF16Into to reject a q length mismatch")
	}
}

// TestMtpAttn_SDPACausalBF16Into_Ugly proves the too-small-capacity path: when cap(out) is
// smaller than the required output, SDPACausalBF16Into must allocate fresh storage rather than
// write out of bounds, and still return the correct attention output.
func TestMtpAttn_SDPACausalBF16Into_Ugly(t *testing.T) {
	requireNativeRuntime(t)

	const H, Hkv, qL, kL, D = 2, 1, 4, 4, 64
	scale := sdpaScale(D)
	q := toBF16Bytes(syntheticFloat32(H*qL*D, 3))
	k := toBF16Bytes(syntheticFloat32(Hkv*kL*D, 5))
	v := toBF16Bytes(syntheticFloat32(Hkv*kL*D, 7))
	want, err := SDPACausalBF16(q, k, v, H, Hkv, qL, kL, D, scale)
	if err != nil {
		t.Fatalf("SDPACausalBF16 reference: %v", err)
	}

	tooSmall := make([]byte, 1)
	got, err := SDPACausalBF16Into(tooSmall, q, k, v, H, Hkv, qL, kL, D, scale)
	if err != nil {
		t.Fatalf("SDPACausalBF16Into (undersized out): %v", err)
	}
	eqBytes(t, "SDPACausalBF16Into (undersized out)", got, want)
}
