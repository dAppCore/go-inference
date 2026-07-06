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

// TestSdpa_SDPA_Good pins the trivial single-key base case: with only one key/value row,
// softmax collapses to certainty and the output must equal V exactly, for every head.
func TestSdpa_SDPA_Good(t *testing.T) {
	requireNativeRuntime(t)

	const b, nHeads, nKV, headDim, kvLen = 1, 2, 1, 64, 1
	q := toBF16Bytes(syntheticFloat32(b*nHeads*headDim, 3))
	k := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 5))
	v := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 7))
	got, err := SDPA(q, k, v, b, nHeads, nKV, headDim, kvLen, 1)
	if err != nil {
		t.Fatalf("SDPA: %v", err)
	}
	want := append(append([]byte(nil), v...), v...)
	if !bytes.Equal(got, want) {
		t.Fatalf("single-value SDPA = %v, want repeated V %v", bf16Floats(got), bf16Floats(want))
	}
}

// TestSdpa_SDPA_Bad proves the GQA guard: nHeads must be a multiple of nKVHeads.
func TestSdpa_SDPA_Bad(t *testing.T) {
	requireNativeRuntime(t)

	x := toBF16Bytes(syntheticFloat32(64, 3))
	if _, err := SDPA(x, x, x, 1, 3, 2, 64, 1, 1); err == nil {
		t.Fatal("expected SDPA to reject nHeads not divisible by nKVHeads")
	}
}

// sdpaBF16Reference computes single-query scaled-dot-product attention on bf16 q/k/v (head-major
// [nHeads,1,headDim] / [nKVHeads,kvLen,headDim]) in independent float64 host maths — the GQA
// head-mapping + softmax correctness gate, distinct from the package's own on-device kernel.
func sdpaBF16Reference(q, k, v []byte, nHeads, nKVHeads, headDim, kvLen int, scale float32) []byte {
	gqa := nHeads / nKVHeads
	out := make([]byte, nHeads*headDim*bf16Size)
	bf16At := func(b []byte, i int) float64 { return float64(bf16ToF32(b[i*2], b[i*2+1])) }
	for h := range nHeads {
		hk := h / gqa
		scores := make([]float64, kvLen)
		maxS := math.Inf(-1)
		for j := range kvLen {
			var dot float64
			for d := range headDim {
				dot += bf16At(q, h*headDim+d) * bf16At(k, (hk*kvLen+j)*headDim+d)
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
		for d := range headDim {
			var acc float64
			for j := range kvLen {
				acc += scores[j] / denom * bf16At(v, (hk*kvLen+j)*headDim+d)
			}
			b := f32ToBF16(float32(acc))
			base := h*headDim + d
			out[base*bf16Size], out[base*bf16Size+1] = byte(b), byte(b>>8)
		}
	}
	return out
}

// TestSdpa_SDPA_Ugly proves real GQA attention correctness (nHeads>nKVHeads, kvLen>1 — beyond
// the trivial single-key/one-KV-head base case) against an independent host reference.
func TestSdpa_SDPA_Ugly(t *testing.T) {
	requireNativeRuntime(t)

	const b, nHeads, nKV, headDim, kvLen = 1, 8, 2, 64, 16
	scale := float32(0.125)
	q := toBF16Bytes(syntheticFloat32(b*nHeads*headDim, 3))
	k := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 5))
	v := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 7))

	got, err := SDPA(q, k, v, b, nHeads, nKV, headDim, kvLen, scale)
	if err != nil {
		t.Fatalf("SDPA: %v", err)
	}
	want := sdpaBF16Reference(q, k, v, nHeads, nKV, headDim, kvLen, scale)
	if cos := cosineBF16(got, want); cos < 0.999 {
		t.Fatalf("GQA SDPA cosine=%.6f vs host reference, want ~1", cos)
	}
}

func TestSDPABF16ScratchPoolKeepsDimensionsResident(t *testing.T) {
	requireNativeRuntime(t)

	small, err := getSDPABF16Scratch(128, 256, 256, 128)
	if err != nil {
		t.Fatalf("get small SDPA scratch: %v", err)
	}
	putSDPABF16Scratch(small)

	large, err := getSDPABF16Scratch(256, 512, 512, 256)
	if err != nil {
		t.Fatalf("get large SDPA scratch: %v", err)
	}
	putSDPABF16Scratch(large)
	forceNativeGC()
	forceNativeGC()

	gotSmall, err := getSDPABF16Scratch(128, 256, 256, 128)
	if err != nil {
		t.Fatalf("get small SDPA scratch again: %v", err)
	}
	defer putSDPABF16Scratch(gotSmall)
	if gotSmall != small {
		t.Fatal("SDPA scratch pool evicted the small scratch after using a larger scratch")
	}

	gotLarge, err := getSDPABF16Scratch(256, 512, 512, 256)
	if err != nil {
		t.Fatalf("get large SDPA scratch again: %v", err)
	}
	defer putSDPABF16Scratch(gotLarge)
	if gotLarge != large {
		t.Fatal("SDPA scratch pool evicted the large scratch after reusing the small scratch")
	}
}

func TestSDPABF16ScratchBuffersUseCallerBackingAfterWarmup(t *testing.T) {
	requireNativeRuntime(t)

	const b, nHeads, nKV, headDim, kvLen = 1, 2, 1, 64, 5
	q := toBF16Bytes(syntheticFloat32(b*nHeads*headDim, 3))
	k := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 5))
	v := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 7))
	outBytes := b * nHeads * headDim * bf16Size
	scratch, err := getSDPABF16Scratch(len(q), len(k), len(v), outBytes)
	if err != nil {
		t.Fatalf("get SDPA scratch: %v", err)
	}
	defer putSDPABF16Scratch(scratch)
	var qBuf, kBuf, vBuf metal.MTLBuffer
	for range 3 {
		qBuf, kBuf, vBuf, _, err = scratch.buffers(q, k, v)
		if err != nil {
			t.Fatalf("SDPA scratch buffers: %v", err)
		}
	}
	if got, want := uintptr(qBuf.Contents()), uintptr(unsafe.Pointer(&q[0])); got != want {
		t.Fatalf("q buffer pointer = %#x, want caller backing %#x", got, want)
	}
	if got, want := uintptr(kBuf.Contents()), uintptr(unsafe.Pointer(&k[0])); got != want {
		t.Fatalf("k buffer pointer = %#x, want caller backing %#x", got, want)
	}
	if got, want := uintptr(vBuf.Contents()), uintptr(unsafe.Pointer(&v[0])); got != want {
		t.Fatalf("v buffer pointer = %#x, want caller backing %#x", got, want)
	}
}

func TestSDPAAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const b, nHeads, nKV, headDim, kvLen = 1, 8, 4, 64, 16
	q := toBF16Bytes(syntheticFloat32(b*nHeads*headDim, 3))
	k := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 5))
	v := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 7))
	if _, err := SDPA(q, k, v, b, nHeads, nKV, headDim, kvLen, 0.125); err != nil {
		t.Fatalf("SDPA warmup: %v", err)
	}

	var sdpaErr error
	allocs := testing.AllocsPerRun(5, func() {
		_, sdpaErr = SDPA(q, k, v, b, nHeads, nKV, headDim, kvLen, 0.125)
	})
	if sdpaErr != nil {
		t.Fatalf("SDPA: %v", sdpaErr)
	}
	if allocs > 10 {
		t.Fatalf("SDPA allocations = %.0f, want <= 10", allocs)
	}
}

// TestSdpa_SDPAInto_Good proves SDPAInto returns caller-owned output backing and matches the
// allocating wrapper byte-for-byte.
func TestSdpa_SDPAInto_Good(t *testing.T) {
	requireNativeRuntime(t)

	const b, nHeads, nKV, headDim, kvLen = 1, 8, 4, 64, 16
	q := toBF16Bytes(syntheticFloat32(b*nHeads*headDim, 3))
	k := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 5))
	v := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 7))
	out := make([]byte, b*nHeads*headDim*bf16Size)
	for i := range out {
		out[i] = 0xA5
	}

	got, err := SDPAInto(out, q, k, v, b, nHeads, nKV, headDim, kvLen, 0.125)
	if err != nil {
		t.Fatalf("SDPAInto: %v", err)
	}
	if len(got) != len(out) {
		t.Fatalf("SDPAInto len = %d, want %d", len(got), len(out))
	}
	if unsafe.Pointer(&got[0]) != unsafe.Pointer(&out[0]) {
		t.Fatal("SDPAInto did not return caller-owned output backing")
	}
	want, err := SDPA(q, k, v, b, nHeads, nKV, headDim, kvLen, 0.125)
	if err != nil {
		t.Fatalf("SDPA reference: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("SDPAInto output differs from allocating wrapper")
	}
}

// TestSdpa_SDPAInto_Bad mirrors SDPA's GQA guard through the caller-output entry point.
func TestSdpa_SDPAInto_Bad(t *testing.T) {
	requireNativeRuntime(t)

	x := toBF16Bytes(syntheticFloat32(64, 3))
	out := make([]byte, 64*bf16Size)
	if _, err := SDPAInto(out, x, x, x, 1, 3, 2, 64, 1, 1); err == nil {
		t.Fatal("expected SDPAInto to reject nHeads not divisible by nKVHeads")
	}
}

// TestSdpa_SDPAInto_Ugly proves the too-small-capacity path: when cap(out) is smaller than the
// required output, SDPAInto must allocate fresh storage rather than write out of bounds, and
// still return the correct attention output.
func TestSdpa_SDPAInto_Ugly(t *testing.T) {
	requireNativeRuntime(t)

	const b, nHeads, nKV, headDim, kvLen = 1, 8, 4, 64, 16
	q := toBF16Bytes(syntheticFloat32(b*nHeads*headDim, 3))
	k := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 5))
	v := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 7))
	want, err := SDPA(q, k, v, b, nHeads, nKV, headDim, kvLen, 0.125)
	if err != nil {
		t.Fatalf("SDPA reference: %v", err)
	}

	tooSmall := make([]byte, 1)
	got, err := SDPAInto(tooSmall, q, k, v, b, nHeads, nKV, headDim, kvLen, 0.125)
	if err != nil {
		t.Fatalf("SDPAInto (undersized out): %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("SDPAInto (undersized out) output differs from allocating wrapper")
	}
}

// TestSdpa_SDPA2Pass_Good cross-checks the two-pass long-context kernel against the proven
// single-pass SDPA at the same inputs — both implement the same online-softmax maths, differing
// only in how the cache reduction parallelises, so they must agree.
func TestSdpa_SDPA2Pass_Good(t *testing.T) {
	requireNativeRuntime(t)

	const b, nHeads, nKV, headDim, kvLen = 1, 4, 2, 64, 8
	scale := float32(0.125)
	q := toBF16Bytes(syntheticFloat32(b*nHeads*headDim, 3))
	k := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 5))
	v := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 7))

	got, err := SDPA2Pass(q, k, v, b, nHeads, nKV, headDim, kvLen, scale)
	if err != nil {
		t.Fatalf("SDPA2Pass: %v", err)
	}
	want, err := SDPA(q, k, v, b, nHeads, nKV, headDim, kvLen, scale)
	if err != nil {
		t.Fatalf("SDPA reference: %v", err)
	}
	if cos := cosineBF16(got, want); cos < 0.999 {
		t.Fatalf("SDPA2Pass cosine=%.6f vs single-pass SDPA, want ~1", cos)
	}
}

// TestSdpa_SDPA2Pass_Bad proves the GQA guard: nHeads must be a multiple of nKVHeads.
func TestSdpa_SDPA2Pass_Bad(t *testing.T) {
	requireNativeRuntime(t)

	x := toBF16Bytes(syntheticFloat32(64, 3))
	if _, err := SDPA2Pass(x, x, x, 1, 3, 2, 64, 1, 1); err == nil {
		t.Fatal("expected SDPA2Pass to reject nHeads not divisible by nKVHeads")
	}
}

// TestSdpa_SDPA2Pass_Ugly proves the batched (b>1) two-pass path: every prior 2-pass test in
// this package runs b=1, so this pins that a batch dimension routes correctly against the
// single-pass reference.
func TestSdpa_SDPA2Pass_Ugly(t *testing.T) {
	requireNativeRuntime(t)

	const b, nHeads, nKV, headDim, kvLen = 3, 4, 2, 64, 8
	scale := float32(0.125)
	q := toBF16Bytes(syntheticFloat32(b*nHeads*headDim, 3))
	k := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 5))
	v := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 7))

	got, err := SDPA2Pass(q, k, v, b, nHeads, nKV, headDim, kvLen, scale)
	if err != nil {
		t.Fatalf("SDPA2Pass (batched): %v", err)
	}
	want, err := SDPA(q, k, v, b, nHeads, nKV, headDim, kvLen, scale)
	if err != nil {
		t.Fatalf("SDPA reference: %v", err)
	}
	if cos := cosineBF16(got, want); cos < 0.999 {
		t.Fatalf("batched SDPA2Pass cosine=%.6f vs single-pass SDPA, want ~1", cos)
	}
}

// TestSdpa_SDPA2PassInto_Good proves SDPA2PassInto returns caller-owned output backing and
// matches the allocating wrapper byte-for-byte.
func TestSdpa_SDPA2PassInto_Good(t *testing.T) {
	requireNativeRuntime(t)

	const b, nHeads, nKV, headDim, kvLen = 1, 4, 2, 64, 8
	scale := float32(0.125)
	q := toBF16Bytes(syntheticFloat32(b*nHeads*headDim, 3))
	k := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 5))
	v := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 7))
	out := make([]byte, b*nHeads*headDim*bf16Size)
	for i := range out {
		out[i] = 0xA5
	}

	got, err := SDPA2PassInto(out, q, k, v, b, nHeads, nKV, headDim, kvLen, scale)
	if err != nil {
		t.Fatalf("SDPA2PassInto: %v", err)
	}
	if unsafe.Pointer(&got[0]) != unsafe.Pointer(&out[0]) {
		t.Fatal("SDPA2PassInto did not return caller-owned output backing")
	}
	want, err := SDPA2Pass(q, k, v, b, nHeads, nKV, headDim, kvLen, scale)
	if err != nil {
		t.Fatalf("SDPA2Pass reference: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("SDPA2PassInto output differs from allocating wrapper")
	}
}

// TestSdpa_SDPA2PassInto_Bad mirrors SDPA2Pass's GQA guard through the caller-output entry point.
func TestSdpa_SDPA2PassInto_Bad(t *testing.T) {
	requireNativeRuntime(t)

	x := toBF16Bytes(syntheticFloat32(64, 3))
	out := make([]byte, 64*bf16Size)
	if _, err := SDPA2PassInto(out, x, x, x, 1, 3, 2, 64, 1, 1); err == nil {
		t.Fatal("expected SDPA2PassInto to reject nHeads not divisible by nKVHeads")
	}
}

// TestSdpa_SDPA2PassInto_Ugly proves the too-small-capacity path: when cap(out) is smaller than
// the required output, SDPA2PassInto must allocate fresh storage rather than write out of
// bounds, and still return the correct attention output.
func TestSdpa_SDPA2PassInto_Ugly(t *testing.T) {
	requireNativeRuntime(t)

	const b, nHeads, nKV, headDim, kvLen = 1, 4, 2, 64, 8
	scale := float32(0.125)
	q := toBF16Bytes(syntheticFloat32(b*nHeads*headDim, 3))
	k := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 5))
	v := toBF16Bytes(syntheticFloat32(b*nKV*kvLen*headDim, 7))
	want, err := SDPA2Pass(q, k, v, b, nHeads, nKV, headDim, kvLen, scale)
	if err != nil {
		t.Fatalf("SDPA2Pass reference: %v", err)
	}

	tooSmall := make([]byte, 1)
	got, err := SDPA2PassInto(tooSmall, q, k, v, b, nHeads, nKV, headDim, kvLen, scale)
	if err != nil {
		t.Fatalf("SDPA2PassInto (undersized out): %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("SDPA2PassInto (undersized out) output differs from allocating wrapper")
	}
}
