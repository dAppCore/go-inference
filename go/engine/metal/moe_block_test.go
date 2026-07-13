// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"os"
	"testing"
	"unsafe"

	"github.com/tmc/apple/metal"
)

// buildMoEWeights makes a MoELayerWeights with deterministic pseudo-random bf16
// weights of the correct shapes — a fixture for the executor-wiring test.
func buildMoEWeights(numExperts, topK, dModel, dFF, expertDFF, salt int) *MoELayerWeights {
	gen := func(n, s int) []byte {
		f := make([]float32, n)
		for i := range f {
			f[i] = float32((i*s+13)%97-48) * 0.02
		}
		return toBF16Bytes(f)
	}
	scale := make([]float32, numExperts)
	for i := range scale {
		scale[i] = 0.5 + float32(i)*0.1
	}
	return &MoELayerWeights{
		NumExperts: numExperts, TopK: topK, ExpertDFF: expertDFF,
		PreFFNormW: gen(dModel, salt+1), PreFFNorm2W: gen(dModel, salt+2),
		PostFFNorm1W: gen(dModel, salt+3), PostFFNorm2W: gen(dModel, salt+4),
		PostFFNormW: gen(dModel, salt+5),
		WGate:       gen(dFF*dModel, salt+6), WUp: gen(dFF*dModel, salt+7), WDown: gen(dModel*dFF, salt+8),
		RouterNormWScaled: gen(dModel, salt+9), RouterW: gen(numExperts*dModel, salt+10),
		PerExpertScale: toBF16Bytes(scale),
		ExpGateW:       gen(numExperts*expertDFF*dModel, salt+11), ExpUpW: gen(numExperts*expertDFF*dModel, salt+12),
		ExpDownW: gen(numExperts*dModel*expertDFF, salt+13),
	}
}

// moeBlockRef is the oracle for MoEBlockBF16: it rebuilds BOTH branches from the
// parity-proven primitives (local MLP inline; expert branch via moeExpertsRef) and
// wires the five norms + dual-branch sum + residual exactly as
// pkg/metal/model/gemma4 decoder_layer.go's MoE branch. It calls the SAME MoERouter
// as the block, so the expert accumulation order — and thus the bf16 rounding —
// matches, allowing a byte-for-byte gate.
func moeBlockRef(t *testing.T, h []byte, w MoELayerWeights, dModel, dFF int, eps float32) []byte {
	t.Helper()
	numExperts, topK, expertDFF := w.NumExperts, w.TopK, w.ExpertDFF
	must := func(b []byte, err error) []byte {
		if err != nil {
			t.Fatalf("moeBlockRef op: %v", err)
		}
		return b
	}
	idx, weights, err := MoERouter(h, w.RouterNormWScaled, w.RouterW, w.PerExpertScale, numExperts, topK, dModel, eps)
	if err != nil {
		t.Fatalf("moeBlockRef router: %v", err)
	}
	// local dense MLP branch, rebuilt from primitives (no residual).
	h1In := must(RMSNormBF16(h, w.PreFFNormW, 1, dModel, eps))
	g := must(MatVecBF16(w.WGate, h1In, dFF, dModel))
	u := must(MatVecBF16(w.WUp, h1In, dFF, dModel))
	h1 := must(MatVecBF16(w.WDown, must(GeluGateMulBF16(g, u)), dModel, dFF))
	// expert branch on the separately-normed input, rebuilt via moeExpertsRef.
	h2In := must(RMSNormBF16(h, w.PreFFNorm2W, 1, dModel, eps))
	h2 := moeExpertsRef(t, h2In, idx, weights, w.ExpGateW, w.ExpUpW, w.ExpDownW, numExperts, topK, dModel, expertDFF)
	// independent norms, sum, post-norm, residual.
	h1n := must(RMSNormBF16(h1, w.PostFFNorm1W, 1, dModel, eps))
	h2n := must(RMSNormBF16(h2, w.PostFFNorm2W, 1, dModel, eps))
	ff := must(RMSNormBF16(must(AddBF16(h1n, h2n)), w.PostFFNormW, 1, dModel, eps))
	return must(AddBF16(h, ff))
}

// denseFFNRef is the NON-MoE feed-forward (decoder_layer.go's else branch): a single
// dense MLP, rms(h, PreFFNorm) → MLP → rms(·, PostFFNorm) → + h. Used only to prove
// MoEBlockBF16's expert branch genuinely contributes (the dual-branch output must
// differ from running the local MLP alone).
func denseFFNRef(t *testing.T, h []byte, w MoELayerWeights, dModel, dFF int, eps float32) []byte {
	t.Helper()
	must := func(b []byte, err error) []byte {
		if err != nil {
			t.Fatalf("denseFFNRef op: %v", err)
		}
		return b
	}
	ffIn := must(RMSNormBF16(h, w.PreFFNormW, 1, dModel, eps))
	ff := must(mlpTransformBF16(ffIn, w.WGate, w.WUp, w.WDown, dModel, dFF))
	return must(AddBF16(h, must(RMSNormBF16(ff, w.PostFFNormW, 1, dModel, eps))))
}

// TestMoEBlock gates the dual-branch MoE feed-forward composition. MoEBlockBF16 is
// byte-for-byte the independent reference that rebuilds both branches from primitives
// and wires the five norms + dual-branch sum + residual per the metal rule — proving
// the WIRING (each norm in the right place, both branches live, the single residual)
// since the sub-ops are individually gated elsewhere. A non-vacuous check confirms
// the expert branch actually contributes: the dual-branch output differs from the
// dense-MLP-only FFN. Local dFF and expertDFF deliberately differ (catch a dim mixup).
func TestMoEBlock(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const numExperts, topK, dModel, dFF, expertDFF = 8, 2, 256, 512, 384
	const eps = float32(1e-6)
	mk := func(n, salt int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32((i*salt+13)%97-48) * 0.02
		}
		return s
	}
	h := toBF16Bytes(mk(dModel, 29))
	w := MoELayerWeights{
		NumExperts:        numExperts,
		TopK:              topK,
		ExpertDFF:         expertDFF,
		PreFFNormW:        toBF16Bytes(mk(dModel, 3)),
		PreFFNorm2W:       toBF16Bytes(mk(dModel, 5)),
		PostFFNorm1W:      toBF16Bytes(mk(dModel, 7)),
		PostFFNorm2W:      toBF16Bytes(mk(dModel, 11)),
		PostFFNormW:       toBF16Bytes(mk(dModel, 13)),
		WGate:             toBF16Bytes(mk(dFF*dModel, 17)),
		WUp:               toBF16Bytes(mk(dFF*dModel, 19)),
		WDown:             toBF16Bytes(mk(dModel*dFF, 23)),
		RouterNormWScaled: toBF16Bytes(mk(dModel, 2)),
		RouterW:           toBF16Bytes(mk(numExperts*dModel, 43)),
		PerExpertScale:    toBF16Bytes([]float32{1.0, 0.5, 2.0, 0.25, 1.5, 0.75, 3.0, 0.1}),
		ExpGateW:          toBF16Bytes(mk(numExperts*expertDFF*dModel, 53)),
		ExpUpW:            toBF16Bytes(mk(numExperts*expertDFF*dModel, 71)),
		ExpDownW:          toBF16Bytes(mk(numExperts*dModel*expertDFF, 47)),
	}

	got, err := MoEBlockBF16(h, w, dModel, dFF, eps)
	if err != nil {
		t.Fatalf("MoEBlockBF16: %v", err)
	}
	want := moeBlockRef(t, h, w, dModel, dFF, eps)
	eqBytes(t, "MoEBlockBF16", got, want)

	// non-vacuous: the dual-branch output must differ from the dense-MLP-only FFN
	// (i.e. the expert branch is genuinely summed in, not silently dropped).
	dense := denseFFNRef(t, h, w, dModel, dFF, eps)
	same := len(dense) == len(got)
	for i := range got {
		if i < len(dense) && got[i] != dense[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("MoEBlockBF16 output equals the dense-MLP-only FFN — the expert branch did not contribute")
	}
	t.Logf("MoEBlock (%d experts, top-%d, dFF %d / expertDFF %d): dual-branch ≡ composed reference and differs from dense-only FFN", numExperts, topK, dFF, expertDFF)
}

func TestMLPTransformBF16AllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const dModel, dFF = 64, 128
	x := toBF16Bytes(syntheticFloat32(dModel, 37))
	wGate := toBF16Bytes(syntheticFloat32(dFF*dModel, 17))
	wUp := toBF16Bytes(syntheticFloat32(dFF*dModel, 19))
	wDown := toBF16Bytes(syntheticFloat32(dModel*dFF, 23))
	if _, err := mlpTransformBF16(x, wGate, wUp, wDown, dModel, dFF); err != nil {
		t.Fatalf("mlpTransformBF16 warmup: %v", err)
	}

	var transformErr error
	allocs := testing.AllocsPerRun(5, func() {
		_, transformErr = mlpTransformBF16(x, wGate, wUp, wDown, dModel, dFF)
	})
	if transformErr != nil {
		t.Fatalf("mlpTransformBF16: %v", transformErr)
	}
	if allocs > 582 {
		t.Fatalf("mlpTransformBF16 allocations = %.0f, want <= 582", allocs)
	}
}

func TestMLPTransformBF16WritesDirectlyToReturnedOutput(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, dFF = 64, 128
	x := toBF16Bytes(syntheticFloat32(dModel, 37))
	wGate := toBF16Bytes(syntheticFloat32(dFF*dModel, 17))
	wUp := toBF16Bytes(syntheticFloat32(dFF*dModel, 19))
	wDown := toBF16Bytes(syntheticFloat32(dModel*dFF, 23))

	scratch, err := getMLPTransformScratch(dModel, dFF)
	if err != nil {
		t.Fatalf("getMLPTransformScratch: %v", err)
	}
	scratchOut := unsafe.Slice((*byte)(scratch.mlp.down.Contents()), dModel*bf16Size)
	sentinel := bytes.Repeat([]byte{0x7d}, len(scratchOut))
	copy(scratchOut, sentinel)
	putMLPTransformScratch(scratch)

	got, err := mlpTransformBF16(x, wGate, wUp, wDown, dModel, dFF)
	if err != nil {
		t.Fatalf("mlpTransformBF16: %v", err)
	}
	want, err := mlpTransformBF16Into(make([]byte, dModel*bf16Size), x, wGate, wUp, wDown, dModel, dFF)
	if err != nil {
		t.Fatalf("mlpTransformBF16Into reference: %v", err)
	}
	eqBytes(t, "mlpTransformBF16 direct output", got, want)

	scratch, err = getMLPTransformScratch(dModel, dFF)
	if err != nil {
		t.Fatalf("getMLPTransformScratch after call: %v", err)
	}
	defer putMLPTransformScratch(scratch)
	scratchOut = unsafe.Slice((*byte)(scratch.mlp.down.Contents()), dModel*bf16Size)
	if !bytes.Equal(scratchOut, sentinel) {
		t.Fatal("mlpTransformBF16 wrote through pooled scratch output instead of returned output")
	}
}

func TestMLPTransformScratchPoolKeepsDimensionsResident(t *testing.T) {
	requireNativeRuntime(t)

	small, err := getMLPTransformScratch(64, 128)
	if err != nil {
		t.Fatalf("get small MLPTransform scratch: %v", err)
	}
	putMLPTransformScratch(small)
	large, err := getMLPTransformScratch(96, 192)
	if err != nil {
		t.Fatalf("get large MLPTransform scratch: %v", err)
	}
	putMLPTransformScratch(large)
	forceNativeGC()
	forceNativeGC()

	gotSmall, err := getMLPTransformScratch(64, 128)
	if err != nil {
		t.Fatalf("get small MLPTransform scratch again: %v", err)
	}
	defer putMLPTransformScratch(gotSmall)
	if gotSmall != small {
		t.Fatal("MLPTransform scratch pool evicted the small dimension after using a larger dimension")
	}
	gotLarge, err := getMLPTransformScratch(96, 192)
	if err != nil {
		t.Fatalf("get large MLPTransform scratch again: %v", err)
	}
	defer putMLPTransformScratch(gotLarge)
	if gotLarge != large {
		t.Fatal("MLPTransform scratch pool evicted the large dimension after reusing the small dimension")
	}
}

func TestMLPTransformMegaScratchPoolKeepsDimensionsResident(t *testing.T) {
	requireNativeRuntime(t)

	small, err := getMLPTransformMegaScratch(64, 128)
	if err != nil {
		t.Fatalf("get small MLPTransformMega scratch: %v", err)
	}
	putMLPTransformMegaScratch(small)
	large, err := getMLPTransformMegaScratch(96, 192)
	if err != nil {
		t.Fatalf("get large MLPTransformMega scratch: %v", err)
	}
	putMLPTransformMegaScratch(large)
	forceNativeGC()
	forceNativeGC()

	gotSmall, err := getMLPTransformMegaScratch(64, 128)
	if err != nil {
		t.Fatalf("get small MLPTransformMega scratch again: %v", err)
	}
	defer putMLPTransformMegaScratch(gotSmall)
	if gotSmall != small {
		t.Fatal("MLPTransformMega scratch pool evicted the small dimension after using a larger dimension")
	}
	gotLarge, err := getMLPTransformMegaScratch(96, 192)
	if err != nil {
		t.Fatalf("get large MLPTransformMega scratch again: %v", err)
	}
	defer putMLPTransformMegaScratch(gotLarge)
	if gotLarge != large {
		t.Fatal("MLPTransformMega scratch pool evicted the large dimension after reusing the small dimension")
	}
}

func TestMoEBlockBF16AllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const numExperts, topK, dModel, dFF, expertDFF = 4, 2, 64, 128, 96
	const eps = float32(1e-5)
	h := toBF16Bytes(syntheticFloat32(dModel, 29))
	w := *buildMoEWeights(numExperts, topK, dModel, dFF, expertDFF, 3)
	if _, err := MoEBlockBF16(h, w, dModel, dFF, eps); err != nil {
		t.Fatalf("MoEBlockBF16 warmup: %v", err)
	}

	var blockErr error
	allocs := testing.AllocsPerRun(3, func() {
		_, blockErr = MoEBlockBF16(h, w, dModel, dFF, eps)
	})
	if blockErr != nil {
		t.Fatalf("MoEBlockBF16: %v", blockErr)
	}
	if allocs > 4 {
		t.Fatalf("MoEBlockBF16 allocations = %.0f, want <= 4", allocs)
	}
}

func TestMoEBlockBF16IntoWritesDirectlyToCallerOutput(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const numExperts, topK, dModel, dFF, expertDFF = 4, 2, 64, 128, 96
	const eps = float32(1e-5)
	h := toBF16Bytes(syntheticFloat32(dModel, 29))
	w := *buildMoEWeights(numExperts, topK, dModel, dFF, expertDFF, 3)
	want, err := MoEBlockBF16(h, w, dModel, dFF, eps)
	if err != nil {
		t.Fatalf("MoEBlockBF16: %v", err)
	}

	scratch, err := getMoEBlockBF16Scratch(dModel, dFF, expertDFF, topK)
	if err != nil {
		t.Fatalf("getMoEBlockBF16Scratch: %v", err)
	}
	sentinel := bytes.Repeat([]byte{0xa5}, len(scratch.out.bytes))
	copy(scratch.out.bytes, sentinel)
	seededScratch := scratch
	putMoEBlockBF16Scratch(scratch)

	out := make([]byte, dModel*bf16Size)
	outPtr := unsafe.Pointer(&out[0])
	got, err := MoEBlockBF16Into(out, h, w, dModel, dFF, eps)
	if err != nil {
		t.Fatalf("MoEBlockBF16Into: %v", err)
	}
	if len(got) != dModel*bf16Size || unsafe.Pointer(&got[0]) != outPtr {
		t.Fatal("MoEBlockBF16Into did not reuse caller-owned output backing")
	}
	eqBytes(t, "MoEBlockBF16Into direct output", got, want)

	if !bytes.Equal(seededScratch.out.bytes, sentinel) {
		t.Fatal("MoEBlockBF16Into wrote through pooled block output instead of caller output")
	}
}

func TestMoEBlockBF16WithBufferOutputWritesDirectlyToProvidedBuffer(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const numExperts, topK, dModel, dFF, expertDFF = 4, 2, 64, 128, 96
	const eps = float32(1e-5)
	h := toBF16Bytes(syntheticFloat32(dModel, 29))
	w := *buildMoEWeights(numExperts, topK, dModel, dFF, expertDFF, 3)
	want, err := MoEBlockBF16(h, w, dModel, dFF, eps)
	if err != nil {
		t.Fatalf("MoEBlockBF16: %v", err)
	}

	scratch, err := getMoEBlockBF16Scratch(dModel, dFF, expertDFF, topK)
	if err != nil {
		t.Fatalf("getMoEBlockBF16Scratch: %v", err)
	}
	sentinel := bytes.Repeat([]byte{0x3c}, len(scratch.out.bytes))
	copy(scratch.out.bytes, sentinel)
	seededScratch := scratch
	putMoEBlockBF16Scratch(scratch)

	input, err := newPinnedNoCopyBytes(len(h))
	if err != nil {
		t.Fatalf("newPinnedNoCopyBytes input: %v", err)
	}
	defer input.Close()
	hBuf, err := input.copyBuffer(h)
	if err != nil {
		t.Fatalf("copy input buffer: %v", err)
	}
	out, err := newPinnedNoCopyBytes(dModel * bf16Size)
	if err != nil {
		t.Fatalf("newPinnedNoCopyBytes output: %v", err)
	}
	defer out.Close()

	if err := moeBlockBF16WithBufferOutputInPool(h, hBuf, out.buf, w, dModel, dFF, eps); err != nil {
		t.Fatalf("moeBlockBF16WithBufferOutputInPool: %v", err)
	}
	eqBytes(t, "MoEBlockBF16 direct Metal output", out.bytes, want)

	if !bytes.Equal(seededScratch.out.bytes, sentinel) {
		t.Fatal("moeBlockBF16WithBufferOutputInPool wrote through pooled block output")
	}
}

func TestMoEBlockScratchInputViewUsesCallerBacking(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, dFF, expertDFF, topK = 64, 128, 96, 2
	h := toBF16Bytes(syntheticFloat32(dModel, 29))
	scratch, err := getMoEBlockBF16Scratch(dModel, dFF, expertDFF, topK)
	if err != nil {
		t.Fatalf("getMoEBlockBF16Scratch: %v", err)
	}
	defer scratch.Close()

	buf, ok := scratch.inputView(h)
	if !ok {
		t.Fatal("inputView ok = false")
	}
	if got, want := uintptr(buf.Contents()), uintptr(unsafe.Pointer(&h[0])); got != want {
		t.Fatalf("inputView buffer pointer = %#x, want caller backing %#x", got, want)
	}
	reused, ok := scratch.inputView(h)
	if !ok {
		t.Fatal("reused inputView ok = false")
	}
	if reused.GetID() != buf.GetID() {
		t.Fatal("inputView did not reuse the cached no-copy buffer")
	}
}

func TestMoEBlockScratchWeightsViewUsesCallerBacking(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, dFF, expertDFF, topK = 64, 128, 96, 2
	weights := toBF16Bytes([]float32{0.75, 0.25})
	scratch, err := getMoEBlockBF16Scratch(dModel, dFF, expertDFF, topK)
	if err != nil {
		t.Fatalf("getMoEBlockBF16Scratch: %v", err)
	}
	defer scratch.Close()

	buf, ok := scratch.weightsView(weights)
	if !ok {
		t.Fatal("weightsView ok = false")
	}
	if got, want := uintptr(buf.Contents()), uintptr(unsafe.Pointer(&weights[0])); got != want {
		t.Fatalf("weightsView buffer pointer = %#x, want caller backing %#x", got, want)
	}
	reused, ok := scratch.weightsView(weights)
	if !ok {
		t.Fatal("reused weightsView ok = false")
	}
	if reused.GetID() != buf.GetID() {
		t.Fatal("weightsView did not reuse the cached no-copy buffer")
	}
}

func TestMoEBlockBF16AfterRouterRejectsInvalidInputs(t *testing.T) {
	requireNativeRuntime(t)

	const numExperts, topK, dModel, dFF, expertDFF = 4, 2, 64, 128, 96
	const eps = float32(1e-5)
	h := toBF16Bytes(syntheticFloat32(dModel, 29))
	idx := []int32{0, 1}
	weights := toBF16Bytes([]float32{0.75, 0.25})
	w := *buildMoEWeights(numExperts, topK, dModel, dFF, expertDFF, 3)
	if _, err := moeBlockBF16AfterRouter(h[:len(h)-bf16Size], idx, weights, nil, w, dModel, dFF, eps); err == nil {
		t.Fatal("expected moeBlockBF16AfterRouter to reject short residual")
	}
	bad := w
	bad.ExpGateW = bad.ExpGateW[:len(bad.ExpGateW)-bf16Size]
	if _, err := moeBlockBF16AfterRouter(h, idx, weights, nil, bad, dModel, dFF, eps); err == nil {
		t.Fatal("expected moeBlockBF16AfterRouter to reject short expert gate weight")
	}
	if _, err := moeBlockBF16AfterRouter(nil, nil, nil, nil, MoELayerWeights{}, 0, 0, eps); err != nil {
		t.Fatalf("moeBlockBF16AfterRouter zero dimensions: %v", err)
	}
}

func TestMoEBlockBF16AfterRouterUsesProvidedHiddenBuffer(t *testing.T) {
	requireNativeRuntime(t)

	const numExperts, topK, dModel, dFF, expertDFF = 4, 2, 64, 128, 96
	const eps = float32(1e-5)
	hostH := toBF16Bytes(syntheticFloat32(dModel, 7))
	bufferH := toBF16Bytes(syntheticFloat32(dModel, 29))
	idx := []int32{0, 1}
	weights := toBF16Bytes([]float32{0.75, 0.25})
	w := *buildMoEWeights(numExperts, topK, dModel, dFF, expertDFF, 3)

	pinned, err := newPinnedNoCopyBytes(len(bufferH))
	if err != nil {
		t.Fatalf("newPinnedNoCopyBytes: %v", err)
	}
	defer pinned.Close()
	hBuf, err := pinned.copyBuffer(bufferH)
	if err != nil {
		t.Fatalf("copyBuffer: %v", err)
	}

	want, err := moeBlockBF16AfterRouter(bufferH, idx, weights, nil, w, dModel, dFF, eps)
	if err != nil {
		t.Fatalf("moeBlockBF16AfterRouter: %v", err)
	}
	got, err := moeBlockBF16AfterRouterWithBuffer(hostH, hBuf, idx, weights, nil, w, dModel, dFF, eps)
	if err != nil {
		t.Fatalf("moeBlockBF16AfterRouterWithBuffer: %v", err)
	}
	eqBytes(t, "MoEBlockBF16 provided hidden buffer", got, want)
}

func TestMoEBlockBF16ScratchClose(t *testing.T) {
	requireNativeRuntime(t)

	s, err := newMoEBlockBF16Scratch(64, 128, 96, 2)
	if err != nil {
		t.Fatalf("newMoEBlockBF16Scratch: %v", err)
	}
	if s.h == nil || s.h.buf == nil || s.weights == nil || s.weights.buf == nil || s.out == nil || s.out.buf == nil {
		t.Fatal("newMoEBlockBF16Scratch did not allocate pinned buffers")
	}
	s.Close()
	if s.h != nil || s.weights != nil || s.out != nil || s.dModel != 0 || s.dFF != 0 || s.expertDFF != 0 || s.topK != 0 {
		t.Fatal("Close did not clear pinned buffers and dimensions")
	}
	s.Close()
}

func TestMoEBlockBF16ScratchPoolKeepsShapesResident(t *testing.T) {
	requireNativeRuntime(t)

	small, err := getMoEBlockBF16Scratch(64, 128, 96, 2)
	if err != nil {
		t.Fatalf("get small MoEBlockBF16 scratch: %v", err)
	}
	putMoEBlockBF16Scratch(small)
	large, err := getMoEBlockBF16Scratch(96, 192, 144, 3)
	if err != nil {
		t.Fatalf("get large MoEBlockBF16 scratch: %v", err)
	}
	putMoEBlockBF16Scratch(large)
	forceNativeGC()
	forceNativeGC()

	gotSmall, err := getMoEBlockBF16Scratch(64, 128, 96, 2)
	if err != nil {
		t.Fatalf("get small MoEBlockBF16 scratch again: %v", err)
	}
	defer putMoEBlockBF16Scratch(gotSmall)
	if gotSmall != small {
		t.Fatal("MoEBlockBF16 scratch pool evicted the small shape after using a larger shape")
	}
	gotLarge, err := getMoEBlockBF16Scratch(96, 192, 144, 3)
	if err != nil {
		t.Fatalf("get large MoEBlockBF16 scratch again: %v", err)
	}
	defer putMoEBlockBF16Scratch(gotLarge)
	if gotLarge != large {
		t.Fatal("MoEBlockBF16 scratch pool evicted the large shape after reusing the small shape")
	}
}

func TestMoEBlockPostCombineRejectsInvalidInputs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel = 64
	h := toBF16Bytes(syntheticFloat32(dModel, 29))
	h1 := toBF16Bytes(syntheticFloat32(dModel, 31))
	h2 := toBF16Bytes(syntheticFloat32(dModel, 37))
	post1 := toBF16Bytes(syntheticFloat32(dModel, 41))
	post2 := toBF16Bytes(syntheticFloat32(dModel, 43))
	post := toBF16Bytes(syntheticFloat32(dModel, 47))
	if _, err := moeBlockPostCombineBF16(h[:len(h)-bf16Size], h1, h2, post1, bufView{}, post2, bufView{}, post, bufView{}, dModel, 1e-5); err == nil {
		t.Fatal("expected moeBlockPostCombineBF16 to reject short residual")
	}
	if _, err := moeBlockPostCombineBF16(h, h1, h2, post1[:len(post1)-bf16Size], bufView{}, post2, bufView{}, post, bufView{}, dModel, 1e-5); err == nil {
		t.Fatal("expected moeBlockPostCombineBF16 to reject short post norm")
	}
	zero, err := moeBlockPostCombineBF16(nil, nil, nil, nil, bufView{}, nil, bufView{}, nil, bufView{}, 0, 1e-5)
	if err != nil {
		t.Fatalf("moeBlockPostCombineBF16 zero dimensions: %v", err)
	}
	if len(zero) != 0 {
		t.Fatalf("moeBlockPostCombineBF16 zero dimensions len = %d, want 0", len(zero))
	}
}

func TestMoEBlockPostCombineScratchClose(t *testing.T) {
	requireNativeRuntime(t)

	s, err := newMoEBlockPostCombineScratch(64)
	if err != nil {
		t.Fatalf("newMoEBlockPostCombineScratch: %v", err)
	}
	if s.h == nil || s.h1 == nil || s.h2 == nil || s.out == nil {
		t.Fatal("newMoEBlockPostCombineScratch did not allocate pinned buffers")
	}
	s.Close()
	if s.h != nil || s.h1 != nil || s.h2 != nil || s.out != nil || s.dModel != 0 {
		t.Fatal("Close did not clear pinned buffers and dimensions")
	}
	s.Close()
}

func TestMoEBlockPostCombineScratchPoolKeepsDimensionsResident(t *testing.T) {
	requireNativeRuntime(t)

	small, err := getMoEBlockPostCombineScratch(64)
	if err != nil {
		t.Fatalf("get small MoEBlockPostCombine scratch: %v", err)
	}
	putMoEBlockPostCombineScratch(small)
	large, err := getMoEBlockPostCombineScratch(96)
	if err != nil {
		t.Fatalf("get large MoEBlockPostCombine scratch: %v", err)
	}
	putMoEBlockPostCombineScratch(large)
	forceNativeGC()
	forceNativeGC()

	gotSmall, err := getMoEBlockPostCombineScratch(64)
	if err != nil {
		t.Fatalf("get small MoEBlockPostCombine scratch again: %v", err)
	}
	defer putMoEBlockPostCombineScratch(gotSmall)
	if gotSmall != small {
		t.Fatal("MoEBlockPostCombine scratch pool evicted the small dimension after using a larger dimension")
	}
	gotLarge, err := getMoEBlockPostCombineScratch(96)
	if err != nil {
		t.Fatalf("get large MoEBlockPostCombine scratch again: %v", err)
	}
	defer putMoEBlockPostCombineScratch(gotLarge)
	if gotLarge != large {
		t.Fatal("MoEBlockPostCombine scratch pool evicted the large dimension after reusing the small dimension")
	}
}

func TestMoEBlockPostCombineScratchInputViewsUseCallerBacking(t *testing.T) {
	requireNativeRuntime(t)

	const dModel = 64
	h := toBF16Bytes(syntheticFloat32(dModel, 29))
	h1 := toBF16Bytes(syntheticFloat32(dModel, 31))
	h2 := toBF16Bytes(syntheticFloat32(dModel, 37))
	scratch, err := getMoEBlockPostCombineScratch(dModel)
	if err != nil {
		t.Fatalf("getMoEBlockPostCombineScratch: %v", err)
	}
	defer scratch.Close()

	cases := []struct {
		name string
		in   []byte
		view func([]byte) (metal.MTLBuffer, bool)
	}{
		{name: "residual", in: h, view: scratch.residualView},
		{name: "branch1", in: h1, view: scratch.branch1View},
		{name: "branch2", in: h2, view: scratch.branch2View},
	}
	for _, tc := range cases {
		buf, ok := tc.view(tc.in)
		if !ok {
			t.Fatalf("%s view ok = false", tc.name)
		}
		if got, want := uintptr(buf.Contents()), uintptr(unsafe.Pointer(&tc.in[0])); got != want {
			t.Fatalf("%s view buffer pointer = %#x, want caller backing %#x", tc.name, got, want)
		}
		reused, ok := tc.view(tc.in)
		if !ok {
			t.Fatalf("reused %s view ok = false", tc.name)
		}
		if reused.GetID() != buf.GetID() {
			t.Fatalf("%s view did not reuse the cached no-copy buffer", tc.name)
		}
	}
}

func TestMoEBlockBF16CachesLocalDenseWeightsWithExperts(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const numExperts, topK, dModel, dFF, expertDFF = 4, 2, 64, 128, 96
	h := toBF16Bytes(syntheticFloat32(dModel, 29))
	w := moeLayerWeightsFixture(numExperts, topK, dModel, dFF, expertDFF, 3)
	idx, _, err := MoERouter(h, w.RouterNormWScaled, w.RouterW, w.PerExpertScale, numExperts, topK, dModel, 1e-5)
	if err != nil {
		t.Fatalf("MoERouter: %v", err)
	}
	resetResidentBufsForTest()

	if _, err := MoEBlockBF16(h, w, dModel, dFF, 1e-5); err != nil {
		t.Fatalf("MoEBlockBF16: %v", err)
	}

	key := func(b []byte) uintptr {
		return uintptr(unsafe.Pointer(&b[0]))
	}
	residentBufMu.Lock()
	got := len(residentBufs)
	required := map[uintptr]string{
		key(w.WGate):    "local gate",
		key(w.WUp):      "local up",
		key(w.WDown):    "local down",
		key(w.ExpGateW): "expert gate",
		key(w.ExpUpW):   "expert up",
		key(w.ExpDownW): "expert down",
	}
	missing := []string{}
	for k, name := range required {
		if _, ok := residentBufs[k]; !ok {
			missing = append(missing, name)
		}
	}
	expertGateSz, expertDownSz := expertDFF*dModel*bf16Size, dModel*expertDFF*bf16Size
	selectedSliceHits := 0
	for _, e32 := range idx {
		e := int(e32)
		if _, ok := residentBufs[key(w.ExpGateW[e*expertGateSz:(e+1)*expertGateSz])]; ok {
			selectedSliceHits++
		}
		if _, ok := residentBufs[key(w.ExpUpW[e*expertGateSz:(e+1)*expertGateSz])]; ok {
			selectedSliceHits++
		}
		if _, ok := residentBufs[key(w.ExpDownW[e*expertDownSz:(e+1)*expertDownSz])]; ok {
			selectedSliceHits++
		}
	}
	residentBufMu.Unlock()

	if len(missing) > 0 {
		t.Fatalf("MoEBlockBF16 missing resident weights %v (resident=%d)", missing, got)
	}
	if selectedSliceHits > 0 {
		t.Fatalf("MoEBlockBF16 cached %d selected expert slices; want whole expert tensors only", selectedSliceHits)
	}
	if got < len(required) {
		t.Fatalf("resident weights = %d, want at least %d local dense + whole expert tensors", got, len(required))
	}
}

type encMoEBlockQuantDeviceFixture struct {
	h             []byte
	w             MoEQuantLayerWeights
	input, output *pinnedNoCopyBytes
	router        *routerDeviceScratch
	scratch       *moeBlockBF16Scratch
}

func newEncMoEBlockQuantDeviceFixture(t *testing.T, bits int, fusedExperts bool) *encMoEBlockQuantDeviceFixture {
	t.Helper()
	const dModel, dFF, expertDFF, numExperts, topK, groupSize = 64, 128, 96, 4, 2, 32
	f := &encMoEBlockQuantDeviceFixture{
		h: toBF16Bytes(syntheticFloat32(dModel, 29)),
		w: quantMoELayerWeightsGuard(t, numExperts, topK, dModel, dFF, expertDFF, groupSize, bits),
	}
	if fusedExperts {
		f.w.ExpGateUp = fuseExpertGateUpQuant(f.w.ExpGate, f.w.ExpUp, numExperts, expertDFF, dModel, groupSize, bits)
		f.w.ExpGate, f.w.ExpUp = QuantWeight{}, QuantWeight{}
	}
	var err error
	f.input, err = newPinnedNoCopyBytes(len(f.h))
	if err != nil {
		t.Fatalf("new pinned MoE input: %v", err)
	}
	if _, err := f.input.copyBuffer(f.h); err != nil {
		f.input.Close()
		t.Fatalf("copy MoE input: %v", err)
	}
	f.output, err = newPinnedNoCopyBytes(dModel * bf16Size)
	if err != nil {
		f.input.Close()
		t.Fatalf("new pinned MoE output: %v", err)
	}
	f.router, err = getRouterDeviceScratch(dModel, numExperts, topK)
	if err != nil {
		f.input.Close()
		f.output.Close()
		t.Fatalf("get router device scratch: %v", err)
	}
	f.scratch, err = getMoEBlockBF16Scratch(dModel, dFF, expertDFF, topK)
	if err != nil {
		putRouterDeviceScratch(f.router)
		f.input.Close()
		f.output.Close()
		t.Fatalf("get MoE block scratch: %v", err)
	}
	return f
}

func (f *encMoEBlockQuantDeviceFixture) Close() {
	if f == nil {
		return
	}
	putMoEBlockBF16Scratch(f.scratch)
	putRouterDeviceScratch(f.router)
	if f.output != nil {
		f.output.Close()
	}
	if f.input != nil {
		f.input.Close()
	}
}

func assertEncMoEBlockQuantDeviceOutput(t *testing.T, f *encMoEBlockQuantDeviceFixture) {
	t.Helper()
	const dModel, dFF = 64, 128
	want, err := MoEBlockQuant(f.h, f.w, dModel, dFF, 1e-5)
	if err != nil {
		t.Fatalf("MoEBlockQuant host reference: %v", err)
	}
	if cos := cosineBF16(f.output.bytes, want); cos < 0.9999 {
		t.Fatalf("encoded device MoE block cosine = %.6f, want >= 0.9999", cos)
	}
}

func TestEncMoEBlockQuantDevice_B2(t *testing.T) {
	requireNativeRuntime(t)
	f := newEncMoEBlockQuantDeviceFixture(t, 2, false)
	defer f.Close()
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	before := moeConcurrentBlocks.Load()
	enc, encConc, handled, err := encMoEBlockQuantDevice(enc, cb, nil, false, f.router, f.scratch, f.input.buf, f.output.buf, f.w, 64, 128, 1e-5)
	if err != nil || !handled {
		endEncodingFast(enc)
		t.Fatalf("encMoEBlockQuantDevice b2: handled=%v err=%v", handled, err)
	}
	if !encConc || moeConcurrentBlocks.Load() == before {
		endEncodingFast(enc)
		t.Fatalf("encMoEBlockQuantDevice b2 did not engage the concurrent encoder (encConc=%v)", encConc)
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	assertEncMoEBlockQuantDeviceOutput(t, f)
}

func TestEncMoEBlockQuantDevice_B3(t *testing.T) {
	requireNativeRuntime(t)
	f := newEncMoEBlockQuantDeviceFixture(t, 3, false)
	defer f.Close()
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	enc, _, handled, err := encMoEBlockQuantDevice(enc, cb, nil, false, f.router, f.scratch, f.input.buf, f.output.buf, f.w, 64, 128, 1e-5)
	if err != nil || !handled {
		endEncodingFast(enc)
		t.Fatalf("encMoEBlockQuantDevice b3: handled=%v err=%v", handled, err)
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	assertEncMoEBlockQuantDeviceOutput(t, f)
}

func TestEncMoEBlockQuantDevice_B4(t *testing.T) {
	requireNativeRuntime(t)
	f := newEncMoEBlockQuantDeviceFixture(t, 4, false)
	defer f.Close()
	oldConcurrentDisabled := moeConcurrentDisabled
	moeConcurrentDisabled = true
	defer func() { moeConcurrentDisabled = oldConcurrentDisabled }()
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	enc, encConc, handled, err := encMoEBlockQuantDevice(enc, cb, nil, false, f.router, f.scratch, f.input.buf, f.output.buf, f.w, 64, 128, 1e-5)
	if err != nil || !handled {
		endEncodingFast(enc)
		t.Fatalf("encMoEBlockQuantDevice b4: handled=%v err=%v", handled, err)
	}
	if encConc {
		endEncodingFast(enc)
		t.Fatal("encMoEBlockQuantDevice b4 left a concurrent encoder open while concurrency was disabled")
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	assertEncMoEBlockQuantDeviceOutput(t, f)
}

func TestEncMoEBlockQuantDevice_B5(t *testing.T) {
	requireNativeRuntime(t)
	f := newEncMoEBlockQuantDeviceFixture(t, 5, false)
	defer f.Close()
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	enc, _, handled, err := encMoEBlockQuantDevice(enc, cb, nil, false, f.router, f.scratch, f.input.buf, f.output.buf, f.w, 64, 128, 1e-5)
	if err != nil || !handled {
		endEncodingFast(enc)
		t.Fatalf("encMoEBlockQuantDevice b5: handled=%v err=%v", handled, err)
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	assertEncMoEBlockQuantDeviceOutput(t, f)
}

func TestEncMoEBlockQuantDevice_B6(t *testing.T) {
	requireNativeRuntime(t)
	f := newEncMoEBlockQuantDeviceFixture(t, 6, false)
	defer f.Close()
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	enc, _, handled, err := encMoEBlockQuantDevice(enc, cb, nil, false, f.router, f.scratch, f.input.buf, f.output.buf, f.w, 64, 128, 1e-5)
	if err != nil || !handled {
		endEncodingFast(enc)
		t.Fatalf("encMoEBlockQuantDevice b6: handled=%v err=%v", handled, err)
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	assertEncMoEBlockQuantDeviceOutput(t, f)
}

func TestEncMoEBlockQuantDevice_B8(t *testing.T) {
	requireNativeRuntime(t)
	f := newEncMoEBlockQuantDeviceFixture(t, 8, false)
	defer f.Close()
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	enc, _, handled, err := encMoEBlockQuantDevice(enc, cb, nil, false, f.router, f.scratch, f.input.buf, f.output.buf, f.w, 64, 128, 1e-5)
	if err != nil || !handled {
		endEncodingFast(enc)
		t.Fatalf("encMoEBlockQuantDevice b8: handled=%v err=%v", handled, err)
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	assertEncMoEBlockQuantDeviceOutput(t, f)
}

func TestEncMoEBlockQuantDevice_FusedExpertGateUp(t *testing.T) {
	requireNativeRuntime(t)
	f := newEncMoEBlockQuantDeviceFixture(t, 4, true)
	defer f.Close()
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	enc, _, handled, err := encMoEBlockQuantDevice(enc, cb, nil, false, f.router, f.scratch, f.input.buf, f.output.buf, f.w, 64, 128, 1e-5)
	if err != nil || !handled {
		endEncodingFast(enc)
		t.Fatalf("encMoEBlockQuantDevice fused gate_up: handled=%v err=%v", handled, err)
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	assertEncMoEBlockQuantDeviceOutput(t, f)
}

func TestEncMoEBlockQuantDevice_NilBuffersDecline(t *testing.T) {
	var inputEnc metal.MTLComputeCommandEncoderObject
	var cb metal.MTLCommandBufferObject
	enc, encConc, handled, err := encMoEBlockQuantDevice(inputEnc, cb, nil, true, nil, nil, nil, nil, MoEQuantLayerWeights{}, 64, 128, 1e-5)
	if enc.GetID() != 0 || !encConc || handled || err != nil {
		t.Fatalf("encMoEBlockQuantDevice nil buffers = encoder id=%d encConc=%v handled=%v err=%v, want 0/true/false/nil", enc.GetID(), encConc, handled, err)
	}
}

func TestEncMoEBlockQuantDevice_SizeFloorDeclines(t *testing.T) {
	requireNativeRuntime(t)
	f := newEncMoEBlockQuantDeviceFixture(t, 4, false)
	defer f.Close()
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	enc, _, handled, err := encMoEBlockQuantDevice(enc, cb, nil, false, f.router, f.scratch, f.input.buf, f.output.buf, f.w, 0, 128, 1e-5)
	endEncodingFast(enc)
	if handled || err != nil {
		t.Fatalf("encMoEBlockQuantDevice zero dModel = handled=%v err=%v, want decline", handled, err)
	}
}

func TestEncMoEBlockQuantDevice_InvalidLocalWeightDeclines(t *testing.T) {
	requireNativeRuntime(t)
	f := newEncMoEBlockQuantDeviceFixture(t, 4, false)
	defer f.Close()
	f.w.LocalGate.Packed = f.w.LocalGate.Packed[:len(f.w.LocalGate.Packed)-1]
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	enc, _, handled, err := encMoEBlockQuantDevice(enc, cb, nil, false, f.router, f.scratch, f.input.buf, f.output.buf, f.w, 64, 128, 1e-5)
	endEncodingFast(enc)
	if handled || err != nil {
		t.Fatalf("encMoEBlockQuantDevice malformed local weight = handled=%v err=%v, want decline", handled, err)
	}
}

func TestEncMoEBlockQuantDevice_MissingPipelineDeclines(t *testing.T) {
	requireNativeRuntime(t)
	f := newEncMoEBlockQuantDeviceFixture(t, 4, false)
	defer f.Close()
	withWrongMainLibrary(t, func() {
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		enc, _, handled, err := encMoEBlockQuantDevice(enc, cb, nil, false, f.router, f.scratch, f.input.buf, f.output.buf, f.w, 64, 128, 1e-5)
		endEncodingFast(enc)
		if handled || err != nil {
			t.Fatalf("encMoEBlockQuantDevice missing pipeline = handled=%v err=%v, want decline", handled, err)
		}
	})
}
