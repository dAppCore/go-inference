// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"os"
	"testing"

	core "dappco.re/go"
	sharedmodel "dappco.re/go/inference/model"
)

// hipQwen3MoESkipUnlessHardware gates a test on the same GO_ROCM_RUN_HIP_TESTS convention
// every other real-GPU test in this package uses (hip_gemma4_q4_oracle_test.go); a custom
// kernel HSACO is resolved automatically from GO_ROCM_KERNEL_HSACO or a packaged sidecar
// (hip_kernel_module.go) — no extra wiring needed once a driver is constructed.
func hipQwen3MoESkipUnlessHardware(t *testing.T) nativeHIPDriver {
	t.Helper()
	if os.Getenv("GO_ROCM_RUN_HIP_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_HIP_TESTS=1 (and GO_ROCM_KERNEL_HSACO=<path>) to run ROCm hardware tests")
	}
	driver := newSystemHIPDriver()
	if driver == nil || !driver.Available() {
		t.Skip("HIP driver is not available on this host")
	}
	return driver
}

// TestHIPQwen3MoEApplyQKNormRoPE_Hardware_Good is TestHIPQwen3MoEApplyQKNormRoPE_Good on
// the real GPU: the fused per-head QK-norm+RoPE kernel dispatched against the compiled
// gfx1101 HSACO, not the fakeHIPDriver host simulation every other test in this package
// runs against by default.
func TestHIPQwen3MoEApplyQKNormRoPE_Hardware_Good(t *testing.T) {
	driver := hipQwen3MoESkipUnlessHardware(t)
	heads, headDim := 2, 4
	values := []float32{1, 0, 3, 4, 0, 2, 5, 12}
	weight := []float32{1, 2, 1, 1}

	normBuf, err := hipQwen3MoEUploadFloat32(driver, "hw test qnorm weight", weight)
	core.AssertNoError(t, err)
	defer func() { _ = normBuf.Close() }()

	position, ropeTheta, eps := 3, float32(10000), float32(0)
	got, err := hipQwen3MoEApplyQKNormRoPE(context.Background(), driver, values, normBuf, heads, headDim, position, ropeTheta, eps)
	core.AssertNoError(t, err)

	var want []float32
	for head := 0; head < heads; head++ {
		start := head * headDim
		normed, err := hipReferenceRMSNorm(values[start:start+headDim], weight, eps)
		core.AssertNoError(t, err)
		rotated, err := hipReferenceRoPENeoXWithFrequencyDim(normed, position, float64(ropeTheta), headDim, headDim)
		core.AssertNoError(t, err)
		want = append(want, rotated...)
	}
	assertFloat32SlicesNear(t, want, got, 0.0001)
}

// TestHIPQwen3MoEExpertForward_Hardware_Good is TestHIPQwen3MoEExpertForward_Good on the
// real GPU: rocm_swiglu's first production caller, dispatched for real — proving the
// kernel that has sat with zero production callers actually computes the right thing on
// gfx1101 hardware, not just under the fake-driver host simulation.
func TestHIPQwen3MoEExpertForward_Hardware_Good(t *testing.T) {
	driver := hipQwen3MoESkipUnlessHardware(t)
	d, ff := 2, 3
	x := []float32{1, 2}
	gate := []float32{1, 0, 0, 1, 1, 1}
	up := []float32{1, 1, 1, 0, 0, 1}
	down := []float32{1, 0, 1, 0, 1, 1}

	gateBuf, err := hipQwen3MoEUploadFloat32(driver, "hw test expert gate", gate)
	core.AssertNoError(t, err)
	defer func() { _ = gateBuf.Close() }()
	upBuf, err := hipQwen3MoEUploadFloat32(driver, "hw test expert up", up)
	core.AssertNoError(t, err)
	defer func() { _ = upBuf.Close() }()
	downBuf, err := hipQwen3MoEUploadFloat32(driver, "hw test expert down", down)
	core.AssertNoError(t, err)
	defer func() { _ = downBuf.Close() }()

	cfg := hipQwen3MoEConfig{HiddenSize: d, ExpertFF: ff}
	lw := &hipQwen3MoELayerWeights{
		ExpertGate: []*hipDeviceByteBuffer{gateBuf},
		ExpertUp:   []*hipDeviceByteBuffer{upBuf},
		ExpertDown: []*hipDeviceByteBuffer{downBuf},
	}

	got, err := hipQwen3MoEExpertForward(context.Background(), driver, cfg, lw, x, 0)
	core.AssertNoError(t, err)

	want := hipQwen3MoESwiGLUHostReference(x, gate, up, down, d, ff)
	assertFloat32SlicesNear(t, want, got, 0.0001)
}

// TestHIPQwen3MoEModel_GenerateGreedy_Hardware_Good is the full greedy-generate smoke
// (TestHIPQwen3MoEModel_GenerateGreedy_Good) driven on real gfx1101 hardware: weight
// upload, embed, N-layer attention (device QKV projection, fused QK-norm+RoPE, causal
// multi-head attention) and MoE FFN (device router projection, host top-k select, device
// SwiGLU expert matvecs), final norm, LM head, and greedy sampling — every device kernel
// this family wires, dispatched for real.
func TestHIPQwen3MoEModel_GenerateGreedy_Hardware_Good(t *testing.T) {
	driver := hipQwen3MoESkipUnlessHardware(t)
	geometry := tinyQwen3MoEGeometry()
	tensors := tinyQwen3MoETensors(geometry.HiddenSize, geometry.VocabSize, geometry.ExpertFF, geometry.Heads, geometry.KVHeads, geometry.HeadDim, geometry.NumExperts, geometry.NumLayers)
	weights, err := loadHIPQwen3MoEWeights(driver, tensors, geometry)
	core.AssertNoError(t, err)
	defer weights.Close()

	model := &hipQwen3MoEModel{driver: driver, cfg: geometry, weights: weights}
	prompt := []int32{1, 2, 3}
	generated, err := sharedmodel.Generate(model, prompt, 4, -1)
	core.AssertNoError(t, err)
	core.AssertEqual(t, 4, len(generated))
	for _, id := range generated {
		if id < 0 || int(id) >= geometry.VocabSize {
			t.Fatalf("generated token %d is out of vocabulary range [0,%d)", id, geometry.VocabSize)
		}
	}
}
