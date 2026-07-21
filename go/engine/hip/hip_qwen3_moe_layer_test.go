// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"testing"

	core "dappco.re/go"
)

// TestHIPQwen3MoEApplyQKNormRoPE_Good pins the device path (fused per-head RMSNorm+RoPE,
// the SAME kernel gemma4's own QK-norm uses) against the package's own reference
// functions composed independently: per-head hipReferenceRMSNorm followed by per-head
// hipReferenceRoPENeoXWithFrequencyDim (the split-half convention — see
// hip_qwen3_moe_layer.go's hipRMSNormLaunchFlagRoPENeoX comment for why NeoX, not the
// kernel's other interleaved-pairs branch).
func TestHIPQwen3MoEApplyQKNormRoPE_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	heads, headDim := 2, 4
	values := []float32{1, 0, 3, 4, 0, 2, 5, 12}
	weight := []float32{1, 2, 1, 1}

	normBuf, err := hipQwen3MoEUploadFloat32(driver, "test qnorm weight", weight)
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

// TestHIPQwen3MoEApplyQKNormRoPE_DifferentPositions_Good proves position actually
// changes the rotation (a regression guard against an accidentally-hardcoded position).
func TestHIPQwen3MoEApplyQKNormRoPE_DifferentPositions_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	heads, headDim := 1, 4
	values := []float32{1, 2, 3, 4}
	weight := []float32{1, 1, 1, 1}
	normBuf, err := hipQwen3MoEUploadFloat32(driver, "test qnorm weight", weight)
	core.AssertNoError(t, err)
	defer func() { _ = normBuf.Close() }()

	at0, err := hipQwen3MoEApplyQKNormRoPE(context.Background(), driver, values, normBuf, heads, headDim, 0, 10000, 0)
	core.AssertNoError(t, err)
	at5, err := hipQwen3MoEApplyQKNormRoPE(context.Background(), driver, values, normBuf, heads, headDim, 5, 10000, 0)
	core.AssertNoError(t, err)

	// position 0 applies angle 0 to every pair: rotation is the identity, so at0 must
	// equal the plain per-head RMSNorm with no rotation.
	normed, err := hipReferenceRMSNorm(values, weight, 0)
	core.AssertNoError(t, err)
	assertFloat32SlicesNear(t, normed, at0, 0.0001)

	diff := false
	for i := range at0 {
		if math32Abs(at0[i]-at5[i]) > 0.0001 {
			diff = true
			break
		}
	}
	if !diff {
		t.Fatalf("position 0 and position 5 produced identical output — position is not reaching the kernel")
	}
}

func math32Abs(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

// TestHIPQwen3MoEExpertForward_Good pins the device expert transform (gate/up
// projection, rocm_swiglu's SiLU-gate multiply — its first production caller — down
// projection) against hipQwen3MoESwiGLUHostReference, the plain-float64 host oracle.
func TestHIPQwen3MoEExpertForward_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	d, ff := 2, 3
	x := []float32{1, 2}
	gate := []float32{1, 0, 0, 1, 1, 1} // [ff,d] row-major: rows [1,0] [0,1] [1,1]
	up := []float32{1, 1, 1, 0, 0, 1}   // rows [1,1] [1,0] [0,1]
	down := []float32{1, 0, 1, 0, 1, 1} // [d,ff] row-major: rows [1,0,1] [0,1,1]

	gateBuf, err := hipQwen3MoEUploadFloat32(driver, "test expert gate", gate)
	core.AssertNoError(t, err)
	defer func() { _ = gateBuf.Close() }()
	upBuf, err := hipQwen3MoEUploadFloat32(driver, "test expert up", up)
	core.AssertNoError(t, err)
	defer func() { _ = upBuf.Close() }()
	downBuf, err := hipQwen3MoEUploadFloat32(driver, "test expert down", down)
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

// TestHIPQwen3MoEExpertForward_IndexOutOfRange_Bad rejects a selected expert index
// outside the loaded expert weight slices.
func TestHIPQwen3MoEExpertForward_IndexOutOfRange_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg := hipQwen3MoEConfig{HiddenSize: 2, ExpertFF: 2}
	lw := &hipQwen3MoELayerWeights{ExpertGate: []*hipDeviceByteBuffer{nil}, ExpertUp: []*hipDeviceByteBuffer{nil}, ExpertDown: []*hipDeviceByteBuffer{nil}}

	_, err := hipQwen3MoEExpertForward(context.Background(), driver, cfg, lw, []float32{1, 2}, 5)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "out of range")
}
