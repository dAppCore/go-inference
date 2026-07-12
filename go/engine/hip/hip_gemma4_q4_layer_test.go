// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"math"
	"testing"

	core "dappco.re/go"
)

// TestHIPGemma4Q4Layer_ValueNormDeviceKernel_PerHead_Good pins the gemma4 value
// RMSNorm to a PER-HEAD normalisation (one no-scale RMSNorm over each headDim
// slice), matching the prefill value-norm and the metal reference. Normalising
// over the whole kvDim collapses every kv head under a single RMS — a no-op when
// there is one kv head (E2B) but a ~sqrt(headCount) value shrink plus a head
// rebalance once there are several (the dense 12B's sliding layers carry 8 kv
// heads), which starved attention and drove the 12B into repetition garbage.
func TestHIPGemma4Q4Layer_ValueNormDeviceKernel_PerHead_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	// Two kv heads of DIFFERENT magnitude so per-head and whole-buffer diverge:
	// head0 = {3,4} (rms sqrt(12.5)), head1 = {6,8} (rms sqrt(50)).
	payload, err := hipFloat32Payload([]float32{3, 4, 6, 8})
	core.AssertNoError(t, err)
	value, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "value", payload, 4)
	core.AssertNoError(t, err)
	defer value.Close()

	perHead, err := hipRunGemma4Q4ValueNormDeviceKernel(context.Background(), driver, value, 2, 2, 0)
	core.AssertNoError(t, err)
	defer perHead.Close()
	got, err := hipReadFloat32DeviceOutput(perHead, hipGemma4Q4Layer0Operation, "per-head value norm", 4)
	core.AssertNoError(t, err)
	// head0: 3/sqrt(12.5), 4/sqrt(12.5); head1: 6/sqrt(50), 8/sqrt(50).
	assertFloat32SlicesNear(t, []float32{0.8485, 1.1314, 0.8485, 1.1314}, got, 0.0001)

	// The pre-fix whole-kvDim norm folds all four values under one RMS
	// (sqrt(31.25)=5.5902) and diverges from the per-head result — the bug.
	reupload, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "value", payload, 4)
	core.AssertNoError(t, err)
	defer reupload.Close()
	whole, err := hipRunGemma4Q4RMSNormNoScaleDeviceKernel(context.Background(), driver, reupload, 0)
	core.AssertNoError(t, err)
	defer whole.Close()
	wholeGot, err := hipReadFloat32DeviceOutput(whole, hipGemma4Q4Layer0Operation, "whole-buffer value norm", 4)
	core.AssertNoError(t, err)
	assertFloat32SlicesNear(t, []float32{0.5367, 0.7155, 1.0733, 1.4311}, wholeGot, 0.0001)
	if math.Abs(float64(got[3]-wholeGot[3])) < 0.1 {
		t.Fatalf("per-head and whole-buffer value norm must differ for multi-kv-head values: per-head=%v whole=%v", got, wholeGot)
	}

	core.AssertEqual(t, hipKernelNameRMSNormHeads, driver.launches[len(driver.launches)-2].Name)
	core.AssertEqual(t, uint32(2), driver.launches[len(driver.launches)-2].GridX)
}

// TestHIPGemma4Q4Layer_ValueNormDeviceKernel_ShapeMismatch_Bad rejects a value
// buffer whose length is not headDim*headCount.
func TestHIPGemma4Q4Layer_ValueNormDeviceKernel_ShapeMismatch_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	payload, err := hipFloat32Payload([]float32{1, 2, 3})
	core.AssertNoError(t, err)
	value, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "value", payload, 3)
	core.AssertNoError(t, err)
	defer value.Close()

	_, err = hipRunGemma4Q4ValueNormDeviceKernel(context.Background(), driver, value, 2, 2, 0)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "shape mismatch")
}
