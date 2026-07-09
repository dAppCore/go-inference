// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"
)

// zz_cover_ensureinit_test.go closes two residual leg families:
//   - the `ensureInit()` failure legs at the top of the ICB probes + profile
//     helpers (squareICB, gemvICB, rebindProbeICB, qmvICB, AttentionBlockICB,
//     NormProjectICB, dispatchProfile, rebindCostProbe, qmvBF16Profile,
//     gemvProfile, mlpTransformBF16/Quant). The per-op guard suite nulls the
//     library but the init once already succeeded, so ensureInit returns its
//     cached nil — these legs need the runtime genuinely un-initialised, which
//     withBrokenRuntime (metallib env unset + init globals reset) provides.
//   - the ICB pipeline-build error legs that fire AFTER ensureInit succeeds: with
//     the library pointed at the wrong metallib the pipelineForICB call inside
//     these probes errors, surfacing the `if err != nil` leg.

// TestCoverEnsureInitLegs covers the ensureInit failure legs by driving each
// init-guarded entry point under a broken runtime.
func TestCoverEnsureInitLegs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, kvLen, dFF = 64, 2, 1, 64, 1, 128
	const gs, bits = 64, 4
	const eps = float32(1e-6)
	f32 := syntheticFloat32(dModel, 3)
	xb := toBF16Bytes(f32)
	mat := syntheticFloat32(dModel*dModel, 5)
	qw := quantWeightFixture(t, dModel, dModel, gs, bits, 7)
	normB := toBF16Bytes(syntheticFloat32(dModel, 9))
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 11)
	kb := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 13))
	vb := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 15))
	moeQ := quantMoELayerWeightsGuard(t, 1, 1, dModel, dFF, dFF, gs, bits)

	withBrokenRuntime(t, func() {
		if _, e := squareICB([]float32{1, 2}); e == nil {
			t.Fatal("squareICB: expected ensureInit failure")
		}
		if _, e := gemvICB(mat, f32, dModel, dModel); e == nil {
			t.Fatal("gemvICB: expected ensureInit failure")
		}
		if _, e := rebindProbeICB(mat, f32, dModel, dModel, 1); e == nil {
			t.Fatal("rebindProbeICB: expected ensureInit failure")
		}
		if _, e := qmvICB(xb, qw.Packed, qw.Scales, qw.Biases, dModel, dModel, gs, bits); e == nil {
			t.Fatal("qmvICB: expected ensureInit failure")
		}
		if _, e := AttentionBlockICB(xb, normB, layer.WQ, layer.WO, kb, vb, dModel, nHeads, nKV, headDim, kvLen, 10000, 0.125, 0, eps, 1); e == nil {
			t.Fatal("AttentionBlockICB: expected ensureInit failure")
		}
		if _, e := NormProjectICB([]float32{1, 2}, []float32{1, 1}, []float32{1, 2, 3, 4}, 2, 2, eps, 1); e == nil {
			t.Fatal("NormProjectICB: expected ensureInit failure")
		}
		if _, _, _, e := dispatchProfile(1, dModel); e == nil {
			t.Fatal("dispatchProfile: expected ensureInit failure")
		}
		if _, e := rebindCostProbe(1); e == nil {
			t.Fatal("rebindCostProbe: expected ensureInit failure")
		}
		if _, _, e := qmvBF16Profile(dModel, dModel, gs, 1); e == nil {
			t.Fatal("qmvBF16Profile: expected ensureInit failure")
		}
		if _, _, e := gemvProfile(dModel, dModel, 1); e == nil {
			t.Fatal("gemvProfile: expected ensureInit failure")
		}
		if _, e := mlpTransformBF16(xb, layer.WGate, layer.WUp, layer.WDown, dModel, dFF); e == nil {
			t.Fatal("mlpTransformBF16: expected ensureInit failure")
		}
		if _, e := mlpTransformQuant(xb, moeQ.LocalGate, moeQ.LocalUp, moeQ.LocalDown, dModel, dFF, gs, bits); e == nil {
			t.Fatal("mlpTransformQuant: expected ensureInit failure")
		}
	})
}

// TestCoverICBProbePipelineBuildLegs covers the `if err != nil` legs after the
// pipelineForICB calls inside gemvICB / rebindProbeICB / qmvICB / AttentionBlockICB
// / NormProjectICB by pointing the library at the wrong metallib (ensureInit has
// already succeeded, so it is the pipeline build that fails).
func TestCoverICBProbePipelineBuildLegs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, kvLen, dFF = 64, 2, 1, 64, 1, 128
	const gs, bits = 64, 4
	const eps = float32(1e-6)
	f32 := syntheticFloat32(dModel, 3)
	xb := toBF16Bytes(f32)
	mat := syntheticFloat32(dModel*dModel, 5)
	qw := quantWeightFixture(t, dModel, dModel, gs, bits, 7)
	normB := toBF16Bytes(syntheticFloat32(dModel, 9))
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 11)
	kb := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 13))
	vb := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 15))

	withWrongMainLibrary(t, func() {
		if _, e := gemvICB(mat, f32, dModel, dModel); e == nil {
			t.Fatal("gemvICB: expected pipeline-build failure")
		}
		if _, e := rebindProbeICB(mat, f32, dModel, dModel, 1); e == nil {
			t.Fatal("rebindProbeICB: expected pipeline-build failure")
		}
		if _, e := qmvICB(xb, qw.Packed, qw.Scales, qw.Biases, dModel, dModel, gs, bits); e == nil {
			t.Fatal("qmvICB: expected pipeline-build failure")
		}
		if _, e := AttentionBlockICB(xb, normB, layer.WQ, layer.WO, kb, vb, dModel, nHeads, nKV, headDim, kvLen, 10000, 0.125, 0, eps, 1); e == nil {
			t.Fatal("AttentionBlockICB: expected pipeline-build failure")
		}
		if _, e := NormProjectICB([]float32{1, 2}, []float32{1, 1}, []float32{1, 2, 3, 4}, 2, 2, eps, 1); e == nil {
			t.Fatal("NormProjectICB: expected pipeline-build failure")
		}
		if _, e := squareICB([]float32{1, 2}); e == nil {
			t.Fatal("squareICB: expected pipeline-build failure")
		}
	})
}

// TestCoverProfilePipelineBuildLegs covers the profile helpers' pipeline-build
// error legs under the wrong library (the guard suite already nulls the library;
// this keeps the float32 gemv/qmv probe build legs covered alongside).
func TestCoverProfilePipelineBuildLegs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, gs = 64, 64
	withWrongMainLibrary(t, func() {
		if _, _, _, e := dispatchProfile(1, dModel); e == nil {
			t.Fatal("dispatchProfile: expected pipeline-build failure")
		}
		if _, e := rebindCostProbe(1); e == nil {
			t.Fatal("rebindCostProbe: expected pipeline-build failure")
		}
		if _, _, e := qmvBF16Profile(dModel, dModel, gs, 1); e == nil {
			t.Fatal("qmvBF16Profile: expected pipeline-build failure")
		}
		if _, _, e := gemvProfile(dModel, dModel, 1); e == nil {
			t.Fatal("gemvProfile: expected pipeline-build failure")
		}
	})
}
