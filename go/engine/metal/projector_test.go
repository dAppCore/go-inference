// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func TestProjectorHasVReflectsOptionalWeight(t *testing.T) {
	if (bf16Projector{}).hasV() {
		t.Fatal("bf16Projector without wV reported hasV")
	}
	if (qmvProjector{}).hasV() {
		t.Fatal("qmvProjector without V weight reported hasV")
	}
	requireNativeRuntime(t)
	if !(bf16Projector{wV: copyView(toBF16Bytes([]float32{1}))}).hasV() {
		t.Fatal("bf16Projector with wV did not report hasV")
	}
	qw := quantWeightFixture(t, 64, 64, 64, 4, 3)
	qv := qmvWeight{wq: copyView(qw.Packed), scales: copyView(qw.Scales), biases: copyView(qw.Biases)}
	if !(qmvProjector{v: qv}).hasV() {
		t.Fatal("qmvProjector with V weight did not report hasV")
	}
}

func TestProjectorRejectsBadProjectionIndex(t *testing.T) {
	if err := (bf16Projector{}).project(nil, nil, nil, 0, projIndex(99)); err == nil {
		t.Fatal("expected bf16Projector to reject bad projection index")
	}
	if err := (qmvProjector{}).project(nil, nil, nil, 0, projIndex(99)); err == nil {
		t.Fatal("expected qmvProjector to reject bad projection index")
	}
}
