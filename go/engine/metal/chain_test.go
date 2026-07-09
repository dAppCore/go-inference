// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func TestNormProjectAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const dIn, dOut = 128, 256
	x := syntheticFloat32(dIn, 3)
	normW := syntheticFloat32(dIn, 5)
	projW := syntheticFloat32(dOut*dIn, 7)
	if _, err := NormProject(x, normW, projW, dIn, dOut, 1e-5); err != nil {
		t.Fatalf("NormProject warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := NormProject(x, normW, projW, dIn, dOut, 1e-5); err != nil {
			t.Fatalf("NormProject: %v", err)
		}
	})
	if allocs > 150 {
		t.Fatalf("NormProject allocations = %.0f, want <= 150", allocs)
	}
}

func TestMLPBlockAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, dFF = 64, 128
	x := syntheticFloat32(dModel, 3)
	normW := syntheticFloat32(dModel, 5)
	wGate := syntheticFloat32(dFF*dModel, 7)
	wUp := syntheticFloat32(dFF*dModel, 11)
	wDown := syntheticFloat32(dModel*dFF, 13)
	if _, err := MLPBlock(x, normW, wGate, wUp, wDown, dModel, dFF, 1e-5); err != nil {
		t.Fatalf("MLPBlock warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := MLPBlock(x, normW, wGate, wUp, wDown, dModel, dFF, 1e-5); err != nil {
			t.Fatalf("MLPBlock: %v", err)
		}
	})
	if allocs > 1140 {
		t.Fatalf("MLPBlock allocations = %.0f, want <= 1140", allocs)
	}
}

func TestNormProjectMatchesComposedOps(t *testing.T) {
	requireNativeRuntime(t)

	x := []float32{3, 4}
	normW := []float32{1, 1}
	projW := []float32{
		1, 0,
		0, 1,
		1, 1,
	}
	got, err := NormProject(x, normW, projW, 2, 3, 0)
	if err != nil {
		t.Fatalf("NormProject: %v", err)
	}
	normed, err := RMSNorm(x, normW, 1, 2, 0)
	if err != nil {
		t.Fatalf("RMSNorm: %v", err)
	}
	want, err := MatVec(projW, normed, 3, 2)
	if err != nil {
		t.Fatalf("MatVec: %v", err)
	}
	assertFloat32Near(t, "NormProject", got, want, 1e-5)
}

func TestMLPBlockMatchesComposedOps(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, dFF = 2, 2
	x := []float32{1, -1}
	normW := []float32{1, 1}
	wGate := []float32{1, 0, 0, 1}
	wUp := []float32{1, 0, 0, 1}
	wDown := []float32{1, 0, 0, 1}
	got, err := MLPBlock(x, normW, wGate, wUp, wDown, dModel, dFF, 0)
	if err != nil {
		t.Fatalf("MLPBlock: %v", err)
	}
	normed, err := RMSNorm(x, normW, 1, dModel, 0)
	if err != nil {
		t.Fatalf("RMSNorm: %v", err)
	}
	gate, err := MatVec(wGate, normed, dFF, dModel)
	if err != nil {
		t.Fatalf("gate MatVec: %v", err)
	}
	up, err := MatVec(wUp, normed, dFF, dModel)
	if err != nil {
		t.Fatalf("up MatVec: %v", err)
	}
	gated, err := GeluGateMul(gate, up)
	if err != nil {
		t.Fatalf("GeluGateMul: %v", err)
	}
	down, err := MatVec(wDown, gated, dModel, dFF)
	if err != nil {
		t.Fatalf("down MatVec: %v", err)
	}
	want, err := Add(x, down)
	if err != nil {
		t.Fatalf("Add: %v", err)
	}
	assertFloat32Near(t, "MLPBlock", got, want, 1e-5)
}

func TestNormProjectRejectsShapeMismatch(t *testing.T) {
	requireNativeRuntime(t)

	if _, err := NormProject([]float32{1}, []float32{1}, []float32{1}, 2, 1, 1e-5); err == nil {
		t.Fatal("expected NormProject to reject mismatched shapes")
	}
}
