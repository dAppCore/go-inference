// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func TestSquareICBMatchesUnarySquare(t *testing.T) {
	requireNativeRuntime(t)
	in := []float32{1, -2, 3, -4}
	got, err := squareICB(in)
	if err != nil {
		t.Fatalf("squareICB: %v", err)
	}
	for i, v := range in {
		if want := v * v; got[i] != want {
			t.Fatalf("squareICB[%d] = %v, want %v", i, got[i], want)
		}
	}
}

func TestSquareICBAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	in := syntheticFloat32(64, 19)
	if _, err := squareICB(in); err != nil {
		t.Fatalf("squareICB warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := squareICB(in); err != nil {
			t.Fatalf("squareICB: %v", err)
		}
	})
	if allocs > 155 {
		t.Fatalf("squareICB allocations = %.0f, want <= 155", allocs)
	}
}

func TestGemvICBMatchesMatVec(t *testing.T) {
	requireNativeRuntime(t)
	const outDim, inDim = 16, 64
	mat := syntheticFloat32(outDim*inDim, 37)
	vec := syntheticFloat32(inDim, 53)
	want, err := MatVec(mat, vec, outDim, inDim)
	if err != nil {
		t.Fatalf("MatVec: %v", err)
	}
	got, err := gemvICB(mat, vec, outDim, inDim)
	if err != nil {
		t.Fatalf("gemvICB: %v", err)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("gemvICB[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestGemvICBAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const outDim, inDim = 16, 64
	mat := syntheticFloat32(outDim*inDim, 37)
	vec := syntheticFloat32(inDim, 53)
	if _, err := gemvICB(mat, vec, outDim, inDim); err != nil {
		t.Fatalf("gemvICB warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := gemvICB(mat, vec, outDim, inDim); err != nil {
			t.Fatalf("gemvICB: %v", err)
		}
	})
	if allocs > 180 {
		t.Fatalf("gemvICB allocations = %.0f, want <= 180", allocs)
	}
}

func TestRebindProbeICBWritesEachReplayRow(t *testing.T) {
	requireNativeRuntime(t)
	const outDim, inDim, nRows = 16, 64, 3
	mat := syntheticFloat32(outDim*inDim, 37)
	vec := syntheticFloat32(inDim, 53)
	want, err := MatVec(mat, vec, outDim, inDim)
	if err != nil {
		t.Fatalf("MatVec: %v", err)
	}
	got, err := rebindProbeICB(mat, vec, outDim, inDim, nRows)
	if err != nil {
		t.Fatalf("rebindProbeICB: %v", err)
	}
	for row := 0; row < nRows; row++ {
		for i := range want {
			if got[row*outDim+i] != want[i] {
				t.Fatalf("rebind row %d value %d = %v, want %v", row, i, got[row*outDim+i], want[i])
			}
		}
	}
}

func TestRebindProbeICBAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const outDim, inDim, nRows = 16, 64, 3
	mat := syntheticFloat32(outDim*inDim, 37)
	vec := syntheticFloat32(inDim, 53)
	if _, err := rebindProbeICB(mat, vec, outDim, inDim, nRows); err != nil {
		t.Fatalf("rebindProbeICB warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := rebindProbeICB(mat, vec, outDim, inDim, nRows); err != nil {
			t.Fatalf("rebindProbeICB: %v", err)
		}
	})
	if allocs > 220 {
		t.Fatalf("rebindProbeICB allocations = %.0f, want <= 220", allocs)
	}
}

func TestQMVICBMatchesQMVBF16(t *testing.T) {
	requireNativeRuntime(t)
	const outDim, inDim, groupSize, bits = 16, 64, 32, 4
	qw := quantWeightFixture(t, outDim, inDim, groupSize, bits, 37)
	x := toBF16Bytes(syntheticFloat32(inDim, 53))
	want, err := QMVBF16(x, qw.Packed, qw.Scales, qw.Biases, outDim, inDim, groupSize, bits)
	if err != nil {
		t.Fatalf("QMVBF16: %v", err)
	}
	got, err := qmvICB(x, qw.Packed, qw.Scales, qw.Biases, outDim, inDim, groupSize, bits)
	if err != nil {
		t.Fatalf("qmvICB: %v", err)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("qmvICB byte %d = %#x, want %#x", i, got[i], want[i])
		}
	}
}

func TestQMVICBAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const outDim, inDim, groupSize, bits = 16, 64, 32, 4
	qw := quantWeightFixture(t, outDim, inDim, groupSize, bits, 37)
	x := toBF16Bytes(syntheticFloat32(inDim, 53))
	if _, err := qmvICB(x, qw.Packed, qw.Scales, qw.Biases, outDim, inDim, groupSize, bits); err != nil {
		t.Fatalf("qmvICB warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := qmvICB(x, qw.Packed, qw.Scales, qw.Biases, outDim, inDim, groupSize, bits); err != nil {
			t.Fatalf("qmvICB: %v", err)
		}
	})
	if allocs > 155 {
		t.Fatalf("qmvICB allocations = %.0f, want <= 155", allocs)
	}
}

func TestRopeFreqsPipelineICBBuildsVariants(t *testing.T) {
	requireNativeRuntime(t)
	for _, traditional := range []bool{false, true} {
		pso, err := ropeFreqsPipelineICB(traditional)
		if err != nil {
			t.Fatalf("ropeFreqsPipelineICB(%v): %v", traditional, err)
		}
		if pso == nil || pso.GetID() == 0 {
			t.Fatalf("ropeFreqsPipelineICB(%v) returned nil pipeline", traditional)
		}
	}
}

func TestRoPEPipelineICBWarmedLookupAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	if _, err := ropePipelineICB(false); err != nil {
		t.Fatalf("ropePipelineICB warmup: %v", err)
	}

	var pipeErr error
	allocs := testing.AllocsPerRun(10, func() {
		_, pipeErr = ropePipelineICB(false)
	})
	if pipeErr != nil {
		t.Fatalf("ropePipelineICB: %v", pipeErr)
	}
	if allocs > 0 {
		t.Fatalf("ropePipelineICB warmed lookup allocations = %.0f, want 0", allocs)
	}
}

func TestRoPEFreqsPipelineICBWarmedLookupAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	if _, err := ropeFreqsPipelineICB(false); err != nil {
		t.Fatalf("ropeFreqsPipelineICB warmup: %v", err)
	}

	var pipeErr error
	allocs := testing.AllocsPerRun(10, func() {
		_, pipeErr = ropeFreqsPipelineICB(false)
	})
	if pipeErr != nil {
		t.Fatalf("ropeFreqsPipelineICB: %v", pipeErr)
	}
	if allocs > 0 {
		t.Fatalf("ropeFreqsPipelineICB warmed lookup allocations = %.0f, want 0", allocs)
	}
}
