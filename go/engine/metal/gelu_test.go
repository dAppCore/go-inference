// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"os"
	"testing"
)

// geluRefF32 is the CPU reference for the tanh-approximation GELU, the exact
// composition native.Gelu drives on the GPU. It anchors the Good cases so a
// both-paths-return-zero false pass cannot slip through.
func geluRefF32(x float32) float32 {
	x3 := x * x * x
	inner := x + 0.044715*x3
	t := float32(math.Tanh(float64(0.7978845608028654 * inner)))
	return 0.5 * x * (1 + t)
}

// TestGelu_Gelu_Good drives the composed GPU GELU over a deterministic spread and
// checks each element against the CPU reference within fp32 tolerance.
func TestGelu_Gelu_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Fatal(err)
	}
	const n = 1024
	x := make([]float32, n)
	for i := range x {
		x[i] = float32((i*37)%160-80) * 0.05
	}
	got, err := Gelu(x)
	if err != nil {
		t.Fatalf("Gelu: %v", err)
	}
	if len(got) != n {
		t.Fatalf("Gelu length: got %d want %d", len(got), n)
	}
	var maxMag float64
	for i := range x {
		ref := geluRefF32(x[i])
		if d := got[i] - ref; d > 1e-2 || d < -1e-2 {
			t.Fatalf("Gelu wrong at [%d]: gpu %v, cpu-ref %v", i, got[i], ref)
		}
		if m := math.Abs(float64(ref)); m > maxMag {
			maxMag = m
		}
	}
	if maxMag < 1e-3 {
		t.Fatalf("Gelu reference ~zero (maxMag %g) — kernel not exercised", maxMag)
	}
}

// TestGelu_Gelu_Bad feeds the empty input: a degenerate-but-valid shape that must
// return an empty result, not panic on the &x[0] address-of.
func TestGelu_Gelu_Bad(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Fatal(err)
	}
	got, err := Gelu(nil)
	if err != nil {
		t.Fatalf("Gelu(nil): %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("Gelu(nil) length: got %d want 0", len(got))
	}
}

// TestGelu_Gelu_Ugly checks GELU at zero — gelu(0) = 0 exactly — and at a large
// positive value, where the tanh saturates to +1 and gelu(x) -> x.
func TestGelu_Gelu_Ugly(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Fatal(err)
	}
	got, err := Gelu([]float32{0, 8})
	if err != nil {
		t.Fatalf("Gelu: %v", err)
	}
	if got[0] != 0 {
		t.Fatalf("Gelu(0) = %v, want 0", got[0])
	}
	if d := got[1] - 8; d > 1e-2 || d < -1e-2 {
		t.Fatalf("Gelu(8) = %v, want ~8 (saturated)", got[1])
	}
}

// TestGelu_GeluGateMul_Good checks gelu(gate)*up against the composed CPU
// reference over a deterministic spread.
func TestGelu_GeluGateMul_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Fatal(err)
	}
	const n = 512
	gate := make([]float32, n)
	up := make([]float32, n)
	for i := range gate {
		gate[i] = float32((i*31)%120-60) * 0.05
		up[i] = float32((i*17)%90-45) * 0.04
	}
	got, err := GeluGateMul(gate, up)
	if err != nil {
		t.Fatalf("GeluGateMul: %v", err)
	}
	var maxMag float64
	for i := range gate {
		ref := geluRefF32(gate[i]) * up[i]
		if d := got[i] - ref; d > 1e-2 || d < -1e-2 {
			t.Fatalf("GeluGateMul wrong at [%d]: gpu %v, cpu-ref %v", i, got[i], ref)
		}
		if m := math.Abs(float64(ref)); m > maxMag {
			maxMag = m
		}
	}
	if maxMag < 1e-3 {
		t.Fatalf("GeluGateMul reference ~zero (maxMag %g) — kernel not exercised", maxMag)
	}
}

// TestGelu_GeluGateMul_Bad feeds empty gate and up: a degenerate-but-valid shape.
func TestGelu_GeluGateMul_Bad(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Fatal(err)
	}
	got, err := GeluGateMul(nil, nil)
	if err != nil {
		t.Fatalf("GeluGateMul(nil,nil): %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("GeluGateMul(nil,nil) length: got %d want 0", len(got))
	}
}

// TestGelu_GeluGateMul_Ugly multiplies a saturated gate by a zero up vector — the
// gate path is fully exercised but the product must be exactly zero everywhere.
func TestGelu_GeluGateMul_Ugly(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Fatal(err)
	}
	got, err := GeluGateMul([]float32{8, -8, 1}, []float32{0, 0, 0})
	if err != nil {
		t.Fatalf("GeluGateMul: %v", err)
	}
	for i, v := range got {
		if v != 0 {
			t.Fatalf("GeluGateMul gate*0 at [%d] = %v, want 0", i, v)
		}
	}
}
