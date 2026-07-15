// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"math"
	"testing"
)

func TestRunUnaryAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	in := syntheticFloat32(1024, 3)
	if _, err := Square(in); err != nil {
		t.Fatalf("Square warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := Square(in); err != nil {
			t.Fatalf("Square: %v", err)
		}
	})
	if allocs > 10 {
		t.Fatalf("Square allocations = %.0f, want <= 10", allocs)
	}
}

func TestRunUnaryBF16AllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	in := toBF16Bytes(syntheticFloat32(1024, 3))
	if _, err := SigmoidBF16(in); err != nil {
		t.Fatalf("SigmoidBF16 warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := SigmoidBF16(in); err != nil {
			t.Fatalf("SigmoidBF16: %v", err)
		}
	})
	if allocs > 10 {
		t.Fatalf("SigmoidBF16 allocations = %.0f, want <= 10", allocs)
	}
}

func TestRunUnaryBF16IntoUsesCallerOutput(t *testing.T) {
	requireNativeRuntime(t)

	in := toBF16Bytes(syntheticFloat32(1024, 3))
	out := make([]byte, len(in))
	for i := range out {
		out[i] = 0xA5
	}

	if err := RunUnaryBF16Into("v_Sigmoidbfloat16bfloat16", in, out); err != nil {
		t.Fatalf("RunUnaryBF16Into: %v", err)
	}
	want, err := SigmoidBF16(in)
	if err != nil {
		t.Fatalf("SigmoidBF16 reference: %v", err)
	}
	if !bytes.Equal(out, want) {
		t.Fatal("RunUnaryBF16Into output differs from allocating wrapper")
	}
}

func TestRunUnaryIntoBypassesScratchOutput(t *testing.T) {
	requireNativeRuntime(t)

	in := syntheticFloat32(1024, 3)
	want, err := Square(in)
	if err != nil {
		t.Fatalf("Square reference: %v", err)
	}

	out := make([]float32, len(in))
	scratch, err := getQMVFloatScratch(len(in), len(in))
	if err != nil {
		t.Fatalf("getQMVFloatScratch: %v", err)
	}
	sentinel := bytes.Repeat([]byte{0xa5}, len(scratch.out.bytes))
	copy(scratch.out.bytes, sentinel)
	putQMVFloatScratch(scratch)

	if err := RunUnaryInto("v_Squarefloat32float32", in, out); err != nil {
		t.Fatalf("RunUnaryInto: %v", err)
	}
	if !bytes.Equal(float32Bytes(out), float32Bytes(want)) {
		t.Fatal("RunUnaryInto output differs from allocating wrapper")
	}

	scratch, err = getQMVFloatScratch(len(in), len(in))
	if err != nil {
		t.Fatalf("getQMVFloatScratch after call: %v", err)
	}
	defer putQMVFloatScratch(scratch)
	if !bytes.Equal(scratch.out.bytes, sentinel) {
		t.Fatal("RunUnaryInto wrote through pooled scratch output instead of caller output")
	}
}

func TestUnaryFloat32Kernels(t *testing.T) {
	requireNativeRuntime(t)

	tests := []struct {
		name string
		in   []float32
		fn   func([]float32) ([]float32, error)
		want []float32
		tol  float32
	}{
		{name: "Square", in: []float32{-3, -2, 0, 4}, fn: Square, want: []float32{9, 4, 0, 16}},
		{name: "Abs", in: []float32{-3, -2, 0, 4}, fn: Abs, want: []float32{3, 2, 0, 4}},
		{name: "Negative", in: []float32{-3, -2, 0, 4}, fn: Negative, want: []float32{3, 2, 0, -4}},
		{name: "Sqrt", in: []float32{1, 4, 9, 16}, fn: Sqrt, want: []float32{1, 2, 3, 4}},
		{name: "Rsqrt", in: []float32{1, 4, 16, 25}, fn: Rsqrt, want: []float32{1, 0.5, 0.25, 0.2}, tol: 1e-6},
		{name: "Log", in: []float32{1, 2, 4, 8}, fn: Log, want: []float32{0, float32(math.Log(2)), float32(math.Log(4)), float32(math.Log(8))}, tol: 1e-6},
		{name: "Exp", in: []float32{-1, 0, 1, 2}, fn: Exp, want: []float32{float32(math.Exp(-1)), 1, float32(math.E), float32(math.Exp(2))}, tol: 1e-5},
		{name: "Sigmoid", in: []float32{-2, 0, 2}, fn: Sigmoid, want: []float32{1 / (1 + float32(math.Exp(2))), 0.5, 1 / (1 + float32(math.Exp(-2)))}, tol: 1e-6},
		{name: "Tanh", in: []float32{-2, 0, 2}, fn: Tanh, want: []float32{float32(math.Tanh(-2)), 0, float32(math.Tanh(2))}, tol: 1e-6},
		{name: "Round", in: []float32{-1.6, -0.4, 0.5, 1.5, 2.6}, fn: Round, want: []float32{-2, 0, 0, 2, 3}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.fn(tt.in)
			if err != nil {
				t.Fatalf("%s: %v", tt.name, err)
			}
			assertFloat32Near(t, tt.name, got, tt.want, tt.tol)
		})
	}
}
