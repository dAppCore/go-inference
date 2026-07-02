// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import "testing"

// dispatchBlock returns deterministic values sized to one block of the
// format under test (32 for the _0 family, 256 for K formats).
func dispatchBlock(blockSize int) []float32 {
	values := make([]float32, blockSize)
	for i := range values {
		values[i] = float32(i%13) - 6.5
	}
	return values
}

// TestQuantizeDispatch_AppendQuantize_Good — every format's exported
// dispatch is byte-identical to its private kernel, and appending onto
// an existing prefix preserves it.
func TestQuantizeDispatch_AppendQuantize_Good(t *testing.T) {
	kernels := map[QuantizeFormat]func([]float32) []byte{
		QuantizeQ8_0: quantizeQ8_0,
		QuantizeQ4_0: quantizeQ4_0,
		QuantizeQ5_0: quantizeQ5_0,
		QuantizeQ4_K: quantizeQ4_K,
		QuantizeQ5_K: quantizeQ5_K,
		QuantizeQ6_K: quantizeQ6_K,
		QuantizeQ8_K: quantizeQ8_K,
		QuantizeQ3_K: quantizeQ3_K,
		QuantizeQ2_K: quantizeQ2_K,
	}
	for format, kernel := range kernels {
		_, blockSize, _, err := ggufQuantizeLayout(format)
		if err != nil {
			t.Fatalf("%s: layout: %v", format, err)
		}
		values := dispatchBlock(blockSize)
		want := kernel(values)

		got, err := AppendQuantize(format, nil, values)
		if err != nil {
			t.Fatalf("%s: AppendQuantize: %v", format, err)
		}
		if string(got) != string(want) {
			t.Fatalf("%s: dispatch bytes differ from the private kernel", format)
		}

		prefix := []byte{0xAA, 0xBB}
		appended, err := AppendQuantize(format, append([]byte(nil), prefix...), values)
		if err != nil {
			t.Fatalf("%s: AppendQuantize with prefix: %v", format, err)
		}
		if string(appended[:2]) != string(prefix) || string(appended[2:]) != string(want) {
			t.Fatalf("%s: append did not preserve the prefix + payload", format)
		}
	}
}

// TestQuantizeDispatch_Quantize_Good — the convenience form matches the
// append form from a nil destination.
func TestQuantizeDispatch_Quantize_Good(t *testing.T) {
	values := dispatchBlock(32)
	direct, err := Quantize(QuantizeQ8_0, values)
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}
	appended, err := AppendQuantize(QuantizeQ8_0, nil, values)
	if err != nil {
		t.Fatalf("AppendQuantize: %v", err)
	}
	if string(direct) != string(appended) {
		t.Fatal("Quantize and AppendQuantize(nil, …) diverge")
	}
}

// TestQuantizeDispatch_AppendQuantize_Bad — an unknown format is an
// error, never a silent empty payload.
func TestQuantizeDispatch_AppendQuantize_Bad(t *testing.T) {
	if _, err := AppendQuantize(QuantizeFormat("q99_z"), nil, dispatchBlock(32)); err == nil {
		t.Fatal("unknown format quantised without error")
	}
	if _, err := Quantize(QuantizeFormat(""), dispatchBlock(32)); err == nil {
		t.Fatal("empty format quantised without error")
	}
}

// TestQuantizeDispatch_AppendQuantize_Ugly — values not divisible by
// the format's block size are rejected before any kernel runs.
func TestQuantizeDispatch_AppendQuantize_Ugly(t *testing.T) {
	if _, err := AppendQuantize(QuantizeQ8_0, nil, dispatchBlock(31)); err == nil {
		t.Fatal("31 values accepted for a 32-block format")
	}
	if _, err := AppendQuantize(QuantizeQ4_K, nil, dispatchBlock(255)); err == nil {
		t.Fatal("255 values accepted for a 256-block K format")
	}
}
