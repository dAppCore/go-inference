// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"testing"

	core "dappco.re/go"
)

// TestQuantizeLane_RegisterQuantizeLane_Good covers the ordinary case: a
// registered lane's Detect matches its own config.json and lookupQuantizeLane
// resolves it.
func TestQuantizeLane_RegisterQuantizeLane_Good(t *testing.T) {
	RegisterQuantizeLane("lane-good9", QuantizeLane{
		Detect: func(configJSON []byte) bool { return string(configJSON) == "lane-good9-config" },
		Quantize: func(Source, []byte, []DenseSafetensor, QuantizeFormat) ([]Tensor, []MetadataEntry, error) {
			return []Tensor{{Name: "lane-good9-marker"}}, nil, nil
		},
	})
	lane, ok := lookupQuantizeLane([]byte("lane-good9-config"))
	if !ok {
		t.Fatal("lookupQuantizeLane(lane-good9-config): not found, want registered lane-good9")
	}
	tensors, _, err := lane.Quantize(Source{}, nil, nil, "")
	if err != nil || len(tensors) != 1 || tensors[0].Name != "lane-good9-marker" {
		t.Fatalf("resolved lane.Quantize = (%v, %v), want the lane-good9 marker tensor", tensors, err)
	}
}

// TestQuantizeLane_RegisterQuantizeLane_Bad covers no lane matching: an
// unrecognised config.json misses every registered Detect and
// lookupQuantizeLane reports ok=false.
func TestQuantizeLane_RegisterQuantizeLane_Bad(t *testing.T) {
	if _, ok := lookupQuantizeLane([]byte("never-registered-lane-config")); ok {
		t.Fatal("lookupQuantizeLane(unrecognised config) = found, want ok=false")
	}
}

// TestQuantizeLane_RegisterQuantizeLane_Ugly covers re-registration: the
// registry is Open (overwrite), so registering the SAME name twice replaces
// the prior lane rather than erroring or duplicating — matching
// model.RegisterArch's re-registration semantics one level up.
func TestQuantizeLane_RegisterQuantizeLane_Ugly(t *testing.T) {
	RegisterQuantizeLane("lane-ugly9", QuantizeLane{
		Detect:         func(configJSON []byte) bool { return string(configJSON) == "lane-ugly9-config" },
		SupportsFormat: func(QuantizeFormat) bool { return false },
	})
	RegisterQuantizeLane("lane-ugly9", QuantizeLane{
		Detect:         func(configJSON []byte) bool { return string(configJSON) == "lane-ugly9-config" },
		SupportsFormat: func(QuantizeFormat) bool { return true },
	})
	lane, ok := lookupQuantizeLane([]byte("lane-ugly9-config"))
	if !ok {
		t.Fatal("lookupQuantizeLane(lane-ugly9-config): not found after re-registration")
	}
	if !lane.SupportsFormat(QuantizeQ8_0) {
		t.Fatal("re-registration did not overwrite: SupportsFormat still the first lane's")
	}
}

// TestQuantizeLane_lookupQuantizeLane_FirstMatchWins covers ordering: when two
// registered lanes would both match the same config.json, lookupQuantizeLane
// resolves the one registered first (Registry.Each's insertion order), not
// the last.
func TestQuantizeLane_lookupQuantizeLane_FirstMatchWins(t *testing.T) {
	matchBoth := func(configJSON []byte) bool { return string(configJSON) == "lane-order9-config" }
	RegisterQuantizeLane("lane-order9-first", QuantizeLane{
		Detect:         matchBoth,
		SupportsFormat: func(QuantizeFormat) bool { return true },
	})
	RegisterQuantizeLane("lane-order9-second", QuantizeLane{
		Detect:         matchBoth,
		SupportsFormat: func(QuantizeFormat) bool { return false },
	})
	lane, ok := lookupQuantizeLane([]byte("lane-order9-config"))
	if !ok {
		t.Fatal("lookupQuantizeLane(lane-order9-config): not found")
	}
	if !lane.SupportsFormat(QuantizeQ8_0) {
		t.Fatal("lookupQuantizeLane resolved the second-registered lane, want the first")
	}
}

// TestQuantizeLane_lookupQuantizeLane_NilDetectSkipped covers the documented
// nil-Detect guard: a registered lane with no Detect func can never match
// (lookupQuantizeLane must not dereference a nil Detect) and a real lane
// registered after it still resolves correctly — the nil entry does not
// short-circuit or panic the scan.
func TestQuantizeLane_lookupQuantizeLane_NilDetectSkipped(t *testing.T) {
	RegisterQuantizeLane("lane-nildetect9-a", QuantizeLane{}) // Detect is nil
	RegisterQuantizeLane("lane-nildetect9-b", QuantizeLane{
		Detect:         func(configJSON []byte) bool { return string(configJSON) == "lane-nildetect9-config" },
		SupportsFormat: func(QuantizeFormat) bool { return true },
	})
	lane, ok := lookupQuantizeLane([]byte("lane-nildetect9-config"))
	if !ok {
		t.Fatal("lookupQuantizeLane(lane-nildetect9-config): not found — a preceding nil-Detect entry must not block later lanes")
	}
	if !lane.SupportsFormat(QuantizeQ8_0) {
		t.Fatal("lookupQuantizeLane resolved the wrong lane")
	}
	if _, ok := lookupQuantizeLane([]byte("no-lane-should-match-this-unique-string-9")); ok {
		t.Fatal("lookupQuantizeLane matched a nil-Detect lane against an unrelated config")
	}
}

// TestQuantizeModelPack_RegisteredLane_Good proves QuantizeModelPack's
// dispatch (quantize.go) actually routes through a registered lane end-to-end
// rather than the generic per-tensor path, using a fake lane so this package
// (model/gguf) never needs to import an arch to prove the seam works
// (AX-8) — model/gemma4/gguf's real registration is verified separately
// (relocation #59 receipts) by running the actual lem quant -gguf CLI.
func TestQuantizeModelPack_RegisteredLane_Good(t *testing.T) {
	dir := t.TempDir()
	if result := core.WriteFile(core.PathJoin(dir, "config.json"), []byte(`{"model_type":"quantizelanetest9"}`), 0o644); !result.OK {
		t.Fatalf("write config.json: %v", result.Err())
	}
	writeTestSafetensors(t, core.PathJoin(dir, "model.safetensors"), map[string][]float32{"w": {1, 2, 3, 4}}, map[string][]int{"w": {4}})

	RegisterQuantizeLane("quantizelanetest9", QuantizeLane{
		Detect:         func(configJSON []byte) bool { return string(configJSON) == `{"model_type":"quantizelanetest9"}` },
		SupportsFormat: func(format QuantizeFormat) bool { return format == QuantizeQ8_0 },
		UnsupportedFormatError: func(format QuantizeFormat) error {
			return core.NewError("quantizelanetest9: unsupported " + string(format))
		},
		Quantize: func(source Source, configJSON []byte, tensors []DenseSafetensor, format QuantizeFormat) ([]Tensor, []MetadataEntry, error) {
			return []Tensor{{Name: "lane-marker.weight", Type: TensorTypeF32, Shape: []uint64{1}, Data: []byte{0, 0, 0, 0}}},
				[]MetadataEntry{{Key: "general.architecture", ValueType: ValueTypeString, Value: "quantizelanetest9"}},
				nil
		},
	})

	out := core.PathJoin(dir, "out")
	result, err := QuantizeModelPack(nil, QuantizeOptions{
		SourcePack: Source{Root: dir, WeightFiles: []string{core.PathJoin(dir, "model.safetensors")}},
		OutputPath: out,
		Format:     QuantizeQ8_0,
	})
	if err != nil {
		t.Fatalf("QuantizeModelPack: %v", err)
	}
	if result.Info.Architecture != "quantizelanetest9" {
		t.Fatalf("Info.Architecture = %q, want quantizelanetest9 (proves the registered lane, not the generic path, produced this GGUF)", result.Info.Architecture)
	}
	if result.QuantizedTensors != 1 {
		t.Fatalf("QuantizedTensors = %d, want 1 (the lane's single marker tensor)", result.QuantizedTensors)
	}

	// The rejected-format path: the SAME lane, a format its SupportsFormat
	// refuses, must surface the lane's own UnsupportedFormatError verbatim
	// rather than the generic pipeline running anyway.
	_, err = QuantizeModelPack(nil, QuantizeOptions{
		SourcePack: Source{Root: dir, WeightFiles: []string{core.PathJoin(dir, "model.safetensors")}},
		OutputPath: core.PathJoin(dir, "out-rejected"),
		Format:     QuantizeQ4_0,
	})
	if err == nil || !core.Contains(err.Error(), "quantizelanetest9: unsupported q4_0") {
		t.Fatalf("QuantizeModelPack(unsupported format) error = %v, want the lane's UnsupportedFormatError", err)
	}
}
