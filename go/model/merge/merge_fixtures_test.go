// SPDX-Licence-Identifier: EUPL-1.2

package merge

import (
	"testing"

	core "dappco.re/go"

	"dappco.re/go/inference/model/modelmgmt"
)

func requireResultOK(t testing.TB, r core.Result) {
	t.Helper()
	if !r.OK {
		t.Fatalf("unexpected result error: %s", r.Error())
	}
}

func assertResultError(t testing.TB, r core.Result) {
	t.Helper()
	if r.OK {
		t.Fatalf("expected result error, got OK value %#v", r.Value)
	}
}

// writeSourceFixture builds a minimal model-pack directory (config.json,
// tokenizer.json, model.safetensors) under dir and returns a ready-to-use
// Source. tokenizerContent controls the tokenizer hash used by
// validatePackCompatibility; values become the pack's F32 tensors.
func writeSourceFixture(t testing.TB, dir, architecture, tokenizerContent string, values map[string][]float32) Source {
	t.Helper()
	requireResultOK(t, core.WriteFile(core.PathJoin(dir, "config.json"), []byte(`{"model_type":"test"}`), 0o644))
	tokenizerPath := core.PathJoin(dir, "tokenizer.json")
	requireResultOK(t, core.WriteFile(tokenizerPath, []byte(tokenizerContent), 0o644))

	tensors := make(map[string]modelmgmt.SafetensorsTensorInfo, len(values))
	data := make(map[string][]byte, len(values))
	for name, vals := range values {
		tensors[name] = modelmgmt.SafetensorsTensorInfo{Dtype: "F32", Shape: []int{len(vals)}}
		data[name] = modelmgmt.EncodeFloat32(vals)
	}
	weightPath := core.PathJoin(dir, "model.safetensors")
	requireResultOK(t, modelmgmt.WriteSafetensors(weightPath, tensors, data))

	return Source{
		Root:          dir,
		Architecture:  architecture,
		TokenizerPath: tokenizerPath,
		WeightFiles:   []string{weightPath},
	}
}

// readMergedTensor reads back one F32 tensor from a merged output pack.
func readMergedTensor(t testing.TB, weightPath, name string) []float32 {
	t.Helper()
	read := modelmgmt.ReadSafetensors(weightPath)
	requireResultOK(t, read)
	data := read.Value.(modelmgmt.SafetensorsData)
	info, ok := data.Tensors[name]
	if !ok {
		t.Fatalf("tensor %q not present in merged output", name)
	}
	raw := modelmgmt.GetTensorData(info, data.Data)
	values, err := modelmgmt.DecodeFloat32(info.Dtype, raw, shapeElements(info.Shape))
	if err != nil {
		t.Fatalf("decode merged tensor %q: %v", name, err)
	}
	return values
}
