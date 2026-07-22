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

// writeSourceFixtureSharded is writeSourceFixture's multi-shard sibling: it
// splits values round-robin (by sorted tensor name) across len(shardNames)
// safetensors files instead of one model.safetensors — simulating a real HF
// sharded checkpoint (model.safetensors.index.json + N
// model-NNNNN-of-NNNNN.safetensors shards), the layout Packs/ComparePacks
// now stream tensor-by-tensor through Source.WeightFiles rather than
// requiring one blob per source. A shard name with no tensor assigned to it
// (more shard names than tensors) is skipped — neither written nor listed
// in the returned Source's WeightFiles.
func writeSourceFixtureSharded(t testing.TB, dir, architecture, tokenizerContent string, values map[string][]float32, shardNames []string) Source {
	t.Helper()
	requireResultOK(t, core.WriteFile(core.PathJoin(dir, "config.json"), []byte(`{"model_type":"test"}`), 0o644))
	tokenizerPath := core.PathJoin(dir, "tokenizer.json")
	requireResultOK(t, core.WriteFile(tokenizerPath, []byte(tokenizerContent), 0o644))

	names := make([]string, 0, len(values))
	for name := range values {
		names = append(names, name)
	}
	core.SliceSort(names)

	shardValues := make([]map[string][]float32, len(shardNames))
	for i := range shardValues {
		shardValues[i] = map[string][]float32{}
	}
	for i, name := range names {
		shard := i % len(shardNames)
		shardValues[shard][name] = values[name]
	}

	weightFiles := make([]string, 0, len(shardNames))
	for i, shardName := range shardNames {
		if len(shardValues[i]) == 0 {
			continue
		}
		tensors := make(map[string]modelmgmt.SafetensorsTensorInfo, len(shardValues[i]))
		data := make(map[string][]byte, len(shardValues[i]))
		for name, vals := range shardValues[i] {
			tensors[name] = modelmgmt.SafetensorsTensorInfo{Dtype: "F32", Shape: []int{len(vals)}}
			data[name] = modelmgmt.EncodeFloat32(vals)
		}
		path := core.PathJoin(dir, shardName)
		requireResultOK(t, modelmgmt.WriteSafetensors(path, tensors, data))
		weightFiles = append(weightFiles, path)
	}

	return Source{
		Root:          dir,
		Architecture:  architecture,
		TokenizerPath: tokenizerPath,
		WeightFiles:   weightFiles,
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
