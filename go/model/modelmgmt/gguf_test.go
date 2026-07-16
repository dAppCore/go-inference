package modelmgmt

import (
	"dappco.re/go"
	"dappco.re/go/inference/model/gguf"
	coreio "dappco.re/go/io"
)

func writeOllamaManifest(t *core.T, root string, model string) {
	t.Helper()
	manifest := core.JoinPath(root, "manifests", "registry.ollama.ai", "library", model, "latest")
	core.RequireNoError(t, coreio.Local.EnsureDir(core.PathDir(manifest)))
	core.RequireNoError(t, coreio.Local.Write(manifest, `{"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:abc"}]}`))
}

// writeGGUFFixture writes a minimal but valid GGUF file at path via this
// repo's own gguf.WriteFile, carrying one architecture-prefixed metadata
// key of each kind ReadGGUFInfo's ggufKeyOfInterest filter retains
// (vocab_size/embedding_length/block_count/context_length) plus one F32
// tensor — enough for a real, non-error ReadGGUFInfo round trip.
func writeGGUFFixture(t *core.T, path, architecture string) {
	t.Helper()
	metadata := []gguf.MetadataEntry{
		{Key: "general.architecture", ValueType: gguf.ValueTypeString, Value: architecture},
		{Key: "general.file_type", ValueType: gguf.ValueTypeUint32, Value: uint32(0)},
		{Key: architecture + ".vocab_size", ValueType: gguf.ValueTypeUint32, Value: uint32(32000)},
		{Key: architecture + ".embedding_length", ValueType: gguf.ValueTypeUint32, Value: uint32(4096)},
		{Key: architecture + ".block_count", ValueType: gguf.ValueTypeUint32, Value: uint32(32)},
		{Key: architecture + ".context_length", ValueType: gguf.ValueTypeUint32, Value: uint32(8192)},
	}
	tensors := []gguf.Tensor{
		{Name: "blk.0.weight", Type: gguf.TensorTypeF32, Shape: []uint64{4}, Data: make([]byte, 16)},
	}
	core.RequireNoError(t, gguf.WriteFile(path, metadata, tensors))
}

// TestGguf_ReadGGUFInfo_Good drives the real happy path: a valid GGUF file
// header must parse into the expected GGUFInfo fields.
func TestGguf_ReadGGUFInfo_Good(t *core.T) {
	path := core.JoinPath(t.TempDir(), "model.gguf")
	writeGGUFFixture(t, path, "llama")

	r := ReadGGUFInfo(path)
	requireResultOK(t, r)
	info := r.Value.(GGUFInfo)
	core.AssertEqual(t, "llama", info.Architecture)
	core.AssertEqual(t, 32000, info.VocabSize)
	core.AssertEqual(t, 4096, info.HiddenSize)
	core.AssertEqual(t, 32, info.NumLayers)
	core.AssertEqual(t, 8192, info.ContextLength)
	core.AssertEqual(t, 1, info.TensorCount)
}

// TestGguf_ReadGGUFInfo_Bad covers invalid input distinct from a missing
// path: an empty path, and a file whose content is not a GGUF at all
// (wrong magic bytes).
func TestGguf_ReadGGUFInfo_Bad(t *core.T) {
	assertResultError(t, ReadGGUFInfo(""))
	badMagic := core.JoinPath(t.TempDir(), "corrupt.gguf")
	core.RequireNoError(t, coreio.Local.Write(badMagic, "NOTGGUF!"))
	assertResultError(t, ReadGGUFInfo(badMagic))
}

// TestGguf_ReadGGUFInfo_Ugly covers path-shape edges that are neither empty
// nor content-invalid: an existing directory with no GGUF inside, and a
// single non-existent .gguf file path.
func TestGguf_ReadGGUFInfo_Ugly(t *core.T) {
	dir := t.TempDir()
	assertResultError(t, ReadGGUFInfo(dir))
	assertResultError(t, ReadGGUFInfo(core.JoinPath(t.TempDir(), "missing.gguf")))
}

func TestGguf_DiscoverModels_Good(t *core.T) {
	models := DiscoverModels(t.TempDir())
	core.AssertEmpty(t, models)
	core.AssertEqual(t, 0, len(models))
}

func TestGguf_DiscoverModels_Bad(t *core.T) {
	models := DiscoverModels(core.JoinPath(t.TempDir(), "missing"))
	core.AssertEmpty(t, models)
	core.AssertEqual(t, 0, len(models))
}

func TestGguf_DiscoverModels_Ugly(t *core.T) {
	dir := t.TempDir()
	core.RequireNoError(t, coreio.Local.Write(core.JoinPath(dir, "note.txt"), "not a model"))
	models := DiscoverModels(dir)
	core.AssertEmpty(t, models)
}

func TestGguf_MLXTensorToGGUF_Good(t *core.T) {
	r := MLXTensorToGGUF("model.layers.0.self_attn.q_proj.lora_a")
	requireResultOK(t, r)
	core.AssertEqual(t, "blk.0.attn_q.weight.lora_a", r.Value.(string))
}

// TestGguf_MLXTensorToGGUF_Bad covers both error branches: a key the regex
// does not recognise at all, and one that matches the shape but names a
// module with no GGUF mapping — plus the empty-string extreme.
func TestGguf_MLXTensorToGGUF_Bad(t *core.T) {
	assertResultError(t, MLXTensorToGGUF("bad.key"))
	assertResultError(t, MLXTensorToGGUF("model.layers.0.unknown_module.lora_a"))
	assertResultError(t, MLXTensorToGGUF(""))
}

func TestGguf_MLXTensorToGGUF_Ugly(t *core.T) {
	r := MLXTensorToGGUF("model.layers.2.mlp.down_proj.lora_b")
	requireResultOK(t, r)
	core.AssertEqual(t, "blk.2.ffn_down.weight.lora_b", r.Value.(string))
}

func TestGguf_SafetensorsDtypeToGGML_Good(t *core.T) {
	r := SafetensorsDtypeToGGML("F32")
	requireResultOK(t, r)
	core.AssertEqual(t, uint32(0), r.Value.(uint32))
}

// TestGguf_SafetensorsDtypeToGGML_Bad covers an unrecognised dtype, the
// empty string, and a case-mismatched valid name (the switch is a literal
// Go string switch, case-sensitive by construction).
func TestGguf_SafetensorsDtypeToGGML_Bad(t *core.T) {
	assertResultError(t, SafetensorsDtypeToGGML("I8"))
	assertResultError(t, SafetensorsDtypeToGGML(""))
	assertResultError(t, SafetensorsDtypeToGGML("f32"))
}

func TestGguf_SafetensorsDtypeToGGML_Ugly(t *core.T) {
	r := SafetensorsDtypeToGGML("BF16")
	requireResultOK(t, r)
	core.AssertEqual(t, uint32(30), r.Value.(uint32))
}

func TestGguf_ConvertMLXtoGGUFLoRA_Good(t *core.T) {
	sf, cfg := writeSafetensorsFixture(t)
	out := core.JoinPath(t.TempDir(), "adapter.gguf")
	requireResultOK(t, ConvertMLXtoGGUFLoRA(sf, cfg, out, "gemma3"))
	core.AssertTrue(t, coreio.Local.IsFile(out))
}

func TestGguf_ConvertMLXtoGGUFLoRA_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	assertResultError(t, ConvertMLXtoGGUFLoRA(core.JoinPath(t.TempDir(), "missing.safetensors"), core.JoinPath(t.TempDir(), "missing.cfg"), core.JoinPath(t.TempDir(), "out.gguf"), "gemma3"))
}

func TestGguf_ConvertMLXtoGGUFLoRA_Ugly(t *core.T) {
	sf, _ := writeSafetensorsFixture(t)
	cfg := core.JoinPath(t.TempDir(), "bad.cfg")
	core.RequireNoError(t, coreio.Local.Write(cfg, "bad"))
	assertResultError(t, ConvertMLXtoGGUFLoRA(sf, cfg, core.JoinPath(t.TempDir(), "out.gguf"), "gemma3"))
}

func TestGguf_DetectArchFromConfig_Good(t *core.T) {
	cases := map[string]struct {
		json string
		want string
	}{
		"gemma3 via architectures[0]": {`{"architectures":["Gemma3ForCausalLM"]}`, "gemma3"},
		"gemma4 via model_type":       {`{"model_type":"gemma4"}`, "gemma4"},
		"qwen3.5 via model_type":      {`{"model_type":"qwen3_5"}`, "qwen3_6"},
	}
	for name, c := range cases {
		cfg := core.JoinPath(t.TempDir(), "config.json")
		core.RequireNoError(t, coreio.Local.Write(cfg, c.json))
		r := DetectArchFromConfig(cfg)
		requireResultOK(t, r)
		core.AssertEqual(t, c.want, r.Value.(string), name)
	}
}

func TestGguf_DetectArchFromConfig_Bad(t *core.T) {
	cfg := core.JoinPath(t.TempDir(), "config.json")
	core.RequireNoError(t, coreio.Local.Write(cfg, "not json"))
	assertResultError(t, DetectArchFromConfig(cfg), "parse")
}

func TestGguf_DetectArchFromConfig_Ugly(t *core.T) {
	// Absent file: read error.
	assertResultError(t, DetectArchFromConfig(core.JoinPath(t.TempDir(), "missing.json")), "read")

	// A syntactically valid config that carries no architecture signal — the
	// original bug's exact trigger shape (an adapter_config.json passed where
	// a base model's config.json belongs).
	_, adapterCfg := writeSafetensorsFixture(t)
	assertResultError(t, DetectArchFromConfig(adapterCfg), "no model_type or architectures")
}

func TestGguf_ModelTagToGGUFArch_Good(t *core.T) {
	cases := map[string]string{
		"gemma-3-1b":  "gemma3",
		"gemma-4-e2b": "gemma4",
	}
	for tag, want := range cases {
		r := ModelTagToGGUFArch(tag)
		requireResultOK(t, r)
		core.AssertEqual(t, want, r.Value.(string), tag)
	}
}

func TestGguf_ModelTagToGGUFArch_Bad(t *core.T) {
	// qwen3_5 has no registered GGUF architecture (no writer in this repo for
	// the hybrid linear-attention family) — an unmapped tag errors rather
	// than silently reporting gemma3.
	assertResultError(t, ModelTagToGGUFArch("qwen-3-5"))
	assertResultError(t, ModelTagToGGUFArch("unknown"))
	assertResultError(t, ModelTagToGGUFArch("Gemma-3-1B")) // map lookup is case-sensitive
}

// TestGguf_ModelTagToGGUFArch_Ugly covers empty and partial-match tags — a
// prefix of a real tag is not itself a registered tag.
func TestGguf_ModelTagToGGUFArch_Ugly(t *core.T) {
	assertResultError(t, ModelTagToGGUFArch(""))
	assertResultError(t, ModelTagToGGUFArch("gemma-3"))
	assertResultError(t, ModelTagToGGUFArch(" gemma-3-1b"))
}

func TestGguf_GGUFModelBlobPath_Good(t *core.T) {
	root := t.TempDir()
	writeOllamaManifest(t, root, "gemma")
	r := GGUFModelBlobPath(root, "gemma")
	requireResultOK(t, r)
	core.AssertContains(t, r.Value.(string), "sha256-abc")
}

// TestGguf_GGUFModelBlobPath_Bad covers a missing manifest (default tag and
// an explicit tag) and, closing a previously-untested branch, a manifest
// file that exists but is not valid JSON (the "parse manifest" error path).
func TestGguf_GGUFModelBlobPath_Bad(t *core.T) {
	assertResultError(t, GGUFModelBlobPath(t.TempDir(), "missing"))
	assertResultError(t, GGUFModelBlobPath(t.TempDir(), "missing:v2"))

	root := t.TempDir()
	manifest := core.JoinPath(root, "manifests", "registry.ollama.ai", "library", "broken", "latest")
	core.RequireNoError(t, coreio.Local.EnsureDir(core.PathDir(manifest)))
	core.RequireNoError(t, coreio.Local.Write(manifest, "not json"))
	assertResultError(t, GGUFModelBlobPath(root, "broken"))
}

func TestGguf_GGUFModelBlobPath_Ugly(t *core.T) {
	root := t.TempDir()
	manifest := core.JoinPath(root, "manifests", "registry.ollama.ai", "library", "gemma", "test")
	core.RequireNoError(t, coreio.Local.EnsureDir(core.PathDir(manifest)))
	core.RequireNoError(t, coreio.Local.Write(manifest, `{"layers":[]}`))
	assertResultError(t, GGUFModelBlobPath(root, "gemma:test"))
}

func TestGguf_ParseLayerFromTensorName_Good(t *core.T) {
	r := ParseLayerFromTensorName("blk.12.attn_q.weight")
	requireResultOK(t, r)
	core.AssertEqual(t, 12, r.Value.(int))
}

// TestGguf_ParseLayerFromTensorName_Bad covers three no-match shapes: no
// "blk." at all, the empty string, and a "blk." with no digits after it.
func TestGguf_ParseLayerFromTensorName_Bad(t *core.T) {
	assertResultError(t, ParseLayerFromTensorName("attn_q.weight"))
	assertResultError(t, ParseLayerFromTensorName(""))
	assertResultError(t, ParseLayerFromTensorName("blk.weight"))
}

func TestGguf_ParseLayerFromTensorName_Ugly(t *core.T) {
	r := ParseLayerFromTensorName("prefix.blk.0.value")
	requireResultOK(t, r)
	core.AssertEqual(t, 0, r.Value.(int))
}
