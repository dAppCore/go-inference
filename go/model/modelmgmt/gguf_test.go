package modelmgmt

import (
	"dappco.re/go"
	coreio "dappco.re/go/io"
)

func writeOllamaManifest(t *core.T, root string, model string) {
	t.Helper()
	manifest := core.JoinPath(root, "manifests", "registry.ollama.ai", "library", model, "latest")
	core.RequireNoError(t, coreio.Local.EnsureDir(core.PathDir(manifest)))
	core.RequireNoError(t, coreio.Local.Write(manifest, `{"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:abc"}]}`))
}

func TestGguf_ReadGGUFInfo_Good(t *core.T) {
	file := core.JoinPath(t.TempDir(), "missing.gguf")
	assertResultError(t, ReadGGUFInfo(file))
}

func TestGguf_ReadGGUFInfo_Bad(t *core.T) {
	assertResultError(t, ReadGGUFInfo(""))
}

func TestGguf_ReadGGUFInfo_Ugly(t *core.T) {
	dir := t.TempDir()
	assertResultError(t, ReadGGUFInfo(dir))
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

func TestGguf_MLXTensorToGGUF_Bad(t *core.T) {
	assertResultError(t, MLXTensorToGGUF("bad.key"))
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

func TestGguf_SafetensorsDtypeToGGML_Bad(t *core.T) {
	assertResultError(t, SafetensorsDtypeToGGML("I8"))
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
	_, cfg := writeSafetensorsFixture(t)
	got := DetectArchFromConfig(cfg)
	core.AssertEqual(t, "gemma3", got)
}

func TestGguf_DetectArchFromConfig_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	got := DetectArchFromConfig(core.JoinPath(t.TempDir(), "missing.cfg"))
	core.AssertEqual(t, "gemma3", got)
}

func TestGguf_DetectArchFromConfig_Ugly(t *core.T) {
	cfg := core.JoinPath(t.TempDir(), "empty.cfg")
	core.RequireNoError(t, coreio.Local.Write(cfg, "{}"))
	got := DetectArchFromConfig(cfg)
	core.AssertEqual(t, "gemma3", got)
}

func TestGguf_ModelTagToGGUFArch_Good(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	got := ModelTagToGGUFArch("gemma-3-1b")
	core.AssertEqual(t, "gemma3", got)
}

func TestGguf_ModelTagToGGUFArch_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	got := ModelTagToGGUFArch("unknown")
	core.AssertEqual(t, "gemma3", got)
}

func TestGguf_ModelTagToGGUFArch_Ugly(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	got := ModelTagToGGUFArch("")
	core.AssertEqual(t, "gemma3", got)
}

func TestGguf_GGUFModelBlobPath_Good(t *core.T) {
	root := t.TempDir()
	writeOllamaManifest(t, root, "gemma")
	r := GGUFModelBlobPath(root, "gemma")
	requireResultOK(t, r)
	core.AssertContains(t, r.Value.(string), "sha256-abc")
}

func TestGguf_GGUFModelBlobPath_Bad(t *core.T) {
	assertResultError(t, GGUFModelBlobPath(t.TempDir(), "missing"))
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

func TestGguf_ParseLayerFromTensorName_Bad(t *core.T) {
	assertResultError(t, ParseLayerFromTensorName("attn_q.weight"))
}

func TestGguf_ParseLayerFromTensorName_Ugly(t *core.T) {
	r := ParseLayerFromTensorName("prefix.blk.0.value")
	requireResultOK(t, r)
	core.AssertEqual(t, 0, r.Value.(int))
}
