package modelmgmt

import (
	"dappco.re/go"
	coreio "dappco.re/go/io"
)

func TestConvert_RenameMLXKey_Good(t *core.T) {
	got := RenameMLXKey("model.layers.0.self_attn.q_proj.lora_a")
	core.AssertContains(t, got, "lora_A.default.weight")
	core.AssertContains(t, got, "base_model.model.")
}

func TestConvert_RenameMLXKey_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	got := RenameMLXKey("plain.weight")
	core.AssertEqual(t, "base_model.model.plain.weight", got)
}

func TestConvert_RenameMLXKey_Ugly(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	got := RenameMLXKey("x.lora_b")
	core.AssertContains(t, got, "lora_B.default.weight")
}

func TestConvert_ReadSafetensors_Good(t *core.T) {
	sf, _ := writeSafetensorsFixture(t)
	r := ReadSafetensors(sf)
	requireResultOK(t, r)
	sd := r.Value.(SafetensorsData)
	core.AssertLen(t, sd.Tensors, 1)
	core.AssertLen(t, sd.Data, 4)
}

func TestConvert_ReadSafetensors_Bad(t *core.T) {
	assertResultError(t, ReadSafetensors(core.JoinPath(t.TempDir(), "missing.safetensors")))
}

func TestConvert_ReadSafetensors_Ugly(t *core.T) {
	file := core.JoinPath(t.TempDir(), "bad.safetensors")
	core.RequireNoError(t, coreio.Local.Write(file, "short"))
	assertResultError(t, ReadSafetensors(file))
}

func TestConvert_GetTensorData_Good(t *core.T) {
	info := SafetensorsTensorInfo{DataOffsets: [2]int{1, 3}}
	got := GetTensorData(info, []byte{0, 1, 2, 3})
	core.AssertEqual(t, []byte{1, 2}, got)
}

func TestConvert_GetTensorData_Bad(t *core.T) {
	info := SafetensorsTensorInfo{DataOffsets: [2]int{0, 0}}
	got := GetTensorData(info, []byte{1, 2})
	core.AssertEmpty(t, got)
}

func TestConvert_GetTensorData_Ugly(t *core.T) {
	info := SafetensorsTensorInfo{DataOffsets: [2]int{0, 4}}
	got := GetTensorData(info, []byte{1, 2, 3, 4})
	core.AssertLen(t, got, 4)
}

func TestConvert_TransposeFloat32_Good(t *core.T) {
	data := []byte{1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0}
	got := TransposeFloat32(data, 2, 2)
	core.AssertEqual(t, []byte{1, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0}, got)
}

func TestConvert_TransposeFloat32_Bad(t *core.T) {
	data := []byte{1, 2, 3}
	got := TransposeFloat32(data, 2, 2)
	core.AssertEqual(t, data, got)
}

func TestConvert_TransposeFloat32_Ugly(t *core.T) {
	data := []byte{1, 0, 0, 0}
	got := TransposeFloat32(data, 1, 1)
	core.AssertEqual(t, data, got)
}

func TestConvert_TransposeFloat16_Good(t *core.T) {
	data := []byte{1, 0, 2, 0, 3, 0, 4, 0}
	got := TransposeFloat16(data, 2, 2)
	core.AssertEqual(t, []byte{1, 0, 3, 0, 2, 0, 4, 0}, got)
}

func TestConvert_TransposeFloat16_Bad(t *core.T) {
	data := []byte{1, 2, 3}
	got := TransposeFloat16(data, 2, 2)
	core.AssertEqual(t, data, got)
}

func TestConvert_TransposeFloat16_Ugly(t *core.T) {
	data := []byte{1, 0}
	got := TransposeFloat16(data, 1, 1)
	core.AssertEqual(t, data, got)
}

func TestConvert_TransposeBFloat16_Good(t *core.T) {
	data := []byte{1, 0, 2, 0, 3, 0, 4, 0}
	got := TransposeBFloat16(data, 2, 2)
	core.AssertEqual(t, []byte{1, 0, 3, 0, 2, 0, 4, 0}, got)
}

func TestConvert_TransposeBFloat16_Bad(t *core.T) {
	data := []byte{1, 2, 3}
	got := TransposeBFloat16(data, 2, 2)
	core.AssertEqual(t, data, got)
}

func TestConvert_TransposeBFloat16_Ugly(t *core.T) {
	data := []byte{9, 0}
	got := TransposeBFloat16(data, 1, 1)
	core.AssertEqual(t, data, got)
}

func TestConvert_WriteSafetensors_Good(t *core.T) {
	file := core.JoinPath(t.TempDir(), "out.safetensors")
	requireResultOK(t, WriteSafetensors(file, map[string]SafetensorsTensorInfo{"a": {Dtype: "F32", Shape: []int{1}}}, map[string][]byte{"a": {1, 2, 3, 4}}))
	core.AssertTrue(t, coreio.Local.IsFile(file))
}

func TestConvert_WriteSafetensors_Bad(t *core.T) {
	dir := core.JoinPath(t.TempDir(), "blocked")
	core.RequireNoError(t, coreio.Local.EnsureDir(dir))
	assertResultError(t, WriteSafetensors(dir, map[string]SafetensorsTensorInfo{}, map[string][]byte{}))
}

func TestConvert_WriteSafetensors_Ugly(t *core.T) {
	file := core.JoinPath(t.TempDir(), "empty.safetensors")
	requireResultOK(t, WriteSafetensors(file, map[string]SafetensorsTensorInfo{}, map[string][]byte{}))
	core.AssertTrue(t, coreio.Local.IsFile(file))
}

func TestConvert_ConvertMLXtoPEFT_Good(t *core.T) {
	sf, cfg := writeSafetensorsFixture(t)
	out := core.JoinPath(t.TempDir(), "peft")
	requireResultOK(t, ConvertMLXtoPEFT(sf, cfg, out, "base-model"))
	core.AssertTrue(t, coreio.Local.IsFile(core.JoinPath(out, "adapter_model.safetensors")))
}

func TestConvert_ConvertMLXtoPEFT_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	assertResultError(t, ConvertMLXtoPEFT(core.JoinPath(t.TempDir(), "missing.safetensors"), core.JoinPath(t.TempDir(), "missing.cfg"), t.TempDir(), "base"))
}

func TestConvert_ConvertMLXtoPEFT_Ugly(t *core.T) {
	sf, _ := writeSafetensorsFixture(t)
	cfg := core.JoinPath(t.TempDir(), "bad.cfg")
	core.RequireNoError(t, coreio.Local.Write(cfg, "not object"))
	assertResultError(t, ConvertMLXtoPEFT(sf, cfg, core.JoinPath(t.TempDir(), "peft"), "base"))
}
