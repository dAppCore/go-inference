// SPDX-Licence-Identifier: EUPL-1.2

package safetensors

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
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

// writeFixture writes a minimal single-tensor F32 safetensors file and
// returns its path — the same fixture shape modelmgmt's conversion tests
// build for the alias layer.
func writeFixture(t testing.TB) string {
	t.Helper()
	key := "model.layers.0.self_attn.q_proj.lora_a"
	path := core.JoinPath(t.TempDir(), "adapter_model.safetensors")
	tensors := map[string]SafetensorsTensorInfo{
		key: {Dtype: "F32", Shape: []int{1, 1}},
	}
	data := map[string][]byte{
		key: {1, 2, 3, 4},
	}
	requireResultOK(t, WriteSafetensors(path, tensors, data))
	return path
}

func TestSafetensors_ReadSafetensors_Good(t *core.T) {
	r := ReadSafetensors(writeFixture(t))
	requireResultOK(t, r)
	sd := r.Value.(SafetensorsData)
	core.AssertLen(t, sd.Tensors, 1)
	core.AssertLen(t, sd.Data, 4)
}

func TestSafetensors_ReadSafetensors_Bad(t *core.T) {
	assertResultError(t, ReadSafetensors(core.JoinPath(t.TempDir(), "missing.safetensors")))
}

func TestSafetensors_ReadSafetensors_Ugly(t *core.T) {
	file := core.JoinPath(t.TempDir(), "bad.safetensors")
	core.RequireNoError(t, coreio.Local.Write(file, "short"))
	assertResultError(t, ReadSafetensors(file))
}

func TestSafetensors_GetTensorData_Good(t *core.T) {
	info := SafetensorsTensorInfo{DataOffsets: [2]int{1, 3}}
	got := GetTensorData(info, []byte{0, 1, 2, 3})
	core.AssertEqual(t, []byte{1, 2}, got)
}

func TestSafetensors_GetTensorData_Bad(t *core.T) {
	info := SafetensorsTensorInfo{DataOffsets: [2]int{0, 0}}
	got := GetTensorData(info, []byte{1, 2})
	core.AssertEmpty(t, got)
}

func TestSafetensors_GetTensorData_Ugly(t *core.T) {
	info := SafetensorsTensorInfo{DataOffsets: [2]int{0, 4}}
	got := GetTensorData(info, []byte{1, 2, 3, 4})
	core.AssertLen(t, got, 4)
}

func TestSafetensors_WriteSafetensors_Good(t *core.T) {
	file := core.JoinPath(t.TempDir(), "out.safetensors")
	requireResultOK(t, WriteSafetensors(file, map[string]SafetensorsTensorInfo{"a": {Dtype: "F32", Shape: []int{1}}}, map[string][]byte{"a": {1, 2, 3, 4}}))
	core.AssertTrue(t, coreio.Local.IsFile(file))
}

func TestSafetensors_WriteSafetensors_Bad(t *core.T) {
	dir := core.JoinPath(t.TempDir(), "blocked")
	core.RequireNoError(t, coreio.Local.EnsureDir(dir))
	assertResultError(t, WriteSafetensors(dir, map[string]SafetensorsTensorInfo{}, map[string][]byte{}))
}

func TestSafetensors_WriteSafetensors_Ugly(t *core.T) {
	file := core.JoinPath(t.TempDir(), "empty.safetensors")
	requireResultOK(t, WriteSafetensors(file, map[string]SafetensorsTensorInfo{}, map[string][]byte{}))
	core.AssertTrue(t, coreio.Local.IsFile(file))
}

// TestSafetensors_WriteSafetensors_HeaderGolden pins the EXACT header JSON WriteSafetensors
// emits — sorted keys, dtype/shape/data_offsets field order, sequential offsets, and crucially
// the nil-shape → "null" vs empty-shape → "[]" distinction that any direct emitter must preserve
// byte-for-byte against the previous reflection marshal. The round-trip tests only prove the
// tensors survive, not that the wire header is unchanged.
func TestSafetensors_WriteSafetensors_HeaderGolden(t *core.T) {
	file := core.JoinPath(t.TempDir(), "golden.safetensors")
	tensors := map[string]SafetensorsTensorInfo{
		"nilshape":   {Dtype: "F32", Shape: nil},
		"emptyshape": {Dtype: "U8", Shape: []int{}},
		"normal":     {Dtype: "BF16", Shape: []int{2, 3}},
	}
	data := map[string][]byte{
		"nilshape":   {1, 2, 3, 4},
		"emptyshape": {9},
		"normal":     {5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17},
	}
	requireResultOK(t, WriteSafetensors(file, tensors, data))
	raw, err := coreio.Local.Read(file)
	core.RequireNoError(t, err)
	blob := []byte(raw)
	hsz := int(blob[0]) | int(blob[1])<<8 | int(blob[2])<<16 | int(blob[3])<<24
	gotHeader := string(blob[8 : 8+hsz])
	const wantHeader = `{"emptyshape":{"dtype":"U8","shape":[],"data_offsets":[0,1]},` +
		`"nilshape":{"dtype":"F32","shape":null,"data_offsets":[1,5]},` +
		`"normal":{"dtype":"BF16","shape":[2,3],"data_offsets":[5,17]}}`
	if gotHeader != wantHeader {
		t.Fatalf("WriteSafetensors header diverged:\n got=%s\nwant=%s", gotHeader, wantHeader)
	}
}

// TestSafetensors_ReadSafetensors_RoundTrip proves Write→Read is lossless
// over the tensor directory and data section for a multi-tensor file.
func TestSafetensors_ReadSafetensors_RoundTrip(t *core.T) {
	file := core.JoinPath(t.TempDir(), "round.safetensors")
	tensors := map[string]SafetensorsTensorInfo{
		"a": {Dtype: "F32", Shape: []int{2}},
		"b": {Dtype: "F16", Shape: []int{1, 2}},
	}
	data := map[string][]byte{
		"a": EncodeFloat32([]float32{1, -1}),
		"b": {0x00, 0x3C, 0x00, 0x40}, // f16 1.0, 2.0
	}
	requireResultOK(t, WriteSafetensors(file, tensors, data))

	r := ReadSafetensors(file)
	requireResultOK(t, r)
	sd := r.Value.(SafetensorsData)
	core.AssertLen(t, sd.Tensors, 2)
	core.AssertEqual(t, data["a"], GetTensorData(sd.Tensors["a"], sd.Data))
	core.AssertEqual(t, data["b"], GetTensorData(sd.Tensors["b"], sd.Data))
}
