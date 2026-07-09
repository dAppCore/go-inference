// SPDX-Licence-Identifier: EUPL-1.2

package modelmgmt

import (
	"testing"

	core "dappco.re/go"
)

// benchSinkResult keeps benchmarked Results live so the optimiser cannot
// elide the call under test.
var benchSinkResult core.Result

// benchAdapterFixture writes a realistic Gemma-3 LoRA adapter safetensors file
// (34 layers × 7 modules × {lora_a,lora_b} = 476 tensors) and returns its path.
// The header — not the tensor bytes — is what ReadSafetensors parses, so data
// blobs are kept small while the tensor count stays realistic.
func benchAdapterFixture(b *testing.B) (string, map[string]SafetensorsTensorInfo, map[string][]byte) {
	b.Helper()
	modules := []string{
		"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
		"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
	}
	tensors := make(map[string]SafetensorsTensorInfo, len(modules)*34*2)
	data := make(map[string][]byte, len(modules)*34*2)
	for layer := range 34 {
		for _, mod := range modules {
			for _, ab := range []string{"lora_a", "lora_b"} {
				key := core.Sprintf("model.layers.%d.%s.%s", layer, mod, ab)
				tensors[key] = SafetensorsTensorInfo{Dtype: "F16", Shape: []int{8, 64}}
				data[key] = make([]byte, 8*64*2)
			}
		}
	}
	path := core.JoinPath(b.TempDir(), "adapter_model.safetensors")
	if r := WriteSafetensors(path, tensors, data); !r.OK {
		b.Fatalf("write fixture: %s", r.Error())
	}
	return path, tensors, data
}

func BenchmarkReadSafetensors(b *testing.B) {
	path, _, _ := benchAdapterFixture(b)
	b.ReportAllocs()
	for b.Loop() {
		r := ReadSafetensors(path)
		if !r.OK {
			b.Fatalf("read: %s", r.Error())
		}
		benchSinkResult = r
	}
}

func BenchmarkWriteSafetensors(b *testing.B) {
	_, tensors, data := benchAdapterFixture(b)
	path := core.JoinPath(b.TempDir(), "out.safetensors")
	b.ReportAllocs()
	for b.Loop() {
		r := WriteSafetensors(path, tensors, data)
		if !r.OK {
			b.Fatalf("write: %s", r.Error())
		}
		benchSinkResult = r
	}
}

// Per-tensor pack/unpack — hit hundreds-to-thousands of times during a
// single LoRA conversion. RenameMLXKey allocates a regex-replaced string;
// GetTensorData is a slice view; TransposeFloat32/16 build a new buffer.

func BenchmarkRenameMLXKey(b *testing.B) {
	key := "layers.0.self_attn.q_proj.lora_a"
	b.ReportAllocs()
	for b.Loop() {
		RenameMLXKey(key)
	}
}

func BenchmarkGetTensorData(b *testing.B) {
	info := SafetensorsTensorInfo{DataOffsets: [2]int{8, 24}}
	allData := make([]byte, 32)
	b.ReportAllocs()
	for b.Loop() {
		_ = GetTensorData(info, allData)
	}
}

func BenchmarkTransposeFloat32_Tiny(b *testing.B) {
	data := make([]byte, 8*8*4)
	b.ReportAllocs()
	for b.Loop() {
		_ = TransposeFloat32(data, 8, 8)
	}
}

func BenchmarkTransposeFloat32_LoRARank(b *testing.B) {
	// 8x256 — typical LoRA-A rank=8 weight on a 256-dim projection.
	data := make([]byte, 8*256*4)
	b.ResetTimer()
	b.ReportAllocs()
	for b.Loop() {
		_ = TransposeFloat32(data, 8, 256)
	}
}

func BenchmarkTransposeFloat16_Tiny(b *testing.B) {
	data := make([]byte, 8*8*2)
	b.ReportAllocs()
	for b.Loop() {
		_ = TransposeFloat16(data, 8, 8)
	}
}

func BenchmarkTransposeFloat16_LoRARank(b *testing.B) {
	// 8x256 fp16 — the common quantised LoRA-A.
	data := make([]byte, 8*256*2)
	b.ResetTimer()
	b.ReportAllocs()
	for b.Loop() {
		_ = TransposeFloat16(data, 8, 256)
	}
}
