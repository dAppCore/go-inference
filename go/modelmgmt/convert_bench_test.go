// SPDX-Licence-Identifier: EUPL-1.2

package modelmgmt

import "testing"

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
