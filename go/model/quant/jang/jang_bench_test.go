// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the driver-neutral JANG / JANGTQ quant primitives.
// Per AX-11 — NewPackedTensorDescriptor fires per tensor at model
// load (Minimax-M2 carries hundreds of routed-expert tensors).
// BuildPackedProfile + ClonePackedProfile fire per profile lifted
// across runtime boundaries. ValidatePackedTensor runs per kernel
// dispatch on the CPU parity path. ParseConfig + ReadConfig hit on
// every model load.
//
// Run:    go test -bench='Benchmark' -benchmem -run='^$' ./quant/jang

package jang

import "testing"

// Sinks defeat compiler DCE.
var (
	jangSinkInfo       *Info
	jangSinkDescriptor PackedTensorDescriptor
	jangSinkProfile    *PackedProfile
	jangSinkClonedProf *PackedProfile
	jangSinkBits       int
	jangSinkPacked     []byte
	jangSinkValues     []float32
	jangSinkErr        error
)

// benchInfo returns the same JANGTQ profile shape the test suite
// uses — 4-bit groups with a mixed-bit role table.
func benchInfo() *Info {
	return &Info{
		Version:          2,
		WeightFormat:     "mxtq",
		Profile:          "JANGTQ",
		Method:           "affine+mxtq",
		GroupSize:        64,
		BitsDefault:      2,
		AttentionBits:    8,
		SharedExpertBits: 8,
		RoutedExpertBits: 2,
		EmbedTokensBits:  8,
		LMHeadBits:       8,
	}
}

// --- ParseConfig (per-model load) ---

func BenchmarkJang_ParseConfig_Minimal(b *testing.B) {
	data := []byte(`{
		"version": 2,
		"weight_format": "mxtq",
		"profile": "JANGTQ",
		"source_model": {
			"name": "MiniMax-M2",
			"org": "MiniMaxAI",
			"architecture": "MiniMaxM2"
		},
		"mxtq_bits": {
			"attention": 8,
			"shared_expert": 8,
			"routed_expert": 2,
			"embed_tokens": 8,
			"lm_head": 8
		},
		"quantization": {
			"method": "affine+mxtq",
			"group_size": 64,
			"bits_default": 2
		}
	}`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jangSinkInfo, jangSinkErr = ParseConfig(data)
	}
}

func BenchmarkJang_ParseConfig_WithCapabilities(b *testing.B) {
	data := []byte(`{
		"version": 2,
		"weight_format": "mxtq",
		"profile": "JANGTQ",
		"source_model": {
			"name": "MiniMax-M2",
			"org": "MiniMaxAI",
			"architecture": "MiniMaxM2"
		},
		"mxtq_bits": {
			"attention": 8,
			"shared_expert": 8,
			"routed_expert": 2,
			"embed_tokens": 8,
			"lm_head": 8
		},
		"quantization": {
			"method": "affine+mxtq",
			"group_size": 64,
			"bits_default": 2
		},
		"capabilities": {
			"reasoning_parser": "qwen-think",
			"tool_parser": "qwen-tool",
			"think_in_template": true,
			"supports_tools": true,
			"supports_thinking": true,
			"family": "minimax_m2",
			"modality": "text",
			"cache_type": "paged-q8"
		}
	}`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jangSinkInfo, jangSinkErr = ParseConfig(data)
	}
}

// --- NewPackedTensorDescriptor (per-tensor at model load) ---

func BenchmarkJang_NewPackedTensorDescriptor_RoutedExpert_Small(b *testing.B) {
	info := benchInfo()
	shape := []uint64{2048, 2048}
	name := "model.layers.0.block_sparse_moe.experts.0.w1.weight"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jangSinkDescriptor, jangSinkErr = NewPackedTensorDescriptor(name, shape, info)
	}
}

func BenchmarkJang_NewPackedTensorDescriptor_RoutedExpert_Large(b *testing.B) {
	info := benchInfo()
	shape := []uint64{6144, 6144}
	name := "model.layers.0.block_sparse_moe.experts.0.w1.weight"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jangSinkDescriptor, jangSinkErr = NewPackedTensorDescriptor(name, shape, info)
	}
}

func BenchmarkJang_NewPackedTensorDescriptor_Attention(b *testing.B) {
	info := benchInfo()
	shape := []uint64{4096, 4096}
	name := "model.layers.0.self_attn.q_proj.weight"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jangSinkDescriptor, jangSinkErr = NewPackedTensorDescriptor(name, shape, info)
	}
}

func BenchmarkJang_NewPackedTensorDescriptor_EmbedTokens(b *testing.B) {
	info := benchInfo()
	shape := []uint64{262144, 4096}
	name := "model.embed_tokens.weight"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jangSinkDescriptor, jangSinkErr = NewPackedTensorDescriptor(name, shape, info)
	}
}

// --- BuildPackedProfile (per profile cross-runtime) ---

func BenchmarkJang_BuildPackedProfile(b *testing.B) {
	info := benchInfo()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jangSinkProfile = BuildPackedProfile(info)
	}
}

// --- ClonePackedProfile (per runtime hand-off) ---

func BenchmarkJang_ClonePackedProfile(b *testing.B) {
	profile := BuildPackedProfile(benchInfo())
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jangSinkClonedProf = ClonePackedProfile(profile)
	}
}

// --- ProfileBits (per-role table build) ---

func BenchmarkJang_ProfileBits_JANGTQ(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jangSinkBits = ProfileBits("JANGTQ")
	}
}

func BenchmarkJang_ProfileBits_JANG_4(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jangSinkBits = ProfileBits("JANG_4M")
	}
}

func BenchmarkJang_ProfileBits_Unknown(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jangSinkBits = ProfileBits("unknown")
	}
}

// --- ValidatePackedTensor (per kernel dispatch) ---

func BenchmarkJang_ValidatePackedTensor_2bit(b *testing.B) {
	info := benchInfo()
	desc, err := NewPackedTensorDescriptor("model.layers.0.block_sparse_moe.experts.0.w1.weight", []uint64{64, 64}, info)
	if err != nil {
		b.Fatal(err)
	}
	packed := make([]byte, desc.PackedBytes)
	scales := make([]float32, desc.ScaleCount)
	biases := make([]float32, desc.BiasCount)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jangSinkErr = ValidatePackedTensor(desc, packed, scales, biases)
	}
}

func BenchmarkJang_ValidatePackedTensor_8bit(b *testing.B) {
	info := benchInfo()
	desc, err := NewPackedTensorDescriptor("model.layers.0.self_attn.q_proj.weight", []uint64{64, 64}, info)
	if err != nil {
		b.Fatal(err)
	}
	packed := make([]byte, desc.PackedBytes)
	scales := make([]float32, desc.ScaleCount)
	biases := make([]float32, desc.BiasCount)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jangSinkErr = ValidatePackedTensor(desc, packed, scales, biases)
	}
}

// --- PackQuantizedValues (CPU parity-test path) ---
// 2-bit / 4-bit / 8-bit shapes; values per byte differs across bit
// widths so the pack hot loop sees all three.

func BenchmarkJang_PackQuantizedValues_2bit_256(b *testing.B) {
	info := benchInfo()
	desc, err := NewPackedTensorDescriptor("model.layers.0.block_sparse_moe.experts.0.w1.weight", []uint64{16, 16}, info)
	if err != nil {
		b.Fatal(err)
	}
	values := make([]uint8, desc.Elements)
	for i := range values {
		values[i] = uint8(i % 4)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jangSinkPacked, jangSinkErr = PackQuantizedValues(desc, values)
	}
}

func BenchmarkJang_PackQuantizedValues_8bit_256(b *testing.B) {
	info := benchInfo()
	desc, err := NewPackedTensorDescriptor("model.layers.0.self_attn.q_proj.weight", []uint64{16, 16}, info)
	if err != nil {
		b.Fatal(err)
	}
	values := make([]uint8, desc.Elements)
	for i := range values {
		values[i] = uint8(i % 256)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jangSinkPacked, jangSinkErr = PackQuantizedValues(desc, values)
	}
}

func BenchmarkJang_PackQuantizedValues_2bit_4096(b *testing.B) {
	info := benchInfo()
	desc, err := NewPackedTensorDescriptor("model.layers.0.block_sparse_moe.experts.0.w1.weight", []uint64{64, 64}, info)
	if err != nil {
		b.Fatal(err)
	}
	values := make([]uint8, desc.Elements)
	for i := range values {
		values[i] = uint8(i % 4)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jangSinkPacked, jangSinkErr = PackQuantizedValues(desc, values)
	}
}

// --- DequantizePackedTensor (CPU parity-test path) ---

func BenchmarkJang_DequantizePackedTensor_2bit_256(b *testing.B) {
	info := benchInfo()
	desc, err := NewPackedTensorDescriptor("model.layers.0.block_sparse_moe.experts.0.w1.weight", []uint64{16, 16}, info)
	if err != nil {
		b.Fatal(err)
	}
	values := make([]uint8, desc.Elements)
	for i := range values {
		values[i] = uint8(i % 4)
	}
	packed, err := PackQuantizedValues(desc, values)
	if err != nil {
		b.Fatal(err)
	}
	scales := make([]float32, desc.ScaleCount)
	biases := make([]float32, desc.BiasCount)
	for i := range scales {
		scales[i] = 0.125
		biases[i] = -1
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jangSinkValues, jangSinkErr = DequantizePackedTensor(desc, packed, scales, biases)
	}
}

func BenchmarkJang_DequantizePackedTensor_2bit_4096(b *testing.B) {
	info := benchInfo()
	desc, err := NewPackedTensorDescriptor("model.layers.0.block_sparse_moe.experts.0.w1.weight", []uint64{64, 64}, info)
	if err != nil {
		b.Fatal(err)
	}
	values := make([]uint8, desc.Elements)
	for i := range values {
		values[i] = uint8(i % 4)
	}
	packed, err := PackQuantizedValues(desc, values)
	if err != nil {
		b.Fatal(err)
	}
	scales := make([]float32, desc.ScaleCount)
	biases := make([]float32, desc.BiasCount)
	for i := range scales {
		scales[i] = 0.125
		biases[i] = -1
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jangSinkValues, jangSinkErr = DequantizePackedTensor(desc, packed, scales, biases)
	}
}

func BenchmarkJang_DequantizePackedTensor_8bit_256(b *testing.B) {
	info := benchInfo()
	desc, err := NewPackedTensorDescriptor("model.layers.0.self_attn.q_proj.weight", []uint64{16, 16}, info)
	if err != nil {
		b.Fatal(err)
	}
	values := make([]uint8, desc.Elements)
	for i := range values {
		values[i] = uint8(i % 256)
	}
	packed, err := PackQuantizedValues(desc, values)
	if err != nil {
		b.Fatal(err)
	}
	scales := make([]float32, desc.ScaleCount)
	biases := make([]float32, desc.BiasCount)
	for i := range scales {
		scales[i] = 0.0625
		biases[i] = -2
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jangSinkValues, jangSinkErr = DequantizePackedTensor(desc, packed, scales, biases)
	}
}

// benchInfoBits returns a benchInfo where the routed-expert bits override
// is set to the requested width. NewPackedTensorDescriptor routes a tensor
// matching block_sparse_moe.experts to RoutedExpertBits, so we can exercise
// any width in {1, 2, 3, 4, 8} through the same name.
func benchInfoBits(bits int) *Info {
	info := benchInfo()
	info.RoutedExpertBits = bits
	info.BitsDefault = bits
	return info
}

func benchDequantize(b *testing.B, bits, elements int) {
	desc, err := NewPackedTensorDescriptor("model.layers.0.block_sparse_moe.experts.0.w1.weight", []uint64{uint64(elements)}, benchInfoBits(bits))
	if err != nil {
		b.Fatal(err)
	}
	maxValue := uint8((1 << bits) - 1)
	values := make([]uint8, desc.Elements)
	for i := range values {
		values[i] = uint8(i) & maxValue
	}
	packed, err := PackQuantizedValues(desc, values)
	if err != nil {
		b.Fatal(err)
	}
	scales := make([]float32, desc.ScaleCount)
	biases := make([]float32, desc.BiasCount)
	for i := range scales {
		scales[i] = 0.0625
		biases[i] = -2
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jangSinkValues, jangSinkErr = DequantizePackedTensor(desc, packed, scales, biases)
	}
}

func BenchmarkJang_DequantizePackedTensor_1bit_4096(b *testing.B) {
	benchDequantize(b, 1, 4096)
}

func BenchmarkJang_DequantizePackedTensor_2bit_16384(b *testing.B) {
	benchDequantize(b, 2, 16384)
}

func BenchmarkJang_DequantizePackedTensor_3bit_4096(b *testing.B) {
	benchDequantize(b, 3, 4096)
}

func BenchmarkJang_DequantizePackedTensor_4bit_4096(b *testing.B) {
	benchDequantize(b, 4, 4096)
}

func BenchmarkJang_DequantizePackedTensor_8bit_4096(b *testing.B) {
	benchDequantize(b, 8, 4096)
}
