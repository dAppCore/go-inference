// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the driver-neutral VQ-codebook quant primitives.
// Per AX-11 — ParseProfile + NewTensorDescriptor fire once per
// tensor at model load (hundreds of tensors per Gemma/Qwen-class
// model). ValidateTensorPayload runs per kernel dispatch on the
// CPU parity path. CloneProfile fires per profile lifted across
// runtime boundaries. The reference MatVec is the CPU parity
// path used by parity tests against the native Metal kernel.
//
// Run:    go test -bench='Benchmark' -benchmem -run='^$' ./quant/codebook

package codebook

import (
	"testing"

	core "dappco.re/go"
)

// Sinks defeat compiler DCE.
var (
	codebookSinkProfile     *Profile
	codebookSinkDescriptor  TensorDescriptor
	codebookSinkMatVec      []float32
	codebookSinkErr         error
	codebookSinkProfileVal  Profile
	codebookSinkClonedProf  *Profile
)

// benchProfile builds a Profile with the requested codebook size and
// a single tensor of the requested shape. Used as a shared fixture
// across the bench surfaces.
func benchProfile(codebookSize, codeDim, indexBits int, outDim, inDim uint64) Profile {
	desc, _ := NewTensorDescriptor("model.layers.0.mlp.down_proj.weight", []uint64{outDim, inDim}, Profile{
		Format:       FormatVQ,
		CodebookSize: codebookSize,
		CodeDim:      codeDim,
		IndexBits:    indexBits,
	})
	return Profile{
		Type:         Type,
		Format:       FormatVQ,
		CodebookSize: codebookSize,
		CodeDim:      codeDim,
		IndexBits:    indexBits,
		Tensors:      []TensorDescriptor{desc},
	}
}

// benchMatVecInputs builds the codes + codebook + bias slices a
// MatVec parity check needs for a given descriptor.
func benchMatVecInputs(desc TensorDescriptor) ([]float32, []uint32, []float32, []float32) {
	input := make([]float32, int(desc.Shape[1]))
	for i := range input {
		input[i] = float32(i%7) * 0.125
	}
	codes := make([]uint32, desc.CodeCount)
	for i := range codes {
		codes[i] = uint32(i % desc.CodebookSize)
	}
	table := make([]float32, desc.CodebookSize*desc.CodeDim)
	for i := range table {
		table[i] = float32(i%11) * 0.25
	}
	bias := make([]float32, int(desc.Shape[0]))
	for i := range bias {
		bias[i] = float32(i%3) * 0.5
	}
	return input, codes, table, bias
}

// --- NewTensorDescriptor (per-tensor at model load) ---

func BenchmarkCodebook_NewTensorDescriptor_Small(b *testing.B) {
	profile := Profile{
		Format:       FormatVQ,
		CodebookSize: 256,
		CodeDim:      4,
		IndexBits:    8,
	}
	shape := []uint64{1024, 1024}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		codebookSinkDescriptor, codebookSinkErr = NewTensorDescriptor("model.layers.0.mlp.down_proj.weight", shape, profile)
	}
}

func BenchmarkCodebook_NewTensorDescriptor_Large(b *testing.B) {
	profile := Profile{
		Format:       FormatVQ,
		CodebookSize: 4096,
		CodeDim:      8,
		IndexBits:    16,
	}
	shape := []uint64{4096, 4096}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		codebookSinkDescriptor, codebookSinkErr = NewTensorDescriptor("model.layers.0.mlp.down_proj.weight", shape, profile)
	}
}

// --- ParseProfile (per-model load) ---

func BenchmarkCodebook_ParseProfile_Small(b *testing.B) {
	data := []byte(`{
		"type": "codebook",
		"format": "vq",
		"codebook_size": 256,
		"code_dim": 4,
		"index_bits": 8,
		"tensors": [
			{
				"name": "model.layers.0.mlp.down_proj.weight",
				"shape": [1024, 1024]
			}
		]
	}`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		codebookSinkProfile, codebookSinkErr = ParseProfile(data)
	}
}

func BenchmarkCodebook_ParseProfile_Large(b *testing.B) {
	data := []byte(`{
		"type": "codebook",
		"format": "vq",
		"codebook_size": 4096,
		"code_dim": 8,
		"index_bits": 16,
		"tensors": [
			{
				"name": "model.layers.0.mlp.down_proj.weight",
				"shape": [4096, 4096]
			},
			{
				"name": "model.layers.0.mlp.gate_proj.weight",
				"shape": [4096, 4096]
			},
			{
				"name": "model.layers.0.mlp.up_proj.weight",
				"shape": [4096, 4096]
			},
			{
				"name": "model.layers.0.self_attn.q_proj.weight",
				"shape": [4096, 4096]
			},
			{
				"name": "model.layers.0.self_attn.k_proj.weight",
				"shape": [4096, 4096]
			},
			{
				"name": "model.layers.0.self_attn.v_proj.weight",
				"shape": [4096, 4096]
			},
			{
				"name": "model.layers.0.self_attn.o_proj.weight",
				"shape": [4096, 4096]
			}
		]
	}`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		codebookSinkProfile, codebookSinkErr = ParseProfile(data)
	}
}

// --- ValidateProfile (per-profile across runtime boundaries) ---

func BenchmarkCodebook_ValidateProfile_Small(b *testing.B) {
	profile := benchProfile(256, 4, 8, 1024, 1024)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		codebookSinkErr = ValidateProfile(profile)
	}
}

func BenchmarkCodebook_ValidateProfile_Large(b *testing.B) {
	profile := benchProfile(4096, 8, 16, 4096, 4096)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		codebookSinkErr = ValidateProfile(profile)
	}
}

// --- ValidateTensorDescriptor (per-tensor across runtime boundaries) ---

func BenchmarkCodebook_ValidateTensorDescriptor_Small(b *testing.B) {
	desc, err := NewTensorDescriptor("model.layers.0.mlp.down_proj.weight", []uint64{1024, 1024}, Profile{
		Format:       FormatVQ,
		CodebookSize: 256,
		CodeDim:      4,
		IndexBits:    8,
	})
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		codebookSinkErr = ValidateTensorDescriptor(desc)
	}
}

func BenchmarkCodebook_ValidateTensorDescriptor_Large(b *testing.B) {
	desc, err := NewTensorDescriptor("model.layers.0.mlp.down_proj.weight", []uint64{4096, 4096}, Profile{
		Format:       FormatVQ,
		CodebookSize: 4096,
		CodeDim:      8,
		IndexBits:    16,
	})
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		codebookSinkErr = ValidateTensorDescriptor(desc)
	}
}

// --- ValidateTensorPayload (per kernel dispatch) ---

func BenchmarkCodebook_ValidateTensorPayload_Small(b *testing.B) {
	desc, err := NewTensorDescriptor("model.layers.0.mlp.down_proj.weight", []uint64{64, 64}, Profile{
		Format:       FormatVQ,
		CodebookSize: 256,
		CodeDim:      4,
		IndexBits:    8,
	})
	if err != nil {
		b.Fatal(err)
	}
	_, codes, table, bias := benchMatVecInputs(desc)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		codebookSinkErr = ValidateTensorPayload(desc, codes, table, bias)
	}
}

func BenchmarkCodebook_ValidateTensorPayload_Large(b *testing.B) {
	desc, err := NewTensorDescriptor("model.layers.0.mlp.down_proj.weight", []uint64{256, 256}, Profile{
		Format:       FormatVQ,
		CodebookSize: 4096,
		CodeDim:      8,
		IndexBits:    16,
	})
	if err != nil {
		b.Fatal(err)
	}
	_, codes, table, bias := benchMatVecInputs(desc)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		codebookSinkErr = ValidateTensorPayload(desc, codes, table, bias)
	}
}

// --- CloneProfile (per runtime hand-off) ---

func BenchmarkCodebook_CloneProfile_Small(b *testing.B) {
	profile := benchProfile(256, 4, 8, 1024, 1024)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		codebookSinkClonedProf = CloneProfile(&profile)
	}
}

func BenchmarkCodebook_CloneProfile_Large(b *testing.B) {
	profile := benchProfile(4096, 8, 16, 4096, 4096)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		codebookSinkClonedProf = CloneProfile(&profile)
	}
}

// --- MatVec (reference CPU parity path) ---
// Sizes intentionally small — the CPU loop is O(out*in) and is the
// parity-test path, not the production hot loop. Keeping the inputs
// modest keeps the bench under 100ms per case while still exercising
// the per-row + per-col dispatch + table lookup.

func BenchmarkCodebook_MatVec_64x64_CB256(b *testing.B) {
	desc, err := NewTensorDescriptor("ok.weight", []uint64{64, 64}, Profile{
		Format:       FormatVQ,
		CodebookSize: 256,
		CodeDim:      4,
		IndexBits:    8,
	})
	if err != nil {
		b.Fatal(err)
	}
	input, codes, table, bias := benchMatVecInputs(desc)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		codebookSinkMatVec, codebookSinkErr = MatVec(desc, input, codes, table, bias)
	}
}

func BenchmarkCodebook_MatVec_128x128_CB4096(b *testing.B) {
	desc, err := NewTensorDescriptor("ok.weight", []uint64{128, 128}, Profile{
		Format:       FormatVQ,
		CodebookSize: 4096,
		CodeDim:      8,
		IndexBits:    16,
	})
	if err != nil {
		b.Fatal(err)
	}
	input, codes, table, bias := benchMatVecInputs(desc)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		codebookSinkMatVec, codebookSinkErr = MatVec(desc, input, codes, table, bias)
	}
}

// --- core.Contains diagnostic-string path (validation error formatting) ---
// Reject paths still cost real wall time when the producer hits a
// guarded shape; bench the error-format hot loop on the unaligned
// branch the test file already covers.

func BenchmarkCodebook_NewTensorDescriptor_RejectUnaligned(b *testing.B) {
	profile := Profile{
		Format:       FormatVQ,
		CodebookSize: 16,
		CodeDim:      4,
		IndexBits:    8,
	}
	shape := []uint64{3, 3}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		codebookSinkDescriptor, codebookSinkErr = NewTensorDescriptor("bad.weight", shape, profile)
	}
	_ = core.Contains // keep the import resolved when reject paths don't fire
}
