// SPDX-Licence-Identifier: EUPL-1.2

package codebook

import (
	"testing"

	core "dappco.re/go"
)

func TestCodebook_DescriptorValidatesAndMatVec_Good(t *testing.T) {
	profile := Profile{
		Format:       FormatVQ,
		CodebookSize: 3,
		CodeDim:      2,
		IndexBits:    16,
	}

	desc, err := NewTensorDescriptor("model.layers.0.mlp.down_proj.weight", []uint64{2, 4}, profile)
	if err != nil {
		t.Fatalf("NewTensorDescriptor() error = %v", err)
	}
	if desc.Elements != 8 || desc.CodeCount != 4 || desc.CodebookSize != 3 || desc.CodeDim != 2 {
		t.Fatalf("descriptor = %+v, want 8 elements, 4 codes, 3-entry codebook with 2D vectors", desc)
	}
	if desc.IndexBytes != 8 {
		t.Fatalf("IndexBytes = %d, want four 16-bit indices", desc.IndexBytes)
	}

	got, err := MatVec(desc, []float32{3, 4, 5, 6}, []uint32{0, 1, 2, 1}, []float32{
		1, 0,
		0, 1,
		2, -1,
	}, []float32{0.5, -1})
	if err != nil {
		t.Fatalf("MatVec() error = %v", err)
	}
	assertCloseSlice(t, got, []float32{9.5, 7}, 1e-5)
}

func TestCodebook_CodeDim_Bad(t *testing.T) {
	_, err := NewTensorDescriptor("bad.weight", []uint64{3, 3}, Profile{
		Format:       FormatVQ,
		CodebookSize: 16,
		CodeDim:      4,
		IndexBits:    8,
	})
	if err == nil || !core.Contains(err.Error(), "divisible") {
		t.Fatalf("error = %v, want code-dim divisibility diagnostic", err)
	}
}

func TestCodebook_CodebookSize_Bad(t *testing.T) {
	desc, err := NewTensorDescriptor("ok.weight", []uint64{1, 2}, Profile{
		Format:       FormatVQ,
		CodebookSize: 2,
		CodeDim:      1,
		IndexBits:    8,
	})
	if err != nil {
		t.Fatalf("NewTensorDescriptor() error = %v", err)
	}

	_, err = MatVec(desc, []float32{1, 2}, []uint32{0, 4}, []float32{1, 2}, nil)
	if err == nil || !core.Contains(err.Error(), "code id") {
		t.Fatalf("error = %v, want out-of-range code diagnostic", err)
	}
}

func TestCodebook_ParseProfile_Good(t *testing.T) {
	profile, err := ParseProfile([]byte(`{
		"type": "codebook",
		"format": "vq",
		"codebook_size": 4,
		"code_dim": 2,
		"index_bits": 8,
		"tensors": [
			{
				"name": "model.layers.0.mlp.down_proj.weight",
				"shape": [2, 4],
				"codes": "model.layers.0.mlp.down_proj.weight.codes",
				"codebook": "model.layers.0.mlp.down_proj.weight.codebook"
			}
		]
	}`))
	if err != nil {
		t.Fatalf("ParseProfile() error = %v", err)
	}
	if profile.Type != Type || profile.Format != FormatVQ || len(profile.Tensors) != 1 {
		t.Fatalf("profile = %+v, want one VQ tensor", profile)
	}
	if tensor := profile.Tensors[0]; tensor.CodeCount != 4 || tensor.CodesName == "" || tensor.CodebookName == "" {
		t.Fatalf("tensor = %+v, want resolved sidecar names and code count", tensor)
	}
}

// --- ReadProfile (file-system boundary) ---

func TestCodebook_ReadProfile_Good(t *testing.T) {
	dir := t.TempDir()
	configPath := core.PathJoin(dir, "codebook_config.json")
	data := []byte(`{
		"type": "codebook",
		"format": "vq",
		"codebook_size": 4,
		"code_dim": 2,
		"index_bits": 8,
		"tensors": [
			{"name": "model.layers.0.mlp.down_proj.weight", "shape": [2, 4]}
		]
	}`)
	if result := core.WriteFile(configPath, data, 0o600); !result.OK {
		t.Fatalf("WriteFile() error = %v", result.Value)
	}

	profile, err := ReadProfile(dir)
	if err != nil {
		t.Fatalf("ReadProfile() error = %v", err)
	}
	if profile == nil || len(profile.Tensors) != 1 || profile.Tensors[0].CodeCount != 4 {
		t.Fatalf("profile = %+v, want one parsed tensor with 4 codes", profile)
	}
}

func TestCodebook_ReadProfile_Bad(t *testing.T) {
	dir := t.TempDir()
	// A directory in place of the config file fails the read with
	// something other than "not exist" (EISDIR), exercising the raw
	// error-propagation branch distinct from the missing-file branch.
	configPath := core.PathJoin(dir, "codebook_config.json")
	if result := core.Mkdir(configPath, 0o755); !result.OK {
		t.Fatalf("Mkdir() error = %v", result.Value)
	}

	profile, err := ReadProfile(dir)
	if err == nil {
		t.Fatalf("ReadProfile() error = nil, profile = %+v, want a read error", profile)
	}
}

func TestCodebook_ReadProfile_Ugly(t *testing.T) {
	dir := t.TempDir()
	profile, err := ReadProfile(dir)
	if err != nil || profile != nil {
		t.Fatalf("ReadProfile() = %+v, %v, want nil, nil for a missing config", profile, err)
	}
}

// --- NewTensorDescriptor (additional branch coverage) ---

func TestCodebook_NewTensorDescriptor_Good(t *testing.T) {
	t.Run("default format when omitted", func(t *testing.T) {
		desc, err := NewTensorDescriptor("model.layers.0.mlp.gate_proj.weight", []uint64{2, 4}, Profile{
			CodebookSize: 4,
			CodeDim:      2,
			IndexBits:    8,
		})
		if err != nil {
			t.Fatalf("NewTensorDescriptor() error = %v", err)
		}
		if desc.Format != FormatVQ {
			t.Fatalf("Format = %q, want default %q", desc.Format, FormatVQ)
		}
	})

	t.Run("codes and codebook sidecar shapes", func(t *testing.T) {
		desc, err := NewTensorDescriptor("model.layers.0.mlp.up_proj.weight", []uint64{4, 4}, Profile{
			Format:       FormatVQ,
			CodebookSize: 8,
			CodeDim:      4,
			IndexBits:    32,
		})
		if err != nil {
			t.Fatalf("NewTensorDescriptor() error = %v", err)
		}
		if desc.CodesName != "model.layers.0.mlp.up_proj.weight.codes" || desc.CodebookName != "model.layers.0.mlp.up_proj.weight.codebook" {
			t.Fatalf("sidecar names = codes:%q codebook:%q", desc.CodesName, desc.CodebookName)
		}
		if len(desc.CodesShape) != 1 || desc.CodesShape[0] != uint64(desc.CodeCount) {
			t.Fatalf("CodesShape = %v, want [%d]", desc.CodesShape, desc.CodeCount)
		}
		if len(desc.CodebookShape) != 2 || desc.CodebookShape[0] != 8 || desc.CodebookShape[1] != 4 {
			t.Fatalf("CodebookShape = %v, want [8 4]", desc.CodebookShape)
		}
	})
}

func TestCodebook_NewTensorDescriptor_Bad(t *testing.T) {
	base := Profile{Format: FormatVQ, CodebookSize: 4, CodeDim: 2, IndexBits: 8}

	t.Run("empty name", func(t *testing.T) {
		_, err := NewTensorDescriptor("", []uint64{2, 4}, base)
		if err == nil || !core.Contains(err.Error(), "name is required") {
			t.Fatalf("error = %v, want name-required diagnostic", err)
		}
	})

	t.Run("unsupported format", func(t *testing.T) {
		profile := base
		profile.Format = "gptq"
		_, err := NewTensorDescriptor("bad.weight", []uint64{2, 4}, profile)
		if err == nil || !core.Contains(err.Error(), "unsupported format") {
			t.Fatalf("error = %v, want unsupported-format diagnostic", err)
		}
	})

	t.Run("wrong shape rank", func(t *testing.T) {
		_, err := NewTensorDescriptor("bad.weight", []uint64{4}, base)
		if err == nil || !core.Contains(err.Error(), "shape must be") {
			t.Fatalf("error = %v, want shape-rank diagnostic", err)
		}
	})

	t.Run("non-positive codebook size", func(t *testing.T) {
		profile := base
		profile.CodebookSize = 0
		_, err := NewTensorDescriptor("bad.weight", []uint64{2, 4}, profile)
		if err == nil || !core.Contains(err.Error(), "codebook size must be positive") {
			t.Fatalf("error = %v, want codebook-size diagnostic", err)
		}
	})

	t.Run("non-positive code dim", func(t *testing.T) {
		profile := base
		profile.CodeDim = 0
		_, err := NewTensorDescriptor("bad.weight", []uint64{2, 4}, profile)
		if err == nil || !core.Contains(err.Error(), "code_dim must be positive") {
			t.Fatalf("error = %v, want code_dim diagnostic", err)
		}
	})

	t.Run("unsupported index bits", func(t *testing.T) {
		profile := base
		profile.IndexBits = 24
		_, err := NewTensorDescriptor("bad.weight", []uint64{2, 4}, profile)
		if err == nil || !core.Contains(err.Error(), "unsupported index bits") {
			t.Fatalf("error = %v, want index-bits diagnostic", err)
		}
	})
}

func TestCodebook_NewTensorDescriptor_Ugly(t *testing.T) {
	base := Profile{Format: FormatVQ, CodebookSize: 4, CodeDim: 2, IndexBits: 8}

	t.Run("zero-width dimension", func(t *testing.T) {
		_, err := NewTensorDescriptor("bad.weight", []uint64{0, 4}, base)
		if err == nil || !core.Contains(err.Error(), "shape must be") {
			t.Fatalf("error = %v, want shape diagnostic for a zero dimension", err)
		}
	})
}

// --- ValidateProfile ---

func TestCodebook_ValidateProfile_Good(t *testing.T) {
	desc, err := NewTensorDescriptor("model.layers.0.mlp.down_proj.weight", []uint64{2, 4}, Profile{
		Format:       FormatVQ,
		CodebookSize: 4,
		CodeDim:      2,
		IndexBits:    8,
	})
	if err != nil {
		t.Fatalf("NewTensorDescriptor() error = %v", err)
	}
	profile := Profile{
		Type:         Type,
		Format:       FormatVQ,
		CodebookSize: 4,
		CodeDim:      2,
		IndexBits:    8,
		Tensors:      []TensorDescriptor{desc},
	}
	if err := ValidateProfile(profile); err != nil {
		t.Fatalf("ValidateProfile() error = %v", err)
	}
}

func TestCodebook_ValidateProfile_Bad(t *testing.T) {
	base := Profile{Format: FormatVQ, CodebookSize: 4, CodeDim: 2, IndexBits: 8}

	t.Run("wrong type", func(t *testing.T) {
		profile := base
		profile.Type = "not-codebook"
		if err := ValidateProfile(profile); err == nil || !core.Contains(err.Error(), "unsupported type") {
			t.Fatalf("error = %v, want unsupported-type diagnostic", err)
		}
	})

	t.Run("wrong format", func(t *testing.T) {
		profile := base
		profile.Format = "gptq"
		if err := ValidateProfile(profile); err == nil || !core.Contains(err.Error(), "unsupported format") {
			t.Fatalf("error = %v, want unsupported-format diagnostic", err)
		}
	})

	t.Run("non-positive codebook size", func(t *testing.T) {
		profile := base
		profile.CodebookSize = 0
		if err := ValidateProfile(profile); err == nil || !core.Contains(err.Error(), "codebook size must be positive") {
			t.Fatalf("error = %v, want codebook-size diagnostic", err)
		}
	})

	t.Run("non-positive code dim", func(t *testing.T) {
		profile := base
		profile.CodeDim = 0
		if err := ValidateProfile(profile); err == nil || !core.Contains(err.Error(), "code_dim must be positive") {
			t.Fatalf("error = %v, want code_dim diagnostic", err)
		}
	})

	t.Run("unsupported index bits", func(t *testing.T) {
		profile := base
		profile.IndexBits = 3
		if err := ValidateProfile(profile); err == nil || !core.Contains(err.Error(), "unsupported index bits") {
			t.Fatalf("error = %v, want index-bits diagnostic", err)
		}
	})

	t.Run("invalid nested tensor", func(t *testing.T) {
		profile := base
		profile.Tensors = []TensorDescriptor{{}}
		if err := ValidateProfile(profile); err == nil || !core.Contains(err.Error(), "tensor name is required") {
			t.Fatalf("error = %v, want propagated tensor diagnostic", err)
		}
	})
}

func TestCodebook_ValidateProfile_Ugly(t *testing.T) {
	// Type, Format, and IndexBits left at zero value fall back to the
	// package defaults rather than being rejected outright.
	profile := Profile{CodebookSize: 4, CodeDim: 2}
	if err := ValidateProfile(profile); err != nil {
		t.Fatalf("ValidateProfile() error = %v, want empty type/format/index_bits to be tolerated", err)
	}
}

// --- ValidateTensorDescriptor ---

func TestCodebook_ValidateTensorDescriptor_Good(t *testing.T) {
	desc, err := NewTensorDescriptor("model.layers.0.mlp.down_proj.weight", []uint64{2, 4}, Profile{
		Format:       FormatVQ,
		CodebookSize: 4,
		CodeDim:      2,
		IndexBits:    8,
	})
	if err != nil {
		t.Fatalf("NewTensorDescriptor() error = %v", err)
	}
	if err := ValidateTensorDescriptor(desc); err != nil {
		t.Fatalf("ValidateTensorDescriptor() error = %v", err)
	}
}

func TestCodebook_ValidateTensorDescriptor_Bad(t *testing.T) {
	valid := TensorDescriptor{
		Name: "ok.weight", Format: FormatVQ, Shape: []uint64{2, 4}, Elements: 8,
		CodebookSize: 4, CodeDim: 2, CodeCount: 4, IndexBits: 8,
	}

	t.Run("empty name", func(t *testing.T) {
		desc := valid
		desc.Name = ""
		if err := ValidateTensorDescriptor(desc); err == nil || !core.Contains(err.Error(), "name is required") {
			t.Fatalf("error = %v, want name-required diagnostic", err)
		}
	})

	t.Run("wrong format", func(t *testing.T) {
		desc := valid
		desc.Format = "gptq"
		if err := ValidateTensorDescriptor(desc); err == nil || !core.Contains(err.Error(), "format must be vq") {
			t.Fatalf("error = %v, want format diagnostic", err)
		}
	})

	t.Run("wrong shape rank", func(t *testing.T) {
		desc := valid
		desc.Shape = []uint64{4}
		if err := ValidateTensorDescriptor(desc); err == nil || !core.Contains(err.Error(), "shape must be") {
			t.Fatalf("error = %v, want shape diagnostic", err)
		}
	})

	t.Run("non-positive codebook fields", func(t *testing.T) {
		desc := valid
		desc.CodeCount = 0
		if err := ValidateTensorDescriptor(desc); err == nil || !core.Contains(err.Error(), "codebook_size, code_dim, and code_count") {
			t.Fatalf("error = %v, want codebook-fields diagnostic", err)
		}
	})

	t.Run("unsupported index bits", func(t *testing.T) {
		desc := valid
		desc.IndexBits = 3
		if err := ValidateTensorDescriptor(desc); err == nil || !core.Contains(err.Error(), "unsupported index bits") {
			t.Fatalf("error = %v, want index-bits diagnostic", err)
		}
	})

	t.Run("element count mismatches shape", func(t *testing.T) {
		desc := valid
		desc.Elements = 99
		if err := ValidateTensorDescriptor(desc); err == nil || !core.Contains(err.Error(), "element count does not match shape") {
			t.Fatalf("error = %v, want element-count diagnostic", err)
		}
	})

	t.Run("code count mismatches code dim", func(t *testing.T) {
		desc := valid
		desc.CodeCount = 3
		if err := ValidateTensorDescriptor(desc); err == nil || !core.Contains(err.Error(), "code count does not match code_dim") {
			t.Fatalf("error = %v, want code-count diagnostic", err)
		}
	})
}

// --- MatVec (additional branch coverage) ---

func TestCodebook_MatVec_Bad(t *testing.T) {
	desc, err := NewTensorDescriptor("ok.weight", []uint64{2, 4}, Profile{
		Format: FormatVQ, CodebookSize: 4, CodeDim: 2, IndexBits: 8,
	})
	if err != nil {
		t.Fatalf("NewTensorDescriptor() error = %v", err)
	}
	codes := []uint32{0, 1, 2, 3}
	table := []float32{1, 0, 0, 1, 2, -1, -1, 2}

	_, err = MatVec(desc, []float32{1, 2, 3}, codes, table, nil)
	if err == nil || !core.Contains(err.Error(), "not divisible by input width") {
		t.Fatalf("error = %v, want input-width diagnostic", err)
	}
}

func TestCodebook_MatVec_Good(t *testing.T) {
	desc, err := NewTensorDescriptor("ok.weight", []uint64{1, 3}, Profile{
		Format: FormatVQ, CodebookSize: 3, CodeDim: 1, IndexBits: 8,
	})
	if err != nil {
		t.Fatalf("NewTensorDescriptor() error = %v", err)
	}
	// code_dim=1 maps codes straight onto weight positions: weight[i] =
	// table[codes[i]]. codes=[0,1,2] against table=[2,-3,5] selects the
	// whole table in order, so out = dot(input, table) with no bias term.
	got, err := MatVec(desc, []float32{1, 1, 1}, []uint32{0, 1, 2}, []float32{2, -3, 5}, nil)
	if err != nil {
		t.Fatalf("MatVec() error = %v", err)
	}
	assertCloseSlice(t, got, []float32{4}, 1e-6)
}

// --- ValidateTensorPayload ---

func TestCodebook_ValidateTensorPayload_Good(t *testing.T) {
	desc, err := NewTensorDescriptor("ok.weight", []uint64{2, 4}, Profile{
		Format: FormatVQ, CodebookSize: 4, CodeDim: 2, IndexBits: 8,
	})
	if err != nil {
		t.Fatalf("NewTensorDescriptor() error = %v", err)
	}
	codes := []uint32{0, 1, 2, 3}
	table := make([]float32, 8)

	t.Run("no bias", func(t *testing.T) {
		if err := ValidateTensorPayload(desc, codes, table, nil); err != nil {
			t.Fatalf("ValidateTensorPayload() error = %v", err)
		}
	})

	t.Run("matching bias", func(t *testing.T) {
		if err := ValidateTensorPayload(desc, codes, table, []float32{0, 1}); err != nil {
			t.Fatalf("ValidateTensorPayload() error = %v", err)
		}
	})
}

func TestCodebook_ValidateTensorPayload_Bad(t *testing.T) {
	desc, err := NewTensorDescriptor("ok.weight", []uint64{2, 4}, Profile{
		Format: FormatVQ, CodebookSize: 4, CodeDim: 2, IndexBits: 8,
	})
	if err != nil {
		t.Fatalf("NewTensorDescriptor() error = %v", err)
	}
	validCodes := []uint32{0, 1, 2, 3}
	validTable := make([]float32, 8)

	t.Run("invalid descriptor propagates", func(t *testing.T) {
		bad := desc
		bad.Name = ""
		if err := ValidateTensorPayload(bad, validCodes, validTable, nil); err == nil || !core.Contains(err.Error(), "name is required") {
			t.Fatalf("error = %v, want propagated descriptor diagnostic", err)
		}
	})

	t.Run("wrong code count", func(t *testing.T) {
		if err := ValidateTensorPayload(desc, []uint32{0, 1}, validTable, nil); err == nil || !core.Contains(err.Error(), "code count") {
			t.Fatalf("error = %v, want code-count diagnostic", err)
		}
	})

	t.Run("wrong codebook value count", func(t *testing.T) {
		if err := ValidateTensorPayload(desc, validCodes, []float32{0, 1}, nil); err == nil || !core.Contains(err.Error(), "value count") {
			t.Fatalf("error = %v, want value-count diagnostic", err)
		}
	})

	t.Run("code id out of range", func(t *testing.T) {
		if err := ValidateTensorPayload(desc, []uint32{0, 1, 2, 9}, validTable, nil); err == nil || !core.Contains(err.Error(), "code id") {
			t.Fatalf("error = %v, want code-id diagnostic", err)
		}
	})

	t.Run("wrong bias length", func(t *testing.T) {
		if err := ValidateTensorPayload(desc, validCodes, validTable, []float32{1, 2, 3}); err == nil || !core.Contains(err.Error(), "bias length") {
			t.Fatalf("error = %v, want bias-length diagnostic", err)
		}
	})
}

// --- CloneProfile ---

func TestCodebook_CloneProfile_Good(t *testing.T) {
	desc, err := NewTensorDescriptor("model.layers.0.mlp.down_proj.weight", []uint64{2, 4}, Profile{
		Format: FormatVQ, CodebookSize: 4, CodeDim: 2, IndexBits: 8,
	})
	if err != nil {
		t.Fatalf("NewTensorDescriptor() error = %v", err)
	}
	original := &Profile{
		Type: Type, Format: FormatVQ, CodebookSize: 4, CodeDim: 2, IndexBits: 8,
		Tensors: []TensorDescriptor{desc},
	}

	cloned := CloneProfile(original)
	if cloned == original {
		t.Fatalf("CloneProfile() returned the same pointer")
	}
	if len(cloned.Tensors) != 1 || cloned.Tensors[0].Name != desc.Name {
		t.Fatalf("cloned tensors = %+v, want a copy of the original tensor", cloned.Tensors)
	}

	// Mutating the clone's backing slices must not reach the original —
	// CloneProfile is documented as a deep copy across runtime boundaries.
	cloned.Tensors[0].Shape[0] = 999
	cloned.Tensors = append(cloned.Tensors, TensorDescriptor{Name: "extra"})
	if original.Tensors[0].Shape[0] == 999 || len(original.Tensors) != 1 {
		t.Fatalf("CloneProfile() aliased the original: original = %+v", original)
	}
}

func TestCodebook_CloneProfile_Ugly(t *testing.T) {
	if got := CloneProfile(nil); got != nil {
		t.Fatalf("CloneProfile(nil) = %+v, want nil", got)
	}
}

// --- small internal helpers ---

func TestCodebook_ValidIndexBits_Good(t *testing.T) {
	for _, bits := range []int{8, 16, 32} {
		if !validIndexBits(bits) {
			t.Errorf("validIndexBits(%d) = false, want true", bits)
		}
	}
}

func TestCodebook_ValidIndexBits_Bad(t *testing.T) {
	for _, bits := range []int{0, 1, 4, 24, 64, -8} {
		if validIndexBits(bits) {
			t.Errorf("validIndexBits(%d) = true, want false", bits)
		}
	}
}

// --- ParseProfile (additional branch coverage) ---

func TestCodebook_ParseProfile_Bad(t *testing.T) {
	t.Run("malformed JSON", func(t *testing.T) {
		_, err := ParseProfile([]byte(`{not json`))
		if err == nil {
			t.Fatalf("ParseProfile() error = nil, want a JSON decode error")
		}
	})

	t.Run("invalid tensor shape", func(t *testing.T) {
		_, err := ParseProfile([]byte(`{
			"codebook_size": 4,
			"code_dim": 2,
			"tensors": [{"name": "bad.weight", "shape": [3, 3]}]
		}`))
		if err == nil || !core.Contains(err.Error(), "divisible") {
			t.Fatalf("error = %v, want propagated tensor diagnostic", err)
		}
	})

	t.Run("invalid overall profile", func(t *testing.T) {
		_, err := ParseProfile([]byte(`{"codebook_size": 0, "code_dim": 2}`))
		if err == nil || !core.Contains(err.Error(), "codebook size must be positive") {
			t.Fatalf("error = %v, want ValidateProfile diagnostic", err)
		}
	})
}

func TestCodebook_ParseProfile_Ugly(t *testing.T) {
	// Per-tensor sidecar names/shapes are honoured verbatim when the JSON
	// supplies them, instead of being derived from the tensor name — use
	// shapes that differ from the derived defaults so the assertion can
	// only pass if the override path actually fired.
	profile, err := ParseProfile([]byte(`{
		"codebook_size": 4,
		"code_dim": 2,
		"tensors": [{
			"name": "model.layers.0.mlp.down_proj.weight",
			"shape": [2, 4],
			"codes": "custom.codes",
			"codebook": "custom.codebook",
			"codes_shape": [2, 2],
			"codebook_shape": [1, 8]
		}]
	}`))
	if err != nil {
		t.Fatalf("ParseProfile() error = %v", err)
	}
	tensor := profile.Tensors[0]
	if tensor.CodesName != "custom.codes" || tensor.CodebookName != "custom.codebook" {
		t.Fatalf("sidecar names = %+v, want explicit overrides honoured", tensor)
	}
	if len(tensor.CodesShape) != 2 || tensor.CodesShape[0] != 2 || tensor.CodesShape[1] != 2 {
		t.Fatalf("CodesShape = %v, want explicit [2 2]", tensor.CodesShape)
	}
	if len(tensor.CodebookShape) != 2 || tensor.CodebookShape[0] != 1 || tensor.CodebookShape[1] != 8 {
		t.Fatalf("CodebookShape = %v, want explicit [1 8]", tensor.CodebookShape)
	}
}

func assertCloseSlice(t *testing.T, got, want []float32, epsilon float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len(got) = %d, want %d", len(got), len(want))
	}
	for i := range got {
		diff := got[i] - want[i]
		if diff < 0 {
			diff = -diff
		}
		if float64(diff) > epsilon {
			t.Fatalf("value[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}
