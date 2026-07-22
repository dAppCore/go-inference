// SPDX-Licence-Identifier: EUPL-1.2

// Package codebook holds the driver-neutral VQ-codebook quant metadata
// + reference CPU matvec for parity tests against native kernels.
//
//	profile, _ := codebook.ParseProfile(data)
//	desc, _ := codebook.NewTensorDescriptor(name, shape, profile)
//	out, _  := codebook.MatVec(desc, input, codes, table, bias)
package codebook

import (
	core "dappco.re/go"
)

const (
	Type     = "codebook"
	FormatVQ = "vq"
)

// profile := codebook.Profile{CodebookSize: 256, CodeDim: 4, IndexBits: 8}
type Profile struct {
	Type         string             `json:"type,omitempty"`
	Format       string             `json:"format,omitempty"`
	CodebookSize int                `json:"codebook_size,omitempty"`
	CodeDim      int                `json:"code_dim,omitempty"`
	IndexBits    int                `json:"index_bits,omitempty"`
	Source       string             `json:"source,omitempty"`
	Tensors      []TensorDescriptor `json:"tensors,omitempty"`
}

// desc, _ := codebook.NewTensorDescriptor(name, []uint64{out, in}, profile)
type TensorDescriptor struct {
	Name          string   `json:"name,omitempty"`
	Format        string   `json:"format,omitempty"`
	Shape         []uint64 `json:"shape,omitempty"`
	Elements      uint64   `json:"elements,omitempty"`
	CodebookSize  int      `json:"codebook_size,omitempty"`
	CodeDim       int      `json:"code_dim,omitempty"`
	CodeCount     int      `json:"code_count,omitempty"`
	IndexBits     int      `json:"index_bits,omitempty"`
	IndexBytes    int      `json:"index_bytes,omitempty"`
	CodesName     string   `json:"codes_name,omitempty"`
	CodebookName  string   `json:"codebook_name,omitempty"`
	CodesShape    []uint64 `json:"codes_shape,omitempty"`
	CodebookShape []uint64 `json:"codebook_shape,omitempty"`
}

type configProbe struct {
	Type         string `json:"type"`
	Format       string `json:"format"`
	CodebookSize int    `json:"codebook_size"`
	CodeDim      int    `json:"code_dim"`
	IndexBits    int    `json:"index_bits"`
	Source       string `json:"source"`
	Tensors      []struct {
		Name          string   `json:"name"`
		Shape         []uint64 `json:"shape"`
		CodesName     string   `json:"codes"`
		CodebookName  string   `json:"codebook"`
		CodesShape    []uint64 `json:"codes_shape"`
		CodebookShape []uint64 `json:"codebook_shape"`
		CodebookSize  int      `json:"codebook_size"`
		CodeDim       int      `json:"code_dim"`
		IndexBits     int      `json:"index_bits"`
	} `json:"tensors"`
}

// profile, _ := codebook.ParseProfile(data)
func ParseProfile(data []byte) (*Profile, error) {
	var probe configProbe
	if result := core.JSONUnmarshal(data, &probe); !result.OK {
		return nil, result.Value.(error)
	}
	profile := Profile{
		Type:         core.Coalesce(probe.Type, Type),
		Format:       core.Coalesce(probe.Format, FormatVQ),
		CodebookSize: probe.CodebookSize,
		CodeDim:      probe.CodeDim,
		IndexBits:    core.FirstPositive(probe.IndexBits, 8),
		Source:       core.Coalesce(probe.Source, "codebook_config.json"),
	}
	// Pre-size to the exact tensor count so the append loop never
	// re-grows. Production profiles carry one descriptor per quantised
	// tensor — hundreds for Gemma/Qwen-class models — and the doubling
	// cascade from cap=0 paid ~7 grows over 100 tensors plus discarded
	// backing arrays.
	if len(probe.Tensors) > 0 {
		profile.Tensors = make([]TensorDescriptor, 0, len(probe.Tensors))
	}
	for _, tensor := range probe.Tensors {
		local := profile
		local.CodebookSize = core.FirstPositive(tensor.CodebookSize, profile.CodebookSize)
		local.CodeDim = core.FirstPositive(tensor.CodeDim, profile.CodeDim)
		local.IndexBits = core.FirstPositive(tensor.IndexBits, profile.IndexBits)
		desc, err := NewTensorDescriptor(tensor.Name, tensor.Shape, local)
		if err != nil {
			return nil, err
		}
		desc.CodesName = core.Coalesce(tensor.CodesName, defaultCodesName(desc.Name))
		desc.CodebookName = core.Coalesce(tensor.CodebookName, defaultTableName(desc.Name))
		if len(tensor.CodesShape) > 0 {
			desc.CodesShape = append([]uint64(nil), tensor.CodesShape...)
		}
		if len(tensor.CodebookShape) > 0 {
			desc.CodebookShape = append([]uint64(nil), tensor.CodebookShape...)
		}
		profile.Tensors = append(profile.Tensors, desc)
	}
	if err := ValidateProfile(profile); err != nil {
		return nil, err
	}
	return &profile, nil
}

// profile, _ := codebook.ReadProfile("/models/foo")
func ReadProfile(root string) (*Profile, error) {
	read := core.ReadFile(core.PathJoin(root, "codebook_config.json"))
	if !read.OK {
		if core.IsNotExist(read.Value.(error)) {
			return nil, nil
		}
		return nil, read.Value.(error)
	}
	return ParseProfile(read.Bytes())
}

// desc, _ := codebook.NewTensorDescriptor("layer0.mlp.w", []uint64{4096, 4096}, profile)
func NewTensorDescriptor(name string, shape []uint64, profile Profile) (TensorDescriptor, error) {
	if name == "" {
		return TensorDescriptor{}, core.NewError("codebook: tensor name is required")
	}
	if profile.Format == "" {
		profile.Format = FormatVQ
	}
	if profile.Format != FormatVQ {
		return TensorDescriptor{}, core.NewError("codebook: unsupported format: " + profile.Format)
	}
	if len(shape) != 2 || shape[0] == 0 || shape[1] == 0 {
		return TensorDescriptor{}, core.NewError("codebook: tensor shape must be [out, in]")
	}
	if profile.CodebookSize <= 0 {
		return TensorDescriptor{}, core.NewError("codebook: codebook size must be positive")
	}
	if profile.CodeDim <= 0 {
		return TensorDescriptor{}, core.NewError("codebook: code_dim must be positive")
	}
	if !validIndexBits(profile.IndexBits) {
		return TensorDescriptor{}, core.NewError(core.Sprintf("codebook: unsupported index bits %d", profile.IndexBits))
	}
	elements := shape[0] * shape[1]
	if elements%uint64(profile.CodeDim) != 0 {
		return TensorDescriptor{}, core.NewError(core.Sprintf("codebook: tensor elements %d must be divisible by code_dim %d", elements, profile.CodeDim))
	}
	codeCount := int(elements / uint64(profile.CodeDim))
	return TensorDescriptor{
		Name:          name,
		Format:        profile.Format,
		Shape:         append([]uint64(nil), shape...),
		Elements:      elements,
		CodebookSize:  profile.CodebookSize,
		CodeDim:       profile.CodeDim,
		CodeCount:     codeCount,
		IndexBits:     profile.IndexBits,
		IndexBytes:    (codeCount*profile.IndexBits + 7) / 8,
		CodesName:     defaultCodesName(name),
		CodebookName:  defaultTableName(name),
		CodesShape:    []uint64{uint64(codeCount)},
		CodebookShape: []uint64{uint64(profile.CodebookSize), uint64(profile.CodeDim)},
	}, nil
}

// err := codebook.ValidateProfile(profile)
func ValidateProfile(profile Profile) error {
	if profile.Type != "" && profile.Type != Type {
		return core.NewError("codebook: unsupported type: " + profile.Type)
	}
	if profile.Format != "" && profile.Format != FormatVQ {
		return core.NewError("codebook: unsupported format: " + profile.Format)
	}
	if profile.CodebookSize <= 0 {
		return core.NewError("codebook: codebook size must be positive")
	}
	if profile.CodeDim <= 0 {
		return core.NewError("codebook: code_dim must be positive")
	}
	if !validIndexBits(core.FirstPositive(profile.IndexBits, 8)) {
		return core.NewError(core.Sprintf("codebook: unsupported index bits %d", profile.IndexBits))
	}
	for _, tensor := range profile.Tensors {
		if err := ValidateTensorDescriptor(tensor); err != nil {
			return err
		}
	}
	return nil
}

// err := codebook.ValidateTensorDescriptor(desc)
func ValidateTensorDescriptor(desc TensorDescriptor) error {
	if desc.Name == "" {
		return core.NewError("codebook: tensor name is required")
	}
	if desc.Format != FormatVQ {
		return core.NewError("codebook: tensor format must be vq")
	}
	if len(desc.Shape) != 2 || desc.Shape[0] == 0 || desc.Shape[1] == 0 {
		return core.NewError("codebook: tensor shape must be [out, in]")
	}
	if desc.CodebookSize <= 0 || desc.CodeDim <= 0 || desc.CodeCount <= 0 {
		return core.NewError("codebook: tensor requires codebook_size, code_dim, and code_count")
	}
	if !validIndexBits(desc.IndexBits) {
		return core.NewError(core.Sprintf("codebook: unsupported index bits %d", desc.IndexBits))
	}
	if desc.Elements != desc.Shape[0]*desc.Shape[1] {
		return core.NewError("codebook: tensor element count does not match shape")
	}
	if int(desc.Elements/uint64(desc.CodeDim)) != desc.CodeCount {
		return core.NewError("codebook: tensor code count does not match code_dim")
	}
	return nil
}

// out, _ := codebook.MatVec(desc, input, codes, table, bias)
func MatVec(desc TensorDescriptor, input []float32, codes []uint32, codebook []float32, bias []float32) ([]float32, error) {
	if err := ValidateTensorPayload(desc, codes, codebook, bias); err != nil {
		return nil, err
	}
	outDim := int(desc.Shape[0])
	inDim := int(desc.Shape[1])
	if len(input) == 0 || len(input)%inDim != 0 {
		return nil, core.NewError(core.Sprintf("codebook: matvec input length %d is not divisible by input width %d", len(input), inDim))
	}
	rows := len(input) / inDim
	out := make([]float32, rows*outDim)
	for row := range rows {
		for outCol := range outDim {
			sum := float32(0)
			for inCol := range inDim {
				weightIndex := outCol*inDim + inCol
				codeIndex := weightIndex / desc.CodeDim
				codeOffset := weightIndex % desc.CodeDim
				codeID := codes[codeIndex]
				weight := codebook[int(codeID)*desc.CodeDim+codeOffset]
				sum += input[row*inDim+inCol] * weight
			}
			if len(bias) > 0 {
				sum += bias[outCol]
			}
			out[row*outDim+outCol] = sum
		}
	}
	return out, nil
}

// err := codebook.ValidateTensorPayload(desc, codes, table, bias)
func ValidateTensorPayload(desc TensorDescriptor, codes []uint32, codebook []float32, bias []float32) error {
	if err := ValidateTensorDescriptor(desc); err != nil {
		return err
	}
	if len(codes) != desc.CodeCount {
		return core.NewError(core.Sprintf("codebook: code count %d, expected %d", len(codes), desc.CodeCount))
	}
	if len(codebook) != desc.CodebookSize*desc.CodeDim {
		return core.NewError(core.Sprintf("codebook: value count %d, expected %d", len(codebook), desc.CodebookSize*desc.CodeDim))
	}
	for i, codeID := range codes {
		if codeID >= uint32(desc.CodebookSize) {
			return core.NewError(core.Sprintf("codebook: code id %d at index %d exceeds codebook size %d", codeID, i, desc.CodebookSize))
		}
	}
	if len(bias) > 0 && len(bias) != int(desc.Shape[0]) {
		return core.NewError(core.Sprintf("codebook: bias length %d, expected %d", len(bias), desc.Shape[0]))
	}
	return nil
}

// clone := codebook.CloneProfile(profile)
func CloneProfile(profile *Profile) *Profile {
	if profile == nil {
		return nil
	}
	cloned := *profile
	cloned.Tensors = append([]TensorDescriptor(nil), profile.Tensors...)
	for i := range cloned.Tensors {
		cloned.Tensors[i].Shape = append([]uint64(nil), profile.Tensors[i].Shape...)
		cloned.Tensors[i].CodesShape = append([]uint64(nil), profile.Tensors[i].CodesShape...)
		cloned.Tensors[i].CodebookShape = append([]uint64(nil), profile.Tensors[i].CodebookShape...)
	}
	return &cloned
}

func validIndexBits(bits int) bool {
	switch bits {
	case 8, 16, 32:
		return true
	default:
		return false
	}
}

func defaultCodesName(name string) string {
	return name + ".codes"
}

func defaultTableName(name string) string {
	return name + ".codebook"
}
