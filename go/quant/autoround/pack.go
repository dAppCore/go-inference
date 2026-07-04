// SPDX-Licence-Identifier: EUPL-1.2

package autoround

import (
	core "dappco.re/go"
	"dappco.re/go/inference/safetensors"
)

const (
	PackConfigFileAutoRound    = "auto_round_config.json"
	PackConfigFileQuantization = "quantization_config.json"

	QuantMethodAutoRound = "auto-round"
	QuantFamilyAutoRound = "auto-round"
)

type PackInfo struct {
	Path             string                 `json:"path,omitempty"`
	Bits             int                    `json:"bits,omitempty"`
	GroupSize        int                    `json:"group_size,omitempty"`
	Symmetric        bool                   `json:"sym,omitempty"`
	DataType         string                 `json:"data_type,omitempty"`
	Iters            int                    `json:"iters,omitempty"`
	NSamples         int                    `json:"nsamples,omitempty"`
	SeqLen           int                    `json:"seqlen,omitempty"`
	AutoRoundVersion string                 `json:"autoround_version,omitempty"`
	QuantMethod      string                 `json:"quant_method,omitempty"`
	PackingFormat    string                 `json:"packing_format,omitempty"`
	Scheme           Scheme                 `json:"scheme,omitempty"`
	ExportFormat     ExportFormat           `json:"export_format,omitempty"`
	Tensors          []PackTensor           `json:"tensors,omitempty"`
	TensorCount      int                    `json:"tensor_count,omitempty"`
	LayerOverrides   map[string]LayerConfig `json:"extra_config,omitempty"`
	LayerOverrideN   int                    `json:"layer_override_count,omitempty"`
}

type PackTensor struct {
	Name        string  `json:"name"`
	Packed      string  `json:"packed"`
	Scales      string  `json:"scales"`
	ZeroPoints  string  `json:"zero_points"`
	Bias        string  `json:"bias,omitempty"`
	Shape       []int32 `json:"shape"`
	Bits        int     `json:"bits,omitempty"`
	GroupSize   int     `json:"group_size,omitempty"`
	Symmetric   bool    `json:"sym,omitempty"`
	PackedBytes int     `json:"packed_bytes,omitempty"`
	Groups      int     `json:"groups,omitempty"`
	QMin        int     `json:"qmin,omitempty"`
	QMax        int     `json:"qmax,omitempty"`
}

type LayerConfig struct {
	Bits      int   `json:"bits,omitempty"`
	GroupSize int   `json:"group_size,omitempty"`
	Symmetric *bool `json:"sym,omitempty"`
}

func ReadPackInfo(root string) (*PackInfo, error) {
	path := core.PathJoin(root, PackConfigFileAutoRound)
	info, err := readPackInfoFile(path, true)
	if err == nil && info != nil {
		return info, nil
	}
	if err != nil && !core.IsNotExist(err) {
		return nil, err
	}
	path = core.PathJoin(root, PackConfigFileQuantization)
	info, err = readPackInfoFile(path, false)
	if err == nil && info != nil {
		return info, nil
	}
	if err != nil && !core.IsNotExist(err) {
		return nil, err
	}
	return nil, nil
}

func ReadPackInfoFile(path string) (*PackInfo, error) {
	return readPackInfoFile(path, true)
}

func readPackInfoFile(path string, requireAutoRound bool) (*PackInfo, error) {
	read := core.ReadFile(path)
	if !read.OK {
		return nil, read.Value.(error)
	}
	var info PackInfo
	if result := core.JSONUnmarshal(read.Value.([]byte), &info); !result.OK {
		return nil, result.Value.(error)
	}
	info.Path = path
	info.normalise()
	if info.QuantMethod != QuantMethodAutoRound && !requireAutoRound {
		return nil, nil
	}
	if err := info.Validate(); err != nil {
		return nil, err
	}
	return &info, nil
}

func ClonePackInfo(info *PackInfo) *PackInfo {
	if info == nil {
		return nil
	}
	cloned := *info
	if len(info.LayerOverrides) > 0 {
		cloned.LayerOverrides = make(map[string]LayerConfig, len(info.LayerOverrides))
		for key, value := range info.LayerOverrides {
			cloned.LayerOverrides[key] = value
		}
	}
	if len(info.Tensors) > 0 {
		cloned.Tensors = make([]PackTensor, len(info.Tensors))
		for i, tensor := range info.Tensors {
			cloned.Tensors[i] = tensor
			cloned.Tensors[i].Shape = core.SliceClone(tensor.Shape)
		}
	}
	return &cloned
}

func (info *PackInfo) normalise() {
	if info == nil {
		return
	}
	info.QuantMethod = normaliseQuantMethod(info.QuantMethod)
	info.PackingFormat = normalisePackingFormat(info.PackingFormat)
	info.DataType = core.Lower(core.Trim(info.DataType))
	if info.Scheme == "" {
		info.Scheme = info.inferScheme()
	} else {
		info.Scheme = normaliseScheme(info.Scheme)
	}
	if info.ExportFormat == "" {
		info.ExportFormat = info.inferExportFormat()
	}
	for i := range info.Tensors {
		info.Tensors[i].normalise(*info)
	}
	info.TensorCount = len(info.Tensors)
	info.LayerOverrideN = len(info.LayerOverrides)
}

func (info PackInfo) Validate() error {
	if info.QuantMethod != QuantMethodAutoRound {
		return core.NewError("autoround: quant_method must be auto-round")
	}
	if info.Bits != 2 && info.Bits != 3 && info.Bits != 4 && info.Bits != 8 {
		return core.NewError("autoround: bits must be one of 2, 3, 4, or 8")
	}
	if info.GroupSize != 0 && info.GroupSize != 16 && info.GroupSize != 32 && info.GroupSize != 64 && info.GroupSize != 128 && info.GroupSize != 256 {
		return core.NewError("autoround: group size must be one of 16, 32, 64, 128, or 256")
	}
	if info.Iters < 0 {
		return core.NewError("autoround: iters must be non-negative")
	}
	if info.NSamples < 0 {
		return core.NewError("autoround: nsamples must be non-negative")
	}
	if info.SeqLen < 0 {
		return core.NewError("autoround: seqlen must be non-negative")
	}
	if info.Scheme != "" {
		if _, ok := ResolveScheme(info.Scheme); !ok {
			return core.NewError("autoround: unsupported scheme: " + string(info.Scheme))
		}
	}
	for _, tensor := range info.Tensors {
		if err := tensor.Validate(); err != nil {
			return err
		}
	}
	return nil
}

func (info PackInfo) NativeFormat() bool {
	format := core.Lower(core.Trim(info.PackingFormat))
	return format == "auto_round" || core.HasPrefix(format, "auto_round:")
}

func (info PackInfo) GGUFExport() bool {
	return info.ExportFormat == FormatGGUFQ4KM || core.Contains(core.Lower(info.PackingFormat), "gguf")
}

func (info PackInfo) NativeTensorMap() bool {
	return info.NativeFormat() && len(info.Tensors) > 0
}

func (tensor *PackTensor) normalise(info PackInfo) {
	if tensor == nil {
		return
	}
	tensor.Name = core.Trim(tensor.Name)
	tensor.Packed = core.Trim(tensor.Packed)
	tensor.Scales = core.Trim(tensor.Scales)
	tensor.ZeroPoints = core.Trim(tensor.ZeroPoints)
	tensor.Bias = core.Trim(tensor.Bias)
	if tensor.Bits == 0 {
		tensor.Bits = info.Bits
	}
	if tensor.GroupSize == 0 {
		tensor.GroupSize = info.GroupSize
	}
	if !tensor.Symmetric {
		tensor.Symmetric = info.Symmetric
	}
	qmin, qmax := quantRange(QuantizeConfig{Bits: tensor.Bits, Symmetric: tensor.Symmetric})
	if tensor.QMin == 0 && tensor.QMax == 0 {
		tensor.QMin = qmin
		tensor.QMax = qmax
	}
	elements, err := packedShapeElements(tensor.Shape)
	if err == nil {
		if tensor.PackedBytes == 0 {
			tensor.PackedBytes = (elements*tensor.Bits + 7) / 8
		}
		if tensor.Groups == 0 && tensor.GroupSize > 0 {
			tensor.Groups = (elements + tensor.GroupSize - 1) / tensor.GroupSize
		}
	}
}

func (tensor PackTensor) Validate() error {
	if tensor.Name == "" {
		return core.NewError("autoround: tensor name is required")
	}
	if tensor.Packed == "" || tensor.Scales == "" || tensor.ZeroPoints == "" {
		return core.NewError("autoround: tensor map requires packed, scales, and zero_points tensors")
	}
	if tensor.Bits != 2 && tensor.Bits != 3 && tensor.Bits != 4 && tensor.Bits != 8 {
		return core.NewError("autoround: tensor bits must be one of 2, 3, 4, or 8")
	}
	if tensor.GroupSize <= 0 {
		return core.NewError("autoround: tensor group size must be positive")
	}
	elements, err := packedShapeElements(tensor.Shape)
	if err != nil {
		return err
	}
	expectedPacked := (elements*tensor.Bits + 7) / 8
	if tensor.PackedBytes != expectedPacked {
		return core.Errorf("autoround: tensor %s packed length %d, expected %d", tensor.Name, tensor.PackedBytes, expectedPacked)
	}
	expectedGroups := (elements + tensor.GroupSize - 1) / tensor.GroupSize
	if tensor.Groups != expectedGroups {
		return core.Errorf("autoround: tensor %s group count %d, expected %d", tensor.Name, tensor.Groups, expectedGroups)
	}
	return nil
}

func ValidateSafetensorsTensorMap(info PackInfo, weightFiles []string) error {
	if !info.NativeTensorMap() {
		return nil
	}
	index, err := safetensors.IndexFiles(weightFiles)
	if err != nil {
		return core.E("autoround.tensor_map", "index safetensors", err)
	}
	for _, tensor := range info.Tensors {
		if err := validateSafetensorsTensor(index, tensor.Packed, "U8", tensor.PackedBytes); err != nil {
			return err
		}
		if err := validateSafetensorsTensor(index, tensor.Scales, "F32", tensor.Groups); err != nil {
			return err
		}
		if err := validateSafetensorsTensor(index, tensor.ZeroPoints, "F32", tensor.Groups); err != nil {
			return err
		}
		if tensor.Bias != "" {
			if err := validateSafetensorsTensor(index, tensor.Bias, "F32", int(tensor.Shape[0])); err != nil {
				return err
			}
		}
	}
	return nil
}

func validateSafetensorsTensor(index safetensors.Index, name, dtype string, elements int) error {
	ref, ok := index.Tensors[name]
	if !ok {
		return core.NewError("autoround: tensor map missing safetensors tensor: " + name)
	}
	if ref.DType != dtype {
		return core.Errorf("autoround: tensor %s dtype %s, expected %s", name, ref.DType, dtype)
	}
	if ref.Elements != elements {
		return core.Errorf("autoround: tensor %s elements %d, expected %d", name, ref.Elements, elements)
	}
	return nil
}

func (info PackInfo) inferScheme() Scheme {
	format := core.Lower(info.PackingFormat)
	if core.Contains(format, "gguf:q4_k_m") || core.Contains(format, "gguf_q4_k_m") {
		return SchemeGGUFQ4KM
	}
	switch info.Bits {
	case 2:
		return SchemeW2A16
	case 4:
		return SchemeW4A16
	case 8:
		if info.DataType == "fp8" || info.DataType == "float8" {
			return SchemeFP8Static
		}
		return SchemeW8A16
	default:
		return ""
	}
}

func (info PackInfo) inferExportFormat() ExportFormat {
	format := core.Lower(info.PackingFormat)
	if core.Contains(format, "gguf:q4_k_m") || core.Contains(format, "gguf_q4_k_m") {
		return FormatGGUFQ4KM
	}
	return FormatAutoRound
}

func normaliseQuantMethod(value string) string {
	value = core.Replace(core.Lower(core.Trim(value)), "_", "-")
	if value == "autoround" {
		return QuantMethodAutoRound
	}
	return value
}

func normalisePackingFormat(value string) string {
	value = core.Trim(value)
	if value == "" {
		return string(FormatAutoRound)
	}
	return core.Lower(core.Replace(value, "_", "_"))
}
