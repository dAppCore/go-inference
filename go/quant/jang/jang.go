// SPDX-Licence-Identifier: EUPL-1.2

// Package jang holds the driver-neutral JANG/JANGTQ quantisation metadata
// + portable packed-tensor descriptor + reference dequant for parity tests.
//
//	info, _ := jang.ReadConfig("/models/minimax-m2-jangtq")
//	desc, _ := jang.NewPackedTensorDescriptor("model.layers.0.self_attn.q_proj.weight", shape, info)
package jang

import (
	core "dappco.re/go"
)

//	info := jang.Info{Profile: "JANGTQ", GroupSize: 64}
type Info struct {
	Version            int           `json:"version,omitempty"`
	WeightFormat       string        `json:"weight_format,omitempty"`
	Profile            string        `json:"profile,omitempty"`
	Method             string        `json:"method,omitempty"`
	GroupSize          int           `json:"group_size,omitempty"`
	BitsDefault        int           `json:"bits_default,omitempty"`
	AttentionBits      int           `json:"attention_bits,omitempty"`
	SharedExpertBits   int           `json:"shared_expert_bits,omitempty"`
	RoutedExpertBits   int           `json:"routed_expert_bits,omitempty"`
	EmbedTokensBits    int           `json:"embed_tokens_bits,omitempty"`
	LMHeadBits         int           `json:"lm_head_bits,omitempty"`
	SourceName         string        `json:"source_name,omitempty"`
	SourceOrg          string        `json:"source_org,omitempty"`
	SourceArchitecture string        `json:"source_architecture,omitempty"`
	Capabilities       Capabilities  `json:"capabilities,omitempty"`
	Packed             *PackedProfile `json:"packed,omitempty"`
}

//	caps := jang.Capabilities{ReasoningParser: "qwen-think", SupportsTools: true}
type Capabilities struct {
	ReasoningParser  string `json:"reasoning_parser,omitempty"`
	ToolParser       string `json:"tool_parser,omitempty"`
	ThinkInTemplate  bool   `json:"think_in_template,omitempty"`
	SupportsTools    bool   `json:"supports_tools,omitempty"`
	SupportsThinking bool   `json:"supports_thinking,omitempty"`
	Family           string `json:"family,omitempty"`
	Modality         string `json:"modality,omitempty"`
	CacheType        string `json:"cache_type,omitempty"`
}

//	role := jang.TensorRoleAttention
type TensorRole string

const (
	TensorRoleDefault      TensorRole = "default"
	TensorRoleAttention    TensorRole = "attention"
	TensorRoleSharedExpert TensorRole = "shared_expert"
	TensorRoleRoutedExpert TensorRole = "routed_expert"
	TensorRoleEmbedTokens  TensorRole = "embed_tokens"
	TensorRoleLMHead       TensorRole = "lm_head"
)

const (
	BitOrderLSB0   = "lsb0"
	EncodingAffine = "affine"
)

//	profile := jang.BuildPackedProfile(&info)
type PackedProfile struct {
	Type          string         `json:"type,omitempty"`
	Format        string         `json:"format,omitempty"`
	Profile       string         `json:"profile,omitempty"`
	Method        string         `json:"method,omitempty"`
	GroupSize     int            `json:"group_size,omitempty"`
	BitsDefault   int            `json:"bits_default,omitempty"`
	RoleBits      map[string]int `json:"role_bits,omitempty"`
	MinBits       int            `json:"min_bits,omitempty"`
	MaxBits       int            `json:"max_bits,omitempty"`
	Mixed         bool           `json:"mixed,omitempty"`
	BitOrder      string         `json:"bit_order,omitempty"`
	Encoding      string         `json:"encoding,omitempty"`
	ValuesPerByte int            `json:"values_per_byte,omitempty"`
}

//	desc, _ := jang.NewPackedTensorDescriptor(name, shape, &info)
type PackedTensorDescriptor struct {
	Name          string     `json:"name,omitempty"`
	Type          string     `json:"type,omitempty"`
	Format        string     `json:"format,omitempty"`
	Profile       string     `json:"profile,omitempty"`
	Role          TensorRole `json:"role,omitempty"`
	Shape         []uint64   `json:"shape,omitempty"`
	Elements      uint64     `json:"elements,omitempty"`
	Bits          int        `json:"bits,omitempty"`
	GroupSize     int        `json:"group_size,omitempty"`
	Groups        int        `json:"groups,omitempty"`
	PackedBytes   int        `json:"packed_bytes,omitempty"`
	ValuesPerByte int        `json:"values_per_byte,omitempty"`
	ScaleCount    int        `json:"scale_count,omitempty"`
	BiasCount     int        `json:"bias_count,omitempty"`
	BitOrder      string     `json:"bit_order,omitempty"`
	Encoding      string     `json:"encoding,omitempty"`
}

type configProbe struct {
	Version      int    `json:"version"`
	WeightFormat string `json:"weight_format"`
	Profile      string `json:"profile"`
	SourceModel  struct {
		Name         string `json:"name"`
		Org          string `json:"org"`
		Architecture string `json:"architecture"`
	} `json:"source_model"`
	MXTQBits struct {
		Attention    int `json:"attention"`
		SharedExpert int `json:"shared_expert"`
		RoutedExpert int `json:"routed_expert"`
		EmbedTokens  int `json:"embed_tokens"`
		LMHead       int `json:"lm_head"`
	} `json:"mxtq_bits"`
	Quantization struct {
		Method      string `json:"method"`
		GroupSize   int    `json:"group_size"`
		BitsDefault int    `json:"bits_default"`
	} `json:"quantization"`
	Capabilities Capabilities `json:"capabilities"`
}

//	info, _ := jang.ReadConfig("/models/minimax-m2")
func ReadConfig(root string) (*Info, error) {
	read := core.ReadFile(core.PathJoin(root, "jang_config.json"))
	if !read.OK {
		if core.IsNotExist(read.Value.(error)) {
			return nil, nil
		}
		return nil, read.Value.(error)
	}
	return ParseConfig(read.Value.([]byte))
}

//	info, _ := jang.ParseConfig(data)
func ParseConfig(data []byte) (*Info, error) {
	var probe configProbe
	if result := core.JSONUnmarshal(data, &probe); !result.OK {
		return nil, result.Value.(error)
	}
	return finalize(&Info{
		Version:            probe.Version,
		WeightFormat:       probe.WeightFormat,
		Profile:            probe.Profile,
		Method:             probe.Quantization.Method,
		GroupSize:          probe.Quantization.GroupSize,
		BitsDefault:        firstPositive(probe.Quantization.BitsDefault, probe.MXTQBits.RoutedExpert, ProfileBits(probe.Profile)),
		AttentionBits:      probe.MXTQBits.Attention,
		SharedExpertBits:   probe.MXTQBits.SharedExpert,
		RoutedExpertBits:   probe.MXTQBits.RoutedExpert,
		EmbedTokensBits:    probe.MXTQBits.EmbedTokens,
		LMHeadBits:         probe.MXTQBits.LMHead,
		SourceName:         probe.SourceModel.Name,
		SourceOrg:          probe.SourceModel.Org,
		SourceArchitecture: normaliseArchitecture(probe.SourceModel.Architecture),
		Capabilities:       probe.Capabilities,
	}), nil
}

//	bits := jang.ProfileBits("JANG_4M")  // returns 4
func ProfileBits(profile string) int {
	profile = core.Lower(profile)
	switch {
	case core.Contains(profile, "jangtq"):
		return 2
	case core.Contains(profile, "jang_1"):
		return 1
	case core.Contains(profile, "jang_2"):
		return 2
	case core.Contains(profile, "jang_3"):
		return 3
	case core.Contains(profile, "jang_4"):
		return 4
	default:
		return 0
	}
}

func quantizationType(info *Info) string {
	if info == nil {
		return ""
	}
	lower := core.Lower(core.Concat(info.Profile, " ", info.WeightFormat, " ", info.Method))
	if core.Contains(lower, "jangtq") || core.Contains(lower, "mxtq") {
		return "jangtq"
	}
	return "jang"
}

func finalize(info *Info) *Info {
	if info == nil {
		return nil
	}
	info.Packed = BuildPackedProfile(info)
	return info
}

//	profile := jang.BuildPackedProfile(&info)
func BuildPackedProfile(info *Info) *PackedProfile {
	if info == nil {
		return nil
	}
	rb := roleBits(info)
	minBits, maxBits := minMaxBits(rb)
	profile := &PackedProfile{
		Type:          quantizationType(info),
		Format:        packedFormat(info),
		Profile:       info.Profile,
		Method:        info.Method,
		GroupSize:     info.GroupSize,
		BitsDefault:   info.BitsDefault,
		RoleBits:      rb,
		MinBits:       minBits,
		MaxBits:       maxBits,
		Mixed:         minBits > 0 && maxBits > minBits,
		BitOrder:      BitOrderLSB0,
		Encoding:      EncodingAffine,
		ValuesPerByte: valuesPerByte(info.BitsDefault),
	}
	if profile.Format == "" {
		profile.Format = profile.Type
	}
	return profile
}

//	clone := jang.ClonePackedProfile(profile)
func ClonePackedProfile(profile *PackedProfile) *PackedProfile {
	if profile == nil {
		return nil
	}
	cloned := *profile
	cloned.RoleBits = cloneRoleBits(profile.RoleBits)
	return &cloned
}

//	desc, _ := jang.NewPackedTensorDescriptor("model.layers.0.q_proj.weight", []uint64{4096, 4096}, &info)
func NewPackedTensorDescriptor(name string, shape []uint64, info *Info) (PackedTensorDescriptor, error) {
	if info == nil {
		return PackedTensorDescriptor{}, core.NewError("jang: packed tensor descriptor requires quantization info")
	}
	role := inferTensorRole(name)
	bits := bitsForRole(info, role)
	elements, err := shapeElements(shape)
	if err != nil {
		return PackedTensorDescriptor{}, err
	}
	if err := validateBits(bits, name); err != nil {
		return PackedTensorDescriptor{}, err
	}
	if info.GroupSize <= 0 {
		return PackedTensorDescriptor{}, core.NewError(core.Sprintf("jang: packed tensor %q has invalid group size %d", name, info.GroupSize))
	}
	if elements > ^uint64(0)/uint64(bits) {
		return PackedTensorDescriptor{}, core.NewError(core.Sprintf("jang: packed tensor %q packed bit count overflows", name))
	}
	packedBits := elements * uint64(bits)
	packedBytes := ceilDivUint64(packedBits, 8)
	if packedBytes > uint64(maxIntValue()) {
		return PackedTensorDescriptor{}, core.NewError(core.Sprintf("jang: packed tensor %q is too large", name))
	}
	groups := ceilDivUint64(elements, uint64(info.GroupSize))
	if groups > uint64(maxIntValue()) {
		return PackedTensorDescriptor{}, core.NewError(core.Sprintf("jang: packed tensor %q has too many groups", name))
	}
	return PackedTensorDescriptor{
		Name:          name,
		Type:          quantizationType(info),
		Format:        packedFormat(info),
		Profile:       info.Profile,
		Role:          role,
		Shape:         append([]uint64(nil), shape...),
		Elements:      elements,
		Bits:          bits,
		GroupSize:     info.GroupSize,
		Groups:        int(groups),
		PackedBytes:   int(packedBytes),
		ValuesPerByte: valuesPerByte(bits),
		ScaleCount:    int(groups),
		BiasCount:     int(groups),
		BitOrder:      BitOrderLSB0,
		Encoding:      EncodingAffine,
	}, nil
}

//	err := jang.ValidatePackedTensor(desc, packed, scales, biases)
func ValidatePackedTensor(desc PackedTensorDescriptor, packed []byte, scales, biases []float32) error {
	if err := validateDescriptor(desc); err != nil {
		return err
	}
	if len(packed) != desc.PackedBytes {
		return core.NewError(core.Sprintf("jang: packed tensor %q packed length %d, expected %d", desc.Name, len(packed), desc.PackedBytes))
	}
	if len(scales) != desc.ScaleCount {
		return core.NewError(core.Sprintf("jang: packed tensor %q scale count %d, expected %d", desc.Name, len(scales), desc.ScaleCount))
	}
	if len(biases) != desc.BiasCount {
		return core.NewError(core.Sprintf("jang: packed tensor %q bias count %d, expected %d", desc.Name, len(biases), desc.BiasCount))
	}
	return nil
}

//	values, _ := jang.DequantizePackedTensor(desc, packed, scales, biases)
func DequantizePackedTensor(desc PackedTensorDescriptor, packed []byte, scales, biases []float32) ([]float32, error) {
	if err := ValidatePackedTensor(desc, packed, scales, biases); err != nil {
		return nil, err
	}
	if desc.Elements > uint64(maxIntValue()) {
		return nil, core.NewError(core.Sprintf("jang: packed tensor %q is too large to dequantize on CPU", desc.Name))
	}
	out := make([]float32, int(desc.Elements))
	groupSize := desc.GroupSize
	// Dispatch by bit-width once outside the loop so the inner unpack
	// becomes a single shift+mask the Go compiler can keep in registers,
	// rather than paying the un-inlinable unpackValue call on every
	// element. The dispatch also lets us hoist scale/bias per group —
	// the original loop re-indexed scales[i/groupSize] + biases[i/groupSize]
	// on every element, which is groupSize-1 redundant indexed reads + a
	// division per group (with groupSize=64, that's a 64× reduction in
	// per-element scale/bias work).
	switch desc.Bits {
	case 8:
		dequantizeBit8(out, packed, scales, biases, groupSize)
	case 4:
		dequantizeBit4(out, packed, scales, biases, groupSize)
	case 2:
		dequantizeBit2(out, packed, scales, biases, groupSize)
	case 1:
		dequantizeBit1(out, packed, scales, biases, groupSize)
	default:
		// Generic walk for non-power-of-2 widths (3-bit and any future
		// awkward width). Inline the bit-walk so we sidestep the
		// fast-path switch in unpackValue — the outer dispatch already
		// proved we won't hit a byte-aligned width here. Outer loop
		// still hoists scale/bias per group.
		dequantizeBitGeneric(out, packed, scales, biases, groupSize, desc.Bits)
	}
	return out, nil
}

// dequantizeBit8 walks the 8-bit-aligned packed path with the unpack
// inlined. One byte per element, no shift required.
func dequantizeBit8(out []float32, packed []byte, scales, biases []float32, groupSize int) {
	for i := 0; i < len(out); {
		group := i / groupSize
		end := (group + 1) * groupSize
		if end > len(out) {
			end = len(out)
		}
		scale := scales[group]
		bias := biases[group]
		for ; i < end; i++ {
			out[i] = float32(packed[i])*scale + bias
		}
	}
}

// dequantizeBit4 walks the 4-bit-nibble-packed path with the unpack
// inlined. Two values per byte; low nibble for even indices, high
// nibble for odd indices.
//
// When the per-group walk lands on a byte boundary we batch 2 elements
// per byte read — amortises the packed-slice load + bounds check across
// both nibble lanes. JANGTQ-style groupSize=64 (== 32 bytes at 4-bit)
// lands on a byte boundary at every group start, so the fast path
// covers the full group body. Single-element prefix + suffix handle
// the rare case where the row's start offset is mid-byte or the group
// runs short at the tensor tail.
//
// The natural if/else for nibble select (rather than a branchless
// bit-mux) avoids the Apple Silicon FCMPD-over-FMOV penalty observed
// when bit-mux-style code regresses against direct branches on M3.
func dequantizeBit4(out []float32, packed []byte, scales, biases []float32, groupSize int) {
	for i := 0; i < len(out); {
		group := i / groupSize
		end := (group + 1) * groupSize
		if end > len(out) {
			end = len(out)
		}
		scale := scales[group]
		bias := biases[group]
		// Drain prefix elements until i is byte-aligned (i&1 == 0).
		if i&1 != 0 && i < end {
			b := packed[i>>1]
			out[i] = float32(b>>4)*scale + bias
			i++
		}
		// Walk 2-at-a-time on byte-aligned boundaries.
		for i+2 <= end {
			b := packed[i>>1]
			out[i] = float32(b&0x0F)*scale + bias
			out[i+1] = float32(b>>4)*scale + bias
			i += 2
		}
		// Drain suffix.
		for ; i < end; i++ {
			b := packed[i>>1]
			if i&1 == 0 {
				out[i] = float32(b&0x0F)*scale + bias
			} else {
				out[i] = float32(b>>4)*scale + bias
			}
		}
	}
}

// dequantizeBit2 walks the 2-bit-packed path with the unpack inlined.
// Four values per byte; the shift is `(i&3)<<1`. This is the dominant
// MiniMax M2 routed-expert weight path.
//
// When the per-group walk lands on a byte boundary we batch 4 elements
// per byte read — amortises the packed-slice load across the four 2-bit
// lanes. The JANGTQ default groupSize=64 (16 bytes at 2-bit) lands on a
// byte boundary at every group start, so the fast path covers the full
// group body. Single-element prefix + suffix handles the (rare) case
// where the group runs short at the tensor tail.
func dequantizeBit2(out []float32, packed []byte, scales, biases []float32, groupSize int) {
	for i := 0; i < len(out); {
		group := i / groupSize
		end := (group + 1) * groupSize
		if end > len(out) {
			end = len(out)
		}
		scale := scales[group]
		bias := biases[group]
		// Drain prefix elements until i is byte-aligned (i&3 == 0).
		for ; i < end && (i&3) != 0; i++ {
			q := (packed[i>>2] >> uint((i&3)<<1)) & 0x03
			out[i] = float32(q)*scale + bias
		}
		// Walk 4-at-a-time on byte-aligned boundaries.
		for i+4 <= end {
			b := packed[i>>2]
			out[i] = float32(b&0x03)*scale + bias
			out[i+1] = float32((b>>2)&0x03)*scale + bias
			out[i+2] = float32((b>>4)&0x03)*scale + bias
			out[i+3] = float32((b>>6)&0x03)*scale + bias
			i += 4
		}
		// Drain suffix.
		for ; i < end; i++ {
			q := (packed[i>>2] >> uint((i&3)<<1)) & 0x03
			out[i] = float32(q)*scale + bias
		}
	}
}

// dequantizeBit1 walks the 1-bit-packed path with the unpack inlined.
// Eight values per byte; mask + shift only.
func dequantizeBit1(out []float32, packed []byte, scales, biases []float32, groupSize int) {
	for i := 0; i < len(out); {
		group := i / groupSize
		end := (group + 1) * groupSize
		if end > len(out) {
			end = len(out)
		}
		scale := scales[group]
		bias := biases[group]
		for ; i < end; i++ {
			q := (packed[i>>3] >> uint(i&7)) & 0x01
			out[i] = float32(q)*scale + bias
		}
	}
}

// dequantizeBitGeneric walks any non-power-of-2 packed width (e.g. 3-bit)
// with the bit-walk inlined directly. The outer DequantizePackedTensor
// dispatch already proved we won't hit a byte-aligned width here, so we
// skip the fast-path switch in unpackValue that would otherwise pay 4
// extra comparisons per element.
func dequantizeBitGeneric(out []float32, packed []byte, scales, biases []float32, groupSize, bits int) {
	for i := 0; i < len(out); {
		group := i / groupSize
		end := (group + 1) * groupSize
		if end > len(out) {
			end = len(out)
		}
		scale := scales[group]
		bias := biases[group]
		for ; i < end; i++ {
			bitOffset := i * bits
			remaining := bits
			shiftOut := 0
			value := uint16(0)
			for remaining > 0 {
				byteIndex := bitOffset / 8
				shiftIn := bitOffset % 8
				take := remaining
				if avail := 8 - shiftIn; avail < take {
					take = avail
				}
				mask := uint16((1 << take) - 1)
				chunk := (uint16(packed[byteIndex]) >> shiftIn) & mask
				value |= chunk << shiftOut
				remaining -= take
				bitOffset += take
				shiftOut += take
			}
			out[i] = float32(uint8(value))*scale + bias
		}
	}
}

//	packed, _ := jang.PackQuantizedValues(desc, values)
func PackQuantizedValues(desc PackedTensorDescriptor, values []uint8) ([]byte, error) {
	if err := validateDescriptor(desc); err != nil {
		return nil, err
	}
	if uint64(len(values)) != desc.Elements {
		return nil, core.NewError(core.Sprintf("jang: packed tensor %q value count %d, expected %d", desc.Name, len(values), desc.Elements))
	}
	out := make([]byte, desc.PackedBytes)
	maxValue := uint8((1 << desc.Bits) - 1)
	for i, value := range values {
		if value > maxValue {
			return nil, core.NewError(core.Sprintf("jang: packed tensor %q value %d exceeds %d-bit max %d", desc.Name, value, desc.Bits, maxValue))
		}
		writeValue(out, i, desc.Bits, value)
	}
	return out, nil
}

func inferTensorRole(name string) TensorRole {
	lower := core.Lower(name)
	switch {
	case core.Contains(lower, "embed_tokens"):
		return TensorRoleEmbedTokens
	case core.Contains(lower, "lm_head"):
		return TensorRoleLMHead
	case core.Contains(lower, "shared_expert"):
		return TensorRoleSharedExpert
	case core.Contains(lower, "experts.") || core.Contains(lower, "block_sparse_moe"):
		return TensorRoleRoutedExpert
	case core.Contains(lower, "self_attn") || core.Contains(lower, ".attention.") || core.Contains(lower, ".q_proj") || core.Contains(lower, ".k_proj") || core.Contains(lower, ".v_proj") || core.Contains(lower, ".o_proj"):
		return TensorRoleAttention
	default:
		return TensorRoleDefault
	}
}

func bitsForRole(info *Info, role TensorRole) int {
	switch role {
	case TensorRoleAttention:
		return firstPositive(info.AttentionBits, info.BitsDefault, ProfileBits(info.Profile))
	case TensorRoleSharedExpert:
		return firstPositive(info.SharedExpertBits, info.BitsDefault, ProfileBits(info.Profile))
	case TensorRoleRoutedExpert:
		return firstPositive(info.RoutedExpertBits, info.BitsDefault, ProfileBits(info.Profile))
	case TensorRoleEmbedTokens:
		return firstPositive(info.EmbedTokensBits, info.BitsDefault, ProfileBits(info.Profile))
	case TensorRoleLMHead:
		return firstPositive(info.LMHeadBits, info.BitsDefault, ProfileBits(info.Profile))
	default:
		return firstPositive(info.BitsDefault, ProfileBits(info.Profile))
	}
}

func roleBits(info *Info) map[string]int {
	if info == nil {
		return nil
	}
	roles := []TensorRole{
		TensorRoleDefault,
		TensorRoleAttention,
		TensorRoleSharedExpert,
		TensorRoleRoutedExpert,
		TensorRoleEmbedTokens,
		TensorRoleLMHead,
	}
	out := map[string]int{}
	for _, role := range roles {
		if bits := bitsForRole(info, role); bits > 0 {
			out[string(role)] = bits
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func minMaxBits(rb map[string]int) (int, int) {
	minBits, maxBits := 0, 0
	for _, bits := range rb {
		if bits <= 0 {
			continue
		}
		if minBits == 0 || bits < minBits {
			minBits = bits
		}
		if bits > maxBits {
			maxBits = bits
		}
	}
	return minBits, maxBits
}

func packedFormat(info *Info) string {
	if info == nil {
		return ""
	}
	lower := core.Lower(core.Concat(info.WeightFormat, " ", info.Profile, " ", info.Method))
	switch {
	case core.Contains(lower, "mxtq"):
		return "mxtq"
	case core.Contains(lower, "jangtq"):
		return "jangtq"
	case core.Contains(lower, "jang"):
		return "jang"
	default:
		return core.Lower(info.WeightFormat)
	}
}

func valuesPerByte(bits int) int {
	if bits <= 0 {
		return 0
	}
	return 8 / bits
}

func shapeElements(shape []uint64) (uint64, error) {
	if len(shape) == 0 {
		return 0, core.NewError("jang: packed tensor shape is required")
	}
	elements := uint64(1)
	for _, dim := range shape {
		if dim == 0 {
			return 0, core.NewError("jang: packed tensor shape contains zero dimension")
		}
		if elements > ^uint64(0)/dim {
			return 0, core.NewError("jang: packed tensor shape overflows element count")
		}
		elements *= dim
	}
	return elements, nil
}

func validateDescriptor(desc PackedTensorDescriptor) error {
	if desc.Elements == 0 {
		return core.NewError(core.Sprintf("jang: packed tensor %q has no elements", desc.Name))
	}
	if err := validateBits(desc.Bits, desc.Name); err != nil {
		return err
	}
	if desc.GroupSize <= 0 {
		return core.NewError(core.Sprintf("jang: packed tensor %q has invalid group size %d", desc.Name, desc.GroupSize))
	}
	if desc.PackedBytes <= 0 {
		return core.NewError(core.Sprintf("jang: packed tensor %q has invalid packed byte count %d", desc.Name, desc.PackedBytes))
	}
	if desc.ScaleCount <= 0 || desc.BiasCount <= 0 {
		return core.NewError(core.Sprintf("jang: packed tensor %q has invalid scale/bias counts", desc.Name))
	}
	return nil
}

func validateBits(bits int, name string) error {
	switch bits {
	case 1, 2, 3, 4, 8:
		return nil
	default:
		return core.NewError(core.Sprintf("jang: packed tensor %q has unsupported %d-bit width", name, bits))
	}
}

func unpackValue(packed []byte, index, bits int) uint8 {
	// Fast paths for the byte-aligned bit widths emitted by the JANG
	// packers (1-bit binary, 2-bit JANGTQ routed-expert, 4-bit nibble
	// JANG_4, 8-bit dense). These cover the overwhelming majority of
	// real model-load dequant calls and bypass the generic walk loop,
	// which fires hundreds of millions of times per tensor materialise.
	switch bits {
	case 8:
		return packed[index]
	case 4:
		b := packed[index>>1]
		if index&1 == 0 {
			return b & 0x0F
		}
		return b >> 4
	case 2:
		return (packed[index>>2] >> uint((index&3)<<1)) & 0x03
	case 1:
		return (packed[index>>3] >> uint(index&7)) & 0x01
	}
	bitOffset := index * bits
	remaining := bits
	shiftOut := 0
	value := uint16(0)
	for remaining > 0 {
		byteIndex := bitOffset / 8
		shiftIn := bitOffset % 8
		take := minInt(remaining, 8-shiftIn)
		mask := uint16((1 << take) - 1)
		chunk := (uint16(packed[byteIndex]) >> shiftIn) & mask
		value |= chunk << shiftOut
		remaining -= take
		bitOffset += take
		shiftOut += take
	}
	return uint8(value)
}

func writeValue(out []byte, index, bits int, value uint8) {
	bitOffset := index * bits
	remaining := bits
	raw := uint16(value)
	for remaining > 0 {
		byteIndex := bitOffset / 8
		shift := bitOffset % 8
		take := minInt(remaining, 8-shift)
		mask := uint16((1 << take) - 1)
		out[byteIndex] |= byte((raw & mask) << shift)
		raw >>= take
		remaining -= take
		bitOffset += take
	}
}

func cloneRoleBits(rb map[string]int) map[string]int {
	if len(rb) == 0 {
		return nil
	}
	cloned := make(map[string]int, len(rb))
	for key, value := range rb {
		cloned[key] = value
	}
	return cloned
}

func ceilDivUint64(value, divisor uint64) uint64 {
	if divisor == 0 || value == 0 {
		return 0
	}
	quotient := value / divisor
	if value%divisor != 0 {
		quotient++
	}
	return quotient
}

func maxIntValue() int {
	return int(^uint(0) >> 1)
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func firstPositive(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func normaliseArchitecture(value string) string {
	value = core.Lower(core.Trim(value))
	value = core.Replace(value, "-", "_")
	switch value {
	case "qwen3_5":
		return "qwen3_next"
	case "minimaxm2", "minimax_m2":
		return "minimax_m2"
	case "mixtral":
		return "mixtral"
	case "mistral":
		return "mistral"
	case "phi", "phi3", "phi4":
		return "phi"
	case "deepseek", "deepseek_v3", "deepseek_r1":
		return "deepseek"
	case "gptoss", "gpt_oss", "gpt_oss_model":
		return "gpt_oss"
	case "bert":
		return "bert"
	case "bert_rerank", "bert_cross_encoder":
		return "bert_rerank"
	default:
		return value
	}
}
