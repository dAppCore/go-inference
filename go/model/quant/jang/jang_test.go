// SPDX-Licence-Identifier: EUPL-1.2

package jang

import (
	"testing"

	core "dappco.re/go"
)

func testJANGTQInfo() *Info {
	return &Info{
		Version:          2,
		WeightFormat:     "mxtq",
		Profile:          "JANGTQ",
		Method:           "affine+mxtq",
		GroupSize:        4,
		BitsDefault:      2,
		AttentionBits:    8,
		SharedExpertBits: 8,
		RoutedExpertBits: 2,
		EmbedTokensBits:  8,
		LMHeadBits:       8,
	}
}

func TestJang_TensorRoleRoutedExpert_Good(t *testing.T) {
	desc, err := NewPackedTensorDescriptor("model.layers.0.block_sparse_moe.experts.17.w1.weight", []uint64{2, 4}, testJANGTQInfo())
	if err != nil {
		t.Fatalf("NewPackedTensorDescriptor() error = %v", err)
	}

	if desc.Type != "jangtq" || desc.Format != "mxtq" || desc.Profile != "JANGTQ" {
		t.Fatalf("profile = type:%q format:%q profile:%q", desc.Type, desc.Format, desc.Profile)
	}
	if desc.Role != TensorRoleRoutedExpert || desc.Bits != 2 || desc.GroupSize != 4 {
		t.Fatalf("descriptor = %+v, want routed expert 2-bit group 4", desc)
	}
	if desc.Elements != 8 || desc.Groups != 2 || desc.PackedBytes != 2 || desc.ScaleCount != 2 || desc.BiasCount != 2 {
		t.Fatalf("descriptor sizes = %+v, want 8 elements, 2 groups, 2 packed bytes", desc)
	}
	if desc.BitOrder != BitOrderLSB0 || desc.Encoding != EncodingAffine {
		t.Fatalf("layout = bit_order:%q encoding:%q", desc.BitOrder, desc.Encoding)
	}
}

func TestJang_TensorRoleAttention_Good(t *testing.T) {
	desc, err := NewPackedTensorDescriptor("model.layers.0.self_attn.q_proj.weight", []uint64{2, 4}, testJANGTQInfo())
	if err != nil {
		t.Fatalf("NewPackedTensorDescriptor() error = %v", err)
	}

	if desc.Role != TensorRoleAttention || desc.Bits != 8 || desc.PackedBytes != 8 {
		t.Fatalf("descriptor = %+v, want attention 8-bit un-nibbled bytes", desc)
	}
}

func TestJang_PackedTensorDescriptorBadUnsupportedBits(t *testing.T) {
	info := testJANGTQInfo()
	info.RoutedExpertBits = 5

	_, err := NewPackedTensorDescriptor("model.layers.0.mlp.experts.0.down_proj.weight", []uint64{4, 4}, info)
	if err == nil || !core.Contains(err.Error(), "unsupported") || !core.Contains(err.Error(), "5-bit") {
		t.Fatalf("error = %v, want explicit unsupported 5-bit error", err)
	}
}

func TestJang_DequantizePackedTensor_Good(t *testing.T) {
	desc, err := NewPackedTensorDescriptor("model.layers.0.block_sparse_moe.experts.3.w2.weight", []uint64{8}, testJANGTQInfo())
	if err != nil {
		t.Fatalf("NewPackedTensorDescriptor() error = %v", err)
	}
	packed, err := PackQuantizedValues(desc, []uint8{0, 1, 2, 3, 0, 1, 2, 3})
	if err != nil {
		t.Fatalf("PackQuantizedValues() error = %v", err)
	}

	out, err := DequantizePackedTensor(desc, packed, []float32{0.5, 1}, []float32{-1, 10})
	if err != nil {
		t.Fatalf("DequantizePackedTensor() error = %v", err)
	}

	want := []float32{-1, -0.5, 0, 0.5, 10, 11, 12, 13}
	if len(out) != len(want) {
		t.Fatalf("out length = %d, want %d", len(out), len(want))
	}
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("out[%d] = %v, want %v (all=%v)", i, out[i], want[i], out)
		}
	}
}

func TestJang_ValidatePackedTensorBadPackedLength(t *testing.T) {
	desc, err := NewPackedTensorDescriptor("model.layers.0.block_sparse_moe.experts.3.w2.weight", []uint64{8}, testJANGTQInfo())
	if err != nil {
		t.Fatalf("NewPackedTensorDescriptor() error = %v", err)
	}

	err = ValidatePackedTensor(desc, []byte{0}, []float32{1, 1}, []float32{0, 0})
	if err == nil || !core.Contains(err.Error(), "packed length") {
		t.Fatalf("error = %v, want packed length validation", err)
	}
}

// roundTripFixture builds a descriptor at the requested bit width with the
// MXTQ routed-expert tensor name (the inferTensorRole route that picks up
// RoutedExpertBits) and feeds it crafted values such that every group is
// exercised. Returns descriptor + the values written in.
func roundTripFixture(t *testing.T, bits int, elements int, groupSize int) (PackedTensorDescriptor, []uint8, []byte, []float32, []float32) {
	t.Helper()
	info := &Info{
		Version:          2,
		WeightFormat:     "mxtq",
		Profile:          "JANGTQ",
		Method:           "affine+mxtq",
		GroupSize:        groupSize,
		BitsDefault:      bits,
		RoutedExpertBits: bits,
	}
	desc, err := NewPackedTensorDescriptor("model.layers.0.block_sparse_moe.experts.0.w1.weight", []uint64{uint64(elements)}, info)
	if err != nil {
		t.Fatalf("NewPackedTensorDescriptor(%d-bit): %v", bits, err)
	}
	maxValue := uint8((1 << bits) - 1)
	values := make([]uint8, desc.Elements)
	for i := range values {
		// Walk the full 0..maxValue range so every nibble/lane is touched.
		values[i] = uint8(i) & maxValue
	}
	packed, err := PackQuantizedValues(desc, values)
	if err != nil {
		t.Fatalf("PackQuantizedValues(%d-bit): %v", bits, err)
	}
	// Distinct per-group scale + bias so a regression that mis-indexes groups
	// surfaces as a wrong magnitude, not a hidden silent identity.
	scales := make([]float32, desc.ScaleCount)
	biases := make([]float32, desc.BiasCount)
	for i := range scales {
		scales[i] = 0.25 + float32(i)*0.0625
		biases[i] = -1 - float32(i)*0.5
	}
	return desc, values, packed, scales, biases
}

// expectedDequantize is the smallest possible reference dequant — pure
// per-element arithmetic with the generic unpack walk used by upstream
// before the W10-N specialisation. Used as the bit-exact oracle.
func expectedDequantize(t *testing.T, values []uint8, scales, biases []float32, groupSize int) []float32 {
	t.Helper()
	out := make([]float32, len(values))
	for i, v := range values {
		group := i / groupSize
		out[i] = float32(v)*scales[group] + biases[group]
	}
	return out
}

func TestJang_DequantizePackedTensor_RoundTrip_1bit(t *testing.T) {
	// 4096 elements with groupSize=64 to exercise the multi-group dispatch.
	desc, values, packed, scales, biases := roundTripFixture(t, 1, 4096, 64)
	got, err := DequantizePackedTensor(desc, packed, scales, biases)
	if err != nil {
		t.Fatalf("DequantizePackedTensor(1-bit): %v", err)
	}
	want := expectedDequantize(t, values, scales, biases, desc.GroupSize)
	assertBitExact(t, got, want)
}

func TestJang_DequantizePackedTensor_RoundTrip_2bit(t *testing.T) {
	desc, values, packed, scales, biases := roundTripFixture(t, 2, 4096, 64)
	got, err := DequantizePackedTensor(desc, packed, scales, biases)
	if err != nil {
		t.Fatalf("DequantizePackedTensor(2-bit): %v", err)
	}
	want := expectedDequantize(t, values, scales, biases, desc.GroupSize)
	assertBitExact(t, got, want)
}

func TestJang_DequantizePackedTensor_RoundTrip_3bit(t *testing.T) {
	// 3-bit hits the generic-walk default branch — the dequant must still
	// be bit-exact against the pre-specialisation oracle.
	desc, values, packed, scales, biases := roundTripFixture(t, 3, 4096, 64)
	got, err := DequantizePackedTensor(desc, packed, scales, biases)
	if err != nil {
		t.Fatalf("DequantizePackedTensor(3-bit): %v", err)
	}
	want := expectedDequantize(t, values, scales, biases, desc.GroupSize)
	assertBitExact(t, got, want)
}

func TestJang_DequantizePackedTensor_RoundTrip_4bit(t *testing.T) {
	desc, values, packed, scales, biases := roundTripFixture(t, 4, 4096, 64)
	got, err := DequantizePackedTensor(desc, packed, scales, biases)
	if err != nil {
		t.Fatalf("DequantizePackedTensor(4-bit): %v", err)
	}
	want := expectedDequantize(t, values, scales, biases, desc.GroupSize)
	assertBitExact(t, got, want)
}

func TestJang_DequantizePackedTensor_RoundTrip_8bit(t *testing.T) {
	desc, values, packed, scales, biases := roundTripFixture(t, 8, 4096, 64)
	got, err := DequantizePackedTensor(desc, packed, scales, biases)
	if err != nil {
		t.Fatalf("DequantizePackedTensor(8-bit): %v", err)
	}
	want := expectedDequantize(t, values, scales, biases, desc.GroupSize)
	assertBitExact(t, got, want)
}

// TestJang_DequantizePackedTensor_RoundTrip_2bit_ShortTail exercises the
// case where the tensor's element count is NOT a multiple of groupSize,
// so the final group runs short and the 2-bit suffix-drain path covers
// the tail.
func TestJang_DequantizePackedTensor_RoundTrip_2bit_ShortTail(t *testing.T) {
	// 130 elements with groupSize=64 → 3 groups, last group has 2 elements.
	desc, values, packed, scales, biases := roundTripFixture(t, 2, 130, 64)
	got, err := DequantizePackedTensor(desc, packed, scales, biases)
	if err != nil {
		t.Fatalf("DequantizePackedTensor(2-bit short tail): %v", err)
	}
	want := expectedDequantize(t, values, scales, biases, desc.GroupSize)
	assertBitExact(t, got, want)
}

// TestJang_DequantizePackedTensor_RoundTrip_2bit_GroupSize2 exercises the
// case where groupSize < 4 — the 2-bit batched fast path can't fire on a
// 4-elements-per-byte stride, so the per-element prefix path must cover
// every element.
func TestJang_DequantizePackedTensor_RoundTrip_2bit_GroupSize2(t *testing.T) {
	desc, values, packed, scales, biases := roundTripFixture(t, 2, 32, 2)
	got, err := DequantizePackedTensor(desc, packed, scales, biases)
	if err != nil {
		t.Fatalf("DequantizePackedTensor(2-bit groupSize=2): %v", err)
	}
	want := expectedDequantize(t, values, scales, biases, desc.GroupSize)
	assertBitExact(t, got, want)
}

// TestJang_DequantizePackedTensor_RoundTrip_4bit_ShortTail covers the
// 4-bit prefix + suffix drains around the batched 2-per-byte fast path
// when the final group is shorter than groupSize.
func TestJang_DequantizePackedTensor_RoundTrip_4bit_ShortTail(t *testing.T) {
	// 67 elements with groupSize=64 → last group has 3 elements; the
	// 2-per-byte batched path takes 2 of them, the suffix drains the 1.
	desc, values, packed, scales, biases := roundTripFixture(t, 4, 67, 64)
	got, err := DequantizePackedTensor(desc, packed, scales, biases)
	if err != nil {
		t.Fatalf("DequantizePackedTensor(4-bit short tail): %v", err)
	}
	want := expectedDequantize(t, values, scales, biases, desc.GroupSize)
	assertBitExact(t, got, want)
}

// TestJang_DequantizePackedTensor_RoundTrip_4bit_GroupSize1 covers the
// degenerate case where groupSize=1, forcing every element into the
// suffix-drain path (no batched stride can fire).
func TestJang_DequantizePackedTensor_RoundTrip_4bit_GroupSize1(t *testing.T) {
	desc, values, packed, scales, biases := roundTripFixture(t, 4, 16, 1)
	got, err := DequantizePackedTensor(desc, packed, scales, biases)
	if err != nil {
		t.Fatalf("DequantizePackedTensor(4-bit groupSize=1): %v", err)
	}
	want := expectedDequantize(t, values, scales, biases, desc.GroupSize)
	assertBitExact(t, got, want)
}

// TestJang_DequantizePackedTensor_RoundTrip_1bit_ShortTail covers the
// 1-bit prefix + suffix drains around the batched 8-per-byte fast path
// when the final group is shorter than groupSize.
func TestJang_DequantizePackedTensor_RoundTrip_1bit_ShortTail(t *testing.T) {
	// 133 elements with groupSize=64 → last group has 5 elements; the
	// 8-per-byte batched path can't fire, suffix-drain takes all 5.
	desc, values, packed, scales, biases := roundTripFixture(t, 1, 133, 64)
	got, err := DequantizePackedTensor(desc, packed, scales, biases)
	if err != nil {
		t.Fatalf("DequantizePackedTensor(1-bit short tail): %v", err)
	}
	want := expectedDequantize(t, values, scales, biases, desc.GroupSize)
	assertBitExact(t, got, want)
}

// TestJang_DequantizePackedTensor_RoundTrip_1bit_GroupSize4 covers the
// case where groupSize=4 < 8, so the 8-per-byte batched fast path can
// never fire and the prefix path must cover every element.
func TestJang_DequantizePackedTensor_RoundTrip_1bit_GroupSize4(t *testing.T) {
	desc, values, packed, scales, biases := roundTripFixture(t, 1, 32, 4)
	got, err := DequantizePackedTensor(desc, packed, scales, biases)
	if err != nil {
		t.Fatalf("DequantizePackedTensor(1-bit groupSize=4): %v", err)
	}
	want := expectedDequantize(t, values, scales, biases, desc.GroupSize)
	assertBitExact(t, got, want)
}

func assertBitExact(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("dequant[%d] = %v, want %v (delta=%v)", i, got[i], want[i], got[i]-want[i])
		}
	}
}

func TestJang_BuildPackedProfile_Good(t *testing.T) {
	profile := BuildPackedProfile(testJANGTQInfo())
	if profile == nil {
		t.Fatal("profile = nil")
	}
	if profile.Type != "jangtq" || profile.Format != "mxtq" || !profile.Mixed {
		t.Fatalf("profile = %+v, want JANGTQ/MXTQ mixed profile", profile)
	}
	if profile.MinBits != 2 || profile.MaxBits != 8 || profile.RoleBits[string(TensorRoleRoutedExpert)] != 2 || profile.RoleBits[string(TensorRoleAttention)] != 8 {
		t.Fatalf("role bits = %+v, min/max=%d/%d", profile.RoleBits, profile.MinBits, profile.MaxBits)
	}
}

func TestJang_BuildPackedProfile_Ugly(t *testing.T) {
	if got := BuildPackedProfile(nil); got != nil {
		t.Fatalf("BuildPackedProfile(nil) = %+v, want nil", got)
	}

	// Every fingerprint field empty: Format falls back to Type since
	// packedFormatFromFingerprint's default branch (core.Lower("")) is
	// also empty, and no role resolves a positive bit width.
	profile := BuildPackedProfile(&Info{})
	if profile == nil || profile.Type != "jang" || profile.Format != "jang" {
		t.Fatalf("profile = %+v, want jang/jang default fallback", profile)
	}
	if profile.Mixed {
		t.Fatalf("profile.Mixed = true, want false when no role bits resolve")
	}
}

// --- ReadConfig (file-system boundary) ---

func TestJang_ReadConfig_Good(t *testing.T) {
	dir := t.TempDir()
	configPath := core.PathJoin(dir, "jang_config.json")
	data := []byte(`{
		"version": 2,
		"weight_format": "mxtq",
		"profile": "JANGTQ",
		"quantization": {"method": "affine+mxtq", "group_size": 64, "bits_default": 2}
	}`)
	if result := core.WriteFile(configPath, data, 0o600); !result.OK {
		t.Fatalf("WriteFile() error = %v", result.Value)
	}

	info, err := ReadConfig(dir)
	if err != nil {
		t.Fatalf("ReadConfig() error = %v", err)
	}
	if info == nil || info.Profile != "JANGTQ" || info.GroupSize != 64 {
		t.Fatalf("info = %+v, want parsed JANGTQ config", info)
	}
}

func TestJang_ReadConfig_Bad(t *testing.T) {
	dir := t.TempDir()
	// A directory in place of the config file fails the read with
	// something other than "not exist" (EISDIR), exercising the raw
	// error-propagation branch distinct from the missing-file branch.
	configPath := core.PathJoin(dir, "jang_config.json")
	if result := core.Mkdir(configPath, 0o755); !result.OK {
		t.Fatalf("Mkdir() error = %v", result.Value)
	}

	info, err := ReadConfig(dir)
	if err == nil {
		t.Fatalf("ReadConfig() error = nil, info = %+v, want a read error", info)
	}
}

func TestJang_ReadConfig_Ugly(t *testing.T) {
	dir := t.TempDir()
	info, err := ReadConfig(dir)
	if err != nil || info != nil {
		t.Fatalf("ReadConfig() = %+v, %v, want nil, nil for a missing config", info, err)
	}
}

// --- ParseConfig ---

func TestJang_ParseConfig_Good(t *testing.T) {
	data := []byte(`{
		"version": 2,
		"weight_format": "mxtq",
		"profile": "JANGTQ",
		"source_model": {"name": "MiniMax-M2", "org": "MiniMaxAI", "architecture": "MiniMaxM2"},
		"mxtq_bits": {"attention": 8, "shared_expert": 8, "routed_expert": 2, "embed_tokens": 8, "lm_head": 8},
		"quantization": {"method": "affine+mxtq", "group_size": 64, "bits_default": 2},
		"capabilities": {"reasoning_parser": "qwen-think", "supports_tools": true, "family": "minimax_m2"}
	}`)

	info, err := ParseConfig(data)
	if err != nil {
		t.Fatalf("ParseConfig() error = %v", err)
	}
	if info.Version != 2 || info.WeightFormat != "mxtq" || info.Profile != "JANGTQ" {
		t.Fatalf("info = %+v, want parsed top-level fields", info)
	}
	if info.SourceName != "MiniMax-M2" || info.SourceOrg != "MiniMaxAI" || info.SourceArchitecture != "minimax_m2" {
		t.Fatalf("source info = %+v, want normalised architecture", info)
	}
	if info.AttentionBits != 8 || info.RoutedExpertBits != 2 || info.BitsDefault != 2 {
		t.Fatalf("mxtq bits = %+v, want attention 8 / routed 2 / default 2", info)
	}
	if info.Capabilities.ReasoningParser != "qwen-think" || !info.Capabilities.SupportsTools {
		t.Fatalf("capabilities = %+v, want carried through", info.Capabilities)
	}
	if info.Packed == nil || info.Packed.Type != "jangtq" {
		t.Fatalf("Packed = %+v, want finalize() to build the packed profile", info.Packed)
	}
}

func TestJang_ParseConfig_Bad(t *testing.T) {
	_, err := ParseConfig([]byte(`{not json`))
	if err == nil {
		t.Fatalf("ParseConfig() error = nil, want a JSON decode error")
	}
}

func TestJang_ParseConfig_Ugly(t *testing.T) {
	// bits_default and mxtq_bits.routed_expert both omitted: BitsDefault
	// falls back to ProfileBits(profile) — a config with only enough to
	// resolve defaults from the profile string.
	info, err := ParseConfig([]byte(`{"profile": "JANG_4M"}`))
	if err != nil {
		t.Fatalf("ParseConfig() error = %v", err)
	}
	if info.BitsDefault != 4 {
		t.Fatalf("BitsDefault = %d, want 4 from ProfileBits(JANG_4M) fallback", info.BitsDefault)
	}
}

// --- ProfileBits ---

func TestJang_ProfileBits_Good(t *testing.T) {
	cases := map[string]int{
		"JANGTQ":  2,
		"jangtq":  2,
		"jang_1":  1,
		"JANG_2":  2,
		"jang_3":  3,
		"JANG_4M": 4,
	}
	for profile, want := range cases {
		if got := ProfileBits(profile); got != want {
			t.Errorf("ProfileBits(%q) = %d, want %d", profile, got, want)
		}
	}
}

func TestJang_ProfileBits_Ugly(t *testing.T) {
	for _, profile := range []string{"", "unknown", "jang", "fp16"} {
		if got := ProfileBits(profile); got != 0 {
			t.Errorf("ProfileBits(%q) = %d, want 0", profile, got)
		}
	}
}

// --- quantizationType ---

func TestJang_QuantizationType_Good(t *testing.T) {
	if got := quantizationType(&Info{Profile: "JANGTQ"}); got != "jangtq" {
		t.Errorf("quantizationType(JANGTQ) = %q, want jangtq", got)
	}
	if got := quantizationType(&Info{WeightFormat: "mxtq"}); got != "jangtq" {
		t.Errorf("quantizationType(mxtq) = %q, want jangtq", got)
	}
	if got := quantizationType(&Info{Method: "affine"}); got != "jang" {
		t.Errorf("quantizationType(affine) = %q, want jang default", got)
	}
}

func TestJang_QuantizationType_Ugly(t *testing.T) {
	if got := quantizationType(nil); got != "" {
		t.Errorf("quantizationType(nil) = %q, want empty string", got)
	}
}

// --- finalize ---

func TestJang_Finalize_Good(t *testing.T) {
	info := &Info{Profile: "JANGTQ", GroupSize: 64, BitsDefault: 2}
	got := finalize(info)
	if got != info {
		t.Fatalf("finalize() returned a different pointer")
	}
	if got.Packed == nil || got.Packed.GroupSize != 64 {
		t.Fatalf("Packed = %+v, want BuildPackedProfile() result attached", got.Packed)
	}
}

func TestJang_Finalize_Ugly(t *testing.T) {
	if got := finalize(nil); got != nil {
		t.Fatalf("finalize(nil) = %+v, want nil", got)
	}
}

// --- quantizationTypeFromFingerprint + packedFormatFromFingerprint ---

func TestJang_QuantizationTypeFromFingerprint_Good(t *testing.T) {
	cases := map[string]string{
		"profile jangtq method": "jangtq",
		"weight mxtq format":    "jangtq",
		"plain jang profile":    "jang",
		"":                      "jang",
	}
	for fingerprint, want := range cases {
		if got := quantizationTypeFromFingerprint(fingerprint); got != want {
			t.Errorf("quantizationTypeFromFingerprint(%q) = %q, want %q", fingerprint, got, want)
		}
	}
}

func TestJang_PackedFormatFromFingerprint_Good(t *testing.T) {
	if got := packedFormatFromFingerprint("uses mxtq and jangtq", "ignored"); got != "mxtq" {
		t.Errorf("packedFormatFromFingerprint(mxtq+jangtq) = %q, want mxtq to win", got)
	}
	if got := packedFormatFromFingerprint("uses jangtq only", "ignored"); got != "jangtq" {
		t.Errorf("packedFormatFromFingerprint(jangtq) = %q, want jangtq", got)
	}
	if got := packedFormatFromFingerprint("plain jang profile", "ignored"); got != "jang" {
		t.Errorf("packedFormatFromFingerprint(jang) = %q, want jang", got)
	}
}

func TestJang_PackedFormatFromFingerprint_Ugly(t *testing.T) {
	if got := packedFormatFromFingerprint("no keyword here", "GPTQ"); got != "gptq" {
		t.Errorf("packedFormatFromFingerprint(none) = %q, want lowered weightFormat fallback", got)
	}
	if got := packedFormatFromFingerprint("", ""); got != "" {
		t.Errorf("packedFormatFromFingerprint(empty) = %q, want empty string", got)
	}
}

// --- ClonePackedProfile ---

func TestJang_ClonePackedProfile_Good(t *testing.T) {
	original := BuildPackedProfile(testJANGTQInfo())
	cloned := ClonePackedProfile(original)
	if cloned == original {
		t.Fatalf("ClonePackedProfile() returned the same pointer")
	}
	if cloned.Type != original.Type || cloned.MinBits != original.MinBits {
		t.Fatalf("cloned = %+v, want a copy of %+v", cloned, original)
	}

	cloned.RoleBits[string(TensorRoleAttention)] = 999
	if original.RoleBits[string(TensorRoleAttention)] == 999 {
		t.Fatalf("ClonePackedProfile() aliased the RoleBits map")
	}
}

func TestJang_ClonePackedProfile_Ugly(t *testing.T) {
	if got := ClonePackedProfile(nil); got != nil {
		t.Fatalf("ClonePackedProfile(nil) = %+v, want nil", got)
	}
}

// --- NewPackedTensorDescriptor (additional branch coverage) ---

func TestJang_NewPackedTensorDescriptor_Bad(t *testing.T) {
	t.Run("nil info", func(t *testing.T) {
		_, err := NewPackedTensorDescriptor("model.layers.0.self_attn.q_proj.weight", []uint64{2, 4}, nil)
		if err == nil || !core.Contains(err.Error(), "requires quantization info") {
			t.Fatalf("error = %v, want nil-info diagnostic", err)
		}
	})

	t.Run("non-positive group size", func(t *testing.T) {
		info := testJANGTQInfo()
		info.GroupSize = 0
		_, err := NewPackedTensorDescriptor("model.layers.0.self_attn.q_proj.weight", []uint64{2, 4}, info)
		if err == nil || !core.Contains(err.Error(), "invalid group size") {
			t.Fatalf("error = %v, want group-size diagnostic", err)
		}
	})

	t.Run("empty shape", func(t *testing.T) {
		_, err := NewPackedTensorDescriptor("model.layers.0.self_attn.q_proj.weight", nil, testJANGTQInfo())
		if err == nil || !core.Contains(err.Error(), "shape is required") {
			t.Fatalf("error = %v, want shape-required diagnostic", err)
		}
	})

	t.Run("zero dimension", func(t *testing.T) {
		_, err := NewPackedTensorDescriptor("model.layers.0.self_attn.q_proj.weight", []uint64{2, 0}, testJANGTQInfo())
		if err == nil || !core.Contains(err.Error(), "zero dimension") {
			t.Fatalf("error = %v, want zero-dimension diagnostic", err)
		}
	})

	t.Run("packed bit count overflows", func(t *testing.T) {
		// A default-role tensor (no name match) at 2-bit with an element
		// count of maxUint64 pushes elements*bits past what a uint64 can
		// hold — no allocation happens before this guard fires, so the
		// test is cheap despite the extreme shape.
		info := testJANGTQInfo()
		info.BitsDefault = 2
		_, err := NewPackedTensorDescriptor("model.layers.0.mlp.gate.weight", []uint64{^uint64(0)}, info)
		if err == nil || !core.Contains(err.Error(), "packed bit count overflows") {
			t.Fatalf("error = %v, want overflow diagnostic", err)
		}
	})

	t.Run("too many groups", func(t *testing.T) {
		// 1-bit keeps elements*bits within uint64 range, but with
		// GroupSize=1 the group count equals the element count — which
		// still exceeds what a (64-bit) int can index.
		info := testJANGTQInfo()
		info.BitsDefault = 1
		info.GroupSize = 1
		_, err := NewPackedTensorDescriptor("model.layers.0.mlp.gate.weight", []uint64{^uint64(0)}, info)
		if err == nil || !core.Contains(err.Error(), "too many groups") {
			t.Fatalf("error = %v, want too-many-groups diagnostic", err)
		}
	})
}

// --- ValidatePackedTensor (additional branch coverage) ---

func TestJang_ValidatePackedTensor_Good(t *testing.T) {
	desc, err := NewPackedTensorDescriptor("model.layers.0.block_sparse_moe.experts.3.w2.weight", []uint64{8}, testJANGTQInfo())
	if err != nil {
		t.Fatalf("NewPackedTensorDescriptor() error = %v", err)
	}
	packed := make([]byte, desc.PackedBytes)
	scales := make([]float32, desc.ScaleCount)
	biases := make([]float32, desc.BiasCount)
	if err := ValidatePackedTensor(desc, packed, scales, biases); err != nil {
		t.Fatalf("ValidatePackedTensor() error = %v", err)
	}
}

func TestJang_ValidatePackedTensor_Bad(t *testing.T) {
	desc, err := NewPackedTensorDescriptor("model.layers.0.block_sparse_moe.experts.3.w2.weight", []uint64{8}, testJANGTQInfo())
	if err != nil {
		t.Fatalf("NewPackedTensorDescriptor() error = %v", err)
	}
	packed := make([]byte, desc.PackedBytes)
	scales := make([]float32, desc.ScaleCount)
	biases := make([]float32, desc.BiasCount)

	t.Run("invalid descriptor propagates", func(t *testing.T) {
		bad := desc
		bad.Elements = 0
		if err := ValidatePackedTensor(bad, packed, scales, biases); err == nil || !core.Contains(err.Error(), "has no elements") {
			t.Fatalf("error = %v, want propagated descriptor diagnostic", err)
		}
	})

	t.Run("wrong scale count", func(t *testing.T) {
		if err := ValidatePackedTensor(desc, packed, []float32{1}, biases); err == nil || !core.Contains(err.Error(), "scale count") {
			t.Fatalf("error = %v, want scale-count diagnostic", err)
		}
	})

	t.Run("wrong bias count", func(t *testing.T) {
		if err := ValidatePackedTensor(desc, packed, scales, []float32{1}); err == nil || !core.Contains(err.Error(), "bias count") {
			t.Fatalf("error = %v, want bias-count diagnostic", err)
		}
	})
}

// --- DequantizePackedTensor (additional branch coverage) ---

func TestJang_DequantizePackedTensor_Bad(t *testing.T) {
	desc, err := NewPackedTensorDescriptor("model.layers.0.block_sparse_moe.experts.3.w2.weight", []uint64{8}, testJANGTQInfo())
	if err != nil {
		t.Fatalf("NewPackedTensorDescriptor() error = %v", err)
	}
	_, err = DequantizePackedTensor(desc, []byte{0}, []float32{1, 1}, []float32{0, 0})
	if err == nil || !core.Contains(err.Error(), "packed length") {
		t.Fatalf("error = %v, want propagated ValidatePackedTensor diagnostic", err)
	}
}

func TestJang_DequantizePackedTensor_Ugly(t *testing.T) {
	// A hand-built descriptor whose Elements exceeds what the CPU
	// reference path can materialise — ValidatePackedTensor is satisfied
	// (packed/scales/biases match the descriptor's own counts) but the
	// dequantize path must still refuse rather than attempt a slice of
	// that size.
	desc := PackedTensorDescriptor{
		Name: "huge.weight", Elements: ^uint64(0), Bits: 1, GroupSize: 1,
		PackedBytes: 1, ScaleCount: 1, BiasCount: 1,
	}
	_, err := DequantizePackedTensor(desc, []byte{0}, []float32{0}, []float32{0})
	if err == nil || !core.Contains(err.Error(), "too large to dequantize") {
		t.Fatalf("error = %v, want too-large-to-dequantize diagnostic", err)
	}
}

// TestJang_DequantizePackedTensor_RoundTrip_8bit_ShortTail exercises the
// dequantizeBit8 per-group end-trim branch: 70 elements with groupSize=64
// leaves a 6-element final group instead of a clean multiple.
func TestJang_DequantizePackedTensor_RoundTrip_8bit_ShortTail(t *testing.T) {
	desc, values, packed, scales, biases := roundTripFixture(t, 8, 70, 64)
	got, err := DequantizePackedTensor(desc, packed, scales, biases)
	if err != nil {
		t.Fatalf("DequantizePackedTensor(8-bit short tail): %v", err)
	}
	want := expectedDequantize(t, values, scales, biases, desc.GroupSize)
	assertBitExact(t, got, want)
}

// TestJang_DequantizePackedTensor_RoundTrip_3bit_ShortTail exercises the
// dequantizeBitGeneric per-group end-trim branch: the earlier 3-bit
// round-trip test uses a clean 4096/64 split where the group boundary
// never needs clamping, so the "end > len(out)" trim never fires. 70
// elements with groupSize=64 leaves a 6-element final group.
func TestJang_DequantizePackedTensor_RoundTrip_3bit_ShortTail(t *testing.T) {
	desc, values, packed, scales, biases := roundTripFixture(t, 3, 70, 64)
	got, err := DequantizePackedTensor(desc, packed, scales, biases)
	if err != nil {
		t.Fatalf("DequantizePackedTensor(3-bit short tail): %v", err)
	}
	want := expectedDequantize(t, values, scales, biases, desc.GroupSize)
	assertBitExact(t, got, want)
}

// --- PackQuantizedValues (additional branch coverage) ---

func TestJang_PackQuantizedValues_Bad(t *testing.T) {
	desc, err := NewPackedTensorDescriptor("model.layers.0.block_sparse_moe.experts.3.w2.weight", []uint64{8}, testJANGTQInfo())
	if err != nil {
		t.Fatalf("NewPackedTensorDescriptor() error = %v", err)
	}

	t.Run("invalid descriptor propagates", func(t *testing.T) {
		bad := desc
		bad.Elements = 0
		if _, err := PackQuantizedValues(bad, []uint8{}); err == nil || !core.Contains(err.Error(), "has no elements") {
			t.Fatalf("error = %v, want propagated descriptor diagnostic", err)
		}
	})

	t.Run("wrong value count", func(t *testing.T) {
		if _, err := PackQuantizedValues(desc, []uint8{0, 1, 2}); err == nil || !core.Contains(err.Error(), "value count") {
			t.Fatalf("error = %v, want value-count diagnostic", err)
		}
	})

	t.Run("value exceeds bit width", func(t *testing.T) {
		values := []uint8{0, 1, 2, 3, 0, 1, 2, 4} // routed-expert 2-bit max is 3
		if _, err := PackQuantizedValues(desc, values); err == nil || !core.Contains(err.Error(), "exceeds") {
			t.Fatalf("error = %v, want value-range diagnostic", err)
		}
	})
}

// --- inferTensorRole ---

func TestJang_InferTensorRole_Good(t *testing.T) {
	cases := map[string]TensorRole{
		"model.embed_tokens.weight":                   TensorRoleEmbedTokens,
		"lm_head.weight":                              TensorRoleLMHead,
		"model.layers.0.shared_expert.w1.weight":      TensorRoleSharedExpert,
		"model.layers.0.mlp.experts.7.gate.weight":    TensorRoleRoutedExpert,
		"model.layers.0.block_sparse_moe.gate.weight": TensorRoleRoutedExpert,
		"model.layers.0.self_attn.dense.weight":       TensorRoleAttention,
		"model.layers.0.attention.wo.weight":          TensorRoleAttention,
		"model.layers.0.block.q_proj.weight":          TensorRoleAttention,
		"model.layers.0.block.k_proj.weight":          TensorRoleAttention,
		"model.layers.0.block.v_proj.weight":          TensorRoleAttention,
		"model.layers.0.block.o_proj.weight":          TensorRoleAttention,
		"model.layers.0.mlp.down_proj.weight":         TensorRoleDefault,
	}
	for name, want := range cases {
		if got := inferTensorRole(name); got != want {
			t.Errorf("inferTensorRole(%q) = %q, want %q", name, got, want)
		}
	}
}

func TestJang_InferTensorRole_Ugly(t *testing.T) {
	// Case-insensitive matching: uppercase substrings still resolve.
	if got := inferTensorRole("MODEL.LAYERS.0.SELF_ATTN.Q_PROJ.WEIGHT"); got != TensorRoleAttention {
		t.Errorf("inferTensorRole(uppercase) = %q, want %q", got, TensorRoleAttention)
	}
	if got := inferTensorRole(""); got != TensorRoleDefault {
		t.Errorf("inferTensorRole(empty) = %q, want default", got)
	}
	// embed_tokens is checked before lm_head in the switch — first match wins.
	if got := inferTensorRole("model.embed_tokens.lm_head.weight"); got != TensorRoleEmbedTokens {
		t.Errorf("inferTensorRole(both) = %q, want embed_tokens to win priority", got)
	}
}

// --- roleBits + minMaxBits ---

func TestJang_RoleBits_Ugly(t *testing.T) {
	if got := roleBits(nil); got != nil {
		t.Fatalf("roleBits(nil) = %v, want nil", got)
	}
	// Every role resolves to 0 bits: no per-role override, no default, no
	// profile fallback — the map stays empty and roleBits reports nil
	// rather than an empty-but-non-nil map.
	if got := roleBits(&Info{}); got != nil {
		t.Fatalf("roleBits(all-zero) = %v, want nil", got)
	}
}

func TestJang_MinMaxBits_Good(t *testing.T) {
	min, max := minMaxBits(map[string]int{"a": 2, "b": 8, "c": 4})
	if min != 2 || max != 8 {
		t.Fatalf("minMaxBits() = (%d, %d), want (2, 8)", min, max)
	}
}

func TestJang_MinMaxBits_Ugly(t *testing.T) {
	if min, max := minMaxBits(nil); min != 0 || max != 0 {
		t.Fatalf("minMaxBits(nil) = (%d, %d), want (0, 0)", min, max)
	}
	// Non-positive entries are skipped rather than treated as a new
	// minimum — roleBits() never inserts one, but minMaxBits must still
	// be defensive when called with a hand-built map.
	min, max := minMaxBits(map[string]int{"zero": 0, "negative": -3, "real": 5})
	if min != 5 || max != 5 {
		t.Fatalf("minMaxBits() = (%d, %d), want (5, 5) skipping non-positive entries", min, max)
	}
}

// --- packedFormat ---

func TestJang_PackedFormat_Good(t *testing.T) {
	if got := packedFormat(&Info{WeightFormat: "mxtq"}); got != "mxtq" {
		t.Errorf("packedFormat(mxtq) = %q, want mxtq", got)
	}
	if got := packedFormat(&Info{Profile: "JANGTQ"}); got != "jangtq" {
		t.Errorf("packedFormat(JANGTQ) = %q, want jangtq", got)
	}
	if got := packedFormat(&Info{Method: "jang-affine"}); got != "jang" {
		t.Errorf("packedFormat(jang-affine) = %q, want jang", got)
	}
}

func TestJang_PackedFormat_Ugly(t *testing.T) {
	if got := packedFormat(nil); got != "" {
		t.Errorf("packedFormat(nil) = %q, want empty string", got)
	}
	if got := packedFormat(&Info{WeightFormat: "FP16"}); got != "fp16" {
		t.Errorf("packedFormat(FP16) = %q, want lowered fallback", got)
	}
}

// --- valuesPerByte ---

func TestJang_ValuesPerByte_Good(t *testing.T) {
	cases := map[int]int{8: 1, 4: 2, 2: 4, 1: 8, 3: 2}
	for bits, want := range cases {
		if got := valuesPerByte(bits); got != want {
			t.Errorf("valuesPerByte(%d) = %d, want %d", bits, got, want)
		}
	}
}

func TestJang_ValuesPerByte_Ugly(t *testing.T) {
	for _, bits := range []int{0, -1} {
		if got := valuesPerByte(bits); got != 0 {
			t.Errorf("valuesPerByte(%d) = %d, want 0", bits, got)
		}
	}
}

// --- shapeElements ---

func TestJang_ShapeElements_Good(t *testing.T) {
	if got, err := shapeElements([]uint64{4}); err != nil || got != 4 {
		t.Fatalf("shapeElements([4]) = %d, %v, want 4, nil", got, err)
	}
	if got, err := shapeElements([]uint64{2, 4, 8}); err != nil || got != 64 {
		t.Fatalf("shapeElements([2,4,8]) = %d, %v, want 64, nil", got, err)
	}
}

func TestJang_ShapeElements_Bad(t *testing.T) {
	t.Run("empty shape", func(t *testing.T) {
		if _, err := shapeElements(nil); err == nil || !core.Contains(err.Error(), "shape is required") {
			t.Fatalf("error = %v, want shape-required diagnostic", err)
		}
	})

	t.Run("zero dimension", func(t *testing.T) {
		if _, err := shapeElements([]uint64{4, 0, 2}); err == nil || !core.Contains(err.Error(), "zero dimension") {
			t.Fatalf("error = %v, want zero-dimension diagnostic", err)
		}
	})

	t.Run("overflow", func(t *testing.T) {
		if _, err := shapeElements([]uint64{2, ^uint64(0)}); err == nil || !core.Contains(err.Error(), "overflows") {
			t.Fatalf("error = %v, want overflow diagnostic", err)
		}
	})
}

// --- validateDescriptor ---

func TestJang_ValidateDescriptor_Good(t *testing.T) {
	desc, err := NewPackedTensorDescriptor("model.layers.0.self_attn.q_proj.weight", []uint64{2, 4}, testJANGTQInfo())
	if err != nil {
		t.Fatalf("NewPackedTensorDescriptor() error = %v", err)
	}
	if err := validateDescriptor(desc); err != nil {
		t.Fatalf("validateDescriptor() error = %v", err)
	}
}

func TestJang_ValidateDescriptor_Bad(t *testing.T) {
	valid := PackedTensorDescriptor{
		Name: "ok.weight", Elements: 8, Bits: 2, GroupSize: 4,
		PackedBytes: 2, ScaleCount: 2, BiasCount: 2,
	}

	t.Run("no elements", func(t *testing.T) {
		desc := valid
		desc.Elements = 0
		if err := validateDescriptor(desc); err == nil || !core.Contains(err.Error(), "has no elements") {
			t.Fatalf("error = %v, want no-elements diagnostic", err)
		}
	})

	t.Run("unsupported bits", func(t *testing.T) {
		desc := valid
		desc.Bits = 5
		if err := validateDescriptor(desc); err == nil || !core.Contains(err.Error(), "unsupported") {
			t.Fatalf("error = %v, want unsupported-bits diagnostic", err)
		}
	})

	t.Run("non-positive group size", func(t *testing.T) {
		desc := valid
		desc.GroupSize = 0
		if err := validateDescriptor(desc); err == nil || !core.Contains(err.Error(), "invalid group size") {
			t.Fatalf("error = %v, want group-size diagnostic", err)
		}
	})

	t.Run("non-positive packed bytes", func(t *testing.T) {
		desc := valid
		desc.PackedBytes = 0
		if err := validateDescriptor(desc); err == nil || !core.Contains(err.Error(), "invalid packed byte count") {
			t.Fatalf("error = %v, want packed-byte-count diagnostic", err)
		}
	})

	t.Run("non-positive scale or bias count", func(t *testing.T) {
		desc := valid
		desc.ScaleCount = 0
		if err := validateDescriptor(desc); err == nil || !core.Contains(err.Error(), "invalid scale/bias counts") {
			t.Fatalf("error = %v, want scale/bias diagnostic", err)
		}
	})
}

// --- unpackValue (mirrors writeValue; currently reference-only, not yet
// wired into the specialised per-bit-width dequant paths) ---

func TestJang_UnpackValue_Good(t *testing.T) {
	// unpackValue must invert writeValue for every fast-path width plus
	// the generic bit-walk fallback (3-bit), across a span of indices
	// wide enough to cross byte boundaries for each width.
	for _, bits := range []int{1, 2, 3, 4, 8} {
		maxValue := uint8((1 << bits) - 1)
		perByte := 8 / bits
		count := perByte * 3
		packed := make([]byte, (count*bits+7)/8)
		values := make([]uint8, count)
		for i := range values {
			values[i] = uint8(i) & maxValue
			writeValue(packed, i, bits, values[i])
		}
		for i, want := range values {
			if got := unpackValue(packed, i, bits); got != want {
				t.Errorf("unpackValue(bits=%d, index=%d) = %d, want %d", bits, i, got, want)
			}
		}
	}
}

func TestJang_UnpackValue_Ugly(t *testing.T) {
	// Byte-aligned fast paths (8/4/2/1-bit) against a fixed byte, checked
	// against hand-derived expected nibbles/lanes/bits rather than a
	// round trip, so a regression in the fast-path shift/mask math would
	// not be masked by an equal-and-opposite bug in writeValue.
	packed := []byte{0b10110100}
	if got := unpackValue(packed, 0, 8); got != 0b10110100 {
		t.Errorf("unpackValue(8-bit) = %08b, want %08b", got, 0b10110100)
	}
	if got := unpackValue(packed, 0, 4); got != 0b0100 {
		t.Errorf("unpackValue(4-bit low) = %04b, want %04b", got, 0b0100)
	}
	if got := unpackValue(packed, 1, 4); got != 0b1011 {
		t.Errorf("unpackValue(4-bit high) = %04b, want %04b", got, 0b1011)
	}
	if got := unpackValue(packed, 0, 2); got != 0b00 {
		t.Errorf("unpackValue(2-bit lane 0) = %02b, want %02b", got, 0b00)
	}
	if got := unpackValue(packed, 3, 2); got != 0b10 {
		t.Errorf("unpackValue(2-bit lane 3) = %02b, want %02b", got, 0b10)
	}
	if got := unpackValue(packed, 2, 1); got != 1 {
		t.Errorf("unpackValue(1-bit index 2) = %d, want 1", got)
	}
	if got := unpackValue(packed, 0, 1); got != 0 {
		t.Errorf("unpackValue(1-bit index 0) = %d, want 0", got)
	}
}

// --- cloneRoleBits ---

func TestJang_CloneRoleBits_Good(t *testing.T) {
	original := map[string]int{"attention": 8, "routed_expert": 2}
	cloned := cloneRoleBits(original)
	if len(cloned) != len(original) || cloned["attention"] != 8 || cloned["routed_expert"] != 2 {
		t.Fatalf("cloneRoleBits() = %v, want a copy of %v", cloned, original)
	}
	cloned["attention"] = 999
	if original["attention"] == 999 {
		t.Fatalf("cloneRoleBits() aliased the original map")
	}
}

func TestJang_CloneRoleBits_Ugly(t *testing.T) {
	if got := cloneRoleBits(nil); got != nil {
		t.Fatalf("cloneRoleBits(nil) = %v, want nil", got)
	}
	if got := cloneRoleBits(map[string]int{}); got != nil {
		t.Fatalf("cloneRoleBits(empty) = %v, want nil", got)
	}
}

// --- ceilDivUint64 ---

func TestJang_CeilDivUint64_Good(t *testing.T) {
	if got := ceilDivUint64(8, 2); got != 4 {
		t.Errorf("ceilDivUint64(8, 2) = %d, want 4", got)
	}
	if got := ceilDivUint64(7, 2); got != 4 {
		t.Errorf("ceilDivUint64(7, 2) = %d, want 4 (rounds up)", got)
	}
}

func TestJang_CeilDivUint64_Ugly(t *testing.T) {
	if got := ceilDivUint64(0, 5); got != 0 {
		t.Errorf("ceilDivUint64(0, 5) = %d, want 0", got)
	}
	if got := ceilDivUint64(5, 0); got != 0 {
		t.Errorf("ceilDivUint64(5, 0) = %d, want 0 (guarded divide by zero)", got)
	}
}

// --- firstPositive ---

func TestJang_FirstPositive_Good(t *testing.T) {
	if got := firstPositive(0, -4, 6, 9); got != 6 {
		t.Fatalf("firstPositive() = %d, want 6", got)
	}
}

func TestJang_FirstPositive_Ugly(t *testing.T) {
	if got := firstPositive(0, -1, -2); got != 0 {
		t.Fatalf("firstPositive() = %d, want 0", got)
	}
	if got := firstPositive(); got != 0 {
		t.Fatalf("firstPositive() = %d, want 0 for no arguments", got)
	}
}

// --- normaliseArchitecture ---

func TestJang_NormaliseArchitecture_Good(t *testing.T) {
	cases := map[string]string{
		"Qwen3-5":            "qwen3_next",
		"MiniMaxM2":          "minimax_m2",
		"  minimax-m2  ":     "minimax_m2",
		"Mixtral":            "mixtral",
		"Mistral":            "mistral",
		"Phi":                "phi",
		"Phi3":               "phi",
		"Phi4":               "phi",
		"DeepSeek":           "deepseek",
		"deepseek-v3":        "deepseek",
		"deepseek-r1":        "deepseek",
		"GptOss":             "gpt_oss",
		"gpt-oss":            "gpt_oss",
		"gpt_oss_model":      "gpt_oss",
		"BERT":               "bert",
		"bert-rerank":        "bert_rerank",
		"bert_cross_encoder": "bert_rerank",
	}
	for input, want := range cases {
		if got := normaliseArchitecture(input); got != want {
			t.Errorf("normaliseArchitecture(%q) = %q, want %q", input, got, want)
		}
	}
}

func TestJang_NormaliseArchitecture_Ugly(t *testing.T) {
	// Unrecognised architectures fall through to the normalised (trimmed,
	// lowered, hyphen-to-underscore) value rather than an empty string.
	if got := normaliseArchitecture("LlamaForCausalLM"); got != "llamaforcausallm" {
		t.Errorf("normaliseArchitecture(unknown) = %q, want lowered passthrough", got)
	}
	if got := normaliseArchitecture(""); got != "" {
		t.Errorf("normaliseArchitecture(empty) = %q, want empty string", got)
	}
}
