// SPDX-Licence-Identifier: EUPL-1.2

package autoround

import (
	"context"
	"testing"
)

import core "dappco.re/go"

// TestExportCov_WriteNativePack_NilContext covers the ctx==nil branch of
// WriteNativePack: a nil context is replaced with context.Background() and the
// write proceeds normally.
func TestExportCov_WriteNativePack_NilContext(t *testing.T) {
	dir := t.TempDir()
	projections := []PackedProjection{
		autoRoundTestProjection("model.layers.0.self_attn.q_proj.weight", []byte{0b11100100}, []float32{0.5}, []float32{0}, nil),
	}
	info := PackInfo{
		Bits:          2,
		GroupSize:     32,
		Symmetric:     true,
		QuantMethod:   QuantMethodAutoRound,
		PackingFormat: string(FormatAutoRound),
		Scheme:        SchemeW2A16,
	}
	//nolint:staticcheck // deliberately exercise the nil-context default branch
	result, err := WriteNativePack(nil, dir, info, projections)
	if err != nil {
		t.Fatalf("WriteNativePack(nil ctx) error = %v, want nil-context default", err)
	}
	if result.TensorCount != 1 {
		t.Fatalf("WriteNativePack(nil ctx) tensor count = %d, want 1", result.TensorCount)
	}
}

// TestExportCov_WriteNativePack_InfoValidateFails covers the info.Validate()
// rejection inside WriteNativePack. The projection inherits an unsupported bit
// width into the pack info, so validation fails before any file is written.
func TestExportCov_WriteNativePack_InfoValidateFails(t *testing.T) {
	bad := PackedProjection{
		Tensor: PackTensor{
			Name:       "weight",
			Packed:     "weight.packed",
			Scales:     "weight.scales",
			ZeroPoints: "weight.zeros",
			Shape:      []int32{1, 4},
			Bits:       5, // unsupported → info.Bits inherits 5 → Validate fails
			GroupSize:  32,
		},
		Weights: PackedWeights{Bits: 5, GroupSize: 32, Shape: []int32{1, 4}},
	}
	if _, err := WriteNativePack(context.Background(), t.TempDir(), PackInfo{}, []PackedProjection{bad}); err == nil || !core.Contains(err.Error(), "bits") {
		t.Fatalf("WriteNativePack(invalid bits) error = %v, want bits diagnostic from info.Validate", err)
	}
}

// TestExportCov_WriteNativePack_MkdirFails covers the MkdirAll(root) failure
// branch: when a path component of root is an existing regular file, the
// directory creation cannot proceed.
func TestExportCov_WriteNativePack_MkdirFails(t *testing.T) {
	dir := t.TempDir()
	filePath := core.PathJoin(dir, "not-a-dir")
	if result := core.WriteFile(filePath, []byte("x"), 0o644); !result.OK {
		t.Fatalf("seed file: %v", result.Value)
	}
	// root nests under a regular file → MkdirAll fails with a not-a-directory error.
	root := core.PathJoin(filePath, "sub")
	projections := []PackedProjection{
		autoRoundTestProjection("weight", []byte{0b11100100}, []float32{0.5}, []float32{0}, nil),
	}
	info := PackInfo{Bits: 2, GroupSize: 32, Symmetric: true, Scheme: SchemeW2A16}
	if _, err := WriteNativePack(context.Background(), root, info, projections); err == nil {
		t.Fatal("WriteNativePack(root under a file) error = nil, want MkdirAll failure")
	}
}

// TestExportCov_WriteNativePack_WeightWriteFails covers the
// WritePackedProjectionsSafetensors failure path inside WriteNativePack: the
// pack info validates, but a projection whose packed payload disagrees with its
// declared shape fails when the weight file is encoded.
func TestExportCov_WriteNativePack_WeightWriteFails(t *testing.T) {
	// Tensor metadata is internally valid (so info.Validate passes), but the
	// PackedWeights.Packed length disagrees with the declared shape, so the
	// per-projection weight match check rejects it during the weight write.
	proj := autoRoundTestProjection("weight", []byte{0b11100100}, []float32{0.5}, []float32{0}, nil)
	proj.Weights.Packed = []byte{0, 0, 0} // wrong length for a 1x4 W2 tensor (want 1 byte)
	info := PackInfo{Bits: 2, GroupSize: 32, Symmetric: true, Scheme: SchemeW2A16}
	if _, err := WriteNativePack(context.Background(), t.TempDir(), info, []PackedProjection{proj}); err == nil || !core.Contains(err.Error(), "packed length") {
		t.Fatalf("WriteNativePack(bad packed) error = %v, want packed length diagnostic", err)
	}
}

// TestExportCov_WriteNativePack_ConfigWriteFails covers the WriteFile(config)
// failure path: the model.safetensors write succeeds, but the config path is a
// pre-existing directory, so the JSON config cannot be written to it.
func TestExportCov_WriteNativePack_ConfigWriteFails(t *testing.T) {
	root := t.TempDir()
	// Pre-create the config path as a directory so WriteFile fails on it while
	// MkdirAll(root) and the weight write still succeed.
	configDir := core.PathJoin(root, PackConfigFileAutoRound)
	if result := core.MkdirAll(configDir, 0o755); !result.OK {
		t.Fatalf("seed config dir: %v", result.Value)
	}
	projections := []PackedProjection{
		autoRoundTestProjection("weight", []byte{0b11100100}, []float32{0.5}, []float32{0}, nil),
	}
	info := PackInfo{Bits: 2, GroupSize: 32, Symmetric: true, Scheme: SchemeW2A16}
	if _, err := WriteNativePack(context.Background(), root, info, projections); err == nil {
		t.Fatal("WriteNativePack(config path is a dir) error = nil, want WriteFile failure")
	}
}

// TestExportCov_WriteNativePack_InfoDefaults covers the four inherit-from-
// projection default branches in nativePackInfoForExport: a fully zero-value
// PackInfo inherits bits, group size, symmetry, and scheme from the first
// projection's weights.
func TestExportCov_WriteNativePack_InfoDefaults(t *testing.T) {
	dir := t.TempDir()
	projections := []PackedProjection{
		autoRoundTestProjection("model.layers.0.self_attn.q_proj.weight", []byte{0b11100100}, []float32{0.5}, []float32{0}, nil),
	}
	// PackInfo{} carries no bits/group/symmetry/scheme — all four are inherited.
	result, err := WriteNativePack(context.Background(), dir, PackInfo{}, projections)
	if err != nil {
		t.Fatalf("WriteNativePack(zero info) error = %v, want projection-inherited defaults", err)
	}
	got, err := ReadPackInfo(dir)
	if err != nil {
		t.Fatalf("ReadPackInfo(inherited) error = %v", err)
	}
	if got == nil || got.Bits != 2 || got.GroupSize != 32 || !got.Symmetric || got.Scheme != SchemeW2A16 {
		t.Fatalf("ReadPackInfo(inherited) = %+v, want W2/32/sym inherited from the projection", got)
	}
	_ = result
}

// TestExportCov_PackedProjectionTensors_TensorValidateFails covers the
// tensor.Validate() failure inside packedProjectionSafetensorsTensors (reached
// via WritePackedProjectionsSafetensors): a projection whose tensor names omit
// the required companions cannot be encoded.
func TestExportCov_PackedProjectionTensors_TensorValidateFails(t *testing.T) {
	proj := autoRoundTestProjection("weight", []byte{0b11100100}, []float32{0.5}, []float32{0}, nil)
	proj.Tensor.Scales = "" // drop a required companion name
	err := WritePackedProjectionsSafetensors(context.Background(), core.PathJoin(t.TempDir(), "x.safetensors"), []PackedProjection{proj})
	if err == nil || !core.Contains(err.Error(), "packed, scales, and zero_points") {
		t.Fatalf("WritePackedProjectionsSafetensors(no scales name) error = %v, want companion diagnostic", err)
	}
}

// TestExportCov_PackedProjectionTensors_BiasLengthMismatch covers the bias
// length check: a tensor declaring a bias whose supplied vector length differs
// from the output dimension is rejected.
func TestExportCov_PackedProjectionTensors_BiasLengthMismatch(t *testing.T) {
	// Shape[0] is 1, so a 3-element bias is wrong.
	proj := autoRoundTestProjection("weight", []byte{0b11100100}, []float32{0.5}, []float32{0}, []float32{1, 2, 3})
	err := WritePackedProjectionsSafetensors(context.Background(), core.PathJoin(t.TempDir(), "x.safetensors"), []PackedProjection{proj})
	if err == nil || !core.Contains(err.Error(), "bias length") {
		t.Fatalf("WritePackedProjectionsSafetensors(bad bias length) error = %v, want bias length diagnostic", err)
	}
}

// TestExportCov_ValidateProjectionWeightsMatch covers each field-mismatch
// branch of validateProjectionWeightsMatch (bits, group size, symmetry,
// shape rank, shape element), reached through WritePackedProjectionsSafetensors.
//
// validateProjectionWeightsMatch first calls validatePackedWeights, which
// checks the packed payload against the weights' OWN bits/shape. So each case
// keeps the weights self-consistent and instead makes the TENSOR disagree —
// both the tensor and the weights are individually valid, but their declared
// metadata differs, which is exactly what the match check guards.
func TestExportCov_ValidateProjectionWeightsMatch(t *testing.T) {
	// A self-consistent W2, 1x4 tensor + matching W2 weights (1 packed byte).
	base := func() PackedProjection {
		return PackedProjection{
			Tensor: PackTensor{
				Name:        "weight",
				Packed:      "weight.packed",
				Scales:      "weight.scales",
				ZeroPoints:  "weight.zeros",
				Shape:       []int32{1, 4},
				Bits:        2,
				GroupSize:   32,
				Symmetric:   true,
				PackedBytes: 1,
				Groups:      1,
				QMin:        -2,
				QMax:        1,
			},
			Weights: PackedWeights{
				Bits:       2,
				GroupSize:  32,
				Symmetric:  true,
				Shape:      []int32{1, 4},
				Packed:     []byte{0b11100100},
				Scales:     []float32{0.5},
				ZeroPoints: []float32{0},
				QMin:       -2,
				QMax:       1,
			},
		}
	}
	cases := []struct {
		name   string
		mutate func(p *PackedProjection)
		want   string
	}{
		// Tensor says W4 (2 packed bytes), weights are W2: tensor.Validate needs
		// PackedBytes=2, so set it; both valid individually, bits disagree.
		{"Bits", func(p *PackedProjection) { p.Tensor.Bits = 4; p.Tensor.PackedBytes = 2 }, "packed bits"},
		{"GroupSize", func(p *PackedProjection) { p.Tensor.GroupSize = 64 }, "packed group size"},
		// tensor.normalise resets a false tensor symmetry back to the info's
		// (true) default, so the disagreement must live on the weights side:
		// weights asymmetric vs tensor symmetric.
		{"Symmetric", func(p *PackedProjection) { p.Weights.Symmetric = false }, "packed symmetry"},
		// Weights rank 1 vs tensor rank 2 (weights stay self-consistent: 4 elems, 1 byte).
		{"ShapeRank", func(p *PackedProjection) { p.Weights.Shape = []int32{4} }, "packed shape rank"},
		// Weights {4,1} vs tensor {1,4}: same element count (self-consistent) but
		// the per-dimension values differ.
		{"ShapeElement", func(p *PackedProjection) { p.Weights.Shape = []int32{4, 1} }, "packed shape["},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			proj := base()
			tc.mutate(&proj)
			err := WritePackedProjectionsSafetensors(context.Background(), core.PathJoin(t.TempDir(), "x.safetensors"), []PackedProjection{proj})
			if err == nil || !core.Contains(err.Error(), tc.want) {
				t.Fatalf("WritePackedProjectionsSafetensors(%s mismatch) error = %v, want %q diagnostic", tc.name, err, tc.want)
			}
		})
	}
}

// TestExportCov_WritePackedProjectionsSafetensors_NilContext covers the ctx==nil
// branch of WritePackedProjectionsSafetensors.
func TestExportCov_WritePackedProjectionsSafetensors_NilContext(t *testing.T) {
	projections := []PackedProjection{
		autoRoundTestProjection("weight", []byte{0b11100100}, []float32{0.5}, []float32{0}, nil),
	}
	//nolint:staticcheck // deliberately exercise the nil-context default branch
	if err := WritePackedProjectionsSafetensors(nil, core.PathJoin(t.TempDir(), "x.safetensors"), projections); err != nil {
		t.Fatalf("WritePackedProjectionsSafetensors(nil ctx) error = %v, want nil-context default", err)
	}
}

// --- direct white-box coverage of writeAutoRoundRawSafetensors guards ---

// TestExportCov_WriteRawSafetensors_EmptyName covers the empty-tensor-name guard
// in writeAutoRoundRawSafetensors. The guard is defensive (callers validate
// names upstream), so it is exercised by calling the writer directly.
func TestExportCov_WriteRawSafetensors_EmptyName(t *testing.T) {
	tensors := []packedProjectionTensor{
		{name: "  ", dtype: "U8", shape: []int{1}, raw: []byte{0}},
	}
	err := writeAutoRoundRawSafetensors(context.Background(), core.PathJoin(t.TempDir(), "x.safetensors"), tensors)
	if err == nil || !core.Contains(err.Error(), "tensor name is empty") {
		t.Fatalf("writeAutoRoundRawSafetensors(empty name) error = %v, want empty-name diagnostic", err)
	}
}

// TestExportCov_WriteRawSafetensors_CancelledContext covers the mid-loop
// ctx.Err() guard in writeAutoRoundRawSafetensors. Calling the writer directly
// bypasses the higher-level up-front context check, so a pre-cancelled context
// fires deterministically on the first loop iteration.
func TestExportCov_WriteRawSafetensors_CancelledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	tensors := []packedProjectionTensor{
		{name: "weight.packed", dtype: "U8", shape: []int{1}, raw: []byte{0}},
	}
	if err := writeAutoRoundRawSafetensors(ctx, core.PathJoin(t.TempDir(), "x.safetensors"), tensors); err == nil {
		t.Fatal("writeAutoRoundRawSafetensors(cancelled ctx) error = nil, want context cancellation")
	}
}

// TestExportCov_WriteRawSafetensors_MkdirParentFails covers the MkdirAll(parent)
// failure branch: the parent directory of the output path is an existing regular
// file, so the parent cannot be created.
func TestExportCov_WriteRawSafetensors_MkdirParentFails(t *testing.T) {
	dir := t.TempDir()
	filePath := core.PathJoin(dir, "blocker")
	if result := core.WriteFile(filePath, []byte("x"), 0o644); !result.OK {
		t.Fatalf("seed blocker file: %v", result.Value)
	}
	// The output path nests under a regular file, so PathDir is that file and
	// MkdirAll on it fails.
	out := core.PathJoin(filePath, "nested", "x.safetensors")
	tensors := []packedProjectionTensor{
		{name: "weight.packed", dtype: "U8", shape: []int{1}, raw: []byte{0}},
	}
	if err := writeAutoRoundRawSafetensors(context.Background(), out, tensors); err == nil {
		t.Fatal("writeAutoRoundRawSafetensors(parent under a file) error = nil, want MkdirAll failure")
	}
}

// TestExportCov_WriteRawSafetensors_OpenFileFails covers the OpenFile failure
// branch: the output path is itself an existing directory, so it cannot be
// opened for writing.
func TestExportCov_WriteRawSafetensors_OpenFileFails(t *testing.T) {
	dir := t.TempDir()
	out := core.PathJoin(dir, "isdir")
	if result := core.MkdirAll(out, 0o755); !result.OK {
		t.Fatalf("seed output dir: %v", result.Value)
	}
	tensors := []packedProjectionTensor{
		{name: "weight.packed", dtype: "U8", shape: []int{1}, raw: []byte{0}},
	}
	if err := writeAutoRoundRawSafetensors(context.Background(), out, tensors); err == nil {
		t.Fatal("writeAutoRoundRawSafetensors(output is a dir) error = nil, want OpenFile failure")
	}
}
