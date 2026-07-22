// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"

	"dappco.re/go/inference/model/arch/Qwen/qwen35"
	"dappco.re/go/inference/model/safetensors"
)

// qwen_vision_device_test.go proves the #59 device-tower follow-up (design doc §6): the bf16 device
// forward stays within a pinned band of the host f32/f64 forward on synthetic weights (both tensor
// layouts), the host f32 arrays are genuinely released once the mirror is built, the load seam wires
// the two together (and the LTHN_QWEN_VISION_DEVICE=0 kill-switch un-wires them), and the small
// data-reshape helpers (head split/merge) round-trip correctly in isolation.

// qwenVisionDeviceParityBand is the pinned device-vs-host max-abs-diff ceiling — see
// TestQwenVisionTowerForwardDevice_HostParity_Good's doc comment for how it was measured.
const qwenVisionDeviceParityBand = 0.01

func TestBuildQwenVisionDeviceTower_Shape_Good(t *testing.T) {
	tower := qtRealTower(2)
	dt := buildQwenVisionDeviceTower(tower)
	if dt == nil {
		t.Fatal("buildQwenVisionDeviceTower returned nil")
	}
	if len(dt.Patch.W) != qtHidden*qtPatch*bf16Size {
		t.Fatalf("Patch.W bf16 bytes = %d, want %d", len(dt.Patch.W), qtHidden*qtPatch*bf16Size)
	}
	if len(dt.Patch.B) != qtHidden*bf16Size {
		t.Fatalf("Patch.B bf16 bytes = %d, want %d", len(dt.Patch.B), qtHidden*bf16Size)
	}
	if len(dt.PosEmbed) != 4*qtHidden*bf16Size {
		t.Fatalf("PosEmbed bf16 bytes = %d, want %d", len(dt.PosEmbed), 4*qtHidden*bf16Size)
	}
	if len(dt.Blocks) != 2 {
		t.Fatalf("Blocks = %d, want 2", len(dt.Blocks))
	}
	b0 := dt.Blocks[0]
	if len(b0.Norm1W) != qtHidden*bf16Size || len(b0.Norm1B) != qtHidden*bf16Size {
		t.Fatalf("block 0 norm1 bf16 bytes = %d/%d, want %d each", len(b0.Norm1W), len(b0.Norm1B), qtHidden*bf16Size)
	}
	if b0.Attn.Q.Out != qtHidden || b0.Attn.Q.In != qtHidden {
		t.Fatalf("block 0 attn.Q dims = %d/%d, want %d/%d", b0.Attn.Q.Out, b0.Attn.Q.In, qtHidden, qtHidden)
	}
	if !b0.MLP.GELU {
		t.Fatal("qtRealTower's block MLP must carry the GELU (FC1/FC2) flag")
	}
	if len(b0.MLP.FC1.W) == 0 || len(b0.MLP.Gate.W) != 0 {
		t.Fatal("GELU block must populate FC1 and leave Gate empty")
	}

	guessed := buildQwenVisionDeviceTower(qtGuessedTower())
	gb := guessed.Blocks[0]
	if gb.MLP.GELU {
		t.Fatal("qtGuessedTower's block MLP must NOT carry the GELU flag (SwiGLU)")
	}
	if len(gb.MLP.Gate.W) == 0 || len(gb.MLP.FC1.W) != 0 {
		t.Fatal("SwiGLU block must populate Gate/Up/Down and leave FC1 empty")
	}
	if len(gb.Attn.QNorm) != qtHeadDim || len(gb.Attn.KNorm) != qtHeadDim {
		t.Fatalf("guessed layout must carry per-head QNorm/KNorm of length %d", qtHeadDim)
	}
}

// TestQwenVisionTowerForwardDevice_HostParity_Good is the device-vs-host oracle receipt: the SAME
// synthetic weights and patches run through both QwenVisionTowerForward (host f32/f64) and
// QwenVisionTowerForwardDevice (bf16 device), for both tensor layouts. This is a PRECISION CHANGE,
// not a rounding-noise check like the host-vs-independent-oracle test in qwen_vision_encoder_test.go
// (which pins 1e-5, f32-vs-f32 structural-bug noise): every weight and every intermediate activation
// downcasts to bf16 (≈3 decimal digits, round-to-nearest-even), so the achievable band is looser, though
// nowhere near as loose as bf16's raw ~0.4% relative step might suggest — LayerNorm re-centres each
// block's activations, so error does not compound unboundedly with depth. Measured directly (a depth
// sweep on qtRealTower, logged then deleted — not a committed test, but the numbers are real): max abs
// diff 0.00099 at 2 blocks, 0.00139 at 4, 0.00173 at 8, 0.00116 at 16, 0.00265 at 27 (matching the live
// 27B checkpoint's actual depth) — against activations of order-magnitude 0.3-0.44 (qtSeq draws
// |v|≤0.25 inputs). The real-layout case below therefore runs qtRealTower(27), not a token-sized
// stand-in, so this receipt is measuring the SAME depth the live receipt serves. qwenVisionDeviceParityBand
// (0.01) leaves ~4× margin over the deepest measured case while still catching a structural porting
// slip (a transposed weight or a wrong fused band diverges by order-1, not by bf16 noise) by orders
// of magnitude.
func TestQwenVisionTowerForwardDevice_HostParity_Good(t *testing.T) {
	for _, tc := range []struct {
		name  string
		tower func() *qwen35.VisionTower
	}{
		{"real_gelu_mha_learnedpos_27blocks", func() *qwen35.VisionTower { return qtRealTower(27) }},
		{"guessed_swiglu_gqa_qknorm", qtGuessedTower},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const gridH, gridW = 2, 4
			patches := qtSeq(gridH*gridW*qtPatch, 4242)

			hostTower := tc.tower()
			hostFeatures, hostSoft, err := QwenVisionTowerForward(patches, gridH, gridW, hostTower)
			if err != nil {
				t.Fatalf("host QwenVisionTowerForward: %v", err)
			}

			deviceTower := tc.tower()
			dt := buildQwenVisionDeviceTower(deviceTower)
			devBytes, devSoft, err := QwenVisionTowerForwardDevice(patches, gridH, gridW, dt)
			if err != nil {
				t.Fatalf("QwenVisionTowerForwardDevice: %v", err)
			}
			if devSoft != hostSoft {
				t.Fatalf("softTokens = %d, host = %d", devSoft, hostSoft)
			}
			devFeatures := bf16ToF32Slice(devBytes)
			if len(devFeatures) != len(hostFeatures) {
				t.Fatalf("feature len = %d, host = %d", len(devFeatures), len(hostFeatures))
			}
			var maxAbs float64
			for i := range hostFeatures {
				if d := math.Abs(float64(devFeatures[i] - hostFeatures[i])); d > maxAbs {
					maxAbs = d
				}
			}
			t.Logf("%s: device-vs-host max abs diff = %v (band %v)", tc.name, maxAbs, qwenVisionDeviceParityBand)
			if maxAbs > qwenVisionDeviceParityBand {
				t.Fatalf("device-vs-host divergence: max abs diff %v > band %v", maxAbs, qwenVisionDeviceParityBand)
			}
		})
	}
}

// TestQwenVisionTowerForwardDevice_Deterministic_Good proves the device forward is bit-deterministic
// across repeated runs on the SAME mirror — a scratch-pool aliasing bug would show up as drift here.
func TestQwenVisionTowerForwardDevice_Deterministic_Good(t *testing.T) {
	dt := buildQwenVisionDeviceTower(qtRealTower(2))
	const gridH, gridW = 2, 4
	patches := qtSeq(gridH*gridW*qtPatch, 999)
	first, softTokens, err := QwenVisionTowerForwardDevice(patches, gridH, gridW, dt)
	if err != nil {
		t.Fatalf("first run: %v", err)
	}
	if want := (gridH / 2) * (gridW / 2); softTokens != want {
		t.Fatalf("softTokens = %d, want %d", softTokens, want)
	}
	second, _, err := QwenVisionTowerForwardDevice(patches, gridH, gridW, dt)
	if err != nil {
		t.Fatalf("second run: %v", err)
	}
	if len(first) != len(second) {
		t.Fatalf("byte length changed between runs: %d vs %d", len(first), len(second))
	}
	for i := range first {
		if first[i] != second[i] {
			t.Fatalf("device forward is not bit-deterministic at byte %d: %v vs %v", i, first[i], second[i])
		}
	}
}

func TestQwenVisionTowerForwardDevice_BadInputs_Bad(t *testing.T) {
	dt := buildQwenVisionDeviceTower(qtRealTower(1))
	if _, _, err := QwenVisionTowerForwardDevice(qtSeq(5, 1), 2, 4, dt); err == nil {
		t.Fatal("a patch buffer not matching L·PatchDim must fail loudly")
	}
	if _, _, err := QwenVisionTowerForwardDevice(nil, 0, 0, dt); err == nil {
		t.Fatal("an empty grid must fail loudly")
	}
	if _, _, err := QwenVisionTowerForwardDevice(qtSeq(4*qtPatch, 1), 2, 2, nil); err == nil {
		t.Fatal("a nil device tower must fail loudly")
	}
}

// TestQwenVisionTowerForward_FreedTower_Bad proves the host forward refuses to run on a tower whose
// f32 weights were freed (freeQwenVisionHostWeights) — the guard that keeps a device-resident tower
// from silently computing over empty slices if something calls the host path on it by mistake.
func TestQwenVisionTowerForward_FreedTower_Bad(t *testing.T) {
	tower := qtRealTower(1)
	_ = buildQwenVisionDeviceTower(tower) // stand-in for the mirror; not attached, just proving the free contract
	freeQwenVisionHostWeights(tower)
	const gridH, gridW = 2, 4
	patches := qtSeq(gridH*gridW*qtPatch, 1)
	if _, _, err := QwenVisionTowerForward(patches, gridH, gridW, tower); err == nil {
		t.Fatal("QwenVisionTowerForward on a freed tower must fail cleanly, not compute over empty slices")
	}
}

// TestFreeQwenVisionHostWeights_Good is the RSS receipt: every large f32 weight slice is nil after
// the call, on both tensor layouts — the host allocation the #59 device-tower follow-up's memory
// story claims is actually released, not merely unreferenced-but-still-large.
func TestFreeQwenVisionHostWeights_Good(t *testing.T) {
	for _, tc := range []struct {
		name  string
		tower *qwen35.VisionTower
	}{
		{"real", qtRealTower(2)},
		{"guessed", qtGuessedTower()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tower := tc.tower
			freeQwenVisionHostWeights(tower)
			if len(tower.Patch.W) != 0 || len(tower.Patch.B) != 0 {
				t.Fatal("Patch weights not freed")
			}
			if len(tower.PosEmbed) != 0 {
				t.Fatal("PosEmbed not freed")
			}
			for i := range tower.Blocks {
				b := &tower.Blocks[i]
				if len(b.Norm1W) != 0 || len(b.Norm1B) != 0 || len(b.Norm2W) != 0 || len(b.Norm2B) != 0 {
					t.Fatalf("block %d norms not freed", i)
				}
				for name, l := range map[string]qwen35.VisionLinear{
					"Q": b.Attn.Q, "K": b.Attn.K, "V": b.Attn.V, "O": b.Attn.O,
					"Gate": b.MLP.Gate, "Up": b.MLP.Up, "Down": b.MLP.Down, "FC1": b.MLP.FC1, "FC2": b.MLP.FC2,
				} {
					if len(l.W) != 0 || len(l.B) != 0 {
						t.Fatalf("block %d %s weights not freed", i, name)
					}
				}
			}
			if len(tower.Merger.NormW) != 0 || len(tower.Merger.NormB) != 0 {
				t.Fatal("merger norm not freed")
			}
			if len(tower.Merger.L1.W) != 0 || len(tower.Merger.L2.W) != 0 {
				t.Fatal("merger linears not freed")
			}
		})
	}
}

// TestLoadQwenVisionTower_DeviceResident_Good is the load-seam integration receipt: a tower loaded
// through loadQwenVisionTower (the real production path, not a hand-built qt* fixture) comes back
// with a populated DeviceSeam and freed f32 host weights, by default.
func TestLoadQwenVisionTower_DeviceResident_Good(t *testing.T) {
	dir := writeQwenVisionDir(t, qwenVisionDirConfig, qwenVisionDirTensors())
	dm, err := safetensors.LoadDirMmap(dir)
	if err != nil {
		t.Fatalf("LoadDirMmap: %v", err)
	}
	defer func() { _ = dm.Close() }()
	tower, err := loadQwenVisionTower(dir, dm)
	if err != nil {
		t.Fatalf("loadQwenVisionTower: %v", err)
	}
	dt, ok := tower.DeviceSeam.(*qwenVisionDeviceTower)
	if !ok || dt == nil {
		t.Fatal("loadQwenVisionTower must attach a *qwenVisionDeviceTower to DeviceSeam by default")
	}
	if len(tower.Patch.W) != 0 {
		t.Fatal("loadQwenVisionTower must free the f32 host copy once the bf16 device mirror is built")
	}
	// projectQwenImage must still serve a full image through the device path on this exact tower.
	features, softTokens, err := projectQwenImage(tower, qwenVisionTestPNG(t, 4, 4))
	if err != nil {
		t.Fatalf("projectQwenImage (device path): %v", err)
	}
	if softTokens <= 0 || len(features) != softTokens*tower.Cfg.TextHidden*bf16Size {
		t.Fatalf("device projectQwenImage: softTokens=%d features=%d bytes", softTokens, len(features))
	}
}

// TestLoadQwenVisionTower_DeviceDisabled_Good proves LTHN_QWEN_VISION_DEVICE=0 restores the v1
// host-only behaviour exactly: no DeviceSeam, f32 weights left populated, projectQwenImage still
// serves correctly through the host tower.
func TestLoadQwenVisionTower_DeviceDisabled_Good(t *testing.T) {
	t.Setenv("LTHN_QWEN_VISION_DEVICE", "0")
	dir := writeQwenVisionDir(t, qwenVisionDirConfig, qwenVisionDirTensors())
	dm, err := safetensors.LoadDirMmap(dir)
	if err != nil {
		t.Fatalf("LoadDirMmap: %v", err)
	}
	defer func() { _ = dm.Close() }()
	tower, err := loadQwenVisionTower(dir, dm)
	if err != nil {
		t.Fatalf("loadQwenVisionTower: %v", err)
	}
	if tower.DeviceSeam != nil {
		t.Fatal("LTHN_QWEN_VISION_DEVICE=0 must leave DeviceSeam nil")
	}
	if len(tower.Patch.W) == 0 {
		t.Fatal("LTHN_QWEN_VISION_DEVICE=0 must leave the f32 host weights populated")
	}
	features, softTokens, err := projectQwenImage(tower, qwenVisionTestPNG(t, 4, 4))
	if err != nil {
		t.Fatalf("projectQwenImage (host path): %v", err)
	}
	if softTokens <= 0 || len(features) != softTokens*tower.Cfg.TextHidden*bf16Size {
		t.Fatalf("host projectQwenImage: softTokens=%d features=%d bytes", softTokens, len(features))
	}
}

// TestQwenVisionSplitMergeHeadsBF16_RoundTrip_Good proves qwenVisionSplitHeadsBF16 and
// qwenVisionMergeHeadsBF16 are exact inverses — the token-major/head-major reshape VisionSDPA's
// layout needs, isolated from the rest of the forward so a transpose bug cannot hide behind
// compensating errors elsewhere.
func TestQwenVisionSplitMergeHeadsBF16_RoundTrip_Good(t *testing.T) {
	const heads, l, headDim = 3, 5, 4
	tokenMajor := f32ToBf16Slice(qtSeq(heads*l*headDim, 77))
	headMajor := qwenVisionSplitHeadsBF16(tokenMajor, heads, l, headDim)
	back := qwenVisionMergeHeadsBF16(headMajor, heads, l, headDim)
	if len(back) != len(tokenMajor) {
		t.Fatalf("round-trip length = %d, want %d", len(back), len(tokenMajor))
	}
	for i := range back {
		if back[i] != tokenMajor[i] {
			t.Fatalf("round-trip mismatch at byte %d: %v vs %v", i, back[i], tokenMajor[i])
		}
	}
}

// TestQwenVisionMergeSpatialBF16_MatchesF32_Good proves the bf16 spatial-merge gather
// (qwenVisionMergeSpatialBF16) reorders rows identically to the host f32 gather
// (qwenVisionMergeSpatial) — same permutation, only the element width differs.
func TestQwenVisionMergeSpatialBF16_MatchesF32_Good(t *testing.T) {
	const gridH, gridW, hidden, m = 4, 4, 3, 2
	f32x := qtSeq(gridH*gridW*hidden, 5)
	wantF32 := qwenVisionMergeSpatial(f32x, gridH, gridW, hidden, m)
	got := bf16ToF32Slice(qwenVisionMergeSpatialBF16(f32ToBf16Slice(f32x), gridH, gridW, hidden, m))
	// bf16 round-trip of the f32 oracle's output — compare through the SAME downcast rather than
	// expecting bit-identity to the f32 source (the gather reorders rows; it does not touch values).
	want := bf16ToF32Slice(f32ToBf16Slice(wantF32))
	if len(got) != len(want) {
		t.Fatalf("merged length = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("merged row mismatch at %d: got %v want %v", i, got[i], want[i])
		}
	}
}

// TestQwenVisionSiluGateMulBF16_MatchesHostSilu_Good proves the device SwiGLU gate matches
// qwenVisionSilu's existing f64 formula within bf16 rounding. The want values are computed from the
// SAME bf16-rounded inputs the function under test actually consumes (gBf16/uBf16), so this isolates
// the gate-mul arithmetic from bf16's own input-rounding error.
func TestQwenVisionSiluGateMulBF16_MatchesHostSilu_Good(t *testing.T) {
	g := qtSeq(16, 11)
	u := qtSeq(16, 12)
	gBf16, uBf16 := bf16ToF32Slice(f32ToBf16Slice(g)), bf16ToF32Slice(f32ToBf16Slice(u))
	gBytes, uBytes := f32ToBf16Slice(g), f32ToBf16Slice(u)
	qwenVisionSiluGateMulBF16(gBytes, uBytes)
	got := bf16ToF32Slice(gBytes)
	for i := range g {
		want := float32(qwenVisionSilu(float64(gBf16[i])) * float64(uBf16[i]))
		if d := math.Abs(float64(got[i] - want)); d > 1e-3 {
			t.Fatalf("silu-gate-mul[%d] = %v, want %v (diff %v)", i, got[i], want, d)
		}
	}
}
