// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"testing"

	core "dappco.re/go"
)

func TestLoadGemma4ImageFeatureConfigs_Good(t *testing.T) {
	dir := t.TempDir()
	write := core.WriteFile(core.PathJoin(dir, "processor_config.json"), []byte(`{
		"image_processor": {"patch_size": 14, "max_soft_tokens": 128, "pooling_kernel_size": 2, "rescale_factor": 0.5, "do_resize": true},
		"video_processor": {"max_soft_tokens": 64, "num_frames": 8}
	}`), 0o644)
	if !write.OK {
		t.Fatalf("write processor_config: %v", write.Value)
	}
	imageCfg, videoCfg, err := LoadGemma4ImageFeatureConfigs(dir)
	if err != nil {
		t.Fatalf("LoadGemma4ImageFeatureConfigs: %v", err)
	}
	if imageCfg == nil || videoCfg == nil {
		t.Fatalf("configs = %v/%v, want both", imageCfg, videoCfg)
	}
	if imageCfg.PatchSize != 14 || imageCfg.MaxSoftTokens != 128 || imageCfg.PoolingKernelSize != 2 || imageCfg.RescaleFactor != 0.5 || !imageCfg.DoResize {
		t.Fatalf("image config = %+v", imageCfg)
	}
	if videoCfg.PatchSize != 16 || videoCfg.MaxSoftTokens != 64 || videoCfg.PoolingKernelSize != 3 || videoCfg.RescaleFactor == 0 || videoCfg.NumFrames != 8 {
		t.Fatalf("video config = %+v", videoCfg)
	}
}

// TestLoadGemma4ImageFeatureConfigs_MissingConfig covers the text-serving path: a directory with
// no processor_config.json returns (nil, nil, nil) — a text-only pack, not an error.
func TestLoadGemma4ImageFeatureConfigs_MissingConfig(t *testing.T) {
	imageCfg, videoCfg, err := LoadGemma4ImageFeatureConfigs(t.TempDir())
	if err != nil {
		t.Fatalf("a missing processor_config should not error, got %v", err)
	}
	if imageCfg != nil || videoCfg != nil {
		t.Fatalf("configs = %v/%v, want both nil for a text-only pack", imageCfg, videoCfg)
	}
	t.Logf("LoadGemma4ImageFeatureConfigs: absent processor_config → (nil,nil,nil)")
}

// TestLoadGemma4ImageFeatureConfigs_Malformed covers the parse-failure path: an unparseable
// processor_config.json surfaces an error rather than a half-built config.
func TestLoadGemma4ImageFeatureConfigs_Malformed(t *testing.T) {
	dir := t.TempDir()
	write := core.WriteFile(core.PathJoin(dir, "processor_config.json"), []byte(`{not json`), 0o644)
	if !write.OK {
		t.Fatalf("write processor_config: %v", write.Value)
	}
	if _, _, err := LoadGemma4ImageFeatureConfigs(dir); err == nil {
		t.Fatal("an unparseable processor_config.json should surface an error")
	}
	t.Logf("LoadGemma4ImageFeatureConfigs: malformed JSON → error")
}

// TestNormalizeGemma4ImageFeatureConfig_Nil covers the nil pass-through: an absent processor
// section (a pack declaring only image or only video) normalises to nil, not a defaulted config.
func TestNormalizeGemma4ImageFeatureConfig_Nil(t *testing.T) {
	if got := normalizeGemma4ImageFeatureConfig(nil); got != nil {
		t.Fatalf("normalizeGemma4ImageFeatureConfig(nil) = %+v, want nil", got)
	}
	// End-to-end: a config declaring only image_processor yields a nil video config.
	dir := t.TempDir()
	write := core.WriteFile(core.PathJoin(dir, "processor_config.json"),
		[]byte(`{"image_processor": {"patch_size": 14}}`), 0o644)
	if !write.OK {
		t.Fatalf("write processor_config: %v", write.Value)
	}
	imageCfg, videoCfg, err := LoadGemma4ImageFeatureConfigs(dir)
	if err != nil {
		t.Fatalf("LoadGemma4ImageFeatureConfigs: %v", err)
	}
	if imageCfg == nil || videoCfg != nil {
		t.Fatalf("image-only pack = image:%v video:%v, want image set, video nil", imageCfg, videoCfg)
	}
	t.Logf("normalizeGemma4ImageFeatureConfig: nil → nil; image-only pack leaves video config nil")
}
