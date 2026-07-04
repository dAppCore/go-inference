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
