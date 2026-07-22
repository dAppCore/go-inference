// SPDX-Licence-Identifier: EUPL-1.2

package qwen35

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/internal/enginegate"
	"dappco.re/go/inference/model/hf"
	"dappco.re/go/inference/model/safetensors"
)

// vision_real_test.go is the factory port's reconciliation receipt (carried over from the retired
// composed lane's vision_loader_real_test.go): it opens the REAL mlx-community/Qwen3.6-27B-4bit
// snapshot's safetensors HEADERS ONLY — safetensors.IndexFiles reads just the length prefix + JSON
// header per shard, the tensor DATA is never touched — and proves LoadVisionTower's name-resolution
// covers EVERY vision_tower.* key the checkpoint actually ships: no unmatched tensor and no
// expected-but-missing one. Skips cleanly (enginegate.HFModelPath) when the snapshot isn't in the
// local HF cache. The functional load-and-forward proof at small synthetic dimensions lives in
// vision_loader_test.go; this file is the artifact-fidelity check that those small dimensions still
// mirror the real thing.

// realVisionBlockSuffixes is every vision_tower.blocks.<i>.<suffix> the loader resolves for the REAL
// layout. Kept in lockstep with vision_loader.go by hand — this table IS the "no unmatched / no
// expected-but-missing" half of the receipt.
var realVisionBlockSuffixes = map[string]bool{
	"norm1.weight": true, "norm1.bias": true,
	"norm2.weight": true, "norm2.bias": true,
	"attn.qkv.weight": true, "attn.qkv.bias": true,
	"attn.proj.weight": true, "attn.proj.bias": true,
	"mlp.linear_fc1.weight": true, "mlp.linear_fc1.bias": true,
	"mlp.linear_fc2.weight": true, "mlp.linear_fc2.bias": true,
}

// realVisionRootNames is every non-block vision_tower.* name the loader resolves for the REAL layout.
var realVisionRootNames = map[string]bool{
	"vision_tower.patch_embed.proj.weight": true, "vision_tower.patch_embed.proj.bias": true,
	"vision_tower.pos_embed.weight":   true,
	"vision_tower.merger.norm.weight": true, "vision_tower.merger.norm.bias": true,
	"vision_tower.merger.linear_fc1.weight": true, "vision_tower.merger.linear_fc1.bias": true,
	"vision_tower.merger.linear_fc2.weight": true, "vision_tower.merger.linear_fc2.bias": true,
}

// realBlockSuffixOf splits a "vision_tower.blocks.<N>.<suffix>" name into its suffix, or ("", false)
// when name doesn't have that shape.
func realBlockSuffixOf(name string) (suffix string, ok bool) {
	const prefix = "vision_tower.blocks."
	if !core.HasPrefix(name, prefix) {
		return "", false
	}
	rest := name[len(prefix):] // "<N>.<suffix...>"
	for i := 0; i < len(rest); i++ {
		if rest[i] == '.' {
			return rest[i+1:], true
		}
	}
	return "", false
}

// TestVisionTowerNameCoverage_RealCheckpoint is the reconciliation receipt described in the file doc
// comment above.
func TestVisionTowerNameCoverage_RealCheckpoint(t *testing.T) {
	snap := enginegate.HFModelPath(t, "mlx-community/Qwen3.6-27B-4bit")

	cfgRead := core.ReadFile(core.PathJoin(snap, "config.json"))
	if !cfgRead.OK {
		t.Fatalf("read config.json: %v", cfgRead.Err())
	}
	var raw Config
	if r := core.JSONUnmarshal(cfgRead.Value.([]byte), &raw); !r.OK {
		t.Fatalf("parse config.json: %v", r.Err())
	}
	if raw.VisionConfig == nil {
		t.Fatal("config.json carries no vision_config — wrong snapshot, or the family changed shape")
	}
	if raw.ImageTokenID == 0 {
		t.Fatal("config.json carries no image_token_id — the splice would have no placeholder id")
	}

	var shardPaths []string
	for _, f := range hf.LocalModelFiles(snap) {
		if core.HasSuffix(core.Lower(f.Name), ".safetensors") {
			shardPaths = append(shardPaths, core.PathJoin(snap, f.Name))
		}
	}
	if len(shardPaths) == 0 {
		t.Fatalf("no .safetensors shards found under %s", snap)
	}

	idx, err := safetensors.IndexFiles(shardPaths)
	if err != nil {
		t.Fatalf("IndexFiles (header-only, no tensor data read): %v", err)
	}

	var visionNames []string
	for _, name := range idx.Names {
		if core.HasPrefix(name, "vision_tower.") {
			visionNames = append(visionNames, name)
		}
	}
	if len(visionNames) == 0 {
		t.Fatal("no vision_tower.* tensors in the real checkpoint's header — wrong snapshot?")
	}

	// --- no unmatched tensor: every real name matches a suffix/root-name the loader probes ---
	seenSuffix := map[string]bool{}
	var unmatched []string
	for _, name := range visionNames {
		if realVisionRootNames[name] {
			continue
		}
		suffix, ok := realBlockSuffixOf(name)
		if !ok || !realVisionBlockSuffixes[suffix] {
			unmatched = append(unmatched, name)
			continue
		}
		seenSuffix[suffix] = true
	}
	if len(unmatched) > 0 {
		shown := unmatched
		if len(shown) > 10 {
			shown = shown[:10]
		}
		t.Fatalf("%d real vision_tower.* tensor name(s) not resolved by any loader probe: %v", len(unmatched), shown)
	}

	// --- no expected-but-missing: every suffix/root-name the loader probes actually ships ---
	for suffix := range realVisionBlockSuffixes {
		if !seenSuffix[suffix] {
			t.Errorf("loader probes block suffix %q but the real checkpoint never ships it (expected-but-missing)", suffix)
		}
	}
	for name := range realVisionRootNames {
		if _, ok := idx.Tensors[name]; !ok {
			t.Errorf("loader probes root tensor %q but the real checkpoint never ships it (expected-but-missing)", name)
		}
	}

	// --- geometry spot-checks: tie the raw header shapes to the loader's own derivation formulas ---
	vc := raw.VisionConfig
	patch, ok := idx.Tensors["vision_tower.patch_embed.proj.weight"]
	if !ok {
		t.Fatal("vision_tower.patch_embed.proj.weight missing from the real checkpoint header")
	}
	if len(patch.Shape) < 2 {
		t.Fatalf("patch_embed.proj.weight shape %v has no input dimensions", patch.Shape)
	}
	hidden := int(patch.Shape[0])
	patchDim := 1
	for _, d := range patch.Shape[1:] {
		patchDim *= int(d)
	}
	if vc.PatchSize <= 0 {
		t.Fatalf("vision_config.patch_size = %d, want positive", vc.PatchSize)
	}
	inChannels := vc.InChannels
	if inChannels <= 0 {
		inChannels = 3 // the loader's own fallback — the real config omits in_channels at times
	}
	perFrame := inChannels * vc.PatchSize * vc.PatchSize
	if perFrame <= 0 || patchDim%perFrame != 0 {
		t.Fatalf("patch_embed input width %d is not a multiple of in_channels·patch_size² %d — the temporal derivation would fail loudly on this checkpoint", patchDim, perFrame)
	}

	qkv0, ok := idx.Tensors["vision_tower.blocks.0.attn.qkv.weight"]
	if !ok {
		t.Fatal("vision_tower.blocks.0.attn.qkv.weight missing from the real checkpoint header")
	}
	if len(qkv0.Shape) != 2 || qkv0.Shape[1] != uint64(hidden) {
		t.Fatalf("blocks.0.attn.qkv.weight shape %v, want [*, %d] (fused qkv reads the SAME hidden width patch_embed emits)", qkv0.Shape, hidden)
	}
	if qkv0.Shape[0]%3 != 0 {
		t.Fatalf("blocks.0.attn.qkv.weight OutDim %d is not divisible by 3 — splitFusedQKV's equal-thirds assumption would fail loudly on this checkpoint", qkv0.Shape[0])
	}
	perQKV := int(qkv0.Shape[0]) / 3
	if vc.NumHeads <= 0 || perQKV%vc.NumHeads != 0 {
		t.Fatalf("fused qkv per-branch width %d is not divisible by vision_config.num_heads %d — resolveVisionAttnGeometry's fallback would fail loudly on this checkpoint", perQKV, vc.NumHeads)
	}

	merger1, ok := idx.Tensors["vision_tower.merger.linear_fc1.weight"]
	if !ok {
		t.Fatal("vision_tower.merger.linear_fc1.weight missing from the real checkpoint header")
	}
	if len(merger1.Shape) != 2 || hidden <= 0 || int(merger1.Shape[1])%hidden != 0 {
		t.Fatalf("merger.linear_fc1 input %v is not a multiple of hidden %d — the mergeSize derivation would fail loudly on this checkpoint", merger1.Shape, hidden)
	}
	mergeSq := int(merger1.Shape[1]) / hidden
	if mergeSize := isqrt(mergeSq); mergeSize <= 0 || mergeSize*mergeSize != mergeSq {
		t.Fatalf("merger.linear_fc1 input/hidden = %d is not a perfect square — the isqrt derivation would fail loudly on this checkpoint", mergeSq)
	}

	t.Logf("name-coverage receipt: %d real vision_tower.* tensor(s), all block suffixes seen, %d root names present, 0 unmatched, 0 expected-but-missing; hidden=%d patchDim=%d perQKV=%d image_token_id=%d",
		len(visionNames), len(realVisionRootNames), hidden, patchDim, perQKV, int32(raw.ImageTokenID))
}
