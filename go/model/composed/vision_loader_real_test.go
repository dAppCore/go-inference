// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/internal/enginegate"
	"dappco.re/go/inference/model/hf"
	"dappco.re/go/inference/model/safetensors"
)

// vision_loader_real_test.go is done-gate #1's reconciliation receipt: it opens the REAL
// mlx-community/Qwen3.6-27B-4bit snapshot's safetensors HEADERS ONLY — safetensors.IndexFiles reads just
// the 8-byte little-endian length prefix + the JSON header per shard (safetensors/index.go's ReadIndex/
// ParseHeaderRefs), the tensor DATA is never touched — and proves buildVisionTowerQuant's name-resolution
// covers EVERY vision_tower.* key the checkpoint actually ships: no unmatched tensor (every real name
// matches a suffix the loader probes) and no expected-but-missing one (every suffix the loader probes is
// actually present). Skips cleanly (via enginegate.HFModelPath) when the snapshot isn't in the local HF
// cache, so CI stays green off this machine — the checkpoint is a multi-GB download this test never fetches
// itself. The functional (load-and-forward) proof for this same tensor layout, at small synthetic
// dimensions, lives in vision_loader_test.go's addRealLayoutVisionTensors-based tests; this file is the
// artifact-fidelity check that those small dimensions still mirror the real thing.

// visionBlockSuffixes is every vision_tower.blocks.<i>.<suffix> the loader (buildVisionBlocksQuant,
// loadBlockQKV, loadBlockMLP — vision_loader.go) resolves for the REAL layout. Kept in lockstep with that
// code by hand — this table IS the "no unmatched / no expected-but-missing" half of the receipt, so a
// future loader change that adds/renames a probed suffix must update this table too, and this test will
// catch a drift the moment it runs against the real checkpoint again.
var visionBlockSuffixes = map[string]bool{
	"norm1.weight": true, "norm1.bias": true,
	"norm2.weight": true, "norm2.bias": true,
	"attn.qkv.weight": true, "attn.qkv.bias": true,
	"attn.proj.weight": true, "attn.proj.bias": true,
	"mlp.linear_fc1.weight": true, "mlp.linear_fc1.bias": true,
	"mlp.linear_fc2.weight": true, "mlp.linear_fc2.bias": true,
}

// visionRootNames is every non-block vision_tower.* name the loader resolves for the REAL layout
// (buildVisionTowerQuant's patch_embed/pos_embed probes, buildVisionMergerQuant's vision_tower.merger.*
// alias branch).
var visionRootNames = map[string]bool{
	"vision_tower.patch_embed.proj.weight": true, "vision_tower.patch_embed.proj.bias": true,
	"vision_tower.pos_embed.weight":   true,
	"vision_tower.merger.norm.weight": true, "vision_tower.merger.norm.bias": true,
	"vision_tower.merger.linear_fc1.weight": true, "vision_tower.merger.linear_fc1.bias": true,
	"vision_tower.merger.linear_fc2.weight": true, "vision_tower.merger.linear_fc2.bias": true,
}

// blockSuffixOf splits a "vision_tower.blocks.<N>.<suffix>" name into its suffix, or ("", false) when name
// doesn't have that shape (wrong prefix, or no "." after the block index). Pure string scanning — no
// regexp/strings import needed (both banned by this repo's house rules beyond core's own helpers).
func blockSuffixOf(name string) (suffix string, ok bool) {
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
	var raw loaderConfig
	if r := core.JSONUnmarshal(cfgRead.Value.([]byte), &raw); !r.OK {
		t.Fatalf("parse config.json: %v", r.Err())
	}
	if raw.VisionConfig == nil {
		t.Fatal("config.json carries no vision_config — wrong snapshot, or the family changed shape")
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
		if visionRootNames[name] {
			continue
		}
		suffix, ok := blockSuffixOf(name)
		if !ok || !visionBlockSuffixes[suffix] {
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
	for suffix := range visionBlockSuffixes {
		if !seenSuffix[suffix] {
			t.Errorf("loader probes block suffix %q but the real checkpoint never ships it (expected-but-missing)", suffix)
		}
	}
	for name := range visionRootNames {
		if _, ok := idx.Tensors[name]; !ok {
			t.Errorf("loader probes root tensor %q but the real checkpoint never ships it (expected-but-missing)", name)
		}
	}

	// --- geometry spot-checks: tie the raw header shapes to the loader's OWN derivation formulas, so a
	// divergence in the maths (not just the names) is caught here too, without fabricating gigabytes of
	// tensor data to actually run buildVisionTowerQuant end-to-end (the functional proof, at small synthetic
	// dimensions exercising the identical code paths, lives in vision_loader_test.go). ---
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
	if vc.PatchSize <= 0 || vc.InChannels <= 0 {
		t.Fatalf("vision_config.patch_size/in_channels = %d/%d, want both positive", vc.PatchSize, vc.InChannels)
	}
	perFrame := vc.InChannels * vc.PatchSize * vc.PatchSize
	if perFrame <= 0 || patchDim%perFrame != 0 {
		t.Fatalf("patch_embed input width %d is not a multiple of in_channels·patch_size² %d — buildVisionTowerQuant's temporal derivation would fail loudly on this checkpoint", patchDim, perFrame)
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
		t.Fatalf("merger.linear_fc1 input %v is not a multiple of hidden %d — buildVisionMergerQuant's mergeSize derivation would fail loudly on this checkpoint", merger1.Shape, hidden)
	}
	mergeSq := int(merger1.Shape[1]) / hidden
	if mergeSize := isqrt(mergeSq); mergeSize <= 0 || mergeSize*mergeSize != mergeSq {
		t.Fatalf("merger.linear_fc1 input/hidden = %d is not a perfect square — buildVisionMergerQuant's isqrt derivation would fail loudly on this checkpoint", mergeSq)
	}

	t.Logf("name-coverage receipt: %d real vision_tower.* tensor(s) (%d block-suffixed + %d root), all %d resolved suffixes seen, %d root names present, 0 unmatched, 0 expected-but-missing; hidden=%d patchDim=%d perQKV=%d",
		len(visionNames), len(visionNames)-len(visionRootNames), len(visionRootNames), len(visionBlockSuffixes), len(visionRootNames), hidden, patchDim, perQKV)
}
