// SPDX-Licence-Identifier: EUPL-1.2

package qwen35

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/internal/enginegate"
	"dappco.re/go/inference/model/hf"
	"dappco.re/go/inference/model/safetensors"
)

// mtp_drafter_real_test.go is the drafter arch route's reconciliation receipt (mirrors
// vision_real_test.go's pattern for the vision tower): it opens the REAL
// mlx-community/Qwen3.6-27B-MTP-4bit snapshot's safetensors HEADER ONLY — safetensors.IndexFiles reads
// just the length-prefix + JSON header, tensor DATA is never touched — and proves DrafterTensorNames'
// mapping covers EVERY tensor the checkpoint actually ships: no unmatched tensor, no
// expected-but-missing one. Skips cleanly (enginegate.HFModelPath) when the snapshot isn't in the
// local HF cache. The functional parse/derive proof at real dimensions lives in mtp_drafter_test.go's
// TestParseDrafterConfig_Good/TestConfigDrafterArch_Good (which inline this SAME checkpoint's
// config.json content, captured by hand from the same snapshot this file reads live); this file is the
// artefact-fidelity check that those inlined numbers still mirror the real thing, and that no tensor
// name in the checkpoint was missed.
//
// The real checkpoint is 4-bit affine-quantised: every Linear ships as three header entries
// (<name>.weight + .scales + .biases), while the six norm/combiner vectors ship as a single BF16
// .weight only. quantSidecarBase strips a .scales/.biases suffix so both forms reconcile against the
// SAME DrafterTensorNames() list (which names only the logical .weight tensor, matching how
// model.Assemble's LoadLinear treats scales/biases as optional companions, not separate entries).

// quantSidecarBase strips a trailing ".scales" or ".biases" and returns (base+".weight", true) — the
// logical tensor name a quantised sidecar belongs to — or ("", false) for a name that carries neither
// suffix (a plain, unquantised BF16 tensor, checked against DrafterTensorNames() as-is).
func quantSidecarBase(name string) (string, bool) {
	if core.HasSuffix(name, ".scales") {
		return core.TrimSuffix(name, ".scales") + ".weight", true
	}
	if core.HasSuffix(name, ".biases") {
		return core.TrimSuffix(name, ".biases") + ".weight", true
	}
	return "", false
}

// TestDrafterTensorNames_RealCheckpoint is the reconciliation receipt described in the file doc
// comment above.
func TestDrafterTensorNames_RealCheckpoint(t *testing.T) {
	snap := enginegate.HFModelPath(t, "mlx-community/Qwen3.6-27B-MTP-4bit")

	cfgRead := core.ReadFile(core.PathJoin(snap, "config.json"))
	if !cfgRead.OK {
		t.Fatalf("read config.json: %v", cfgRead.Err())
	}
	cfg, err := ParseDrafterConfig(cfgRead.Value.([]byte))
	if err != nil {
		t.Fatalf("ParseDrafterConfig: %v — wrong snapshot, or the family's drafter shape changed", err)
	}
	nLayers := cfg.effective().MTPNumHiddenLayers
	if nLayers <= 0 {
		t.Fatalf("effective().MTPNumHiddenLayers = %d, want > 0", nLayers)
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
	if len(idx.Names) == 0 {
		t.Fatal("no tensors in the real checkpoint's header — wrong snapshot?")
	}

	want := map[string]bool{}
	for _, n := range DrafterTensorNames(nLayers) {
		want[n] = true
	}

	// --- no unmatched tensor: every real name resolves to a wanted .weight, directly or via a
	// quantised .scales/.biases sidecar ---
	seenWant := map[string]bool{}
	var unmatched []string
	for _, name := range idx.Names {
		if want[name] {
			seenWant[name] = true
			continue
		}
		if base, ok := quantSidecarBase(name); ok && want[base] {
			seenWant[base] = true
			continue
		}
		unmatched = append(unmatched, name)
	}
	if len(unmatched) > 0 {
		shown := unmatched
		if len(shown) > 10 {
			shown = shown[:10]
		}
		t.Fatalf("%d real tensor name(s) not resolved by DrafterTensorNames: %v", len(unmatched), shown)
	}

	// --- no expected-but-missing: every name DrafterTensorNames declares actually ships ---
	for name := range want {
		if !seenWant[name] {
			t.Errorf("DrafterTensorNames declares %q but the real checkpoint never ships it (expected-but-missing)", name)
		}
	}

	// --- geometry spot-checks: tie the raw header shapes to DrafterArch's own derivation ---
	arch, err := cfg.DrafterArch()
	if err != nil {
		t.Fatalf("DrafterArch: %v", err)
	}
	d := arch.Hidden
	fc, ok := idx.Tensors[DrafterFCWeight]
	if !ok {
		t.Fatal("fc.weight missing from the real checkpoint header")
	}
	if len(fc.Shape) != 2 || int(fc.Shape[0]) != d {
		t.Fatalf("fc.weight shape %v, want OutDim %d (the head's own hidden size)", fc.Shape, d)
	}
	// fc.weight is 4-bit packed (8 values/U32 word): unpacked InDim must be exactly 2*hidden (the
	// [normed embed ; normed hidden] concatenation width).
	quant := cfg.ResolvedQuant()
	if quant == nil {
		t.Fatal("config.json carries no quantization block — wrong snapshot (expected the -4bit pack)")
	}
	_, bits := quant.For("fc")
	if bits <= 0 {
		t.Fatalf("quantization.bits = %d, want > 0", bits)
	}
	packFactor := 32 / bits
	unpackedIn := int(fc.Shape[1]) * packFactor
	if unpackedIn != 2*d {
		t.Fatalf("fc.weight unpacked InDim = %d (shape[1]=%d * pack factor %d), want 2*hidden=%d", unpackedIn, fc.Shape[1], packFactor, 2*d)
	}

	qProj, ok := idx.Tensors["layers.0.self_attn.q_proj.weight"]
	if !ok {
		t.Fatal("layers.0.self_attn.q_proj.weight missing from the real checkpoint header")
	}
	wantQOut := arch.Heads * arch.Layer[0].HeadDim
	if arch.AttnOutputGate {
		wantQOut *= 2 // gated q_proj emits [q ; gate] — doubled rows, matching qwen35.go's own comment
	}
	if int(qProj.Shape[0]) != wantQOut {
		t.Fatalf("q_proj.weight OutDim = %d, want %d (heads=%d * head_dim=%d, gated=%v)", qProj.Shape[0], wantQOut, arch.Heads, arch.Layer[0].HeadDim, arch.AttnOutputGate)
	}

	t.Logf("reconciliation receipt: %d real tensor(s) for %d head layer(s), all DrafterTensorNames present, 0 unmatched, 0 expected-but-missing; hidden=%d heads=%d kvHeads=%d headDim=%d FF=%d gated=%v",
		len(idx.Names), nLayers, d, arch.Heads, arch.Layer[0].KVHeads, arch.Layer[0].HeadDim, arch.FF, arch.AttnOutputGate)
}

// realDrafterSnapshotRepo is factored out for TestDrafterTensorNames_RealCheckpoint's sibling
// TestParseDrafterConfig_RealSnapshotSharesBaseHidden below, so both tests name the identical repo id
// once.
const realDrafterSnapshotRepo = "mlx-community/Qwen3.6-27B-MTP-4bit"

// TestParseDrafterConfig_RealSnapshotSharesBaseHidden proves the real drafter's declared hidden size
// matches the real BASE checkpoint's (mlx-community/Qwen3.6-27B-4bit) — the D==base.D attachment
// invariant a future pair loader validates (see design-qwen-mtp-pair.md part 1/5), checked here at the
// config level since no pair loader exists yet. Skips (independently) if either snapshot is absent.
func TestParseDrafterConfig_RealSnapshotSharesBaseHidden(t *testing.T) {
	draftSnap := enginegate.HFModelPath(t, realDrafterSnapshotRepo)
	baseSnap := enginegate.HFModelPath(t, "mlx-community/Qwen3.6-27B-4bit")

	draftCfg := core.ReadFile(core.PathJoin(draftSnap, "config.json"))
	if !draftCfg.OK {
		t.Fatalf("read drafter config.json: %v", draftCfg.Err())
	}
	cfg, err := ParseDrafterConfig(draftCfg.Value.([]byte))
	if err != nil {
		t.Fatalf("ParseDrafterConfig: %v", err)
	}

	baseCfg := core.ReadFile(core.PathJoin(baseSnap, "config.json"))
	if !baseCfg.OK {
		t.Fatalf("read base config.json: %v", baseCfg.Err())
	}
	var base Config
	if r := core.JSONUnmarshal(baseCfg.Value.([]byte), &base); !r.OK {
		t.Fatalf("parse base config.json: %v", r.Err())
	}

	if got, want := cfg.effective().HiddenSize, base.effective().HiddenSize; got != want {
		t.Errorf("drafter hidden_size = %d, want the base's %d (the D==base.D attachment a pair loader must validate)", got, want)
	}
}
