// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"os"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// TestDumpArchFromSnapshot is an env-gated real-config diagnostic: GEMMA4_SNAP=<snapshot dir>
// parses config.json through the SAME lane the engine loader uses (parseGemma4Config → Arch)
// and dumps every attention-geometry field plus the first global/sliding layer specs — the
// one-print check that a derived arch matches the config's declared truth.
func TestDumpArchFromSnapshot(t *testing.T) {
	snap := os.Getenv("GEMMA4_SNAP")
	if snap == "" {
		t.Skip("GEMMA4_SNAP not set")
	}
	raw := core.ReadFile(snap + "/config.json")
	if !raw.OK {
		t.Fatalf("read config: %v", raw.Value)
	}
	cfg, err := parseGemma4Config(raw.Value.([]byte))
	if err != nil {
		t.Fatalf("parseGemma4Config: %v", err)
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	t.Logf("Hidden=%d Layer=%d Heads=%d KVHeads=%d GlobalKVHeads=%d HeadDim=%d GlobalHeadDim=%d FF=%d",
		arch.Hidden, len(arch.Layer), arch.Heads, arch.KVHeads, arch.GlobalKVHeads, arch.HeadDim, arch.GlobalHeadDim, arch.FF)
	t.Logf("SlidingWindow=%d RotaryDim=%d RotaryDimLocal=%d RopeBase=%.0f RopeLocalBase=%.0f ValueNorm=%v Eps=%g RopeFreqs=%d",
		arch.SlidingWindow, arch.RotaryDim, arch.RotaryDimLocal, arch.RopeBase, arch.RopeLocalBase, arch.ValueNorm, arch.Eps, len(arch.RopeFreqs))
	var firstG, firstS = -1, -1
	for li, sp := range arch.Layer {
		if sp.Attention == model.GlobalAttention && firstG < 0 {
			firstG = li
		}
		if sp.Attention != model.GlobalAttention && firstS < 0 {
			firstS = li
		}
		if firstG >= 0 && firstS >= 0 {
			break
		}
	}
	for _, li := range []int{firstS, firstG} {
		if li < 0 {
			continue
		}
		sp := arch.Layer[li]
		t.Logf("L%02d attn=%v KVHeads=%d HeadDim=%d KVShareFrom=%d OwnsCache=%v MoE=%v",
			li, sp.Attention, sp.KVHeads, sp.HeadDim, sp.KVShareFrom, sp.OwnsCache(), sp.MoE)
	}
	// the KV-share map: every sharer must share from an owner with IDENTICAL attention
	// geometry — a cross-type share reads the owner's cache at the wrong kvDim/headDim.
	bad := 0
	for li, sp := range arch.Layer {
		if sp.OwnsCache() {
			continue
		}
		o := arch.Layer[sp.KVShareFrom]
		mark := ""
		if o.Attention != sp.Attention || o.KVHeads != sp.KVHeads || o.HeadDim != sp.HeadDim {
			mark = "  <<< GEOMETRY MISMATCH"
			bad++
		}
		t.Logf("L%02d(attn=%v kv=%d hd=%d) shares from L%02d(attn=%v kv=%d hd=%d)%s",
			li, sp.Attention, sp.KVHeads, sp.HeadDim, sp.KVShareFrom, o.Attention, o.KVHeads, o.HeadDim, mark)
	}
	t.Logf("cross-geometry shares: %d", bad)
}

// TestDumpLoadedLayersFromSnapshot is the loader-level half of the #348 hunt: GEMMA4_SNAP
// assembles the real checkpoint (mmap metadata only, no GPU) and dumps the per-layer weight
// dims the engine will actually run — the instrument that catches a mis-mapped tensor after
// the forward maths itself was exonerated by the host-reference ladder.
func TestDumpLoadedLayersFromSnapshot(t *testing.T) {
	snap := os.Getenv("GEMMA4_SNAP")
	if snap == "" {
		t.Skip("GEMMA4_SNAP not set")
	}
	m, dm, err := model.Load(snap)
	if err != nil {
		t.Fatalf("model.Load: %v", err)
	}
	defer func() { _ = dm.Close() }()
	dims := func(w *model.Linear) string {
		if w == nil {
			return "ABSENT"
		}
		return core.Sprintf("%dx%d quant=%v", w.OutDim, w.InDim, w.Quantised())
	}
	norm := func(b []byte) string {
		if len(b) == 0 {
			return "ABSENT"
		}
		return core.Sprintf("[%d]", len(b)/2)
	}
	for _, li := range []int{0, 5} {
		if li >= len(m.Layers) {
			continue
		}
		L := m.Layers[li]
		t.Logf("L%02d Q=%s K=%s V=%s O=%s", li, dims(L.Q), dims(L.K), dims(L.V), dims(L.O))
		t.Logf("L%02d qNorm=%s kNorm=%s postAttn=%s preFF(mlpNorm)=%s postFF=%s attnNorm=%s",
			li, norm(L.QNorm), norm(L.KNorm), norm(L.PostAttnNorm), norm(L.MLPNorm), norm(L.PostFFNorm), norm(L.AttnNorm))
	}
}
