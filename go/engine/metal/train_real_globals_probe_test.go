// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"dappco.re/go/inference/model"
)

// train_real_globals_probe_test.go is the #42 global-layer INSTRUMENT: the per-layer LoRA host
// mirror (buildRealLayerTemplates + realSharedChainForward) run over the REAL gemma4 E2B bf16
// checkpoint HOST-ONLY — no metal runtime, no engine session — against a reference per-layer
// hidden dump produced by the vendored mlx-lm gemma4_text reference (the ecosystem semantics the
// engine's own decode was cross-validated against). The weights and arch travel the LIVE trainer's
// exact host path — model.Load → loadedToBF16 → buildRealLayerTemplates — so a pass proves every
// host-side link of the live mirror, byte-for-byte. It prints every layer's cosine tagged by
// class (sliding / global / consumer) plus the attention-half cosine, so a divergence names its
// layer class AND its station — the visibility the live harness's single worst-layer number
// lacked when it measured 0.83 on the globals.
//
// Env-gated twice (skips cleanly in CI): E2B_BF16_DIR names the bf16 snapshot dir,
// E2B_MIRROR_ORACLE_DIR a dump directory holding f32 little-endian files produced by
// testdata/e2b_mirror_oracle.py (embeds_scaled.f32 [T,H] — the √hidden-scaled token embeddings;
// pli.f32 [T,NL,PLID] — the COMBINED per-layer-input rows exactly as ArchSession.perLayerInput
// returns them; layer_out.f32 [NL,T,H] — each layer's output hidden; attn_res.f32 [NL,T,H] —
// the residual after the attention half). T is fixed at 8 (the harness's parity ids).
func TestRealChainE2BMirrorVsReference_Good(t *testing.T) {
	dir := os.Getenv("E2B_BF16_DIR")
	oracleDir := os.Getenv("E2B_MIRROR_ORACLE_DIR")
	if dir == "" || oracleDir == "" {
		t.Skip("set E2B_BF16_DIR (bf16 snapshot) and E2B_MIRROR_ORACLE_DIR (reference dump) to run the host mirror probe")
	}

	lm, dm, err := model.Load(dir) // the live loader's front half — pure host (no metal buffers)
	if err != nil {
		t.Fatalf("model.Load(%s): %v", dir, err)
	}
	t.Cleanup(func() { _ = dm.Close() })
	arch, g := lm.Arch, loadedToBF16(lm)
	const T = 8
	H, NL, PLID := arch.Hidden, len(arch.Layer), arch.PerLayerInputHidden

	embeds := readF32File(t, filepath.Join(oracleDir, "embeds_scaled.f32"), T*H)
	pli := readF32File(t, filepath.Join(oracleDir, "pli.f32"), T*NL*PLID)
	want := readF32File(t, filepath.Join(oracleDir, "layer_out.f32"), NL*T*H)
	wantAttn := readF32File(t, filepath.Join(oracleDir, "attn_res.f32"), NL*T*H)

	layers, err := buildRealLayerTemplates(g, arch)
	if err != nil {
		t.Fatalf("buildRealLayerTemplates: %v", err)
	}
	shareFrom := make([]int, NL)
	for li := range arch.Layer {
		shareFrom[li] = arch.Layer[li].KVShareFrom
	}
	sets := make([]layerWeightSet, NL)
	for li, L := range layers {
		L.T = T
		if L.PLIDim > 0 {
			rows := make([]float32, T*PLID)
			for tok := range T {
				copy(rows[tok*PLID:(tok+1)*PLID], pli[tok*NL*PLID+li*PLID:tok*NL*PLID+(li+1)*PLID])
			}
			L.PLEInput = rows
		}
		sets[li] = layerWeightSet{wQ: L.WQ, wK: L.WK, wV: L.WV, wO: L.WO, wGate: L.WGate, wUp: L.WUp, wDown: L.WDown}
	}

	_, tapes, err := realSharedChainForward(embeds, layers, shareFrom, sets)
	if err != nil {
		t.Fatalf("realSharedChainForward: %v", err)
	}

	worst, worstL := 2.0, -1
	worstByClass := map[string]float64{}
	for li := range tapes {
		class := "sliding"
		if arch.Layer[li].Attention == model.GlobalAttention {
			class = "global"
		}
		if shareFrom[li] != li {
			class += "-consumer"
		}
		cos := cosineF32(tapes[li].out, want[li*T*H:(li+1)*T*H])
		attnCos := cosineF32(tapes[li].h1, wantAttn[li*T*H:(li+1)*T*H])
		t.Logf("layer %2d %-16s cosine=%.6f attn-half=%.6f", li, class, cos, attnCos)
		if prev, ok := worstByClass[class]; !ok || cos < prev {
			worstByClass[class] = cos
		}
		if cos < worst {
			worst, worstL = cos, li
		}
	}
	for class, cos := range worstByClass {
		t.Logf("worst %-16s cosine=%.6f", class, cos)
	}
	if worst < 0.999 {
		t.Fatalf("layer %d: host mirror diverges from the gemma4 reference (cosine=%.6f < 0.999)", worstL, worst)
	}
}

// readF32File reads a little-endian f32 dump of exactly n values.
func readF32File(t *testing.T, path string, n int) []float32 {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if len(raw) != n*4 {
		t.Fatalf("%s: %d bytes, want %d (n=%d f32)", path, len(raw), n*4, n)
	}
	out := make([]float32, n)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}

// cosineF32 is the f32 twin of cosineBF16 — cosine similarity with f64 accumulation.
func cosineF32(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return -2
	}
	var dot, na, nb float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		dot += x * y
		na += x * x
		nb += y * y
	}
	if na == 0 || nb == 0 {
		return -2
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
