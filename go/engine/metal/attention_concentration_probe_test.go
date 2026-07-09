// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"math"
	"os"
	"sort"
	"testing"
	"unsafe"

	"dappco.re/go/inference/model"
)

// TestAttentionMassConcentrationRealE2B is the #365 opportunity probe: deep-context
// tok/s drops because every decode step reads the WHOLE KV cache, and that cost grows
// with context. This measures whether gemma4 attention actually CONCENTRATES at depth
// — how few cells carry the attention mass — by reading the paged-SDPA pass-1 partials
// (per-(head,cell) max+sum, persistent per cache, host-readable with the cb idle, no
// flush). If a GLOBAL layer needs only k of C cells for 99% of its softmax mass, then
// ~(1-k/C) of its KV reads are inert — that fraction is the #365 skip ceiling. Unlike
// the FFN (fixed sparsity), this scales with context, so the deeper the context the
// bigger the win. Sliding layers already cap at the window; the GLOBAL layers are the tax.
//
//	LEM_REAL_E2B=1 MLX_METALLIB_PATH=... go test -run TestAttentionMassConcentrationRealE2B -v ./engine/metal/
func TestAttentionMassConcentrationRealE2B(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if os.Getenv("LEM_REAL_E2B") == "" {
		t.Skip("set LEM_REAL_E2B=1 to run the real e2b-4bit attention-concentration probe (loads ~2.7GB)")
	}
	dir := resolveE2B4bitDir(t)
	lm, dm, err := loadRegistered(dir)
	if err != nil {
		t.Fatalf("loadRegistered: %v", err)
	}
	defer func() { _ = dm.Close() }()
	sb, err := buildShardBuffers(dm)
	if err != nil {
		t.Fatalf("buildShardBuffers: %v", err)
	}
	defer func() { _ = sb.Close() }()
	qm, err := loadedToQuant(lm, lm.Embed.GroupSize, lm.Embed.Bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	const ctxLen = 4096
	sess, err := newArchQuantSessionShards(qm, lm.Arch, ctxLen+64, sb)
	if err != nil {
		t.Fatalf("newArchQuantSessionShards: %v", err)
	}

	// non-ICB decode so buildSDPAPagedDecodePlan runs each step (fresh lastCellCount)
	// and the pass-1 partials are filled per token, not replayed from a captured cb.
	prevICB := icbDisabledForTest
	icbDisabledForTest = true
	defer func() { icbDisabledForTest = prevICB }()

	// a long DIVERSE prompt over KNOWN-VALID low vocab (BOS-prefixed) — repetition biases
	// the pattern, but high random IDs tripped an early cap; cycling 100..3100 stays
	// in-vocab while still diverse enough to surface the sink+recency structure.
	prompt := make([]int32, ctxLen)
	prompt[0] = 2
	for i := 1; i < ctxLen; i++ {
		prompt[i] = int32(100 + (i*131)%3000)
	}
	if err := sess.PrefillTokens(prompt); err != nil {
		t.Fatalf("prefill %d: %v", ctxLen, err)
	}
	// decode a few tokens so the paged-DECODE sdpa runs and fills the partials for the
	// last position attending over the full context.
	if _, err := sess.GenerateFromCache(4, -1); err != nil {
		t.Fatalf("generate: %v", err)
	}

	// conc returns k95/k99 = smallest #cells (of cc) holding 95%/99% of head h's softmax
	// mass. mass_c = sum_c·exp(max_c - gmax) — the online-softmax partial re-normalised.
	conc := func(maxs, sums []float32, h, cc int) (k95, k99 int) {
		gmax := float32(math.Inf(-1))
		for c := 0; c < cc; c++ {
			if m := maxs[h*cc+c]; m > gmax {
				gmax = m
			}
		}
		mass := make([]float64, cc)
		var tot float64
		for c := 0; c < cc; c++ {
			w := float64(sums[h*cc+c]) * math.Exp(float64(maxs[h*cc+c]-gmax))
			mass[c] = w
			tot += w
		}
		if tot <= 0 {
			return cc, cc
		}
		sort.Sort(sort.Reverse(sort.Float64Slice(mass)))
		var acc float64
		for c := 0; c < cc; c++ {
			acc += mass[c]
			if k95 == 0 && acc >= 0.95*tot {
				k95 = c + 1
			}
			if acc >= 0.99*tot {
				k99 = c + 1
				break
			}
		}
		return k95, k99
	}

	t.Logf("=== #365 attention-mass concentration — real e2b-4bit, %d ctx tokens ===", ctxLen)
	seen := map[*devicePagedKVCache]bool{}
	reported := 0
	for li := range sess.state.specs {
		cache := sess.state.layerPagedKV(li)
		if cache == nil || seen[cache] || len(cache.sdpaScratch) == 0 || cache.sdpaScratch[0] == nil {
			continue
		}
		seen[cache] = true
		sc := cache.sdpaScratch[0]
		cc := sc.lastCellCount
		totPos, nPages := 0, 0
		if _, _, lens, _, _, _, _, serr := cache.state(); serr == nil {
			for _, l := range lens {
				if l > 0 {
					totPos += l
					nPages++
				}
			}
		}
		if cc <= 1 {
			continue // single-cell (bounded/sliding at this depth) — no cells to concentrate over
		}
		nH := sc.nHeads
		maxs := unsafe.Slice((*float32)(sc.maxs.Contents()), nH*cc)
		sums := unsafe.Slice((*float32)(sc.sums.Contents()), nH*cc)
		var s95, s99 float64
		for h := 0; h < nH; h++ {
			k95, k99 := conc(maxs, sums, h, cc)
			s95 += float64(k95) / float64(cc)
			s99 += float64(k99) / float64(cc)
		}
		kind := "sliding"
		if sess.state.specs[li].Attention == model.GlobalAttention {
			kind = "GLOBAL"
		}
		t.Logf("  L%-2d %-7s pos=%-4d pages=%-2d cells=%-3d  95%%mass=%3.0f%% · 99%%mass=%3.0f%% of cells  -> ~%2.0f%% inert@99",
			li, kind, totPos, nPages, cc, 100*s95/float64(nH), 100*s99/float64(nH), 100*(1-s99/float64(nH)))
		reported++
	}
	if reported == 0 {
		t.Fatal("no multi-cell caches — context too short or partials not filled (ICB path?)")
	}
	t.Logf("  => high 'inert@99' on GLOBAL layers = the deep-context KV-skip ceiling (scales with context)")
}
