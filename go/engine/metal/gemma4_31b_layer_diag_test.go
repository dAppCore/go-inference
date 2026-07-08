// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"os"
	"testing"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// TestRealModelLayerHiddenDump is the env-gated real-model half of the cross-engine
// per-layer divergence hunt (#348): GEMMA4_SNAP names a snapshot, GEMMA4_IDS a
// comma-separated token-id list (the OTHER engine's exact tokenisation). The prompt
// prefills token-by-token through stepToken; the LAST token's per-layer hidden L2/mean/
// absmax print in the same format as the mlx-side dump, so `diff` finds the first layer
// where the engines part company.
func TestRealModelLayerHiddenDump(t *testing.T) {
	snap := os.Getenv("GEMMA4_SNAP")
	idsCSV := os.Getenv("GEMMA4_IDS")
	if snap == "" || idsCSV == "" {
		t.Skip("GEMMA4_SNAP / GEMMA4_IDS not set")
	}
	var ids []int32
	for _, p := range core.Split(idsCSV, ",") {
		r := core.Atoi(core.Trim(p))
		if !r.OK {
			t.Fatalf("bad id %q", p)
		}
		ids = append(ids, int32(r.Value.(int)))
	}
	nm, err := LoadTokenModelDir(snap, 4096)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	ns, err := nm.(model.SessionModel).OpenSession()
	if err != nil {
		t.Fatalf("session: %v", err)
	}
	s := ns.(*ArchSession)
	defer s.Close()
	// batched-lane bisection levers (#348): kill one stage of the fold at a time.
	if os.Getenv("GEMMA4_NO_BATCHED_ROPE") != "" {
		batchedRopeDisabledForTest = true
		defer func() { batchedRopeDisabledForTest = false }()
	}
	if os.Getenv("GEMMA4_NO_FOLD") != "" {
		batchedMLPFoldDisabledForTest = true
		defer func() { batchedMLPFoldDisabledForTest = false }()
	}

	// prefill all but the last id, then step the last with capture on.
	// GEMMA4_STEP_PREFILL=1 routes the prefill token-by-token through the per-op-verified
	// stepToken path instead of the batched prefill lane — the #348 discriminator: if the
	// step-prefilled cache fixes the last-token divergence, the batched lane corrupted K/V.
	if len(ids) > 1 {
		if os.Getenv("GEMMA4_STEP_PREFILL") != "" {
			for i, id := range ids[:len(ids)-1] {
				e, eerr := s.embed(id)
				if eerr != nil {
					t.Fatalf("embed(%d): %v", id, eerr)
				}
				if s.perLayerInput != nil {
					pli, perr := s.perLayerInput(id, e)
					if perr != nil {
						t.Fatalf("perLayerInput(%d): %v", id, perr)
					}
					s.state.perLayerInput = pli
				}
				if err := s.state.stepTokenNoResult(e, i); err != nil {
					t.Fatalf("stepToken prefill @%d: %v", i, err)
				}
			}
			s.pos = len(ids) - 1
		} else if err := s.PrefillTokens(ids[:len(ids)-1]); err != nil {
			t.Fatalf("prefill: %v", err)
		}
	}
	t.Logf("SESSION icb=%v pagedKV=%d specs=%d", s.state.icb != nil, len(s.state.pagedKV), len(s.state.specs))
	// immediate post-prefill probe: row 0 of L0's K in every medium, BEFORE any step/head
	// touches the session — separates "never landed" from "wiped by a later stage".
	if os.Getenv("GEMMA4_CACHE_VS_MLX") != "" {
		l2Of := func(b []byte, n int) float64 {
			var sq float64
			for i := 0; i < n*2; i += 2 {
				v := float64(bf16ToF32(b[i], b[i+1]))
				sq += v * v
			}
			return math.Sqrt(sq)
		}
		kvDim0 := kvHeadsOf(s.state.specs[0], s.state.nKVHeads) * headDimOf(s.state.specs[0], s.state.headDim)
		if s.state.lb[0].kCache != nil {
			t.Logf("POST-PREFILL L00 linear K row0 l2=%.3f", l2Of(s.state.bufferBytes(s.state.lb[0].kCache, kvDim0*bf16Size), kvDim0))
		}
		if pg := s.state.layerPagedKV(0); pg != nil && len(pg.kPagePtrs) > 0 {
			t.Logf("POST-PREFILL L00 paged K page0 l2=%.3f lens=%v", l2Of(unsafe.Slice(pg.kPagePtrs[0], kvDim0*bf16Size), kvDim0), pg.pageLens[:min(3, len(pg.pageLens))])
		}
		if s.state.icb != nil && len(s.state.icb.kCaches) > 0 && s.state.icb.kCaches[0] != nil {
			t.Logf("POST-PREFILL L00 icb K row0 l2=%.3f id=%d", l2Of(s.state.bufferBytes(s.state.icb.kCaches[0], kvDim0*bf16Size), kvDim0), s.state.icb.kCaches[0].GetID())
		}
	}
	last := ids[len(ids)-1]
	emb, eerr := s.embed(last)
	if eerr != nil {
		t.Fatalf("embed(last): %v", eerr)
	}
	{
		var sq float64
		for i := 0; i+1 < len(emb); i += 2 {
			v := float64(bf16ToF32(emb[i], emb[i+1]))
			sq += v * v
		}
		t.Logf("EMBED last id=%d l2=%.4f", last, math.Sqrt(sq))
	}
	if s.perLayerInput != nil {
		pli, perr := s.perLayerInput(last, emb)
		if perr != nil {
			t.Fatalf("perLayerInput: %v", perr)
		}
		s.state.perLayerInput = pli
	}
	capturedLayerHiddens = nil
	capturedAttnHiddens = nil
	captureLayerHiddens = true
	hFinal, serr := s.state.stepToken(emb, s.pos)
	captureLayerHiddens = false
	if serr != nil {
		t.Fatalf("stepToken: %v", serr)
	}
	// HEAD-LANE A/B (#348): the same final hidden through BOTH head doors — the GPU
	// direct-argmax head vs the logits lane + host argmax. Disagreement convicts the
	// direct head; agreement on a garbage token sends the hunt back upstream.
	if len(hFinal) > 0 {
		if next, ok, derr := s.directGreedyFromHiddenInPool(hFinal, nil); derr != nil {
			t.Logf("HEAD direct-greedy err: %v", derr)
		} else if !ok {
			t.Logf("HEAD direct-greedy: unavailable")
		} else {
			t.Logf("HEAD direct-greedy -> id=%d", next)
		}
		logits, herr := s.head(hFinal, true)
		if herr != nil {
			t.Fatalf("head logits: %v", herr)
		}
		next2, gerr := greedyBF16Suppressed(logits, s.arch.Vocab, nil)
		if gerr != nil {
			t.Fatalf("greedy argmax: %v", gerr)
		}
		type tv struct {
			id int
			v  float32
		}
		top := make([]tv, 0, 5)
		for id := range s.arch.Vocab {
			v := bf16ToF32(logits[id*2], logits[id*2+1])
			if len(top) < 5 || v > top[len(top)-1].v {
				top = append(top, tv{id, v})
				for i := len(top) - 1; i > 0 && top[i].v > top[i-1].v; i-- {
					top[i], top[i-1] = top[i-1], top[i]
				}
				if len(top) > 5 {
					top = top[:5]
				}
			}
		}
		t.Logf("HEAD logits-lane -> id=%d top5=%v", next2, top)
	}
	if opsDir := os.Getenv("GEMMA4_OPS"); opsDir != "" {
		for _, li := range []int{0, 5} {
			if li >= len(capturedAttnHiddens) {
				continue
			}
			r := core.ReadFile(core.Sprintf("%s/L%02d.resid_attn.bin", opsDir, li))
			if !r.OK {
				continue
			}
			mb := r.Value.([]byte)
			h := capturedAttnHiddens[li]
			var dot, no, nm float64
			for i := 0; i < len(h)/2 && i*4+3 < len(mb); i++ {
				ov := float64(bf16ToF32(h[i*2], h[i*2+1]))
				bits := uint32(mb[i*4]) | uint32(mb[i*4+1])<<8 | uint32(mb[i*4+2])<<16 | uint32(mb[i*4+3])<<24
				mv := float64(math.Float32frombits(bits))
				dot += ov * mv
				no += ov * ov
				nm += mv * mv
			}
			t.Logf("ATTN-HIDDEN L%02d cos=%.6f l2(ours)=%.2f l2(mlx)=%.2f", li, dot/(math.Sqrt(no)*math.Sqrt(nm)+1e-30), math.Sqrt(no), math.Sqrt(nm))
		}
	}
	// CACHE-vs-MLX audit (#348): after prefill + the final step, compare the engine's
	// K/V cache rows [0, T-1) — the prefilled rows — against mlx's cache dumps, row by
	// row, half by half. The first bad (layer, row, K|V) is the batched lane's defect
	// address. Handles both the paged and plain cache forms.
	if os.Getenv("GEMMA4_CACHE_VS_MLX") != "" && os.Getenv("GEMMA4_OPS") != "" {
		opsDir := os.Getenv("GEMMA4_OPS")
		T := len(ids)
		readMLXRow := func(mb []byte, lkv, lhd, r int) []float32 {
			out := make([]float32, lkv*lhd)
			for h := range lkv {
				base := (h*T + r) * lhd * 4
				for d := range lhd {
					o := base + d*4
					bits := uint32(mb[o]) | uint32(mb[o+1])<<8 | uint32(mb[o+2])<<16 | uint32(mb[o+3])<<24
					out[h*lhd+d] = math.Float32frombits(bits)
				}
			}
			return out
		}
		for _, li := range []int{0, 5} {
			rK := core.ReadFile(core.Sprintf("%s/L%02d.k_cache_full.bin", opsDir, li))
			rV := core.ReadFile(core.Sprintf("%s/L%02d.v_cache_full.bin", opsDir, li))
			if !rK.OK || !rV.OK {
				continue
			}
			mk, mv := rK.Value.([]byte), rV.Value.([]byte)
			spec := s.state.specs[li]
			lkv, lhd := kvHeadsOf(spec, s.state.nKVHeads), headDimOf(spec, s.state.headDim)
			kvDim := lkv * lhd
			paged := s.state.layerPagedKV(li)
			if os.Getenv("GEMMA4_CACHE_LINEAR") != "" {
				paged = nil // read the LINEAR lb cache even on a paged session — the landing-vs-sync split
			}
			icbCache := os.Getenv("GEMMA4_CACHE_ICB") != "" && s.state.icb != nil
			if icbCache {
				paged = nil // read the ICB replay's caches — the medium the live decode loop attends
			}
			readEngRow := func(isK bool, r int) []float32 {
				out := make([]float32, kvDim)
				if icbCache {
					buf := s.state.icb.kCaches[li]
					if !isK {
						buf = s.state.icb.vCaches[li]
					}
					bb := s.state.bufferBytes(buf, (r+1)*kvDim*bf16Size)
					for i := range kvDim {
						o := (r*kvDim + i) * 2
						out[i] = bf16ToF32(bb[o], bb[o+1])
					}
					return out
				}
				if paged !=nil {
					p, slot := r/paged.pageSize, r%paged.pageSize
					ptrs, hs, ss := paged.kPagePtrs, paged.kHeadStrides, paged.kSeqStrides
					if !isK {
						ptrs, hs, ss = paged.vPagePtrs, paged.vHeadStrides, paged.vSeqStrides
					}
					pb := unsafe.Slice(ptrs[p], (hs[p]*(lkv-1)+ss[p]*(paged.pageSize-1)+lhd)*2)
					for h := range lkv {
						for d := range lhd {
							o := (h*hs[p] + slot*ss[p] + d) * 2
							out[h*lhd+d] = bf16ToF32(pb[o], pb[o+1])
						}
					}
					return out
				}
				buf := s.state.lb[li].kCache
				if !isK {
					buf = s.state.lb[li].vCache
				}
				bb := s.state.bufferBytes(buf, (r+1)*kvDim*bf16Size)
				for i := range kvDim {
					o := (r*kvDim + i) * 2
					out[i] = bf16ToF32(bb[o], bb[o+1])
				}
				return out
			}
			for _, half := range []string{"K", "V"} {
				mb := mk
				if half == "V" {
					mb = mv
				}
				bad, worst, worstRow, logged := 0, 2.0, -1, 0
				for r := 0; r < T-1; r++ {
					er := readEngRow(half == "K", r)
					mr := readMLXRow(mb, lkv, lhd, r)
					var dot, ne, nm float64
					for i := range er {
						dot += float64(er[i]) * float64(mr[i])
						ne += float64(er[i]) * float64(er[i])
						nm += float64(mr[i]) * float64(mr[i])
					}
					cos := dot / (math.Sqrt(ne)*math.Sqrt(nm) + 1e-30)
					if cos < 0.999 {
						bad++
						if logged < 3 {
							t.Logf("CACHE L%02d %s row %2d cos=%.4f l2(eng)=%.3f l2(mlx)=%.3f", li, half, r, cos, math.Sqrt(ne), math.Sqrt(nm))
							logged++
						}
					}
					if cos < worst {
						worst, worstRow = cos, r
					}
				}
				t.Logf("CACHE L%02d %s: bad %d/%d worst=%.4f@row%d (paged=%v)", li, half, bad, T-1, worst, worstRow, paged != nil)
			}
		}
	}

	vecDir := os.Getenv("GEMMA4_MLX_VECS")
	for li, h := range capturedLayerHiddens {
		var sum, sq, amax float64
		n := len(h) / 2
		for i := 0; i < len(h); i += 2 {
			bits := uint16(h[i]) | uint16(h[i+1])<<8
			v := float64(math.Float32frombits(uint32(bits) << 16))
			sum += v
			sq += v * v
			if a := math.Abs(v); a > amax {
				amax = a
			}
		}
		cosStr := ""
		if vecDir != "" {
			cosVs := func(lj int) float64 {
				r := core.ReadFile(core.Sprintf("%s/layer%02d.bin", vecDir, lj))
				if !r.OK {
					return -2
				}
				mb := r.Value.([]byte)
				var dot, no, nm float64
				for i := 0; i < len(h)/2 && i*4+3 < len(mb); i++ {
					ov := float64(bf16ToF32(h[i*2], h[i*2+1]))
					bits := uint32(mb[i*4]) | uint32(mb[i*4+1])<<8 | uint32(mb[i*4+2])<<16 | uint32(mb[i*4+3])<<24
					mv := float64(math.Float32frombits(bits))
					dot += ov * mv
					no += ov * ov
					nm += mv * mv
				}
				return dot / (math.Sqrt(no)*math.Sqrt(nm) + 1e-30)
			}
			cosStr = core.Sprintf(" cos[li-1]=%.4f cos[li]=%.4f cos[li+1]=%.4f", cosVs(li-1), cosVs(li), cosVs(li+1))
		}
		t.Logf("L%02d l2=%.4f mean=%+.6f absmax=%.4f%s", li, math.Sqrt(sq), sum/float64(n), amax, cosStr)
	}
}
