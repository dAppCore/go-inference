// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"os"
	"testing"

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

	// prefill all but the last id, then step the last with capture on.
	if len(ids) > 1 {
		if err := s.PrefillTokens(ids[:len(ids)-1]); err != nil {
			t.Fatalf("prefill: %v", err)
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
	_, serr := s.state.stepToken(emb, s.pos)
	captureLayerHiddens = false
	if serr != nil {
		t.Fatalf("stepToken: %v", serr)
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
