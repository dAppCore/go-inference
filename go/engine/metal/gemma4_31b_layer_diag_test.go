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
	if err := s.PrefillTokens(ids[:len(ids)-1]); err != nil {
		t.Fatalf("prefill: %v", err)
	}
	last := ids[len(ids)-1]
	emb, eerr := s.embed(last)
	if eerr != nil {
		t.Fatalf("embed(last): %v", eerr)
	}
	if s.perLayerInput != nil {
		pli, perr := s.perLayerInput(last, emb)
		if perr != nil {
			t.Fatalf("perLayerInput: %v", perr)
		}
		s.state.perLayerInput = pli
	}
	capturedLayerHiddens = nil
	captureLayerHiddens = true
	_, serr := s.state.stepToken(emb, s.pos)
	captureLayerHiddens = false
	if serr != nil {
		t.Fatalf("stepToken: %v", serr)
	}
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
		t.Logf("L%02d l2=%.4f mean=%+.6f absmax=%.4f", li, math.Sqrt(sq), sum/float64(n), amax)
	}
}
