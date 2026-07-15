// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"

	"dappco.re/go/inference/model"
)

// TestQ4ICBvsReencodePerLayer localises the q4 ICB-replay decode bug at REAL scale: it loads the
// e2b-4bit checkpoint and runs ONE token at pos 0 through BOTH the ICB replay (stepBodyCapture) and
// the proven host re-encode (stepToken + captureLayerHiddens), then diffs their per-layer hiddens.
// Both paths compute identical math over fresh caches at pos 0, so a structurally-correct ICB tracks
// the host at cosine 1.0; the first layer whose cosine drops is where the quant ICB recording
// diverges — and its TYPE (owner / sharer / sliding / global) names the culprit. Set E2B_Q4_DIR.
func TestQ4ICBvsReencodePerLayer(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	dir := os.Getenv("E2B_Q4_DIR")
	if dir == "" {
		t.Skip("set E2B_Q4_DIR to the e2b-4bit snapshot dir")
	}
	const maxLen = 64
	nm, err := LoadTokenModelDir(dir, maxLen)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	ns, err := nm.(model.SessionModel).OpenSession()
	if err != nil {
		t.Fatalf("session: %v", err)
	}
	s := ns.(*ArchSession)
	if s.state.icb == nil {
		t.Fatal("expected an ICB-eligible session (icb recorded) — the localiser needs the ICB path")
	}

	const id = int32(2331)
	emb, err := s.embed(id)
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	var pli []byte
	if s.perLayerInput != nil {
		if pli, err = s.perLayerInput(id, emb); err != nil {
			t.Fatalf("perLayerInput: %v", err)
		}
		s.state.perLayerInput = pli
	}

	// host re-encode per-layer (the correct reference)
	capturedLayerHiddens = nil
	captureLayerHiddens = true
	_, serr := s.state.stepToken(emb, 0)
	captureLayerHiddens = false
	if serr != nil {
		t.Fatalf("stepToken: %v", serr)
	}
	reLayers := capturedLayerHiddens

	// ICB replay per-layer (the suspect) — fresh ICB caches, same token, same pos
	_, icbLayers := s.state.icb.stepBodyCapture(emb, 0, pli)

	n := min(len(icbLayers), len(reLayers))
	if n == 0 {
		t.Fatalf("no per-layer captures: reencode=%d icb=%d", len(reLayers), len(icbLayers))
	}
	worst, worstL := 2.0, -1
	for L := range n {
		c := cosineBF16(reLayers[L], icbLayers[L])
		owns := s.state.specs[L].OwnsCache()
		at := "sliding"
		if s.state.specs[L].Attention == model.GlobalAttention {
			at = "GLOBAL "
		}
		t.Logf("L%2d  cos=%.5f  %s owns=%v shareFrom=%d", L, c, at, owns, s.state.specs[L].KVShareFrom)
		if c < worst {
			worst, worstL = c, L
		}
	}
	t.Logf("=> FIRST/WORST divergence: L%d cos=%.5f  owns=%v shareFrom=%d", worstL, worst, s.state.specs[worstL].OwnsCache(), s.state.specs[worstL].KVShareFrom)
}
