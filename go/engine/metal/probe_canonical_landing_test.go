// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"os"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/decode/tokenizer"
)

// TestProbeCanonicalLanding (LTHN_PROBE_MODEL-gated) is the #1846 KV-level
// receipt for the reuse fix: it lands the SAME rows two ways — a fresh whole
// prefill, and the reuse flow (warm prefill → rollback to the shared prefix →
// append the suffix) — and asserts the retained caches are byte-identical
// row-for-row when reuseCanonicalLanding is armed, then shows the batched
// landing DIFFERS without it. The whole-prefill vs reuse cache being byte-
// identical is the mechanism behind TestProbePromptReuseParity's token parity:
// the batched forward is intra-batch tile-position sensitive (a row lands a
// few bf16 ulps apart depending on its offset within the projection/rope/norm
// batch, which the q8 store amplifies into token flips), while the per-token
// canonical lane lands every row at offset 0.
//
//	LTHN_PROBE_MODEL=<snapshot dir> [LTHN_PROBE_CONTEXT=2048] \
//	  go test -tags metal_runtime ./engine/metal/ -run TestProbeCanonicalLanding -v
func TestProbeCanonicalLanding(t *testing.T) {
	requireNativeRuntime(t)
	dir := os.Getenv("LTHN_PROBE_MODEL")
	if dir == "" {
		t.Skip("LTHN_PROBE_MODEL not set")
	}
	maxLen := 2048
	tm, err := LoadTokenModelDir(dir, maxLen)
	if err != nil {
		t.Fatalf("LoadTokenModelDir: %v", err)
	}
	ntm := tm.(*NativeTokenModel)
	defer ntm.Close()

	tok, err := tokenizer.LoadTokenizer(core.PathJoin(dir, "tokenizer.json"))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	warmIDs := tok.Encode("Write the integers from 90001 to 90801 separated by single spaces. Output only the numbers, nothing else.")
	aIDs := tok.Encode("Write the integers from 1 to 800 separated by single spaces. Output only the numbers, nothing else.")
	lcp := 0
	for lcp < len(warmIDs) && lcp < len(aIDs) && warmIDs[lcp] == aIDs[lcp] {
		lcp++
	}
	if lcp == 0 {
		t.Fatalf("prompts share no token prefix")
	}
	t.Logf("warm=%d A=%d shared prefix=%d", len(warmIDs), len(aIDs), lcp)

	openArch := func(canonical bool) *ArchSession {
		stepper, oerr := ntm.OpenSession()
		if oerr != nil {
			t.Fatalf("OpenSession: %v", oerr)
		}
		s := stepper.(*ArchSession)
		s.reuseCanonicalLanding = canonical
		return s
	}
	freshViews := func(canonical bool) []sessionStateLayerView {
		s := openArch(canonical)
		t.Cleanup(func() { s.Close() })
		if err := s.PrefillTokens(aIDs); err != nil {
			t.Fatalf("fresh PrefillTokens: %v", err)
		}
		v, verr := s.stateLayerViews()
		if verr != nil {
			t.Fatalf("fresh stateLayerViews: %v", verr)
		}
		return v
	}
	reuseViews := func(canonical bool) []sessionStateLayerView {
		s := openArch(canonical)
		t.Cleanup(func() { s.Close() })
		if err := s.PrefillTokens(warmIDs); err != nil {
			t.Fatalf("reuse PrefillTokens(warm): %v", err)
		}
		if !s.TruncateTo(lcp) {
			t.Fatalf("reuse TruncateTo(%d) failed", lcp)
		}
		if err := s.truncateSpeculativeKV(lcp); err != nil {
			t.Fatalf("reuse truncateSpeculativeKV: %v", err)
		}
		if err := s.AppendTokens(aIDs[lcp:]); err != nil {
			t.Fatalf("reuse AppendTokens: %v", err)
		}
		v, verr := s.stateLayerViews()
		if verr != nil {
			t.Fatalf("reuse stateLayerViews: %v", verr)
		}
		return v
	}

	rows := len(aIDs)
	compare := func(label string, hard bool, fv, sv []sessionStateLayerView) {
		if len(fv) != len(sv) {
			t.Fatalf("%s: view counts differ fresh=%d reuse=%d", label, len(fv), len(sv))
		}
		kMax, vMax := float32(0), float32(0)
		firstL, firstRow, firstKind := -1, -1, ""
		scan := func(fb, sb []byte, rowBytes int, kind string, layer int, curMax *float32) {
			for r := 0; r < rows; r++ {
				off := r * rowBytes
				if off+rowBytes > len(fb) || off+rowBytes > len(sb) {
					break
				}
				fr := bf16BytesToF32(fb[off : off+rowBytes])
				sr := bf16BytesToF32(sb[off : off+rowBytes])
				for i := range fr {
					d := fr[i] - sr[i]
					if d < 0 {
						d = -d
					}
					if d > *curMax {
						*curMax = d
					}
					if d > 0 && firstL < 0 {
						firstL, firstRow, firstKind = layer, r, kind
					}
				}
			}
		}
		for vi := range fv {
			scan(fv[vi].keyBytes, sv[vi].keyBytes, fv[vi].rowBytes, "K", fv[vi].layer, &kMax)
			scan(fv[vi].valueBytes, sv[vi].valueBytes, fv[vi].rowBytes, "V", fv[vi].layer, &vMax)
		}
		if firstL < 0 {
			t.Logf("%s: reuse cache BYTE-IDENTICAL to fresh over rows 0..%d (all owner layers K+V)", label, rows-1)
			return
		}
		msg := core.Sprintf("%s: reuse cache DIFFERS from fresh first L%02d %s row=%d (prefix boundary=%d) | K max|Δ|=%g V max|Δ|=%g",
			label, firstL, firstKind, firstRow, lcp, kMax, vMax)
		if hard {
			t.Error(msg)
		} else {
			t.Log(msg)
		}
	}

	// Canonical landing: reuse must be byte-identical to a fresh whole prefill.
	compare("canonical", true, freshViews(true), reuseViews(true))
	// Batched landing (the #1845 wobble the decline guards): differs — logged,
	// not fatal (it is the documented tile-position numeric tier).
	compare("batched", false, freshViews(false), reuseViews(false))
}
