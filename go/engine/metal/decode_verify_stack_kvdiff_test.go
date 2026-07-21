// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"bytes"
	"os"
	"testing"
	"strconv"
	"unsafe"

	"dappco.re/go/inference/internal/enginegate"
)

// TestRealE2BVerifyStackKVDiff is the engine-bistability reproducer: generate
// N tokens twice and diff tokens + recorded-ICB cache bytes row by row.
// LTHN_KVDIFF_BOTH_LIVE=1 runs BOTH arms with the stack lane force-disabled —
// the arms still flip a near-tied token (~1 in 2 at 60 tokens, both
// directions, KV bytes clean) which is the proof the nondeterminism lives in
// the live MTP re-engagement path, not the replay. LTHN_KVDIFF_MAXNEW sizes
// the run (40 = stable, 60 = spans the flip).
func TestRealE2BVerifyStackKVDiff(t *testing.T) {
	if os.Getenv("LTHN_VERIFY_STACK_ICB") != "1" {
		t.Skip("set LTHN_VERIFY_STACK_ICB=1 (race-hunt instrument)")
	}
	requireNativeRuntime(t)
	targetDir := enginegate.HFModelPath(t, "mlx-community/gemma-4-e2b-it-4bit")
	assistantDir := enginegate.HFModelPath(t, "mlx-community/gemma-4-E2B-it-assistant-bf16")
	maxNew := 40
	if v := os.Getenv("LTHN_KVDIFF_MAXNEW"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			maxNew = n
		}
	}

	run := func(disable bool) ([]int32, *ArchSession) {
		t.Helper()
		verifyStackICBDisabledForTest = disable
		defer func() { verifyStackICBDisabledForTest = false }()
		target, err := LoadDir(targetDir, 4096)
		if err != nil {
			t.Fatalf("LoadDir: %v", err)
		}
		pair, err := LoadAssistantPairDirs(targetDir, assistantDir)
		if err != nil {
			t.Fatalf("LoadAssistantPairDirs: %v", err)
		}
		t.Cleanup(func() { pair.Close(); _ = target.Close() })
		prompt := realE2BAssistantPrompt(t, targetDir)
		res, err := pair.GenerateFromSession(target, prompt, maxNew, -1, 4, nil)
		if err != nil {
			t.Fatalf("GenerateFromSession: %v", err)
		}
		return res.Tokens, target
	}

	bothLive := os.Getenv("LTHN_KVDIFF_BOTH_LIVE") == "1"
	liveToks, liveSess := run(true)
	laneToks, laneSess := run(!bothLive)
	for i := range liveToks {
		if i < len(laneToks) && liveToks[i] != laneToks[i] {
			t.Logf("token %d differs already: lane %d live %d", i, laneToks[i], liveToks[i])
			break
		}
	}
	lr, rr := liveSess.state.icb, laneSess.state.icb
	if lr == nil || rr == nil {
		t.Fatal("both sessions must be recorded-ICB")
	}
	endPos := min(liveSess.pos, laneSess.pos)
	firstBad := -1
	for li := range liveSess.state.specs {
		if !liveSess.state.specs[li].OwnsCache() {
			continue
		}
		rows := min(endPos, lr.cacheRows[li])
		rb := lr.rowBytes[li]
		for _, pair := range []struct {
			name string
			a, b interface{ Contents() unsafe.Pointer }
		}{{"K", lr.kCaches[li], rr.kCaches[li]}, {"V", lr.vCaches[li], rr.vCaches[li]}} {
			av := unsafe.Slice((*byte)(pair.a.Contents()), rows*rb)
			bv := unsafe.Slice((*byte)(pair.b.Contents()), rows*rb)
			for r := 0; r < rows; r++ {
				if !bytes.Equal(av[r*rb:(r+1)*rb], bv[r*rb:(r+1)*rb]) {
					t.Logf("layer %d %s row %d differs (first byte-divergent row)", li, pair.name, r)
					if firstBad < 0 || r < firstBad {
						firstBad = r
					}
					break
				}
			}
		}
	}
	if firstBad >= 0 {
		t.Fatalf("KV bytes diverge from row %d (tokens agreed to %d)", firstBad, len(liveToks))
	}
}
