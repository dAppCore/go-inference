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

// TestRealE2BVerifyStackKVDiff is the run-to-run determinism guard: generate
// N tokens twice and diff tokens + recorded-ICB cache bytes row by row. The
// greedy MTP loop composes verify and plain stretches on wall-clock
// re-engagement verdicts, so two runs may take DIFFERENT cycle structures —
// the emitted stream and the cache bytes must be invariant to that, which
// holds only while every greedy lane is byte-identical to sequential plain
// decode (the #55 routing, mtpVerifyFoldArmed). LTHN_KVDIFF_BOTH_LIVE=1
// force-disables the stack lane in both arms (the engine-only shape);
// LTHN_KVDIFF_MAXNEW sizes the run — the default spans the re-engagement
// boundary where a non-parity lane flips a near-tied token.
func TestRealE2BVerifyStackKVDiff(t *testing.T) {
	requireNativeRuntime(t)
	targetDir := enginegate.HFModelPath(t, "mlx-community/gemma-4-e2b-it-4bit")
	assistantDir := enginegate.HFModelPath(t, "mlx-community/gemma-4-E2B-it-assistant-bf16")
	maxNew := 60
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
