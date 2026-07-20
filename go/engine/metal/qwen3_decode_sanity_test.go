// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/decode/tokenizer"
)

// qwen3_decode_sanity_test.go is #67's ordinary-decode-sanity receipt: a direct
// ArchSession.Generate call on the real Qwen/Qwen3-4B target (the same checkpoint
// capture_hidden_qwen3_oracle_test.go's oracle compares against), independent of DFlash and the
// capture machinery entirely — the lem-CLI-style probe the brief asked for, gated on the same
// LTHN_DFLASH_ZLAB_TARGET env var the rest of this lane's real-checkpoint tests use.
//
//	MLX_METALLIB_PATH=... LTHN_DFLASH_ZLAB_TARGET=<Qwen/Qwen3-4B snapshot> \
//	  go test -tags metal_runtime -run Qwen3DecodeSanity -v ./engine/metal/

// TestQwen3DecodeSanity_ParisContinuation feeds "The capital of France is" through the
// tokenizer and the production ArchSession.Generate path and requires the continuation to name
// Paris — the same coherence bar TestRealCheckpointGPU_Bonsai1BitRepack_Good uses for the 1-bit
// repack lane. Soft-gated (t.Errorf, not t.Fatalf) on the coherence check only: #67's own
// diagnosis (capture_hidden_qwen3_oracle_test.go's header and
// TestForwardCaptureHiddensQwen3AllLayersVsRealOracle) found a real, deterministic engine fault
// at the FINAL decoder layer (35 of 36) — outside this lane's fence (model/arch/Qwen/qwen3/
// only) — that corrupts the exact hidden state this test's argmax reads. This test is the
// receipt for THAT fix landing, not a claim this lane makes one; a KNOWN-gap soft-fail here
// documents the gap precisely rather than landmining anyone else who runs this suite.
func TestQwen3DecodeSanity_ParisContinuation(t *testing.T) {
	requireNativeRuntime(t)
	targetDir := core.Getenv("LTHN_DFLASH_ZLAB_TARGET")
	if core.Trim(targetDir) == "" {
		t.Skip("set LTHN_DFLASH_ZLAB_TARGET to a local Qwen/Qwen3-4B snapshot (see capture_hidden_qwen3_oracle_test.go's doc comment)")
	}

	tok, err := tokenizer.LoadTokenizer(core.PathJoin(targetDir, "tokenizer.json"))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	ids := tok.Encode("The capital of France is")
	t.Logf("prompt ids: %v", ids)

	sess, err := LoadDir(targetDir, 0)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	defer func() { _ = sess.Close() }()

	const maxNew = 12
	gen, err := sess.Generate(ids, maxNew, -1) // eosID<0: no early stop, want the full continuation
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	text := tok.Decode(gen)
	t.Logf("continuation: %q (ids %v)", text, gen)

	if !strings.Contains(text, "Paris") {
		t.Errorf("continuation %q does not name Paris — ordinary greedy decode on this checkpoint is still corrupted (KNOWN, root-caused OUTSIDE this lane's fence — see capture_hidden_qwen3_oracle_test.go's #67 header)", text)
	}
}
