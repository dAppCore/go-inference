// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/decode/tokenizer"
	"dappco.re/go/inference/internal/enginegate"
	"dappco.re/go/inference/kv/turboquant"
	"dappco.re/go/inference/model"
)

// turboquant_real_kv_capture_test.go is RFC #41 slice S3's live capture
// harness: it loads the actual gemma-4 e2b-4bit checkpoint, decodes a real
// prompt past 512 appended rows, taps every distinct global-attention
// cache-OWNING layer's resident K/V via (*ArchSession).DumpKVRows
// (turboquant_capture_tap.go), measures each layer's real distortion via
// turboquant.MeasureReal, and writes the small committed CI fixture consumed
// by go/kv/turboquant's TestMeasureReal_Ugly. It is gated on LEM_TQ_CAPTURE=1
// — a manual capture run, not part of the routine test:metal lane, since it
// writes repo files as a side effect (mirroring the bench:hip:gemma4-sweep
// receipt test's env-gated write). The per-layer distortion tables land in
// -v test output for the orchestrator to transcribe into the dated receipt —
// composing that receipt's "does K/V match the Gaussian oracle" reading
// needs the actual numbers in hand, not a template this test would have to
// guess at before running. Paths below are relative to this package's
// directory (go/engine/metal), the Taskfile test:metal working directory.
const (
	turboquantCaptureModelRepo  = "mlx-community/gemma-4-e2b-it-4bit"
	turboquantCaptureGenerate   = 550  // tokens generated past the prompt — clears the >=512-row target regardless of prompt length
	turboquantCaptureMaxLen     = 1536 // cache rows to allocate; comfortably covers prompt + turboquantCaptureGenerate
	turboquantCaptureSeed       = 42
	turboquantCaptureCIFixtureN = 32 // rows/side committed to testdata (kept small and within budget; the -v log carries the FULL captured pool's numbers)
	turboquantCaptureKeysPath   = "../../kv/turboquant/testdata/real_kv_keys.bin"
	turboquantCaptureValuesPath = "../../kv/turboquant/testdata/real_kv_values.bin"
)

// turboquantCapturePromptText is a real, on-topic English prompt (not a
// synthetic token-id fixture) — real cache rows follow the model's actual
// language distribution, which is the whole point of this instrument.
const turboquantCapturePromptText = `Explain how a transformer decoder's attention cache works during a long conversation. Every time the model generates a new token, it stores a key vector and a value vector for that token in every attention layer, so future tokens can attend back over the whole conversation history without recomputing earlier work. As a conversation grows across many turns, this cache grows too, and holding every key and value in full precision costs a great deal of memory. Compressing those cached vectors to fewer bits per number can save substantial memory, but only if the compression does not distort the attention scores enough to change which earlier tokens the model chooses to attend to, or how strongly it attends to them, at each new decoding step.`

// TestTurboQuantRealKVCaptureReceipt is the live capture: see the file doc
// comment. Skips unless LEM_TQ_CAPTURE=1.
func TestTurboQuantRealKVCaptureReceipt(t *testing.T) {
	requireNativeRuntime(t)
	if core.Getenv("LEM_TQ_CAPTURE") == "" {
		t.Skip("set LEM_TQ_CAPTURE=1 to run the real-KV distortion capture (loads a model, writes the CI testdata fixture)")
	}
	dir := enginegate.HFModelPath(t, turboquantCaptureModelRepo)

	tok, err := tokenizer.LoadTokenizer(core.PathJoin(dir, "tokenizer.json"))
	if err != nil {
		t.Fatalf("LoadTokenizer(%s): %v", dir, err)
	}
	prompt := tok.Encode(turboquantCapturePromptText)
	if len(prompt) == 0 {
		t.Fatal("capture prompt tokenised to no ids")
	}

	sess, err := LoadDir(dir, turboquantCaptureMaxLen)
	if err != nil {
		t.Fatalf("LoadDir(%s): %v", dir, err)
	}
	defer func() { _ = sess.Close() }()

	if _, err := sess.Generate(prompt, turboquantCaptureGenerate, -1); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if sess.Pos() < 512 {
		t.Fatalf("session position = %d after the capture prompt, want >= 512", sess.Pos())
	}

	layers := turboquantCaptureGlobalOwningLayers(sess.arch)
	if len(layers) == 0 {
		t.Fatalf("found no global-attention owning-cache layers (arch has %d layers) — nothing to capture", len(sess.arch.Layer))
	}
	t.Logf("session position %d; %d global-attention owning-cache layer(s): %v (of %d total layers)",
		sess.Pos(), len(layers), layers, len(sess.arch.Layer))

	var ciKeys, ciValues [][]float32
	for i, layer := range layers {
		keys, values, err := sess.DumpKVRows(layer)
		if err != nil {
			t.Fatalf("DumpKVRows(%d): %v", layer, err)
		}
		measured, err := turboquant.MeasureReal(keys, values, turboquantCaptureSeed)
		if err != nil {
			t.Fatalf("MeasureReal(layer %d): %v", layer, err)
		}
		t.Logf("layer %d (%d rows, d=%d):\n%s", layer, len(keys), measured.D, turboquant.FormatRealReport(measured))

		if i == 0 {
			ciKeys = turboquantCaptureSubsample(keys, turboquantCaptureCIFixtureN)
			ciValues = turboquantCaptureSubsample(values, turboquantCaptureCIFixtureN)
		}
	}

	if err := turboquant.SaveRealKVRows(turboquantCaptureKeysPath, ciKeys); err != nil {
		t.Fatalf("SaveRealKVRows(keys): %v", err)
	}
	if err := turboquant.SaveRealKVRows(turboquantCaptureValuesPath, ciValues); err != nil {
		t.Fatalf("SaveRealKVRows(values): %v", err)
	}
	t.Logf("wrote CI fixtures: %s, %s (%d rows/side)", turboquantCaptureKeysPath, turboquantCaptureValuesPath, len(ciKeys))
}

// turboquantCaptureGlobalOwningLayers returns every layer index that is
// BOTH global-attention AND owns its cache — gemma4-family archs route most
// full_attention-typed layers onto an EARLIER owner's cache via
// num_kv_shared_layers (model.DeriveLayers), so the owning subset can be far
// smaller than the nominal full_attention layer count; querying a
// non-owning index would just read the SAME bytes as its owner under a
// different label, not a distinct sample.
func turboquantCaptureGlobalOwningLayers(arch model.Arch) []int {
	var layers []int
	for i, spec := range arch.Layer {
		if spec.Attention == model.GlobalAttention && spec.OwnsCache() {
			layers = append(layers, i)
		}
	}
	return layers
}

// turboquantCaptureSubsample picks up to n rows from rows at a uniform
// stride, deterministic given the same input — the CI fixture stays small
// and reproducible across capture runs rather than a random draw.
func turboquantCaptureSubsample(rows [][]float32, n int) [][]float32 {
	if n <= 0 || len(rows) == 0 {
		return nil
	}
	if len(rows) <= n {
		return rows
	}
	out := make([][]float32, 0, n)
	stride := len(rows) / n
	for i := 0; i < n; i++ {
		out = append(out, rows[i*stride])
	}
	return out
}
