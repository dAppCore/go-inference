// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"bytes"
	"context"
	"image"
	"image/color"
	"image/png"
	"path/filepath"
	"strings"
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/inference/decode/tokenizer"
)

// vision_chat_smoke_test.go is THE receipt for the engine vision bridge (#274): a
// real gemma-4 E2B checkpoint ANSWERING about an image end-to-end through the
// engine-neutral engine.TextModel.Chat — load → a generated solid-colour PNG on a
// user turn → ProjectImage (preprocess + SigLIP tower) → splice the soft tokens
// over the placeholder run → PrefillTokenEmbeddings → decode. Green compile only
// proves it links; this proves the model reads the pixels. metal_runtime-gated
// (needs MLX_METALLIB_PATH + the cached bf16 E2B checkpoint); skips cleanly when
// either is absent.

// solidColourPNG builds a w×h PNG filled with c — a synthetic image whose subject
// is unambiguous, so the assertion tests the SERVE-through-vision path (does the
// model see the image at all) not the model's fine-grained recognition.
func solidColourPNG(t *testing.T, c color.RGBA, w, h int) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			img.SetRGBA(x, y, c)
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encode png: %v", err)
	}
	return buf.Bytes()
}

func TestEngineVisionChatAnswersImageE2B(t *testing.T) {
	requireNativeRuntime(t)
	dir := gemma4E2BBf16Dir(t)
	const maxLen = 2048

	tm, err := LoadTokenModelDir(dir, maxLen)
	if err != nil {
		t.Fatalf("LoadTokenModelDir: %v", err)
	}
	if c, ok := tm.(interface{ Close() error }); ok {
		defer func() { _ = c.Close() }()
	}
	nm, ok := tm.(*NativeTokenModel)
	if !ok {
		t.Fatalf("loaded model is %T, want *NativeTokenModel", tm)
	}
	// E2B-it ships the SigLIP tower — if the bridge (or the loader) failed to wire
	// it, this is the bug the whole task is about, so fail rather than skip.
	if !nm.AcceptsImageInput() {
		t.Fatal("gemma-4-E2B-it-bf16 reports no vision tower — the vision bridge is not wired")
	}

	// LoadTokenModelDir works in token-id space; the serve boundary attaches the
	// text tokenizer (text↔ids), exactly as serving.NewMLXBackend does.
	tok, err := tokenizer.LoadTokenizer(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	nm.AttachTokenizer(tok)

	model := newNativeTextModel(nm, "gemma4")

	// A solid red image; ask for the dominant colour in one word. Red is the least
	// ambiguous thing a VLM reports, so a correct answer means the pixels reached
	// the model (the serve path), not that it is a strong recogniser.
	red := solidColourPNG(t, color.RGBA{R: 220, G: 20, B: 20, A: 255}, 256, 256)
	messages := []inference.Message{{
		Role:    "user",
		Content: "What is the main colour of this image? Answer with a single word.",
		Images:  [][]byte{red},
	}}

	// Thinking off explicitly: the smoke wants the one-word answer inside a
	// 64-token budget, and gemma4's family default (thinking ON, #1847) would
	// spend the whole budget in the thought channel.
	off := false
	var b strings.Builder
	for tok := range model.Chat(context.Background(), messages, inference.WithMaxTokens(64), inference.WithEnableThinking(&off)) {
		b.WriteString(tok.Text)
	}
	if r := model.Err(); !r.OK {
		t.Fatalf("vision Chat: %v", r.Value)
	}
	answer := strings.TrimSpace(b.String())
	t.Logf("engine vision bridge answered: %q", answer)
	if answer == "" {
		t.Fatal("vision Chat returned an empty answer — the multimodal prefill path is broken")
	}
	if !strings.Contains(strings.ToLower(answer), "red") {
		t.Errorf("expected the answer to name the colour red, got %q", answer)
	}
}
