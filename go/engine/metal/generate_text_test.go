// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"

	core "dappco.re/go"
	g4 "dappco.re/go/inference/model/gemma4"
	"dappco.re/go/inference/decode/tokenizer"
)

// a tiny BPE tokenizer (max id 101 via the specials) — Encode/Decode work, no model load.
const textTestTokenizerJSON = `{
  "model": {
    "type": "BPE",
    "vocab": {"h": 0, "e": 1, "l": 2, "o": 3, "▁": 4, "he": 5, "ll": 6, "▁h": 7},
    "merges": ["h e", "l l"],
    "byte_fallback": false
  },
  "added_tokens": [
    {"id": 100, "content": "<bos>", "special": true},
    {"id": 101, "content": "<eos>", "special": true}
  ]
}`

// TestGenerateText gates the text-in/text-out wrapper: GenerateText encodes the prompt, runs
// the session, and decodes the result — and equals the manual Encode → Generate → Decode
// chain (so the text glue is correct), with no cgo. (Tiny random model → arbitrary text; the
// gate is the glue, not coherence.)
func TestGenerateText(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	// write the tokenizer.json to a temp dir and load it (the shared no-cgo tokenizer).
	dirRes := core.MkdirTemp("", "go-mlx-native-text-*")
	if !dirRes.OK {
		t.Fatalf("MkdirTemp: %v", dirRes.Value)
	}
	dir := dirRes.Value.(string)
	defer core.RemoveAll(dir)
	path := core.PathJoin(dir, "tokenizer.json")
	if r := core.WriteFile(path, []byte(textTestTokenizerJSON), 0o644); !r.OK {
		t.Fatalf("WriteFile: %v", r.Value)
	}
	tok, err := tokenizer.LoadTokenizer(path)
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}

	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 102 // vocab covers the tokenizer's max id (101)
	const maxLen, maxNew = 24, 5
	arch, err := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: 2, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim,
		VocabSize: vocab, RMSNormEps: 1e-6,
	}.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	mk := func(n, salt int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32((i*salt+13)%97-48) * 0.02
		}
		return s
	}
	layers := make([]DecodeLayerWeights, len(arch.Layer))
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
	}
	g := &BF16Model{Layers: layers, Embed: toBF16Bytes(mk(vocab*dModel, 11)), FinalNorm: toBF16Bytes(mk(dModel, 7))}
	g.LMHead, g.Tied = g.Embed, true

	const prompt = "hello"
	ids := tok.Encode(prompt)
	if len(ids) == 0 {
		t.Fatalf("tokenizer encoded %q to no tokens", prompt)
	}

	// text wrapper.
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	got, err := sess.GenerateText(tok, prompt, maxNew)
	if err != nil {
		t.Fatalf("GenerateText: %v", err)
	}

	// manual chain: Encode → Generate(ids) → Decode, on a fresh session.
	sess2, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession 2: %v", err)
	}
	eos := -1
	if tok.HasEOSToken() {
		eos = int(tok.EOSToken())
	}
	gen, err := sess2.Generate(ids, maxNew, eos)
	if err != nil {
		t.Fatalf("manual Generate: %v", err)
	}
	want := tok.Decode(gen)

	if got != want {
		t.Fatalf("GenerateText %q != manual Encode→Generate→Decode %q", got, want)
	}
	t.Logf("text path: %q → ids %v → generate → decode → %q (≡ manual chain) — text in/out, no cgo", prompt, ids, got)
}
