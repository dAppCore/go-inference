// SPDX-Licence-Identifier: EUPL-1.2

package rwkv7

import (
	"encoding/json"
	"math"
	"os"
	"strconv"
	"testing"
	"unicode"
	"unicode/utf8"

	"dappco.re/go/inference/model/safetensors"
)

// real_checkpoint_test.go is the #42 oracle gate PLUS the library-level generation acceptance (#36):
// both env-guarded on RWKV7_SMOKE_DIR (a real RWKV7-Goose-World2.8-0.1B-HF checkpoint directory) so
// neither is part of the normal `go test ./...` suite (no network/download dependency in CI) — the same
// shape as mamba2/smoke_test.go's MAMBA2_SMOKE_DIR. testdata/oracle_fixture.json was produced ONCE by
// testdata/oracle_rwkv7.py (an independent numpy transcription of the reference block math, run over the
// SAME real checkpoint) — see that script's header for provenance.

// oracleFixture mirrors testdata/oracle_rwkv7.py's json.dump shape.
type oracleFixture struct {
	PromptIDs     []int32              `json:"prompt_ids"`
	CaptureLayers []int                `json:"capture_layers"`
	LayerHidden   map[string][]float64 `json:"layer_hidden"`
	Logits        []float64            `json:"logits"`
	Top5IDs       []int32              `json:"top5_ids"`
	Top5Vals      []float64            `json:"top5_vals"`
}

func cosineSim32v64(a []float32, b []float64) float64 {
	var dot, na, nb float64
	for i := range a {
		av, bv := float64(a[i]), b[i]
		dot += av * bv
		na += av * av
		nb += bv * bv
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func top5(v []float32) ([]int, []float32) {
	idx := make([]int, len(v))
	for i := range idx {
		idx[i] = i
	}
	for i := 0; i < 5 && i < len(idx); i++ {
		best := i
		for j := i + 1; j < len(idx); j++ {
			if v[idx[j]] > v[idx[best]] {
				best = j
			}
		}
		idx[i], idx[best] = idx[best], idx[i]
	}
	ids := make([]int, 5)
	vals := make([]float32, 5)
	for i := range 5 {
		ids[i] = idx[i]
		vals[i] = v[idx[i]]
	}
	return ids, vals
}

func loadRealRWKV7Model(t *testing.T, dir string) *RWKV7Model {
	t.Helper()
	dm, err := safetensors.LoadDirMmap(dir)
	if err != nil {
		t.Fatalf("load safetensors: %v", err)
	}
	t.Cleanup(func() { _ = dm.Close() })
	cfgBytes, err := os.ReadFile(dir + "/config.json")
	if err != nil {
		t.Fatalf("read config.json: %v", err)
	}
	m, err := LoadRWKV7Model(dm.Tensors, cfgBytes)
	if err != nil {
		t.Fatalf("LoadRWKV7Model: %v", err)
	}
	return m
}

// TestRWKV7RealCheckpointOracle is the #42 oracle gate: load the REAL checkpoint, run the SAME fixed
// prompt the oracle used, and cross-check the host Go forward against testdata/oracle_fixture.json at
// >=4 layers spread across depth (cosine >= 0.9999) plus a full-forward (all 65536 logits) cosine check.
// It walks layers manually (mirroring RWKV7Session.forwardEmb) rather than calling forwardEmb directly,
// because forwardEmb only returns the FINAL hidden — this needs to snapshot several intermediate layers.
func TestRWKV7RealCheckpointOracle(t *testing.T) {
	dir := os.Getenv("RWKV7_SMOKE_DIR")
	if dir == "" {
		t.Skip("set RWKV7_SMOKE_DIR to the RWKV7-Goose-World2.8-0.1B-HF checkpoint dir")
	}
	fixData, err := os.ReadFile("testdata/oracle_fixture.json")
	if err != nil {
		t.Fatalf("read oracle fixture (run testdata/oracle_rwkv7.py first): %v", err)
	}
	var fix oracleFixture
	if err := json.Unmarshal(fixData, &fix); err != nil {
		t.Fatalf("parse oracle fixture: %v", err)
	}

	m := loadRealRWKV7Model(t, dir)
	t.Logf("loaded real checkpoint: %d layers D=%d vocab=%d FF=%d cfg=%+v eps=%v", len(m.Layers), m.D, m.Vocab, m.FF, m.Cfg, m.Eps)

	D, L := m.D, len(fix.PromptIDs)
	hidden := make([]float32, L*D)
	for i, id := range fix.PromptIDs {
		copy(hidden[i*D:(i+1)*D], m.Embed[int(id)*D:int(id)*D+D])
	}
	captureSet := map[int]bool{}
	for _, li := range fix.CaptureLayers {
		captureSet[li] = true
	}

	s := NewSession(m)
	cur := hidden
	var vFirst []float32
	const gate = 0.9999
	for li := range m.Layers {
		layer := &m.Layers[li]
		base0 := cur
		if layer.PreNormW != nil {
			base0 = layerNormRows(cur, layer.PreNormW, layer.PreNormB, L, D, m.Eps)
		}
		h1 := layerNormRows(base0, layer.AttnNormW, layer.AttnNormB, L, D, m.Eps)
		attnOut, vf, newSt, terr := timeMixForward(h1, layer.Attn, m.Cfg, li, vFirst, timeMixState{WKV: s.wkv[li], Shift: s.shift1[li]}, L, D, m.Eps)
		if terr != nil {
			t.Fatalf("layer %d time-mix: %v", li, terr)
		}
		vFirst = vf
		s.wkv[li], s.shift1[li] = newSt.WKV, newSt.Shift
		x1 := make([]float32, L*D)
		for i := range x1 {
			x1[i] = base0[i] + attnOut[i]
		}
		h2 := layerNormRows(x1, layer.FfnNormW, layer.FfnNormB, L, D, m.Eps)
		ffnOut, newShift2, cerr := channelMixForward(h2, layer.FFN, s.shift2[li], L, D, m.FF)
		if cerr != nil {
			t.Fatalf("layer %d channel-mix: %v", li, cerr)
		}
		s.shift2[li] = newShift2
		x2 := make([]float32, L*D)
		for i := range x2 {
			x2[i] = x1[i] + ffnOut[i]
		}
		cur = x2

		if captureSet[li] {
			want, ok := fix.LayerHidden[strconv.Itoa(li)]
			if !ok {
				t.Fatalf("oracle fixture has no layer_hidden for layer %d", li)
			}
			got := cur[(L-1)*D : L*D]
			cos := cosineSim32v64(got, want)
			t.Logf("layer %d: cosine(go, oracle) = %.8f", li, cos)
			if cos < gate {
				t.Errorf("layer %d cosine %.8f below gate %.4f", li, cos, gate)
			}
		}
	}

	logits := s.headLogits(cur[(L-1)*D : L*D])
	cos := cosineSim32v64(logits, fix.Logits)
	t.Logf("full-forward logits (%d-wide): cosine(go, oracle) = %.8f", len(logits), cos)
	if cos < gate {
		t.Errorf("logits cosine %.8f below gate %.4f", cos, gate)
	}
	gotIDs, gotVals := top5(logits)
	t.Logf("go     top5 ids=%v vals=%v", gotIDs, gotVals)
	t.Logf("oracle top5 ids=%v vals=%v", fix.Top5IDs, fix.Top5Vals)
	if gotIDs[0] != int(fix.Top5IDs[0]) {
		t.Errorf("argmax token = %d, oracle argmax = %d", gotIDs[0], fix.Top5IDs[0])
	}
}

// TestRWKV7RealCheckpointGenerationSmoke is the library-level acceptance gate (#36): load the real
// checkpoint, tokenise a real English prompt with the World tokenizer, greedy-generate a continuation
// through RWKV7Session.Generate, and assert it decodes to well-formed text. The automated assertions are
// the objective floor (non-empty, valid UTF-8, overwhelmingly printable); the qualitative "coherent
// English" judgement is made by reading the t.Logf sample this test prints (see the task report for the
// actual generated text).
func TestRWKV7RealCheckpointGenerationSmoke(t *testing.T) {
	dir := os.Getenv("RWKV7_SMOKE_DIR")
	if dir == "" {
		t.Skip("set RWKV7_SMOKE_DIR to the RWKV7-Goose-World2.8-0.1B-HF checkpoint dir")
	}
	tok, err := LoadWorldTokenizerHex("testdata/rwkv_vocab_v20230424.hex")
	if err != nil {
		t.Fatalf("LoadWorldTokenizerHex: %v", err)
	}
	m := loadRealRWKV7Model(t, dir)

	const prompt = "The capital of France is"
	promptIDs := tok.Encode(prompt)
	if len(promptIDs) == 0 {
		t.Fatal("prompt encoded to zero tokens")
	}
	t.Logf("prompt %q -> %d tokens: %v", prompt, len(promptIDs), promptIDs)

	gen, err := NewSession(m).Generate(promptIDs, 40, -1)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	text := tok.Decode(gen)
	t.Logf("continuation (%d tokens): %q", len(gen), text)
	t.Logf("full sample: %q", prompt+text)

	if !utf8.ValidString(text) {
		t.Fatal("generated continuation is not valid UTF-8")
	}
	if len(text) == 0 {
		t.Fatal("generated continuation is empty")
	}
	total, printable := 0, 0
	for _, r := range text {
		total++
		if unicode.IsPrint(r) || r == '\n' || r == '\t' {
			printable++
		}
	}
	if frac := float64(printable) / float64(total); frac < 0.95 {
		t.Fatalf("continuation is mostly non-printable (%d/%d printable, %.2f) — not well-formed text", printable, total, frac)
	}
}
