// SPDX-Licence-Identifier: EUPL-1.2

package needle

import (
	"math"

	core "dappco.re/go"
)

// Model is a loaded Needle checkpoint: its config, f32 weights and SentencePiece
// tokenizer. It is a read-only host reference — construct with Load and call
// Generate. Concurrent Generate calls are safe (no mutable state is shared).
type Model struct {
	cfg Config
	w   *Weights
	tok *spmTokenizer
}

// Load reads a Needle checkpoint directory (config.json, model.safetensors,
// tokenizer.model) into a runnable host reference.
//
//	m, err := needle.Load("/models/needle")
//	if err != nil { return err }
//	fmt.Println(m.Generate(query, tools, 64))
func Load(dir string) (*Model, error) {
	cfg, err := LoadConfig(dir)
	if err != nil {
		return nil, core.E("needle.Load", "config", err)
	}
	w, err := LoadWeights(dir + "/model.safetensors")
	if err != nil {
		return nil, core.E("needle.Load", "weights", err)
	}
	if _, ok := w.mustGet("model.embed_tokens.weight"); !ok {
		return nil, core.E("needle.Load", "checkpoint missing model.embed_tokens.weight", nil)
	}
	tok, err := loadSPM(dir+"/tokenizer.model", cfg)
	if err != nil {
		return nil, core.E("needle.Load", "tokenizer", err)
	}
	return &Model{cfg: cfg, w: w, tok: tok}, nil
}

// Config returns the model's hyper-parameters.
func (m *Model) Config() Config { return m.cfg }

// embed gathers token-embedding rows and scales them by sqrt(hidden) — the
// embed_scale the reference applies to BOTH the encoder and decoder embeddings
// (the tied lm_head, by contrast, is unscaled). Returns [len(ids), hidden].
func (m *Model) embed(ids []int) []float32 {
	hidden := m.cfg.HiddenSize
	scale := float32(math.Sqrt(float64(hidden)))
	table := m.w.get("model.embed_tokens.weight")
	out := make([]float32, len(ids)*hidden)
	for i, id := range ids {
		row := table[id*hidden : id*hidden+hidden]
		for d := range hidden {
			out[i*hidden+d] = row[d] * scale
		}
	}
	return out
}

// logits projects one hidden vector through the tied lm_head (= embed_tokens),
// giving a score per vocabulary entry: logits[v] = hidden · embed_tokens[v].
func (m *Model) logits(hiddenVec []float32) []float32 {
	hidden := m.cfg.HiddenSize
	return linearNoBias(hiddenVec, m.w.get("model.embed_tokens.weight"), m.cfg.VocabSize, hidden)
}

// encodeInput builds the encoder token stream the reference feeds: the query
// tokens, the <tools> separator, then the tools-JSON tokens — each string
// tokenised independently (its own dummy-prefix), no BOS/EOS added.
func (m *Model) encodeInput(query, tools string) []int {
	ids := m.tok.encode(query)
	ids = append(ids, m.cfg.ToolsTokenID)
	return append(ids, m.tok.encode(tools)...)
}

// generateIDs runs the full greedy decode and returns (encoderInputIDs,
// generatedIDs). The decoder is seeded with the EOS token (Needle's decoder-start
// token), argmax is taken at each step, and generation stops at EOS. The first
// generated token is <tool_call>. Exposed unexported for tests to assert exact
// token parity with the reference without a detokenisation round-trip.
func (m *Model) generateIDs(query, tools string, maxTokens int) (enc []int, gen []int) {
	enc = m.encodeInput(query, tools)
	encoderHidden := m.encode(enc)
	encLen := len(enc)
	hidden := m.cfg.HiddenSize

	dec := []int{m.cfg.EosTokenID}
	for range maxTokens {
		decHidden := m.decode(dec, encoderHidden, encLen)
		last := decHidden[(len(dec)-1)*hidden : len(dec)*hidden]
		next := argmax(m.logits(last))
		if next == m.cfg.EosTokenID {
			break
		}
		gen = append(gen, next)
		dec = append(dec, next)
	}
	return enc, gen
}

// Generate produces a tool call for a query and a tools-JSON string, returning
// the decoded text with the leading <tool_call> marker stripped (mirroring the
// reference's generate()). maxTokens caps the decode length.
//
//	m.Generate("What is the weather in San Francisco?",
//		`[{"name":"get_weather","parameters":{"location":"string"}}]`, 64)
//	// ` [{"name":"get_weather","arguments":{"location":"San Francisco"}}]`
func (m *Model) Generate(query, tools string, maxTokens int) string {
	_, gen := m.generateIDs(query, tools, maxTokens)
	text := m.tok.decode(gen)
	return core.TrimPrefix(text, "<tool_call>")
}

// argmax returns the index of the largest element (first on ties).
func argmax(v []float32) int {
	best := 0
	bestV := float32(math.Inf(-1))
	for i, x := range v {
		if x > bestV {
			bestV = x
			best = i
		}
	}
	return best
}
