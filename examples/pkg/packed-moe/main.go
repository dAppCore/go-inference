// SPDX-Licence-Identifier: EUPL-1.2

// The packed-expert MoE convention, end to end. A Mixture-of-Experts checkpoint on disk (Mixtral's
// HuggingFace layout: .block_sparse_moe.experts.{i}.{w1,w3,w2}.weight — one 2-D tensor PER expert)
// does not match the engine's generic MoE assembler, which wants ONE tensor per role per layer —
// the "every expert lives in one [experts·outDim, inDim] tensor" shape gpt_oss and qwen3_5_moe
// already ship natively. packExperts (model/arch/mistralai/mixtral/weights.go — the same synthesis
// dbrx/olmoe/qwenmoe/llama4 share) bridges the two at load time: it concatenates the N per-expert
// matrices row-major into one packed tensor per role. A quantised checkpoint carries the identical
// synthesis for the .scales/.biases triple (since 542d5484), so a 4-bit MoE pack loads and serves
// exactly like a dense one — through the SAME factory route (model.Load, then inference.LoadModel)
// any other architecture takes.
//
// This example builds a small synthetic 4-bit Mixtral-shaped checkpoint (1 layer, 2 experts,
// top-1) — self-contained, no download, no cgo — writes it to a temp directory in the REAL
// per-expert on-disk layout, loads it once directly (model.Load) to show the packed tensors
// packExperts synthesised, then loads it again through inference.LoadModel — the identical call
// any HF snapshot on disk takes — and Generates a few greedy tokens through the metal engine. The
// checkpoint's weights are synthetic noise (no training), so the generated text is not meant to
// read as language: it is the receipt that the packed-MoE wiring loads and decodes end to end.
//
//	go run ./pkg/packed-moe
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"dappco.re/go/inference"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/quant/mlxaffine"
	"dappco.re/go/inference/model/safetensors"

	_ "dappco.re/go/inference/examples/internal/engine" // registers the platform engine + every built-in arch (incl. mixtral)
)

// The checkpoint geometry — deliberately tiny but real Mixtral shapes: hidden_size/
// num_attention_heads gives headDim 32 (the paged SDPA kernel's floor), and intermediate_size ==
// groupSize so every projection packs to a whole number of affine groups. Mirrors the proven
// fixture go/engine/metal/moe_zoo_generate_test.go's writeZooMixtralQuantDir builds.
const (
	hidden    = 64
	ff        = 32 // intermediate_size — one expert's FFN width
	vocabSize = 32
	heads     = 2
	kvHeads   = 1
	headDim   = hidden / heads
	experts   = 2
	topK      = 1
	groupSize = 32
	bits      = 4
)

const configJSON = `{"model_type":"mixtral","hidden_size":64,"intermediate_size":32,"num_hidden_layers":1,` +
	`"num_attention_heads":2,"num_key_value_heads":1,"num_local_experts":2,"num_experts_per_tok":1,` +
	`"vocab_size":32,"rms_norm_eps":1e-5,"rope_theta":10000,"tie_word_embeddings":false}`

func main() {
	dir, err := os.MkdirTemp("", "go-inference-packed-moe-*")
	if err != nil {
		fmt.Fprintln(os.Stderr, "mkdir temp:", err)
		os.Exit(1)
	}
	defer func() { _ = os.RemoveAll(dir) }()

	if err := writeCheckpoint(dir); err != nil {
		fmt.Fprintln(os.Stderr, "write checkpoint:", err)
		os.Exit(1)
	}
	fmt.Printf("wrote a synthetic %d-bit Mixtral-shaped checkpoint (1 layer, %d experts, top-%d) to %s\n", bits, experts, topK, dir)
	fmt.Println()

	if err := describePacking(dir); err != nil {
		fmt.Fprintln(os.Stderr, "describe packing:", err)
		os.Exit(1)
	}
	fmt.Println()

	fmt.Println("loading the SAME directory through inference.LoadModel (the public factory route) and generating:")
	r := inference.LoadModel(dir)
	if !r.OK {
		fmt.Fprintln(os.Stderr, "load:", r.Value)
		os.Exit(1)
	}
	m := r.Value.(inference.TextModel)
	defer func() { _ = m.Close() }()

	const prompt = "moe"
	fmt.Printf("  prompt %q\n  ", prompt)
	for tok := range m.Generate(context.Background(), prompt, inference.WithMaxTokens(8), inference.WithTemperature(0)) {
		if tok.ID < 0 || tok.ID >= vocabSize {
			fmt.Fprintf(os.Stderr, "\ngenerated id %d outside vocab [0,%d)\n", tok.ID, vocabSize)
			os.Exit(1)
		}
		fmt.Printf("%d:%q ", tok.ID, tok.Text)
	}
	fmt.Println()
	if er := m.Err(); !er.OK {
		fmt.Fprintln(os.Stderr, "generate:", er.Value)
		os.Exit(1)
	}
	fmt.Println()
	fmt.Println("the packed-expert checkpoint served a real Generate call through the unmodified factory route.")
}

// describePacking loads dir directly through model.Load (the same call inference.LoadModel makes
// underneath) and prints the packed-expert convention in action: the REAL per-expert tensors this
// checkpoint ships on disk (still present in dm.Tensors — packExperts ADDS, never replaces —
// see weights.go's packExperts doc), beside the ONE synthesised [experts·outDim, inDim] tensor per
// role the engine actually binds and decodes.
func describePacking(dir string) error {
	lm, dm, err := model.Load(dir)
	if err != nil {
		return fmt.Errorf("model.Load: %w", err)
	}
	defer func() { _ = dm.Close() }()

	fmt.Println("on-disk per-expert tensors (the real Mixtral HF layout) — layer 0, gate_proj role:")
	for e := 0; e < experts; e++ {
		name := fmt.Sprintf("model.layers.0.block_sparse_moe.experts.%d.w1", e)
		t, ok := dm.Tensors[name+".weight"]
		if !ok {
			return fmt.Errorf("missing %s.weight", name)
		}
		fmt.Printf("  %-58s %-4s shape %v  (+ .scales/.biases)\n", name+".weight", t.Dtype, t.Shape)
	}

	if lm.Layers[0].MoE == nil {
		return fmt.Errorf("layer 0 loaded with no MoE block — packExperts did not run")
	}
	moe := lm.Layers[0].MoE
	fmt.Println()
	fmt.Println("packExperts' synthesised packed tensor (what the engine actually binds and decodes):")
	if err := reportPacked("gate_proj", moe.ExpGate, experts*ff, hidden); err != nil {
		return err
	}
	if err := reportPacked("up_proj", moe.ExpUp, experts*ff, hidden); err != nil {
		return err
	}
	if err := reportPacked("down_proj", moe.ExpDown, experts*hidden, ff); err != nil {
		return err
	}
	return nil
}

// reportPacked prints one packed-expert role's assembled shape and checks it against the expected
// [experts·outDim, inDim] convention packExperts documents — the loudly-named failure a genuinely
// broken pack would surface (see packExpertRole's own doc: "a mismatch means a malformed or
// partial checkpoint").
func reportPacked(role string, lin *model.Linear, wantOutDim, wantInDim int) error {
	fmt.Printf("  block_sparse_moe.experts_packed.%-10s outDim=%-4d inDim=%-3d quantised=%-5v kind=%-8q (%d experts x %d rows, stacked row-major)\n",
		role+".weight", lin.OutDim, lin.InDim, lin.Quantised(), lin.Kind, experts, wantOutDim/experts)
	if lin.OutDim != wantOutDim || lin.InDim != wantInDim {
		return fmt.Errorf("%s: assembled shape [%d,%d], want [%d,%d]", role, lin.OutDim, lin.InDim, wantOutDim, wantInDim)
	}
	if !lin.Quantised() || lin.Kind != mlxaffine.Mode {
		return fmt.Errorf("%s: lost its quant triple — packExperts must carry .scales/.biases alongside .weight", role)
	}
	return nil
}

// projSpec is one quantised projection to synthesise: its on-disk tensor-name prefix and logical
// (outDim, inDim) shape.
type projSpec struct {
	prefix        string
	outDim, inDim int
}

// writeCheckpoint materialises a complete, self-contained checkpoint directory — config.json,
// model.safetensors, tokenizer.json — model.Load and inference.LoadModel read straight off disk,
// exactly as they would a real downloaded snapshot.
func writeCheckpoint(dir string) error {
	tensors := map[string]safetensors.Tensor{
		"model.norm.weight":                              bf16Tensor(syntheticWeights(hidden, 1), hidden),
		"model.layers.0.input_layernorm.weight":          bf16Tensor(syntheticWeights(hidden, 2), hidden),
		"model.layers.0.post_attention_layernorm.weight": bf16Tensor(syntheticWeights(hidden, 3), hidden),
	}

	projs := []projSpec{
		{"model.embed_tokens", vocabSize, hidden},
		{"lm_head", vocabSize, hidden},
		{"model.layers.0.self_attn.q_proj", heads * headDim, hidden},
		{"model.layers.0.self_attn.k_proj", kvHeads * headDim, hidden},
		{"model.layers.0.self_attn.v_proj", kvHeads * headDim, hidden},
		{"model.layers.0.self_attn.o_proj", hidden, heads * headDim},
		{"model.layers.0.block_sparse_moe.gate", experts, hidden},
	}
	for e := 0; e < experts; e++ {
		p := fmt.Sprintf("model.layers.0.block_sparse_moe.experts.%d", e)
		projs = append(projs,
			projSpec{p + ".w1", ff, hidden}, // gate_proj
			projSpec{p + ".w3", ff, hidden}, // up_proj
			projSpec{p + ".w2", hidden, ff}, // down_proj
		)
	}
	for i, pr := range projs {
		if err := packQuantised(tensors, pr.prefix, pr.outDim, pr.inDim, i+10); err != nil {
			return err
		}
	}

	blob, err := safetensors.Encode(tensors)
	if err != nil {
		return fmt.Errorf("encode weights: %w", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), blob, 0o644); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(configJSON), 0o644); err != nil {
		return err
	}
	tokJSON, err := buildTokenizerJSON()
	if err != nil {
		return fmt.Errorf("build tokenizer.json: %w", err)
	}
	return os.WriteFile(filepath.Join(dir, "tokenizer.json"), tokJSON, 0o644)
}

// packQuantised affine-quantises a deterministic synthetic (outDim×inDim) weight and adds its
// packed .weight/.scales/.biases triple to tensors at prefix — the on-disk shape a real 4-bit
// checkpoint carries per projection. mlxaffine.QuantizeTensor is the SAME function pkg/quantise's
// native lane (`lem quant`'s default format) calls — production code, not a test-only fixture.
func packQuantised(tensors map[string]safetensors.Tensor, prefix string, outDim, inDim, salt int) error {
	packed, scales, biases, err := mlxaffine.QuantizeTensor(syntheticWeights(outDim*inDim, salt), outDim, inDim, bits, groupSize)
	if err != nil {
		return fmt.Errorf("quantise %s: %w", prefix, err)
	}
	tensors[prefix+".weight"] = safetensors.Tensor{Dtype: "U32", Shape: []int{outDim, mlxaffine.PackedWords(inDim, bits)}, Data: packed}
	tensors[prefix+".scales"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{outDim, inDim / groupSize}, Data: scales}
	tensors[prefix+".biases"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{outDim, inDim / groupSize}, Data: biases}
	return nil
}

// syntheticWeights generates n deterministic pseudo-random values from salt — no I/O, no RNG,
// reproducible across runs (mirrors the fixture convention go/engine/metal's own MoE zoo tests use
// throughout — see moe_zoo_generate_test.go's syntheticFloat32).
func syntheticWeights(n, salt int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32((i*salt+7)%101-50) * 0.02
	}
	return v
}

// bf16Tensor rounds vals to bf16 (round-to-nearest-even, the mlx/safetensors on-disk convention)
// and wraps them as a safetensors.Tensor — every norm in this checkpoint's plain (unquantised) dtype.
func bf16Tensor(vals []float32, shape ...int) safetensors.Tensor {
	data := make([]byte, len(vals)*2)
	for i, v := range vals {
		raw := math.Float32bits(v)
		r := uint16((raw + 0x7fff + ((raw >> 16) & 1)) >> 16)
		data[2*i], data[2*i+1] = byte(r), byte(r>>8)
	}
	return safetensors.Tensor{Dtype: "BF16", Shape: shape, Data: data}
}

// tokenizerFile, tokenizerModel and tokenizerAddedToken are the minimal slice of the HF
// tokenizer.json schema buildTokenizerJSON writes — the same shape (a tiny synthetic vocab, no
// merges) go/engine/hip/mamba2_runtime_test.go's writeHIPMamba2TestModel fixture proves against
// this engine's real text-level Generate path.
type tokenizerFile struct {
	Model       tokenizerModel        `json:"model"`
	AddedTokens []tokenizerAddedToken `json:"added_tokens"`
}
type tokenizerModel struct {
	Type   string           `json:"type"`
	Vocab  map[string]int32 `json:"vocab"`
	Merges []string         `json:"merges"`
}
type tokenizerAddedToken struct {
	ID      int32  `json:"id"`
	Content string `json:"content"`
	Special bool   `json:"special"`
}

// buildTokenizerJSON returns a minimal HF tokenizer.json: a flat character-level vocabulary (ids
// 0-30 = a-z0-4) covering every OTHER id embed_tokens/lm_head can produce, no merges, and id
// vocabSize-1 explicitly reserved as "<eos>". The explicit reservation matters:
// decode/tokenizer's Tokenizer.EOS() returns the tokenizer's eosToken field UNCONDITIONALLY, not
// gated on whether an "<eos>" was ever declared (tokenizer.go's HasEOSToken is a separate, unused-
// by-EOS() bool) — so a tokenizer.json that declares no EOS at all leaves eosToken at its zero
// value, id 0, which engine.TextModel's buildStopTokens then arms as a real stop token. This
// checkpoint's synthetic weights reach greedy id 0 on their very first decode step, so leaving
// "<eos>" undeclared would silently truncate every Generate call to one token. Real subword
// tokenisation is irrelevant here — this checkpoint's weights are synthetic noise, so the only
// thing worth proving is that the text ⇄ token-id plumbing round-trips.
func buildTokenizerJSON() ([]byte, error) {
	vocab := make(map[string]int32, vocabSize)
	for i, r := range "abcdefghijklmnopqrstuvwxyz01234" { // 31 content symbols -> ids 0..30
		vocab[string(r)] = int32(i)
	}
	const eosID = vocabSize - 1 // 31 — the one id reserved OFF the content alphabet above
	vocab["<eos>"] = eosID
	doc := tokenizerFile{
		Model:       tokenizerModel{Type: "BPE", Vocab: vocab, Merges: []string{}},
		AddedTokens: []tokenizerAddedToken{{ID: eosID, Content: "<eos>", Special: true}},
	}
	return json.Marshal(doc)
}
