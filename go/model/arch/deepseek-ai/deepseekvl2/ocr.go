// SPDX-Licence-Identifier: EUPL-1.2

package deepseekvl2

import (
	core "dappco.re/go"
	"dappco.re/go/inference/decode/tokenizer"
	"dappco.re/go/inference/model/safetensors"
)

// ocr.go is the top-level orchestration `lem ocr` drives (cli/ocr.go): Load a checkpoint
// directory once, OCR as many images as needed. DeepSeek-OCR never enters
// model.Assemble/model.LookupArch here — this mirrors whisper's "own loader, own session" shape
// (arch/openai/whisper/transcribe.go's doc comment, itself mirroring mamba2's), because a
// vision-encoder-conditioned MoE-decoder forward does not fit the decoder-only causal-LM contract
// model.Load's registered path assembles (config.go's Arch refusal explains why).

// DefaultPrompt is DeepSeek-OCR's own recommended plain-text-extraction prompt (the README's
// first documented example, confirmed against the live checkpoint — see testdata/e2e_golden.json's
// generating script): "<image>\nFree OCR. ". The README's other documented prompt,
// "<image>\n<|grounding|>Convert the document to markdown. ", requests the layout-aware structured
// mode instead — Options.Prompt overrides this default for that or any other prompt.
const DefaultPrompt = "<image>\nFree OCR. "

// DefaultMaxNewTokens bounds a v1 OCR request's generation length absent an explicit
// Options.MaxNewTokens override — generous for a single page of text, short of the reference's
// own 8192 (this lane's no-KV-... — see decoder.go's file doc comment: a cache IS implemented, so
// the bound here is about a sane CLI default, not tractability).
const DefaultMaxNewTokens = 1024

// Model is a fully loaded DeepSeek-OCR checkpoint, ready to run OCR. Weights are read-only,
// host-owned f32 slices (LoadWeights copies every tensor out of the safetensors mmap — see Load's
// doc comment), so one Model is safe to call OCR on repeatedly and concurrently from multiple
// goroutines (each call is independent: its own vision-forward/prompt-embeds/decode-cache state).
type Model struct {
	Config    *Config
	Weights   *Weights
	Tokenizer *tokenizer.Tokenizer
}

// Load reads a DeepSeek-OCR checkpoint directory: config.json, tokenizer.json, and the
// safetensors weights — everything OCR needs. A directory whose config.json is not model_type
// "deepseek_vl_v2" is refused cleanly (the capability-refusal pattern serve's vision 400 route
// uses) rather than failing deeper with a confusing tensor-name error.
func Load(dir string) (*Model, error) {
	configRead := core.ReadFile(core.PathJoin(dir, "config.json"))
	if !configRead.OK {
		return nil, core.E("deepseekvl2.Load", "read config.json", resultErr(configRead))
	}
	cb, ok := configRead.Value.([]byte)
	if !ok {
		return nil, core.NewError("deepseekvl2.Load: config.json read returned non-byte data")
	}
	cfg, err := ParseConfig(cb)
	if err != nil {
		return nil, err
	}
	if cfg.ModelType != "deepseek_vl_v2" {
		return nil, core.NewError("deepseekvl2.Load: " + dir + " is not a DeepSeek-OCR checkpoint (model_type " +
			core.Sprintf("%q", cfg.ModelType) + ") — lem ocr only accepts deepseek_vl_v2 directories")
	}

	tok, err := tokenizer.LoadTokenizer(core.PathJoin(dir, "tokenizer.json"))
	if err != nil {
		return nil, core.E("deepseekvl2.Load", "load tokenizer.json", err)
	}

	dm, err := safetensors.LoadDirMmap(dir)
	if err != nil {
		return nil, core.E("deepseekvl2.Load", "load safetensors", err)
	}
	defer func() { _ = dm.Close() }() // LoadWeights copies every tensor to an owned f32 slice — the mmap needn't outlive it
	w, err := LoadWeights(dm.Tensors, cfg)
	if err != nil {
		return nil, err
	}
	return &Model{Config: cfg, Weights: w, Tokenizer: tok}, nil
}

// Result is one OCR call's output.
type Result struct {
	Text string
}

// Options configures one OCR call.
type Options struct {
	// Prompt overrides DefaultPrompt. Must contain EXACTLY ONE "<image>" placeholder
	// (BuildPromptEmbeds' contract) — "" uses DefaultPrompt.
	Prompt string
	// MaxNewTokens caps the generated content length; <=0 uses DefaultMaxNewTokens.
	MaxNewTokens int
}

// OCR runs the full forward on one image: decode+normalise → the dual-tower vision encoder
// (vision.go's VisionForward) → assemble the prompt embeddings (tokens.go's BuildPromptEmbeds) →
// greedy decode (tokens.go's GreedyDecode) → tokenizer decode.
func (m *Model) OCR(imageBytes []byte, opts Options) (Result, error) {
	if m == nil {
		return Result{}, core.NewError("deepseekvl2.OCR: nil model")
	}
	pixels, err := DecodeAndNormaliseImage(imageBytes)
	if err != nil {
		return Result{}, err
	}
	visionFeatures, err := VisionForward(pixels, m.Weights)
	if err != nil {
		return Result{}, err
	}
	prompt := opts.Prompt
	if prompt == "" {
		prompt = DefaultPrompt
	}
	embeds, _, err := BuildPromptEmbeds(prompt, visionFeatures, m.Tokenizer, m.Weights)
	if err != nil {
		return Result{}, err
	}
	maxNew := opts.MaxNewTokens
	if maxNew <= 0 {
		maxNew = DefaultMaxNewTokens
	}
	contentIDs, err := GreedyDecode(embeds, m.Config, m.Weights, maxNew)
	if err != nil {
		return Result{}, err
	}
	return Result{Text: m.Tokenizer.Decode(contentIDs)}, nil
}

// OCRImage satisfies inference.OCRModel — serve/CLI's capability-discovery surface — by adapting
// OCR's richer Options/Result shape to the primitive strings the root package can name without
// importing this one (the VisionModel/Transcriber stability-contract pattern; see go/inference.go's
// OCRModel doc comment). A separate method rather than changing OCR's own signature: OCR's
// Options/Result shape is the one `lem ocr` and every test in this package already depend on.
func (m *Model) OCRImage(imageBytes []byte, prompt string) (text string, err error) {
	result, err := m.OCR(imageBytes, Options{Prompt: prompt})
	if err != nil {
		return "", err
	}
	return result.Text, nil
}

// resultErr pulls the error out of a failed core.Result for wrapping, tolerating a Result whose
// Value is not an error — verbatim copy of whisper's helper of the same name (arch/openai/
// whisper/mel.go's own doc comment: "mirrors decode/generate/multimodal.go's helper of the same
// name in a sibling package"); duplicated rather than imported (AX-8 — no arch-to-arch imports).
func resultErr(r core.Result) error {
	if err, ok := r.Value.(error); ok {
		return err
	}
	return nil
}
