// SPDX-Licence-Identifier: EUPL-1.2

package glmocr

import (
	core "dappco.re/go"

	"dappco.re/go/inference/decode/tokenizer"
	"dappco.re/go/inference/model/safetensors"
)

// ocr.go is the top-level orchestration a `lem ocr` verb (a sibling lane, wired at merge — see
// register.go's doc comment) drives: Load a checkpoint directory once, OCR as many images as
// needed. GLM-OCR never enters model.Assemble/model.LookupArch here — this mirrors whisper's
// "own loader, own session" shape (arch/openai/whisper/transcribe.go's doc comment), because a
// vision-language OCR forward (image tower + a decoder that scatters vision embeddings into a
// text token stream) does not fit the neutral decoder-only-causal-LM contract model.Load's
// registered path assembles; register.go's Arch/Composed hooks still refuse cleanly for
// generate/serve (the shared paths), unaffected by this file.

// GenerationConfig is the EOS-token subset of a GLM-OCR checkpoint's generation_config.json —
// GLM-OCR carries TWO eos ids (the true end-of-text token and the role-switch "<|user|>" token
// GLM's own generation policy also treats as a stop condition — confirmed against the real
// checkpoint's generation_config.json and reproduced in testdata/e2e_golden.json's
// generated_ids, which end on the role-switch id, not end-of-text).
type GenerationConfig struct {
	EOSTokenIDs []int32
}

type generationConfigJSON struct {
	EOSTokenIDs []int32 `json:"eos_token_id"`
}

func loadGenerationConfig(dir string) (*GenerationConfig, error) {
	path := core.PathJoin(dir, "generation_config.json")
	read := core.ReadFile(path)
	if !read.OK {
		return nil, core.E("glmocr.loadGenerationConfig", "read "+path, resultErr(read))
	}
	data, ok := read.Value.([]byte)
	if !ok {
		return nil, core.NewError("glmocr.loadGenerationConfig: " + path + " read returned non-byte data")
	}
	var g generationConfigJSON
	if r := core.JSONUnmarshal(data, &g); !r.OK {
		return nil, core.NewError("glmocr.loadGenerationConfig: parse " + path)
	}
	if len(g.EOSTokenIDs) == 0 {
		return nil, core.NewError("glmocr.loadGenerationConfig: " + path + " is missing eos_token_id")
	}
	return &GenerationConfig{EOSTokenIDs: g.EOSTokenIDs}, nil
}

// Model is a fully loaded GLM-OCR checkpoint, ready to run OCR. Weights are read-only, host-
// owned f32 slices (LoadWeights copies every tensor out of the safetensors mmap — see Load's
// doc comment), so one Model is safe to call OCR on repeatedly and concurrently from multiple
// goroutines (each call is independent: its own patch grid/vision embeddings/decode state).
type Model struct {
	Config            *Config
	ImagePreprocessor *ImagePreprocessorConfig
	Generation        *GenerationConfig
	Weights           *Weights
	Tokenizer         *tokenizer.Tokenizer
}

// Load reads a GLM-OCR checkpoint directory: config.json, preprocessor_config.json,
// generation_config.json, tokenizer.json, and the safetensors weights — everything OCR needs.
// A directory whose config.json is not model_type "glm_ocr" is refused cleanly (the
// capability-refusal pattern whisper.Load uses) rather than failing deeper with a confusing
// tensor-name error.
func Load(dir string) (*Model, error) {
	configRead := core.ReadFile(core.PathJoin(dir, "config.json"))
	if !configRead.OK {
		return nil, core.E("glmocr.Load", "read config.json", resultErr(configRead))
	}
	cb, ok := configRead.Value.([]byte)
	if !ok {
		return nil, core.NewError("glmocr.Load: config.json read returned non-byte data")
	}
	cfg, err := ParseConfig(cb)
	if err != nil {
		return nil, err
	}
	if cfg.ModelType != "glm_ocr" {
		return nil, core.NewError("glmocr.Load: " + dir + " is not a GLM-OCR checkpoint (model_type " +
			core.Sprintf("%q", cfg.ModelType) + ") — this package only accepts GlmOcrForConditionalGeneration directories")
	}
	if cfg.TextConfig == nil || cfg.VisionConfig == nil {
		return nil, core.NewError("glmocr.Load: " + dir + " config.json is missing text_config/vision_config")
	}

	imgCfg, err := LoadImagePreprocessorConfig(dir)
	if err != nil {
		return nil, err
	}
	gen, err := loadGenerationConfig(dir)
	if err != nil {
		return nil, err
	}
	tok, err := tokenizer.LoadTokenizer(core.PathJoin(dir, "tokenizer.json"))
	if err != nil {
		return nil, core.E("glmocr.Load", "load tokenizer.json", err)
	}

	dm, err := safetensors.LoadDirMmap(dir)
	if err != nil {
		return nil, core.E("glmocr.Load", "load safetensors", err)
	}
	defer func() { _ = dm.Close() }() // LoadWeights copies every tensor to an owned f32 slice — the mmap needn't outlive it
	w, err := LoadWeights(dm.Tensors, cfg)
	if err != nil {
		return nil, err
	}
	return &Model{Config: cfg, ImagePreprocessor: imgCfg, Generation: gen, Weights: w, Tokenizer: tok}, nil
}

// defaultMaxNewTokens caps a call that doesn't set GenerateOptions.MaxNewTokens. This package
// recomputes the WHOLE sequence on every decode step (no growing KV cache — see textdecoder.go's
// doc comment), so wall-clock grows roughly with the CUBE of the token count for a long
// generation; 256 is a conservative default for a host-f32 correctness lane, not a hard model
// limit — pass a larger GenerateOptions.MaxNewTokens for a long document transcription.
const defaultMaxNewTokens = 256

// GenerateOptions configures one OCRWithOptions call.
type GenerateOptions struct {
	// MaxNewTokens caps how many tokens OCR will generate before stopping even if no EOS token
	// was produced. <= 0 uses defaultMaxNewTokens.
	MaxNewTokens int
}

// OCR implements the sibling verb lane's OCR(imageBytes,prompt) shape: image in, recognised
// text out, greedy-decoded (GLM-OCR's generation_config.json declares do_sample=false — this
// package only ever implements that documented default). An empty prompt defaults to "Text
// Recognition:" — GLM-OCR's own README-documented plain-OCR task prompt.
func (m *Model) OCR(imageBytes []byte, prompt string) (string, error) {
	return m.OCRWithOptions(imageBytes, prompt, GenerateOptions{})
}

// OCRWithOptions is OCR with an explicit MaxNewTokens cap — see GenerateOptions.
func (m *Model) OCRWithOptions(imageBytes []byte, prompt string, opts GenerateOptions) (string, error) {
	if m == nil {
		return "", core.NewError("glmocr.OCR: nil model")
	}
	if prompt == "" {
		prompt = "Text Recognition:"
	}
	patches, err := DecodeAndPatchify(imageBytes, m.ImagePreprocessor, m.Config.VisionConfig)
	if err != nil {
		return "", err
	}
	visionEmbeds, numMerged, err := VisionForward(patches, &m.Weights.Vision, m.Config.VisionConfig)
	if err != nil {
		return "", err
	}

	ids, mmType, err := BuildPrompt(m.Tokenizer, m.Config, prompt, numMerged)
	if err != nil {
		return "", err
	}
	tPos, hPos, wPos, err := PositionIDs(mmType, patches.GridT, patches.GridH, patches.GridW, m.Config.VisionConfig.SpatialMergeSize)
	if err != nil {
		return "", err
	}

	hiddenSize := m.Config.TextConfig.HiddenSize
	embeds, err := embedTokens(ids, m.Weights.Text.EmbedTokens, hiddenSize)
	if err != nil {
		return "", err
	}
	imgIdx := 0
	for i, t := range mmType {
		if t == 1 {
			copy(embeds[i*hiddenSize:(i+1)*hiddenSize], visionEmbeds[imgIdx*hiddenSize:(imgIdx+1)*hiddenSize])
			imgIdx++
		}
	}

	eos := make(map[int32]bool, len(m.Generation.EOSTokenIDs))
	for _, e := range m.Generation.EOSTokenIDs {
		eos[e] = true
	}
	maxNew := opts.MaxNewTokens
	if maxNew <= 0 {
		maxNew = defaultMaxNewTokens
	}

	curTPos := append([]int(nil), tPos...)
	curHPos := append([]int(nil), hPos...)
	curWPos := append([]int(nil), wPos...)
	genIDs := make([]int32, 0, maxNew)

	for range maxNew {
		T := len(curTPos)
		out, err := TextForward(embeds, T, m.Config.TextConfig, &m.Weights.Text, curTPos, curHPos, curWPos)
		if err != nil {
			return "", err
		}
		lastRow := out[(T-1)*hiddenSize : T*hiddenSize]
		logits := linearForward(lastRow, m.Weights.Text.LMHead, 1)
		next := int32(argmax32(logits))
		if eos[next] {
			break
		}
		genIDs = append(genIDs, next)

		// The prompt always ends on a plain-text position (BuildPrompt's fixed template never
		// ends on the image span), so every generated token — always text, this lane never
		// re-enters an image span — simply continues the running 1D counter on all three axes.
		nextPos := curTPos[T-1] + 1
		curTPos = append(curTPos, nextPos)
		curHPos = append(curHPos, nextPos)
		curWPos = append(curWPos, nextPos)

		nextEmbed, err := embedTokens([]int32{next}, m.Weights.Text.EmbedTokens, hiddenSize)
		if err != nil {
			return "", err
		}
		embeds = append(embeds, nextEmbed...)
	}
	return m.Tokenizer.Decode(genIDs), nil
}

// OCRImage adapts OCR to the engine-neutral inference.OCRModel shape — satisfied
// structurally, never by import.
func (m *Model) OCRImage(imageBytes []byte, prompt string) (string, error) {
	return m.OCR(imageBytes, prompt)
}
