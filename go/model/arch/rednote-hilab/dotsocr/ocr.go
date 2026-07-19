// SPDX-Licence-Identifier: EUPL-1.2

package dotsocr

import (
	core "dappco.re/go"
	"dappco.re/go/inference/decode/tokenizer"
	"dappco.re/go/inference/model/safetensors"
)

// ocr.go is the top-level orchestration a caller (or the sibling lane's `lem ocr` verb, wired in
// at merge time) drives: Load a checkpoint directory once, OCR as many images as needed. DOTS-OCR
// never enters model.Assemble/model.LookupArch here — this mirrors whisper.Load/
// mamba2.LoadMambaModel's "own loader, own session" shape (register.go's Composed/Arch hooks keep
// refusing unconditionally; that refusal is for the model.Load/model.LoadComposedDir entry points
// engine/metal's decoder-only causal-LM path uses, a genuinely different contract from this
// package's own vision+OCR forward).

// defaultMaxNewTokens bounds one OCR call's generation length when Model.MaxNewTokens is left
// zero — generous enough for a real document-layout response (this lane's own E2E golden runs
// 160 steps on a single short text line) while still refusing to run away unbounded by default.
const defaultMaxNewTokens = 4096

// Model is a fully loaded DOTS-OCR checkpoint, ready to OCR images. Weights are read-only,
// host-owned f32 slices (LoadWeights copies every tensor out of the safetensors mmap — see Load's
// doc comment), so one Model is safe to call OCR on repeatedly and concurrently from multiple
// goroutines (each call is independent: its own patchify/vision/decode-loop state).
type Model struct {
	Config     *Config
	Generation *GenerationConfig
	Preproc    *PreprocessorConfig
	Weights    *Weights
	Tokenizer  *tokenizer.Tokenizer

	// MaxNewTokens bounds OCR's generation length; zero uses defaultMaxNewTokens.
	MaxNewTokens int
}

// Load reads a DOTS-OCR (or its dots_ocr_1_5 successor) checkpoint directory: config.json,
// preprocessor_config.json, generation_config.json, tokenizer.json, and the safetensors weights —
// everything OCR needs. A directory whose config.json is not a recognised DOTS-OCR model_type is
// refused cleanly rather than failing deeper with a confusing tensor-name error (the
// capability-refusal pattern whisper.Load/serve's vision 400 route use).
func Load(dir string) (*Model, error) {
	configRead := core.ReadFile(core.PathJoin(dir, "config.json"))
	if !configRead.OK {
		return nil, core.E("dotsocr.Load", "read config.json", resultErr(configRead))
	}
	cb, ok := configRead.Value.([]byte)
	if !ok {
		return nil, core.NewError("dotsocr.Load: config.json read returned non-byte data")
	}
	cfg, err := ParseConfig(cb)
	if err != nil {
		return nil, err
	}
	if cfg.ModelType != "dots_ocr" && cfg.ModelType != "dots_ocr_1_5" {
		return nil, core.NewError("dotsocr.Load: " + dir + " is not a DOTS-OCR checkpoint (model_type " +
			core.Sprintf("%q", cfg.ModelType) + ") — OCR only accepts dots_ocr/dots_ocr_1_5 directories")
	}

	preproc := &PreprocessorConfig{}
	if pcRead := core.ReadFile(core.PathJoin(dir, "preprocessor_config.json")); pcRead.OK {
		if pb, ok := pcRead.Value.([]byte); ok {
			if parsed, err := ParsePreprocessorConfig(pb); err == nil {
				preproc = parsed
			}
		}
	}
	if preproc.MinPixels <= 0 || preproc.MaxPixels <= 0 {
		return nil, core.NewError("dotsocr.Load: " + dir + " preprocessor_config.json is missing or has non-positive min_pixels/max_pixels")
	}

	gen := &GenerationConfig{}
	if gcRead := core.ReadFile(core.PathJoin(dir, "generation_config.json")); gcRead.OK {
		if gb, ok := gcRead.Value.([]byte); ok {
			if parsed, err := ParseGenerationConfig(gb); err == nil {
				gen = parsed
			}
		}
	}

	tok, err := tokenizer.LoadTokenizer(core.PathJoin(dir, "tokenizer.json"))
	if err != nil {
		return nil, core.E("dotsocr.Load", "load tokenizer.json", err)
	}

	dm, err := safetensors.LoadDirMmap(dir)
	if err != nil {
		return nil, core.E("dotsocr.Load", "load safetensors", err)
	}
	defer func() { _ = dm.Close() }() // LoadWeights copies every tensor to an owned f32 slice — the mmap needn't outlive it
	w, err := LoadWeights(dm.Tensors, cfg)
	if err != nil {
		return nil, err
	}
	return &Model{Config: cfg, Generation: gen, Preproc: preproc, Weights: w, Tokenizer: tok}, nil
}

// resultErr adapts a core.Result's failure into an error for core.E's err argument (core.ReadFile
// returns a Result, not an (T,error) pair — mirrors whisper.Load's identical helper).
func resultErr(r core.Result) error {
	if r.OK {
		return nil
	}
	return core.NewError("read failed")
}

// scatterVisionEmbeds overwrites embeds' rows at every position where ids equals imageTokenID,
// in order, with visionEmb's rows — the masked_scatter DotsOCRForCausalLM.prepare_inputs_embeds
// performs (modeling_dots_ocr.py). Refuses if the count of image-token positions in ids doesn't
// exactly match the number of vision embedding rows produced (never silently truncates/pads,
// matching this lane's "never guessed" discipline).
func scatterVisionEmbeds(embeds []float32, ids []int32, visionEmb []float32, imageTokenID int32, hidden int) error {
	if hidden <= 0 || len(visionEmb)%hidden != 0 {
		return core.NewError("dotsocr.scatterVisionEmbeds: visionEmb is not a whole number of hidden-sized rows")
	}
	numRows := len(visionEmb) / hidden
	row := 0
	for i, id := range ids {
		if id != imageTokenID {
			continue
		}
		if row >= numRows {
			return core.NewError(core.Sprintf("dotsocr.scatterVisionEmbeds: prompt has more image-token positions than the %d vision embedding rows produced", numRows))
		}
		copy(embeds[i*hidden:(i+1)*hidden], visionEmb[row*hidden:(row+1)*hidden])
		row++
	}
	if row != numRows {
		return core.NewError(core.Sprintf("dotsocr.scatterVisionEmbeds: prompt has %d image-token positions, want %d (one per vision embedding row)", row, numRows))
	}
	return nil
}

// OCR runs the full DOTS-OCR forward on one image: Patchify → EncodeImage (the NaViT-style vision
// tower) → build the prompt (buildPrompt) and scatter the vision embeddings into it at the
// image-token positions → GreedyDecode → tokenizer Decode. The structural shape
// `OCR(imageBytes []byte, prompt string) (string, error)` matches the shared verb interface a
// sibling lane defines (see this package's own doc comment) — Load + this method are the whole
// library surface; wiring it into `lem ocr` is the orchestrator's job, not this package's.
func (m *Model) OCR(imageBytes []byte, prompt string) (string, error) {
	if m == nil {
		return "", core.NewError("dotsocr.OCR: nil model")
	}
	vc := m.Config.VisionConfig
	if vc == nil {
		return "", core.NewError("dotsocr.OCR: config has no vision_config")
	}

	pv, err := Patchify(imageBytes, vc, m.Preproc.MinPixels, m.Preproc.MaxPixels)
	if err != nil {
		return "", err
	}
	visionEmb, err := EncodeImage(pv.Values, pv.GridT, pv.GridH, pv.GridW, m.Weights, m.Config)
	if err != nil {
		return "", err
	}

	merge := vc.SpatialMergeSize
	numImageTokens := pv.GridT * (pv.GridH / merge) * (pv.GridW / merge)
	text := buildPrompt(prompt, numImageTokens)
	ids := m.Tokenizer.Encode(text)

	embeds, err := EmbedTokens(ids, m.Weights, m.Config)
	if err != nil {
		return "", err
	}
	if err := scatterVisionEmbeds(embeds, ids, visionEmb, int32(m.Config.ImageTokenID), m.Weights.HiddenSize); err != nil {
		return "", err
	}

	maxNew := m.MaxNewTokens
	if maxNew <= 0 {
		maxNew = defaultMaxNewTokens
	}
	generated, err := GreedyDecode(embeds, len(ids), m.Weights, m.Config, maxNew, m.Generation.EOSSet())
	if err != nil {
		return "", err
	}
	return m.Tokenizer.Decode(generated), nil
}
