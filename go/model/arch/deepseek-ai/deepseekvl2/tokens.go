// SPDX-Licence-Identifier: EUPL-1.2

package deepseekvl2

import (
	core "dappco.re/go"
	"dappco.re/go/inference/decode/tokenizer"
)

// tokens.go is the prompt-token/embedding assembly (modeling_deepseekocr.py's infer(), the
// text-splitting/tokenising half — NOT VisionForward's own image-feature assembly, vision.go)
// plus the greedy decode loop (mirroring whisper's tokens.go: DetectLanguage/BuildInitTokens/
// GreedyDecode's shape, adapted to DeepSeek-OCR's single fixed image-placeholder run instead of
// Whisper's task-token machinery).
//
// V1 SCOPE: plain greedy argmax, no no_repeat_ngram_size (the reference's infer() enables it,
// no_repeat_ngram_size=35, but ONLY as a repetition guard over LONG documents — it cannot alter
// any generation shorter than 35 tokens, which is exactly how this lane's E2E golden was captured
// too, see testdata/e2e_golden.json's generating script; a longer-document guard is a named v2
// follow-on, not a silent gap).

// imageToken is the literal placeholder DeepSeek-OCR's prompt template splits on
// (modeling_deepseekocr.py's infer(), hardcoded image_token = '<image>').
const imageToken = "<image>"

// encodeSegment tokenises one plain-text segment WITHOUT auto-injecting BOS (the reference's
// text_encode(tokenizer, text, bos=False, eos=False) — every segment around the image placeholder
// is encoded this way, with exactly one BOS prepended ONCE to the whole assembled sequence, see
// BuildPromptEmbeds). tokenizer.Tokenizer.Encode always prepends BOS unless text already starts
// with the BOS token's own literal spelling (its shouldPrependBOS rule) — stripping that leading
// token back off (when the caller's text plainly did not itself start with the BOS spelling)
// reproduces add_special_tokens=False for the one case this package's prompts exercise.
func encodeSegment(tok *tokenizer.Tokenizer, text string) []int32 {
	ids := tok.Encode(text)
	if len(ids) > 0 && ids[0] == tok.BOSToken() && !core.HasPrefix(text, tok.IDToken(tok.BOSToken())) {
		ids = ids[1:]
	}
	return ids
}

// BuildPromptEmbeds assembles the full [T,hidden] input embedding sequence for one OCR request:
// BOS, the prompt text split around its ONE "<image>" placeholder (text_encode'd, see
// encodeSegment), the NumImageTokens soft-token run VisionForward computed scattered in at the
// placeholder's position, and the trailing prompt text. Returns the assembled embeds and the
// token ids in the SAME order (the ids' image-run entries all read imageTokenID — GreedyDecode
// never needs them again, but Model.OCR keeps them for parity with the reference's own tokenised-
// prompt shape). Refuses a prompt with zero or more than one "<image>" placeholder — DeepSeek-OCR
// always conditions on exactly one image per request in this v1 lane (multi-image batching is the
// reference's own images=[…] list of MULTIPLE (patches,image_ori) pairs, a distinct v2 slice from
// the single-image dynamic-tiling cut vision.go names).
func BuildPromptEmbeds(prompt string, visionFeatures []float32, tok *tokenizer.Tokenizer, w *Weights) (embeds []float32, ids []int32, err error) {
	if tok == nil || w == nil {
		return nil, nil, core.NewError("deepseekvl2.BuildPromptEmbeds: nil tokenizer/weights")
	}
	// Cheapest, model-state-independent checks first: the caller's own prompt shape, before ever
	// consulting the loaded tokenizer/weights.
	n := core.Count(prompt, imageToken)
	if n != 1 {
		return nil, nil, core.NewError(core.Sprintf("deepseekvl2.BuildPromptEmbeds: prompt has %d %q placeholders, want exactly 1", n, imageToken))
	}
	hidden := w.Decoder.hiddenSize()
	if hidden <= 0 || len(visionFeatures) != NumImageTokens*hidden {
		return nil, nil, core.NewError(core.Sprintf("deepseekvl2.BuildPromptEmbeds: visionFeatures has %d elements, want %d rows of the loaded model's hidden width", len(visionFeatures), NumImageTokens))
	}
	imageTokenID, ok := tok.TokenID(imageToken)
	if !ok {
		return nil, nil, core.NewError("deepseekvl2.BuildPromptEmbeds: tokenizer has no " + imageToken + " token")
	}

	splitIdx := core.Index(prompt, imageToken)
	prefix, suffix := prompt[:splitIdx], prompt[splitIdx+len(imageToken):]

	prefixIDs := encodeSegment(tok, prefix)
	suffixIDs := encodeSegment(tok, suffix)

	total := 1 + len(prefixIDs) + NumImageTokens + len(suffixIDs)
	ids = make([]int32, 0, total)
	ids = append(ids, tok.BOSToken())
	ids = append(ids, prefixIDs...)
	for range NumImageTokens {
		ids = append(ids, imageTokenID)
	}
	ids = append(ids, suffixIDs...)

	embeds = make([]float32, total*hidden)
	row := 0
	putEmbed := func(id int32) {
		copy(embeds[row*hidden:(row+1)*hidden], w.Decoder.EmbedTokens[int(id)*hidden:int(id)*hidden+hidden])
		row++
	}
	putEmbed(tok.BOSToken())
	for _, id := range prefixIDs {
		putEmbed(id)
	}
	copy(embeds[row*hidden:(row+NumImageTokens)*hidden], visionFeatures)
	row += NumImageTokens
	for _, id := range suffixIDs {
		putEmbed(id)
	}
	return embeds, ids, nil
}

// GreedyDecode runs the autoregressive greedy loop from promptEmbeds (BuildPromptEmbeds' output):
// prefill the whole prompt as one DecodeLogitsStep batch, then one new token per step, stopping at
// EOS or maxNewTokens. Returns the generated CONTENT token ids only (the prompt is never echoed
// back).
func GreedyDecode(promptEmbeds []float32, cfg *Config, w *Weights, maxNewTokens int) ([]int32, error) {
	if cfg == nil || w == nil {
		return nil, core.NewError("deepseekvl2.GreedyDecode: nil config/weights")
	}
	hidden := cfg.HiddenSize
	if hidden <= 0 || len(promptEmbeds)%hidden != 0 {
		return nil, core.NewError("deepseekvl2.GreedyDecode: promptEmbeds buffer is not a whole number of hidden-width rows")
	}
	promptLen := len(promptEmbeds) / hidden
	limit := cfg.MaxPositionEmbeddings
	if maxNewTokens > 0 && promptLen+maxNewTokens < limit {
		limit = promptLen + maxNewTokens
	}
	if promptLen >= limit {
		return nil, core.NewError(core.Sprintf("deepseekvl2.GreedyDecode: prompt is already %d tokens, at or past the %d-token bound", promptLen, limit))
	}

	cache := NewSelfAttnCache(len(w.Decoder.Layers))
	logits, err := DecodeLogitsStep(promptEmbeds, 0, cache, cfg, w)
	if err != nil {
		return nil, err
	}
	var content []int32
	pos := promptLen
	for pos < limit {
		next := argmaxF32(logits)
		if int(next) == cfg.EOSTokenID {
			break
		}
		content = append(content, next)
		pos++
		if pos >= limit {
			break
		}
		nextEmbed := w.Decoder.EmbedTokens[int(next)*hidden : int(next)*hidden+hidden]
		logits, err = DecodeLogitsStep(nextEmbed, pos-1, cache, cfg, w)
		if err != nil {
			return nil, err
		}
	}
	return content, nil
}
