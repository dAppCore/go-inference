// SPDX-Licence-Identifier: EUPL-1.2

package glmocr

import (
	core "dappco.re/go"

	"dappco.re/go/inference/decode/tokenizer"
)

// prompt.go builds GLM-OCR's single-image task-prompt token sequence and its 3D mrope position
// ids — the ONLY usage shape GLM-OCR's own README documents (one image, one of a handful of
// fixed task prompts — "Text Recognition:", "Formula Recognition:", "Table Recognition:", or a
// JSON-schema information-extraction instruction — no system turn, no multi-turn history, no
// thinking mode). Multi-turn conversations, a system prompt, and video are a named boundary
// this package does not serve; BuildPrompt refuses rather than guess at their chat-template
// shape.
//
// BuildPrompt's literal token layout —
//
//	[gMASK] <sop> <|user|> "\n" <|begin_of_image|> {image_token_id}×N <|end_of_image|> {prompt} <|assistant|> "\n"
//
// — was NOT derived by hand-tracing chat_template.jinja (GLM's template has conditional
// whitespace this package does not implement a Jinja engine for); it is instead the literal
// text transformers' real Glm46VProcessor.apply_chat_template(...) produces for exactly this
// shape, confirmed token-for-token against testdata/e2e_golden.json's "input_ids" (captured
// from the real processor + tokenizer on this package's own fixture image and the "Text
// Recognition:" prompt) — see prompt_test.go.

// BuildPrompt tokenizes one single-image OCR request into GLM-OCR's exact prompt token
// sequence (see the file doc comment). numImageTokens is the vision tower's merged-token count
// (VisionForward's second return value) — the number of image_token_id placeholders the text
// decoder's image span must carry, one per merged vision embedding, in the SAME order
// VisionForward produced them (see rope.go's visionPosIDs doc comment for why that ordering
// guarantee holds).
func BuildPrompt(tok *tokenizer.Tokenizer, cfg *Config, promptText string, numImageTokens int) (ids, mmType []int32, err error) {
	if tok == nil || cfg == nil {
		return nil, nil, core.NewError("glmocr.BuildPrompt: nil tokenizer/config")
	}
	if numImageTokens <= 0 {
		return nil, nil, core.NewError("glmocr.BuildPrompt: numImageTokens must be positive")
	}
	if cfg.ImageTokenID == 0 {
		return nil, nil, core.NewError("glmocr.BuildPrompt: config.image_token_id is unset")
	}
	need := func(s string) (int32, error) {
		id, ok := tok.TokenID(s)
		if !ok {
			return 0, core.NewError("glmocr.BuildPrompt: tokenizer is missing the special token " + s)
		}
		return id, nil
	}
	gmask, err := need("[gMASK]")
	if err != nil {
		return nil, nil, err
	}
	sop, err := need("<sop>")
	if err != nil {
		return nil, nil, err
	}
	userTok, err := need("<|user|>")
	if err != nil {
		return nil, nil, err
	}
	beginImg, err := need("<|begin_of_image|>")
	if err != nil {
		return nil, nil, err
	}
	endImg, err := need("<|end_of_image|>")
	if err != nil {
		return nil, nil, err
	}
	asstTok, err := need("<|assistant|>")
	if err != nil {
		return nil, nil, err
	}
	newline := tok.Encode("\n")
	promptIDs := tok.Encode(promptText)

	ids = make([]int32, 0, 6+numImageTokens+2*len(newline)+len(promptIDs))
	ids = append(ids, gmask, sop, userTok)
	ids = append(ids, newline...)
	ids = append(ids, beginImg)
	for range numImageTokens {
		ids = append(ids, cfg.ImageTokenID)
	}
	ids = append(ids, endImg)
	ids = append(ids, promptIDs...)
	ids = append(ids, asstTok)
	ids = append(ids, newline...)

	mmType = make([]int32, len(ids))
	for i, id := range ids {
		if id == cfg.ImageTokenID {
			mmType[i] = 1
		}
	}
	return ids, mmType, nil
}

// PositionIDs computes GLM-OCR's 3D (temporal,height,width) mrope position ids for a token
// sequence carrying AT MOST ONE image span (mmType[i]==1 for that span, 0 elsewhere) — a direct
// port of GlmOcrModel.get_rope_index's single-image, no-padding, no-video, batch-1 case, pinned
// against the real function's output in testdata/block_goldens.json's "rope_index" golden (see
// position_test.go). A text run advances all three axes together by 1 per token (ordinary 1D
// rope); the image run's positions follow GlmOcrModel.get_vision_position_ids: temporal
// constant, height repeats each row across the merged width, width tiles across the merged
// height (row-major over the merged (llmGridH,llmGridW) grid) — after the image, the next text
// token resumes at current + max(gridH,gridW)/spatialMerge (the reference's own advance rule,
// NOT current + llmGridH*llmGridW — the position axes stay closer together than the token count
// would suggest, matching how a 2D image "costs" only its longer merged side in 1D sequence
// terms). More than one image span refuses cleanly (a named boundary — see the file doc
// comment).
func PositionIDs(mmType []int32, gridT, gridH, gridW, spatialMerge int) (tPos, hPos, wPos []int, err error) {
	n := len(mmType)
	tPos, hPos, wPos = make([]int, n), make([]int, n), make([]int, n)
	current := 0
	imageUsed := false
	i := 0
	for i < n {
		if mmType[i] == 0 {
			j := i
			for j < n && mmType[j] == 0 {
				j++
			}
			for k := i; k < j; k++ {
				p := current + (k - i)
				tPos[k], hPos[k], wPos[k] = p, p, p
			}
			current += j - i
			i = j
			continue
		}
		if imageUsed {
			return nil, nil, nil, core.NewError("glmocr.PositionIDs: more than one image span is not supported in this lane")
		}
		j := i
		for j < n && mmType[j] == 1 {
			j++
		}
		if spatialMerge <= 0 || gridH%spatialMerge != 0 || gridW%spatialMerge != 0 {
			return nil, nil, nil, core.NewError("glmocr.PositionIDs: grid dimensions must be divisible by spatialMerge")
		}
		llmH, llmW, llmT := gridH/spatialMerge, gridW/spatialMerge, gridT
		expect := llmT * llmH * llmW
		if j-i != expect {
			return nil, nil, nil, core.NewError(core.Sprintf("glmocr.PositionIDs: image span is %d tokens, want %d (grid %dx%dx%d / merge %d)", j-i, expect, gridT, gridH, gridW, spatialMerge))
		}
		idx := i
		for range llmT {
			for h := range llmH {
				for w := range llmW {
					tPos[idx] = current
					hPos[idx] = current + h
					wPos[idx] = current + w
					idx++
				}
			}
		}
		maxDim := gridH
		if gridW > maxDim {
			maxDim = gridW
		}
		current += maxDim / spatialMerge
		imageUsed = true
		i = j
	}
	return tPos, hPos, wPos, nil
}
