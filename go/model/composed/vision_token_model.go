// SPDX-Licence-Identifier: EUPL-1.2

package composed

import core "dappco.re/go"

// vision_token_model.go is composed's side of the repo's ONE multimodal route (go/engine/vision.go):
// *ComposedTokenModel implements the engine.VisionTokenModel method SET (AcceptsImageInput,
// ImagePlaceholderTokenID, ImagePlaceholderBlock, ProjectImage, TokenEmbeddingsWithFeatures) and
// *composedStepper implements the engine.VisionSession method set (PrefillTokenEmbeddings) — BY SHAPE, not
// by import: composed is a model-root leaf and must never import engine (AX-8, "lib never imports
// consumer"), so there is no `var _ engine.VisionTokenModel = (*ComposedTokenModel)(nil)` compile-time
// assertion here the way token_model.go asserts model.TokenModel/model.SessionModel — Go's structural
// typing means the engine's own type assertions (m.tm.(engine.VisionTokenModel), a session's
// .(engine.VisionSession)) pick these methods up once something on the engine side hands the engine a
// *ComposedTokenModel-backed session. Wiring that bridge — engine/metal/inference_register.go's
// composedTextModel/composedEngineSession currently forward only the plain engine.TokenModel/engine.Session
// shapes — is out of this package's scope; see this task's report for the exact boundary.

// AcceptsImageInput reports whether the LOADED checkpoint shipped a vision tower + merger — a live probe
// (ComposedModel.Vision non-nil), not a family declaration: a text-only quant of a vision family, or any
// composed arch whose safetensors carry no vision_tower.* tensors, answers false.
func (tm *ComposedTokenModel) AcceptsImageInput() bool {
	return tm != nil && tm.m != nil && tm.m.Vision != nil
}

// ImagePlaceholderTokenID is the vocabulary id one image soft-token occupies (config.json's top-level
// image_token_id) — 0 when AcceptsImageInput is false.
func (tm *ComposedTokenModel) ImagePlaceholderTokenID() int32 {
	if tm == nil || tm.m == nil {
		return 0
	}
	return tm.m.ImageTokenID
}

// ImagePlaceholderBlock renders the begin/<soft-token>×n/end marker string for one image occupying
// softTokens positions — spliced into the prompt text ahead of the turn it belongs to by the neutral
// splice in go/engine/vision.go (chatMultimodal). Empty when the model carries no vision tower.
func (tm *ComposedTokenModel) ImagePlaceholderBlock(softTokens int) string {
	if tm == nil || tm.m == nil || tm.m.Vision == nil || softTokens <= 0 {
		return ""
	}
	var b core.Builder
	b.Grow(len(tm.m.VisionBeginToken) + len(tm.m.VisionEndToken) + softTokens*len(tm.m.VisionToken))
	b.WriteString(tm.m.VisionBeginToken)
	for range softTokens {
		b.WriteString(tm.m.VisionToken)
	}
	b.WriteString(tm.m.VisionEndToken)
	return b.String()
}

// ProjectImage preprocesses one raw PNG/JPEG image (top-left crop to a patch·merge-aligned grid, patchify)
// and runs it through the vision tower + merger, returning the projected soft-token feature bytes (dModel
// bf16 rows, ready for TokenEmbeddingsWithFeatures) and the soft-token count (the placeholder run length
// for this image).
func (tm *ComposedTokenModel) ProjectImage(image []byte) ([]byte, int, error) {
	if tm == nil || tm.m == nil || tm.m.Vision == nil {
		return nil, 0, core.NewError("composed.ComposedTokenModel.ProjectImage: model has no vision tower")
	}
	patches, gridH, gridW, err := imageToPatchGrid(image, tm.m.Vision.Cfg)
	if err != nil {
		return nil, 0, core.E("composed.ComposedTokenModel.ProjectImage", "patchify", err)
	}
	features, softTokens, err := visionTowerForward(patches, gridH, gridW, tm.m.Vision)
	if err != nil {
		return nil, 0, core.E("composed.ComposedTokenModel.ProjectImage", "tower forward", err)
	}
	return f32ToBF16Bytes(features), softTokens, nil
}

// TokenEmbeddingsWithFeatures gathers scaled token embeddings for ids and splices the projected image
// feature rows into the image-placeholder positions, returning rows ready for
// composedStepper.PrefillTokenEmbeddings. audioFeatures/videoFeatures are accepted for the
// engine.VisionTokenModel signature shape but composed carries no audio/video head — a non-empty one is a
// caller error, not a silent no-op.
func (tm *ComposedTokenModel) TokenEmbeddingsWithFeatures(ids []int32, imageFeatures, audioFeatures, videoFeatures []byte) ([][]byte, error) {
	if tm == nil || tm.m == nil {
		return nil, core.NewError("composed.ComposedTokenModel.TokenEmbeddingsWithFeatures: nil model")
	}
	if len(ids) == 0 {
		return nil, core.NewError("composed.ComposedTokenModel.TokenEmbeddingsWithFeatures: empty token ids")
	}
	if len(audioFeatures) > 0 {
		return nil, core.NewError("composed.ComposedTokenModel.TokenEmbeddingsWithFeatures: composed carries no audio head")
	}
	if len(videoFeatures) > 0 {
		return nil, core.NewError("composed.ComposedTokenModel.TokenEmbeddingsWithFeatures: composed carries no video head")
	}
	row := tm.m.D * 2 // bf16 bytes per embedding row
	stream := make([]byte, len(ids)*row)
	for i, id := range ids {
		e, err := tm.Embed(id)
		if err != nil {
			return nil, err
		}
		copy(stream[i*row:(i+1)*row], e)
	}
	if len(imageFeatures) > 0 {
		if err := spliceComposedImageFeatures(stream, ids, imageFeatures, tm.m.ImageTokenID, row); err != nil {
			return nil, err
		}
	}
	rows := make([][]byte, len(ids))
	for i := range ids {
		rows[i] = stream[i*row : (i+1)*row]
	}
	return rows, nil
}

// spliceComposedImageFeatures overwrites stream's rows at every position whose token id equals
// imageTokenID with the next feature row, in order — the same "N-th placeholder gets the N-th feature row"
// contract engine/metal's spliceTokenFeaturesInto uses.
func spliceComposedImageFeatures(stream []byte, ids []int32, features []byte, imageTokenID int32, row int) error {
	if imageTokenID == 0 {
		return core.NewError("composed.ComposedTokenModel.TokenEmbeddingsWithFeatures: image token id is not configured")
	}
	if len(features)%row != 0 {
		return core.NewError("composed.ComposedTokenModel.TokenEmbeddingsWithFeatures: image feature rows must align to the embedding width")
	}
	featureRows := len(features) / row
	slots := 0
	for _, id := range ids {
		if id == imageTokenID {
			slots++
		}
	}
	if slots != featureRows {
		return core.NewError(core.Sprintf("composed.ComposedTokenModel.TokenEmbeddingsWithFeatures: %d image placeholder(s) in the prompt but %d feature row(s) projected", slots, featureRows))
	}
	fi := 0
	for pos, id := range ids {
		if id != imageTokenID {
			continue
		}
		copy(stream[pos*row:(pos+1)*row], features[fi*row:(fi+1)*row])
		fi++
	}
	return nil
}

// PrefillTokenEmbeddings prefills the session from ALREADY-COMPUTED embedding rows (dModel bf16 bytes
// each) instead of token ids — the engine.VisionSession capability shape (go/engine/vision.go): a
// multimodal turn splices projected image features over the prompt's placeholder positions before
// prefill, so the stack must advance every layer's state from those rows directly rather than re-deriving
// them from a (nonexistent) vocabulary id. ids is accepted for the engine.VisionSession signature — only
// its LENGTH is consulted (a fresh prefill needs the row count, not token identity); PrefillBatch already
// advances every layer's recurrent/KV state from raw embeddings exactly as Step does from one, including
// the head-fuse cache PrefillBatch already maintains for the Head call that follows.
func (st *composedStepper) PrefillTokenEmbeddings(ids []int32, embeddings [][]byte) error {
	if len(ids) != len(embeddings) {
		return core.NewError("composed.composedStepper.PrefillTokenEmbeddings: token and embedding counts differ")
	}
	_, err := st.PrefillBatch(embeddings)
	return err
}
