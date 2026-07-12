// SPDX-Licence-Identifier: EUPL-1.2

package engine

import (
	"context"
	"iter"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/model"
)

// vision.go bridges an engine's vision tower to the engine-neutral serve surface.
// The metal engine carries the intricate tower/splice/prefill work on its
// *NativeTokenModel + *ArchSession; this file exposes exactly the steps
// [TextModel.Chat] drives to serve an image turn behind two optional interfaces,
// so [TextModel] answers image turns without importing a concrete engine. Mirrors
// the go-mlx native vision driver (cmd/mlx/vision.go): per image → project →
// placeholder block; template + encode; verify the placeholder run;
// splice the features over the placeholders; prefill token-embeddings; decode.

// VisionModel is satisfied by [TextModel] so the serve + generate handlers can
// probe image support with a plain type assertion.
var _ inference.VisionModel = (*TextModel)(nil)

// VisionTokenModel is the optional capability a [TokenModel] implements when the
// loaded checkpoint carries a vision tower — the metal engine's *NativeTokenModel
// satisfies it. Keeping the tower + splice work behind this interface is what lets
// [TextModel] stay engine-neutral while still answering image turns.
type VisionTokenModel interface {
	// AcceptsImageInput reports whether the LOADED checkpoint shipped a vision
	// tower — a live probe, not a family declaration (a text-only quant of a
	// vision family answers false).
	AcceptsImageInput() bool
	// ImagePlaceholderTokenID is the token id one image soft-token occupies, used
	// to verify the templated prompt tokenised to the expected placeholder run
	// before any features are spliced.
	ImagePlaceholderTokenID() int32
	// ImagePlaceholderBlock returns the begin/<soft-token>×n/end marker string for
	// one image occupying softTokens positions, spliced into the prompt text ahead
	// of the turn it belongs to.
	ImagePlaceholderBlock(softTokens int) string
	// ProjectImage preprocesses one raw PNG/JPEG image and runs it through the
	// vision tower, returning the projected feature bytes and the number of soft
	// tokens they occupy (the placeholder run length for this image).
	ProjectImage(image []byte) (features []byte, softTokens int, err error)
	// TokenEmbeddingsWithFeatures builds the base token-embedding rows for ids and
	// splices imageFeatures at the image-placeholder positions, returning rows
	// ready for [VisionSession.PrefillTokenEmbeddings].
	TokenEmbeddingsWithFeatures(ids []int32, imageFeatures, audioFeatures, videoFeatures []byte) ([][]byte, error)
}

// VisionSession is the optional [Session] capability to prefill token-embeddings
// (rather than token ids) — the metal engine's *ArchSession satisfies it. An image
// turn prefills the spliced embedding rows here instead of PrefillTokens.
type VisionSession interface {
	PrefillTokenEmbeddings(ids []int32, embeddings [][]byte) error
}

// AudioInputTokenModel is the optional token-model capability for audio turns:
// the loaded checkpoint carries an audio head (a live probe, like
// AcceptsImageInput) and can project raw audio bytes to soft-token features.
// The embeddings splice itself rides [VisionTokenModel.TokenEmbeddingsWithFeatures].
type AudioInputTokenModel interface {
	AcceptsAudioInput() bool
	AudioPlaceholderTokenID() int32
	AudioPlaceholderBlock(softTokens int) string
	// ProjectAudio decodes one audio attachment (WAV, 16-bit PCM mono 16 kHz)
	// and projects it through the audio head, returning the feature bytes and
	// the soft-token count.
	ProjectAudio(audio []byte) (features []byte, softTokens int, err error)
}

// messagesHaveAudios reports whether any turn carries audio attachments.
func messagesHaveAudios(messages []inference.Message) bool {
	for _, m := range messages {
		if len(m.Audios) > 0 {
			return true
		}
	}
	return false
}

// messagesHaveVideos reports whether any turn carries video frames.
func messagesHaveVideos(messages []inference.Message) bool {
	for _, m := range messages {
		if len(m.Videos) > 0 {
			return true
		}
	}
	return false
}

// VideoTokenModel is the optional token-model capability for video turns:
// frames project through the SAME vision path but splice at the video
// placeholder, one timestamped block per frame.
type VideoTokenModel interface {
	VideoPlaceholderTokenID() int32
	VideoPlaceholderBlock(softTokens int) string
}

// AcceptsImages reports whether the loaded checkpoint serves image turns — the
// inference.VisionModel probe the serve + generate handlers gate on. True only
// when the engine's TokenModel is a VisionTokenModel AND the loaded checkpoint
// actually shipped the tower.
func (m *TextModel) AcceptsImages() bool {
	if m == nil || m.tm == nil {
		return false
	}
	v, ok := m.tm.(VisionTokenModel)
	return ok && v.AcceptsImageInput()
}

// AcceptsAudio reports whether the loaded checkpoint serves audio turns — the
// inference.AudioModel probe, mirroring AcceptsImages.
func (m *TextModel) AcceptsAudio() bool {
	if m == nil || m.tm == nil {
		return false
	}
	a, ok := m.tm.(AudioInputTokenModel)
	return ok && a.AcceptsAudioInput()
}

// chatMultimodal serves a chat turn carrying images: it projects each image
// through the vision tower, splices the soft-token features over the prompt's
// placeholder positions, prefills the resulting token-embeddings, and streams the
// completion — the neutral counterpart of the go-mlx cmd/mlx/vision.go driver.
func (m *TextModel) chatMultimodal(ctx context.Context, messages []inference.Message, v VisionTokenModel, a AudioInputTokenModel, cfg inference.GenerateConfig) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		start := time.Now()
		if ctx == nil {
			ctx = context.Background()
		}
		// Project every attachment in turn order: image placeholder blocks lead
		// the turn text (the go-mlx convention), audio blocks FOLLOW it (the
		// gemma4 audio-after-text convention). imageFeatures/audioFeatures
		// accumulate the projected soft tokens in placeholder order.
		var imageFeatures, audioFeatures, videoFeatures []byte
		wantPlaceholders, wantAudioPlaceholders, wantVideoPlaceholders := 0, 0, 0
		rendered := make([]inference.Message, len(messages))
		for i, msg := range messages {
			rendered[i] = msg
			if len(msg.Images) == 0 && len(msg.Audios) == 0 && len(msg.Videos) == 0 {
				continue
			}
			var prefix, suffix core.Builder
			if len(msg.Videos) > 0 {
				vt, vtok := m.tm.(VideoTokenModel)
				if !vtok || vt.VideoPlaceholderTokenID() == 0 {
					m.setErr(core.NewError("engine.TextModel.Chat: model declares no video placeholder tokens"))
					return
				}
				// One timestamped vision block per frame (the reference's
				// "MM:SS {boi}{video_token×N}{eoi}" join); frames are treated
				// as 1 s apart when the caller supplies no timing.
				for f, frame := range msg.Videos {
					features, softTokens, err := v.ProjectImage(frame)
					if err != nil {
						m.setErr(core.E("engine.TextModel.Chat", "project video frame", err))
						return
					}
					if softTokens <= 0 {
						m.setErr(core.NewError("engine.TextModel.Chat: video frame produced no soft tokens"))
						return
					}
					block := vt.VideoPlaceholderBlock(softTokens)
					if block == "" {
						m.setErr(core.NewError("engine.TextModel.Chat: model declares no video placeholder tokens"))
						return
					}
					videoFeatures = append(videoFeatures, features...)
					wantVideoPlaceholders += softTokens
					prefix.WriteString(core.Sprintf("%02d:%02d ", f/60, f%60))
					prefix.WriteString(block)
					prefix.WriteString("\n")
				}
			}
			for _, img := range msg.Images {
				features, softTokens, err := v.ProjectImage(img)
				if err != nil {
					m.setErr(core.E("engine.TextModel.Chat", "project image", err))
					return
				}
				if softTokens <= 0 {
					m.setErr(core.NewError("engine.TextModel.Chat: image produced no soft tokens"))
					return
				}
				block := v.ImagePlaceholderBlock(softTokens)
				if block == "" {
					m.setErr(core.NewError("engine.TextModel.Chat: model declares no image placeholder tokens"))
					return
				}
				imageFeatures = append(imageFeatures, features...)
				wantPlaceholders += softTokens
				prefix.WriteString(block)
				prefix.WriteString("\n")
			}
			for _, aud := range msg.Audios {
				if a == nil {
					m.setErr(core.NewError("engine.TextModel.Chat: audio turn without an audio-capable model"))
					return
				}
				features, softTokens, err := a.ProjectAudio(aud)
				if err != nil {
					m.setErr(core.E("engine.TextModel.Chat", "project audio", err))
					return
				}
				if softTokens <= 0 {
					m.setErr(core.NewError("engine.TextModel.Chat: audio produced no soft tokens"))
					return
				}
				block := a.AudioPlaceholderBlock(softTokens)
				if block == "" {
					m.setErr(core.NewError("engine.TextModel.Chat: model declares no audio placeholder tokens"))
					return
				}
				audioFeatures = append(audioFeatures, features...)
				wantAudioPlaceholders += softTokens
				suffix.WriteString("\n")
				suffix.WriteString(block)
			}
			rendered[i].Content = prefix.String() + msg.Content + suffix.String()
		}

		// Frame the multimodal turn through the model's DECLARED chat template —
		// the same neutral seam the text-only [TextModel.Chat] path drives — so a
		// non-gemma vision family (a qwen-VL / ChatML shape) is framed in its own
		// dialect rather than gemma's. For a gemma model chatTemplate() resolves to
		// GemmaChatTemplate(m.turns, m.thoughtSuppressor), byte-identical to the
		// former formatChatPrompt(m.turnTokens(), …, m.thoughtSuppressor) call.
		ids := m.encode(renderChatTemplate(m.chatTemplate(), rendered, cfg.EnableThinking))
		if len(ids) == 0 {
			m.setErr(core.NewError("engine.TextModel.Chat: empty prompt after tokenisation"))
			return
		}
		// The templated placeholder runs must survive tokenisation exactly, or
		// the feature splice would land on the wrong rows — fail loud rather
		// than answer against a corrupted prefill.
		if got := countTokenID(ids, v.ImagePlaceholderTokenID()); got != wantPlaceholders {
			m.setErr(core.E("engine.TextModel.Chat",
				core.Sprintf("tokenizer produced %d image placeholders, want %d", got, wantPlaceholders), nil))
			return
		}
		if a != nil {
			if got := countTokenID(ids, a.AudioPlaceholderTokenID()); got != wantAudioPlaceholders {
				m.setErr(core.E("engine.TextModel.Chat",
					core.Sprintf("tokenizer produced %d audio placeholders, want %d", got, wantAudioPlaceholders), nil))
				return
			}
		}
		if wantVideoPlaceholders > 0 {
			vt, _ := m.tm.(VideoTokenModel)
			if got := countTokenID(ids, vt.VideoPlaceholderTokenID()); got != wantVideoPlaceholders {
				m.setErr(core.E("engine.TextModel.Chat",
					core.Sprintf("tokenizer produced %d video placeholders, want %d", got, wantVideoPlaceholders), nil))
				return
			}
		}

		rows, err := v.TokenEmbeddingsWithFeatures(ids, imageFeatures, audioFeatures, videoFeatures)
		if err != nil {
			m.setErr(core.E("engine.TextModel.Chat", "splice multimodal features", err))
			return
		}

		sess, err := m.openSession()
		if err != nil {
			m.setErr(err)
			return
		}
		defer func() { _ = sess.Close() }()
		vs, ok := sess.(VisionSession)
		if !ok {
			m.setErr(core.NewError("engine.TextModel.Chat: engine session does not support multimodal prefill"))
			return
		}
		if err := vs.PrefillTokenEmbeddings(ids, rows); err != nil {
			m.setErr(core.E("engine.TextModel.Chat", "prefill image embeddings", err))
			return
		}
		m.decodeFromPrefilled(ctx, sess, len(ids), cfg, start, yield)
	}
}

// countTokenID counts occurrences of id in ids — the placeholder-run check. A
// zero id (no placeholder configured) counts nothing.
func countTokenID(ids []int32, id int32) int {
	if id == 0 {
		return 0
	}
	n := 0
	for _, x := range ids {
		if x == id {
			n++
		}
	}
	return n
}

// messagesHaveImages reports whether any turn carries image bytes — the gate
// [TextModel.Chat] uses to choose the multimodal path.
func messagesHaveImages(messages []inference.Message) bool {
	for _, msg := range messages {
		if len(msg.Images) > 0 {
			return true
		}
	}
	return false
}

// modelSampleParams builds the decode SampleParams from a GenerateConfig — shared
// by the text and multimodal decode paths so sampling behaviour is identical.
func modelSampleParams(cfg inference.GenerateConfig) model.SampleParams {
	return model.SampleParams{
		Temperature:    cfg.Temperature,
		TopK:           cfg.TopK,
		TopP:           cfg.TopP,
		MinP:           cfg.MinP,
		RepeatPenalty:  cfg.RepeatPenalty,
		SuppressTokens: cfg.SuppressTokens,
	}
}
