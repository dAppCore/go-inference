// SPDX-Licence-Identifier: EUPL-1.2

package engine

import (
	"context"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// fakeVisionTokenModel is a configurable [VisionTokenModel] double layered
// over fakeTokenModel — AcceptsImageInput/ImagePlaceholder*/ProjectImage/
// TokenEmbeddingsWithFeatures each return their configured value or error.
// OpenEngineSession hands back session (typically a *fakeVisionSession, so
// the session-side VisionSession probe in chatMultimodal succeeds — but a
// plain *fakeSession is a legitimate fixture too, for the "session lacks the
// VisionSession seam" test).
type fakeVisionTokenModel struct {
	fakeTokenModel

	accepts           bool
	placeholderID     int32
	placeholderBlock  string
	projectFeatures   []byte
	projectSoftTokens int
	projectErr        error
	embedRows         [][]byte
	embedErr          error

	session Session
}

func (f *fakeVisionTokenModel) AcceptsImageInput() bool            { return f.accepts }
func (f *fakeVisionTokenModel) ImagePlaceholderTokenID() int32     { return f.placeholderID }
func (f *fakeVisionTokenModel) ImagePlaceholderBlock(n int) string { return f.placeholderBlock }

func (f *fakeVisionTokenModel) ProjectImage(image []byte) ([]byte, int, error) {
	if f.projectErr != nil {
		return nil, 0, f.projectErr
	}
	return f.projectFeatures, f.projectSoftTokens, nil
}

func (f *fakeVisionTokenModel) TokenEmbeddingsWithFeatures(ids []int32, imageFeatures, audioFeatures, videoFeatures []byte) ([][]byte, error) {
	if f.embedErr != nil {
		return nil, f.embedErr
	}
	return f.embedRows, nil
}

func (f *fakeVisionTokenModel) OpenEngineSession() (Session, error) {
	f.openCalls++
	if f.openErr != nil {
		return nil, f.openErr
	}
	return f.session, nil
}

var _ VisionTokenModel = (*fakeVisionTokenModel)(nil)

// fakeVisionSession layers the [VisionSession] capability over fakeSession.
type fakeVisionSession struct {
	fakeSession

	prefillEmbedErr    error
	prefillEmbedCalled bool
	embedIDs           []int32
	gotEmbedRows       [][]byte
}

func (f *fakeVisionSession) PrefillTokenEmbeddings(ids []int32, embeddings [][]byte) error {
	f.prefillEmbedCalled = true
	f.embedIDs = ids
	f.gotEmbedRows = embeddings
	if f.prefillEmbedErr != nil {
		return f.prefillEmbedErr
	}
	f.pos = len(ids)
	return nil
}

var (
	_ VisionSession = (*fakeVisionSession)(nil)
	_ Session       = (*fakeVisionSession)(nil)
)

// --- AcceptsImages -----------------------------------------------------

// TestVision_TextModel_AcceptsImages_Good pins the positive probe: an engine
// whose loaded checkpoint shipped a vision tower reports true.
func TestVision_TextModel_AcceptsImages_Good(t *testing.T) {
	m := &TextModel{tm: &fakeVisionTokenModel{accepts: true}}
	if !m.AcceptsImages() {
		t.Fatal("AcceptsImages() = false, want true for a vision-capable loaded checkpoint")
	}
}

// TestVision_TextModel_AcceptsImages_Bad pins the no-seam case: an engine
// TokenModel that does not implement VisionTokenModel at all reports false.
func TestVision_TextModel_AcceptsImages_Bad(t *testing.T) {
	m := &TextModel{tm: &fakeTokenModel{}}
	if m.AcceptsImages() {
		t.Fatal("AcceptsImages() = true, want false for an engine with no vision seam")
	}
}

// TestVision_TextModel_AcceptsImages_Ugly pins the live-probe distinction (a
// FAMILY supporting vision does not mean THIS checkpoint shipped the tower)
// and the nil-receiver guard.
func TestVision_TextModel_AcceptsImages_Ugly(t *testing.T) {
	m := &TextModel{tm: &fakeVisionTokenModel{accepts: false}}
	if m.AcceptsImages() {
		t.Fatal("AcceptsImages() = true, want false: the family supports vision but this checkpoint has no tower")
	}
	var nilModel *TextModel
	if nilModel.AcceptsImages() {
		t.Fatal("AcceptsImages() on a nil *TextModel = true, want false")
	}
}

// --- AcceptsAudio ------------------------------------------------------

// fakeAudioTokenModel is a configurable [AudioInputTokenModel] double layered
// over fakeTokenModel — the audio sibling of fakeVisionTokenModel. Only
// AcceptsAudioInput drives the AcceptsAudio probe; the other three methods are
// present to satisfy the interface (a checkpoint that carries an audio head
// declares all four).
type fakeAudioTokenModel struct {
	fakeTokenModel

	accepts bool
}

func (f *fakeAudioTokenModel) AcceptsAudioInput() bool            { return f.accepts }
func (f *fakeAudioTokenModel) AudioPlaceholderTokenID() int32     { return 0 }
func (f *fakeAudioTokenModel) AudioPlaceholderBlock(n int) string { return "" }
func (f *fakeAudioTokenModel) ProjectAudio(a []byte) ([]byte, int, error) {
	return nil, 0, nil
}

var _ AudioInputTokenModel = (*fakeAudioTokenModel)(nil)

// TestVision_TextModel_AcceptsAudio_Good pins the positive probe: an engine
// whose loaded checkpoint shipped an audio head reports true.
func TestVision_TextModel_AcceptsAudio_Good(t *testing.T) {
	m := &TextModel{tm: &fakeAudioTokenModel{accepts: true}}
	if !m.AcceptsAudio() {
		t.Fatal("AcceptsAudio() = false, want true for an audio-capable loaded checkpoint")
	}
}

// TestVision_TextModel_AcceptsAudio_Bad pins the no-seam case: an engine
// TokenModel that does not implement AudioInputTokenModel at all reports false.
func TestVision_TextModel_AcceptsAudio_Bad(t *testing.T) {
	m := &TextModel{tm: &fakeTokenModel{}}
	if m.AcceptsAudio() {
		t.Fatal("AcceptsAudio() = true, want false for an engine with no audio seam")
	}
}

// TestVision_TextModel_AcceptsAudio_Ugly pins the live-probe distinction (a
// FAMILY supporting audio does not mean THIS checkpoint shipped the head) and
// the nil-receiver guard, mirroring AcceptsImages.
func TestVision_TextModel_AcceptsAudio_Ugly(t *testing.T) {
	m := &TextModel{tm: &fakeAudioTokenModel{accepts: false}}
	if m.AcceptsAudio() {
		t.Fatal("AcceptsAudio() = true, want false: the family supports audio but this checkpoint has no head")
	}
	var nilModel *TextModel
	if nilModel.AcceptsAudio() {
		t.Fatal("AcceptsAudio() on a nil *TextModel = true, want false")
	}
}

// --- chatMultimodal (reached only via TextModel.Chat on an image turn) -----

// TestChatMultimodal_Good walks the full image-turn path: project each image,
// splice its soft-token features over the placeholder run, prefill the
// spliced embeddings, and stream the completion — mirroring the go-mlx
// cmd/mlx/vision.go driver this engine-neutral seam replaces.
func TestChatMultimodal_Good(t *testing.T) {
	tok := newFixtureTokenizer(t)
	visionSess := &fakeVisionSession{fakeSession: fakeSession{genIDs: []int32{10, 11}}}
	vtm := &fakeVisionTokenModel{
		accepts:           true,
		placeholderID:     42, // the fixture tokenizer's "z" vocab entry
		placeholderBlock:  "zzz",
		projectFeatures:   []byte{9, 9},
		projectSoftTokens: 3,
		embedRows:         [][]byte{{0}, {1}, {2}, {3}},
		session:           visionSess,
	}
	m := NewTextModel(vtm, tok, "gemma-test", inference.ModelInfo{}, 4096)
	messages := []inference.Message{{Role: "user", Content: "hi", Images: [][]byte{{1, 2, 3}}}}

	var got []inference.Token
	for tk := range m.Chat(context.Background(), messages, inference.WithMaxTokens(2)) {
		got = append(got, tk)
	}
	if r := m.Err(); !r.OK {
		t.Fatalf("Chat over an image turn failed: %+v", r)
	}
	if len(got) != 2 {
		t.Fatalf("Chat produced %d tokens, want 2", len(got))
	}
	if !visionSess.prefillEmbedCalled {
		t.Fatal("chatMultimodal did not prefill token embeddings")
	}
	// 1 BOS + 3 placeholder tokens (one per soft token) — "hi"/boilerplate
	// chars are absent from the fixture vocab and vanish, proving the
	// placeholder count survives tokenisation exactly, independent of noise.
	if len(visionSess.embedIDs) != 4 {
		t.Fatalf("prefilled %d ids, want 4 (1 BOS + 3 placeholder tokens)", len(visionSess.embedIDs))
	}
}

// chatMultimodalErrorCase names one failure branch of chatMultimodal (Bad/
// Ugly variants are folded into this table — the function has more distinct
// failure branches than a bare Good/Bad/Ugly triplet can name, and each one
// guards a real "fail loud rather than answer against a corrupted prefill"
// invariant from the vision.go doc comments).
type chatMultimodalErrorCase struct {
	name    string
	vtm     *fakeVisionTokenModel
	wantErr string
}

// TestChatMultimodal_Errors pins every distinct failure branch chatMultimodal
// guards: each must fail loud with its own named error rather than silently
// answering against a corrupted prefill.
func TestChatMultimodal_Errors(t *testing.T) {
	cases := []chatMultimodalErrorCase{
		{
			name: "ProjectImageFails",
			vtm: &fakeVisionTokenModel{
				accepts:    true,
				projectErr: core.NewError("decode PNG failed"),
			},
			wantErr: "project image",
		},
		{
			name: "ZeroSoftTokens",
			vtm: &fakeVisionTokenModel{
				accepts:           true,
				projectSoftTokens: 0,
			},
			wantErr: "image produced no soft tokens",
		},
		{
			name: "EmptyPlaceholderBlock",
			vtm: &fakeVisionTokenModel{
				accepts:           true,
				projectSoftTokens: 2,
				placeholderBlock:  "",
			},
			wantErr: "declares no image placeholder tokens",
		},
		{
			name: "PlaceholderCountMismatch",
			vtm: &fakeVisionTokenModel{
				accepts:           true,
				projectSoftTokens: 5, // wantPlaceholders=5, but the block only tokenises to 3 "z"s
				placeholderBlock:  "zzz",
				placeholderID:     42,
			},
			wantErr: "image placeholders",
		},
		{
			name: "TokenEmbeddingsWithFeaturesFails",
			vtm: &fakeVisionTokenModel{
				accepts:           true,
				projectSoftTokens: 3,
				placeholderBlock:  "zzz",
				placeholderID:     42,
				embedErr:          core.NewError("splice failed"),
			},
			wantErr: "splice multimodal features",
		},
		{
			name: "SessionOpenFails",
			vtm: &fakeVisionTokenModel{
				accepts:           true,
				projectSoftTokens: 3,
				placeholderBlock:  "zzz",
				placeholderID:     42,
				embedRows:         [][]byte{{0}, {1}, {2}, {3}},
				fakeTokenModel:    fakeTokenModel{openErr: core.NewError("engine offline")},
			},
			wantErr: "",
		},
		{
			name: "PrefillTokenEmbeddingsFails",
			vtm: &fakeVisionTokenModel{
				accepts:           true,
				projectSoftTokens: 3,
				placeholderBlock:  "zzz",
				placeholderID:     42,
				embedRows:         [][]byte{{0}, {1}, {2}, {3}},
				session:           &fakeVisionSession{prefillEmbedErr: core.NewError("device OOM")},
			},
			wantErr: "prefill image embeddings",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tok := newFixtureTokenizer(t)
			if tc.vtm.session == nil {
				tc.vtm.session = &fakeVisionSession{}
			}
			m := NewTextModel(tc.vtm, tok, "gemma-test", inference.ModelInfo{}, 4096)
			messages := []inference.Message{{Role: "user", Content: "hi", Images: [][]byte{{1, 2, 3}}}}
			for range m.Chat(context.Background(), messages) {
				t.Fatal("expected no tokens from a failing multimodal turn")
			}
			r := m.Err()
			if r.OK {
				t.Fatal("want a failure Result")
			}
			if tc.wantErr != "" && !core.Contains(r.Error(), tc.wantErr) {
				t.Fatalf("Err() = %q, want it to contain %q", r.Error(), tc.wantErr)
			}
		})
	}
}

// chatMLVisionTokenModel is a fakeVisionTokenModel that ALSO declares a ChatML
// chat dialect (engine.ChatTemplateDeclarer) — a stand-in for a non-gemma
// vision family (a qwen-VL shape) riding the SAME shared multimodal path. It
// exists to prove the multimodal turn is framed in the model's DECLARED dialect
// rather than gemma's, exactly as the text-only Chat path already is.
type chatMLVisionTokenModel struct {
	fakeVisionTokenModel
}

func (chatMLVisionTokenModel) DeclaredChatTemplate() (ChatTemplate, bool) {
	return ChatTemplate{
		Open: "<|im_start|>", Close: "<|im_end|>",
		UserRole: "user", AssistantRole: "assistant", SystemRole: "system",
	}, true
}

// TestChatMultimodal_ArchNeutral_NonGemmaFraming pins arch-neutrality on the
// shared multimodal path: a vision model that DECLARES a ChatML template must
// have its image turn framed in ChatML, not gemma. The prefilled ids carry the
// declared <|im_start|> marker (a real ChatML added-token id in this fixture),
// which is absent when the path frames the turn in gemma's dialect and the
// gemma markers vanish against a ChatML vocab. This guards the pre-tag question:
// composed/qwen/hip vision checkpoints ride this code and must follow their own
// declaration, not a baked-in gemma fallback.
func TestChatMultimodal_ArchNeutral_NonGemmaFraming(t *testing.T) {
	tok := newChatMLFixtureTokenizer(t)
	imStart, ok := tok.TokenID("<|im_start|>")
	if !ok {
		t.Fatal("ChatML fixture tokenizer is missing <|im_start|>")
	}
	visionSess := &fakeVisionSession{fakeSession: fakeSession{genIDs: []int32{10, 11}}}
	vtm := &chatMLVisionTokenModel{fakeVisionTokenModel{
		accepts:           true,
		placeholderID:     42, // the ChatML fixture tokenizer's "z" vocab entry
		placeholderBlock:  "zzz",
		projectFeatures:   []byte{9, 9},
		projectSoftTokens: 3,
		embedRows:         [][]byte{{0}, {1}, {2}, {3}},
		session:           visionSess,
	}}
	m := NewTextModel(vtm, tok, "chatml-vision-fake", inference.ModelInfo{}, 4096)
	messages := []inference.Message{{Role: "user", Content: "hi", Images: [][]byte{{1, 2, 3}}}}

	for range m.Chat(context.Background(), messages, inference.WithMaxTokens(2)) {
	}
	if r := m.Err(); !r.OK {
		t.Fatalf("ChatML vision turn failed: %+v", r)
	}
	if !visionSess.prefillEmbedCalled {
		t.Fatal("multimodal path did not reach the token-embedding prefill")
	}
	if !tokenInSet(imStart, visionSess.embedIDs) {
		t.Fatalf("multimodal prompt ids %v missing the declared ChatML <|im_start|> id %d — "+
			"the shared multimodal path framed a non-gemma model in gemma's dialect", visionSess.embedIDs, imStart)
	}
}

// multimodalFake composes the vision, video and audio token-model capabilities
// over the shared fakeVisionTokenModel so chatMultimodal's video- and audio-
// splice paths are exercised. Video/audio placeholder blocks are runs of the
// fixture's "z" (id 42) so a block of N tokenises to exactly N placeholder ids;
// the image placeholder id stays 0 so the image-count check no-ops on the
// video-/audio-only turns these tests drive.
type multimodalFake struct {
	fakeVisionTokenModel
	videoPlaceholderID int32
	audioPlaceholderID int32
	audioSoftTokens    int
}

func (f *multimodalFake) VideoPlaceholderTokenID() int32     { return f.videoPlaceholderID }
func (f *multimodalFake) VideoPlaceholderBlock(n int) string { return strings.Repeat("z", n) }
func (f *multimodalFake) AcceptsAudioInput() bool            { return true }
func (f *multimodalFake) AudioPlaceholderTokenID() int32     { return f.audioPlaceholderID }
func (f *multimodalFake) AudioPlaceholderBlock(n int) string { return strings.Repeat("z", n) }
func (f *multimodalFake) ProjectAudio([]byte) ([]byte, int, error) {
	return []byte{7}, f.audioSoftTokens, nil
}

var (
	_ VideoTokenModel      = (*multimodalFake)(nil)
	_ AudioInputTokenModel = (*multimodalFake)(nil)
	_ VisionTokenModel     = (*multimodalFake)(nil)
)

// TestChatMultimodal_VideoTurn_Good walks the video-frame path: each frame
// projects through the SAME vision tower, splices at the video placeholder as
// one timestamped block per frame, and the templated placeholder run must
// survive tokenisation exactly before the embeddings prefill.
func TestChatMultimodal_VideoTurn_Good(t *testing.T) {
	tok := newFixtureTokenizer(t)
	visionSess := &fakeVisionSession{fakeSession: fakeSession{genIDs: []int32{10, 11}}}
	fake := &multimodalFake{
		fakeVisionTokenModel: fakeVisionTokenModel{
			accepts:           true,
			placeholderID:     0, // image-count check no-ops (video-only turn)
			projectFeatures:   []byte{9},
			projectSoftTokens: 2,
			embedRows:         [][]byte{{0}, {1}, {2}, {3}},
			session:           visionSess,
		},
		videoPlaceholderID: 42,
	}
	m := NewTextModel(fake, tok, "gemma-test", inference.ModelInfo{}, 4096)
	msgs := []inference.Message{{Role: "user", Content: "describe", Videos: [][]byte{{1}, {2}}}}

	var got int
	for range m.Chat(context.Background(), msgs, inference.WithMaxTokens(2)) {
		got++
	}
	if r := m.Err(); !r.OK {
		t.Fatalf("video turn failed: %+v", r)
	}
	if !visionSess.prefillEmbedCalled {
		t.Fatal("video path did not reach the token-embedding prefill")
	}
	if got != 2 {
		t.Fatalf("video turn produced %d tokens, want 2", got)
	}
}

// TestChatMultimodal_AudioTurn_Good walks the audio path: the audio head projects
// the attachment to soft-token features, the placeholder block FOLLOWS the turn
// text (the gemma4 audio-after-text convention), and the placeholder run is
// verified against the tokenised prompt before the embeddings prefill.
func TestChatMultimodal_AudioTurn_Good(t *testing.T) {
	tok := newFixtureTokenizer(t)
	visionSess := &fakeVisionSession{fakeSession: fakeSession{genIDs: []int32{10, 11}}}
	fake := &multimodalFake{
		fakeVisionTokenModel: fakeVisionTokenModel{
			accepts:           true,
			placeholderID:     0, // image-count check no-ops (audio-only turn)
			projectFeatures:   []byte{9},
			projectSoftTokens: 2,
			embedRows:         [][]byte{{0}, {1}, {2}},
			session:           visionSess,
		},
		audioPlaceholderID: 42,
		audioSoftTokens:    2,
	}
	m := NewTextModel(fake, tok, "gemma-test", inference.ModelInfo{}, 4096)
	msgs := []inference.Message{{Role: "user", Content: "transcribe", Audios: [][]byte{{1}}}}

	var got int
	for range m.Chat(context.Background(), msgs, inference.WithMaxTokens(2)) {
		got++
	}
	if r := m.Err(); !r.OK {
		t.Fatalf("audio turn failed: %+v", r)
	}
	if !visionSess.prefillEmbedCalled {
		t.Fatal("audio path did not reach the token-embedding prefill")
	}
	if got != 2 {
		t.Fatalf("audio turn produced %d tokens, want 2", got)
	}
}

// TestChatMultimodal_SessionNotVisionCapable pins the session-side probe: an
// engine session that does NOT implement VisionSession is rejected rather
// than silently falling back to a text-only prefill.
func TestChatMultimodal_SessionNotVisionCapable(t *testing.T) {
	tok := newFixtureTokenizer(t)
	// session is a plain *fakeSession (no PrefillTokenEmbeddings) — the
	// engine declares vision support, but the SESSION it hands out that turn
	// cannot serve a multimodal prefill.
	vtm := &fakeVisionTokenModel{
		accepts:           true,
		projectSoftTokens: 3,
		placeholderBlock:  "zzz",
		placeholderID:     42,
		embedRows:         [][]byte{{0}, {1}, {2}, {3}},
		session:           &fakeSession{},
	}
	m := NewTextModel(vtm, tok, "gemma-test", inference.ModelInfo{}, 4096)
	messages := []inference.Message{{Role: "user", Content: "hi", Images: [][]byte{{1, 2, 3}}}}
	for range m.Chat(context.Background(), messages) {
		t.Fatal("expected no tokens: the session cannot serve a multimodal prefill")
	}
	r := m.Err()
	if r.OK || !core.Contains(r.Error(), "does not support multimodal prefill") {
		t.Fatalf("Err() = %+v, want the multimodal-unsupported message", r)
	}
}

// --- unexported helpers ------------------------------------------------

// TestCountTokenID pins the placeholder-run counter: a configured non-zero id
// counts its occurrences; the sentinel zero id (no placeholder configured)
// always counts nothing.
func TestCountTokenID(t *testing.T) {
	ids := []int32{1, 42, 42, 2, 42}
	if got := countTokenID(ids, 42); got != 3 {
		t.Fatalf("countTokenID = %d, want 3", got)
	}
	if got := countTokenID(ids, 0); got != 0 {
		t.Fatalf("countTokenID(..., 0) = %d, want 0 (the no-placeholder sentinel)", got)
	}
	if got := countTokenID(nil, 42); got != 0 {
		t.Fatalf("countTokenID(nil, ...) = %d, want 0", got)
	}
}

// TestMessagesHaveImages pins the multimodal-path gate.
func TestMessagesHaveImages(t *testing.T) {
	if messagesHaveImages([]inference.Message{{Role: "user", Content: "hi"}}) {
		t.Fatal("messagesHaveImages = true for a text-only turn")
	}
	if !messagesHaveImages([]inference.Message{{Role: "user", Images: [][]byte{{1}}}}) {
		t.Fatal("messagesHaveImages = false for a turn carrying an image")
	}
	if messagesHaveImages(nil) {
		t.Fatal("messagesHaveImages(nil) = true, want false")
	}
}

// TestModelSampleParams pins the shared sampling-config builder: the text and
// multimodal decode paths must derive byte-identical sampling behaviour from
// the same GenerateConfig.
func TestModelSampleParams(t *testing.T) {
	cfg := inference.GenerateConfig{Temperature: 0.7, TopK: 40, TopP: 0.9, MinP: 0.05, RepeatPenalty: 1.1, SuppressTokens: []int32{5}}
	got := modelSampleParams(cfg)
	if got.Temperature != cfg.Temperature || got.TopK != cfg.TopK || got.TopP != cfg.TopP ||
		got.MinP != cfg.MinP || got.RepeatPenalty != cfg.RepeatPenalty || len(got.SuppressTokens) != 1 {
		t.Fatalf("modelSampleParams(%+v) = %+v, fields must carry over unchanged", cfg, got)
	}
}
