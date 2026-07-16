// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"bytes"
	"testing"
)

// mkVisionComposedModel loads a full composed hybrid (mkHybridCheckpoint's 4-layer text stack) PAIRED with
// a synthetic vision tower (addVisionTensors) through the real LoadComposed path, so token-model-level
// tests exercise the same object the loader produces rather than a hand-assembled stand-in. imageTokenID
// is stamped into config.json's image_token_id.
func mkVisionComposedModel(t *testing.T, imageTokenID int) *ComposedModel {
	t.Helper()
	ts, _ := mkHybridCheckpoint()
	addVisionTensors(ts, 1, 2, 8, true)
	cfg := []byte(`{"hidden_size":8,"num_hidden_layers":4,"intermediate_size":16,"num_attention_heads":4,
		"num_key_value_heads":2,"head_dim":8,"vocab_size":32,"rms_norm_eps":1e-5,"rope_theta":1000000,
		"partial_rotary_factor":0.5,"full_attention_interval":2,"image_token_id":` + itoa(imageTokenID) + `,
		"vision_config":{"patch_size":2}}`)
	m, err := LoadComposed(ts, cfg)
	if err != nil {
		t.Fatalf("LoadComposed: %v", err)
	}
	if m.Vision == nil {
		t.Fatal("mkVisionComposedModel: loaded model has no vision tower")
	}
	return m
}

func TestComposedTokenModelAcceptsImageInput_Good(t *testing.T) {
	tm := NewTokenModel(mkVisionComposedModel(t, 999))
	if !tm.AcceptsImageInput() {
		t.Fatal("AcceptsImageInput() = false, want true for a vision-carrying model")
	}
}

func TestComposedTokenModelAcceptsImageInput_Bad(t *testing.T) {
	tm := NewTokenModel(mkComposedModel(2, 8, 32, 16)) // text-only fixture, no Vision
	if tm.AcceptsImageInput() {
		t.Fatal("AcceptsImageInput() = true, want false for a text-only model")
	}
	var nilTM *ComposedTokenModel
	if nilTM.AcceptsImageInput() {
		t.Fatal("AcceptsImageInput() on a nil *ComposedTokenModel = true, want false")
	}
}

func TestComposedTokenModelImagePlaceholderTokenID_Good(t *testing.T) {
	tm := NewTokenModel(mkVisionComposedModel(t, 30))
	if got := tm.ImagePlaceholderTokenID(); got != 30 {
		t.Fatalf("ImagePlaceholderTokenID() = %d, want 30", got)
	}
}

func TestComposedTokenModelImagePlaceholderTokenID_Bad(t *testing.T) {
	tm := NewTokenModel(mkComposedModel(1, 8, 32, 16))
	if got := tm.ImagePlaceholderTokenID(); got != 0 {
		t.Fatalf("ImagePlaceholderTokenID() = %d, want 0 for a text-only model", got)
	}
}

func TestComposedTokenModelImagePlaceholderBlock_Good(t *testing.T) {
	tm := NewTokenModel(mkVisionComposedModel(t, 1))
	got := tm.ImagePlaceholderBlock(3)
	want := qwenVisionBeginToken + qwenVisionToken + qwenVisionToken + qwenVisionToken + qwenVisionEndToken
	if got != want {
		t.Fatalf("ImagePlaceholderBlock(3) = %q, want %q", got, want)
	}
}

func TestComposedTokenModelImagePlaceholderBlock_Bad(t *testing.T) {
	tm := NewTokenModel(mkVisionComposedModel(t, 1))
	if got := tm.ImagePlaceholderBlock(0); got != "" {
		t.Fatalf("ImagePlaceholderBlock(0) = %q, want empty", got)
	}
	textOnly := NewTokenModel(mkComposedModel(1, 8, 32, 16))
	if got := textOnly.ImagePlaceholderBlock(3); got != "" {
		t.Fatalf("ImagePlaceholderBlock(3) on a text-only model = %q, want empty", got)
	}
}

// TestComposedTokenModelProjectImage_Good is the done-gate's literal pipeline: image bytes -> patches ->
// tower forward -> projector -> features, checked by shape (softTokens x textHidden as bf16 bytes) and by
// the placeholder-count contract (softTokens matches the grid/merge arithmetic exactly).
func TestComposedTokenModelProjectImage_Good(t *testing.T) {
	m := mkVisionComposedModel(t, 1)
	tm := NewTokenModel(m)
	img := mkSyntheticPNG(t, 8, 8) // PatchSize=2 -> grid 4x4; MergeSize=2 -> softTokens=(4/2)*(4/2)=4

	features, softTokens, err := tm.ProjectImage(img)
	if err != nil {
		t.Fatalf("ProjectImage: %v", err)
	}
	if softTokens != 4 {
		t.Fatalf("softTokens = %d, want 4", softTokens)
	}
	wantLen := softTokens * m.D * 2 // bf16 bytes
	if len(features) != wantLen {
		t.Fatalf("len(features) = %d, want %d ([softTokens x textHidden] bf16)", len(features), wantLen)
	}
	block := tm.ImagePlaceholderBlock(softTokens)
	gotPlaceholders := 0
	for i := 0; i+len(qwenVisionToken) <= len(block); i++ {
		if block[i:i+len(qwenVisionToken)] == qwenVisionToken {
			gotPlaceholders++
			i += len(qwenVisionToken) - 1
		}
	}
	if gotPlaceholders != softTokens {
		t.Fatalf("ImagePlaceholderBlock rendered %d placeholder tokens, want softTokens=%d", gotPlaceholders, softTokens)
	}
}

func TestComposedTokenModelProjectImage_Bad(t *testing.T) {
	tm := NewTokenModel(mkComposedModel(1, 8, 32, 16))
	if _, _, err := tm.ProjectImage(mkSyntheticPNG(t, 8, 8)); err == nil {
		t.Fatal("ProjectImage: want an error for a text-only model, got nil")
	}
}

// TestComposedTokenModelTokenEmbeddingsWithFeatures_Good proves the splice contract: every image-token
// position gets the next projected feature row, in order; every other position gets the plain token
// embedding — the exact input a VisionSession.PrefillTokenEmbeddings-shaped call needs.
func TestComposedTokenModelTokenEmbeddingsWithFeatures_Good(t *testing.T) {
	const imageTokenID = 31 // must be a real (if special) vocabulary id — vocab is 32 in this fixture
	m := mkVisionComposedModel(t, imageTokenID)
	tm := NewTokenModel(m)
	img := mkSyntheticPNG(t, 8, 8)
	features, softTokens, err := tm.ProjectImage(img)
	if err != nil {
		t.Fatalf("ProjectImage: %v", err)
	}
	if softTokens != 4 {
		t.Fatalf("softTokens = %d, want 4", softTokens)
	}

	ids := []int32{1, imageTokenID, imageTokenID, imageTokenID, imageTokenID, 2}
	rows, err := tm.TokenEmbeddingsWithFeatures(ids, features, nil, nil)
	if err != nil {
		t.Fatalf("TokenEmbeddingsWithFeatures: %v", err)
	}
	if len(rows) != len(ids) {
		t.Fatalf("len(rows) = %d, want %d", len(rows), len(ids))
	}
	row := m.D * 2
	fi := 0
	for i, id := range ids {
		if id == imageTokenID {
			want := features[fi*row : (fi+1)*row]
			if !bytes.Equal(rows[i], want) {
				t.Fatalf("row %d (placeholder %d) != projected feature row %d", i, fi, fi)
			}
			fi++
			continue
		}
		want, err := tm.Embed(id)
		if err != nil {
			t.Fatalf("Embed(%d): %v", id, err)
		}
		if !bytes.Equal(rows[i], want) {
			t.Fatalf("row %d (text id %d) != plain Embed", i, id)
		}
	}
}

func TestComposedTokenModelTokenEmbeddingsWithFeatures_Bad(t *testing.T) {
	const imageTokenID = 31 // must be a real (if special) vocabulary id — vocab is 32 in this fixture
	tm := NewTokenModel(mkVisionComposedModel(t, imageTokenID))
	if _, err := tm.TokenEmbeddingsWithFeatures(nil, nil, nil, nil); err == nil {
		t.Fatal("want an error for empty ids, got nil")
	}
	if _, err := tm.TokenEmbeddingsWithFeatures([]int32{1, 2}, nil, []byte{1, 2, 3, 4}, nil); err == nil {
		t.Fatal("want an error for non-empty audioFeatures (composed carries no audio head), got nil")
	}
	if _, err := tm.TokenEmbeddingsWithFeatures([]int32{1, 2}, nil, nil, []byte{1, 2, 3, 4}); err == nil {
		t.Fatal("want an error for non-empty videoFeatures (composed carries no video head), got nil")
	}
	// 4 placeholders in the prompt but only 1 feature row projected — a mismatched splice.
	img := mkSyntheticPNG(t, 8, 8)
	features, softTokens, err := tm.ProjectImage(img)
	if err != nil {
		t.Fatalf("ProjectImage: %v", err)
	}
	if softTokens <= 1 {
		t.Fatalf("softTokens = %d, want > 1 for this mismatch to be meaningful", softTokens)
	}
	row := len(features) / softTokens
	ids := []int32{1, imageTokenID, imageTokenID, imageTokenID, imageTokenID, 2}
	if _, err := tm.TokenEmbeddingsWithFeatures(ids, features[:row], nil, nil); err == nil {
		t.Fatal("want an error for a feature-row/placeholder-slot count mismatch, got nil")
	}
}

// TestComposedStepperPrefillTokenEmbeddings_Good proves state equivalence: prefilling all-but-the-last
// token via PrefillTokenEmbeddings, then Step-ing the last, reproduces the SAME hidden as running the
// WHOLE sequence through DecodeForward — PrefillTokenEmbeddings advances every layer's state exactly as
// PrefillBatch/Step does, the property a multimodal turn's prefill depends on.
func TestComposedStepperPrefillTokenEmbeddings_Good(t *testing.T) {
	m := mkComposedModel(3, 8, 32, 16)
	tm := NewTokenModel(m)
	tokens := []int32{3, 1, 4, 1, 5}
	embs := make([][]byte, len(tokens))
	for i, tok := range tokens {
		e, err := tm.Embed(tok)
		if err != nil {
			t.Fatalf("Embed: %v", err)
		}
		embs[i] = e
	}
	whole, err := tm.DecodeForward(embs)
	if err != nil {
		t.Fatalf("DecodeForward: %v", err)
	}

	sess, err := tm.OpenSession()
	if err != nil {
		t.Fatalf("OpenSession: %v", err)
	}
	st, ok := sess.(*composedStepper)
	if !ok {
		t.Fatalf("OpenSession returned %T, want *composedStepper", sess)
	}
	if err := st.PrefillTokenEmbeddings(tokens[:len(tokens)-1], embs[:len(embs)-1]); err != nil {
		t.Fatalf("PrefillTokenEmbeddings: %v", err)
	}
	last, err := st.Step(embs[len(embs)-1])
	if err != nil {
		t.Fatalf("Step (after PrefillTokenEmbeddings): %v", err)
	}
	want := whole[len(whole)-1]
	if !bytes.Equal(last, want) {
		t.Fatal("PrefillTokenEmbeddings + Step != DecodeForward's last row — state did not advance identically")
	}
}

func TestComposedStepperPrefillTokenEmbeddings_Bad(t *testing.T) {
	m := mkComposedModel(1, 8, 32, 16)
	tm := NewTokenModel(m)
	sess, err := tm.OpenSession()
	if err != nil {
		t.Fatalf("OpenSession: %v", err)
	}
	st := sess.(*composedStepper)
	if err := st.PrefillTokenEmbeddings([]int32{1, 2, 3}, [][]byte{{0}}); err == nil {
		t.Fatal("PrefillTokenEmbeddings: want an error for mismatched ids/embeddings counts, got nil")
	}
}
