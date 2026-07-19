// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/inference/kv"
	"dappco.re/go/inference/model"
)

// TestMetalBackend_LoadSpeculativePair_Good pins the compile-time contract:
// metalBackend satisfies inference.SpeculativePairBackend, so a caller can
// reach it through inference.Get("metal") + a type assertion without importing
// this package (train/tune's discovery seam).
func TestMetalBackend_LoadSpeculativePair_Good(t *testing.T) {
	var b inference.Backend = metalBackend{}
	if _, ok := b.(inference.SpeculativePairBackend); !ok {
		t.Fatal("metalBackend must implement inference.SpeculativePairBackend")
	}
}

// TestMetalBackend_LoadSpeculativePair_Bad pins the delegation: calling the
// method over two empty directories (no config.json — no GPU/metallib touched)
// reaches LoadSpeculativePair and returns ITS error rather than swallowing it
// or panicking.
func TestMetalBackend_LoadSpeculativePair_Bad(t *testing.T) {
	_, err := metalBackend{}.LoadSpeculativePair(t.TempDir(), t.TempDir(), 5)
	if err == nil {
		t.Fatal("LoadSpeculativePair over empty directories must fail, not silently succeed")
	}
}

// bridgeBF16 writes v as the two bf16 bytes the composed seam carries (small
// integers are exact, so the fake round-trips ids losslessly).
func bridgeBF16(v float32) (lo, hi byte) {
	h := uint16(math.Float32bits(v) >> 16)
	return byte(h), byte(h >> 8)
}

func bridgeBF16Val(lo, hi byte) float32 {
	return math.Float32frombits(uint32(uint16(lo)|uint16(hi)<<8) << 16)
}

// bridgePlainModel is a deterministic fake model.SessionModel with NO vision
// capability: Embed writes the id into dim 0, DecodeForward is identity, Head
// one-hots (id+1) mod vocab — the counter walk the model package's own fakes
// use, so bridge outputs are predictable.
type bridgePlainModel struct {
	vocab, dModel int
	prefillRows   *[][]byte // records what PrefillBatch was fed (nil = don't record)
}

func (m bridgePlainModel) Vocab() int { return m.vocab }

func (m bridgePlainModel) Embed(id int32) ([]byte, error) {
	emb := make([]byte, m.dModel*2)
	emb[0], emb[1] = bridgeBF16(float32(id))
	return emb, nil
}

func (m bridgePlainModel) DecodeForward(inputs [][]byte) ([][]byte, error) {
	return inputs, nil
}

func (m bridgePlainModel) Head(hidden []byte) ([]byte, error) {
	id := int(math.Round(float64(bridgeBF16Val(hidden[0], hidden[1]))))
	logits := make([]byte, m.vocab*2)
	target := ((id + 1) % m.vocab) * 2
	logits[target], logits[target+1] = bridgeBF16(1)
	return logits, nil
}

func (m bridgePlainModel) OpenSession() (model.DecodeStepper, error) {
	return &bridgeStepper{prefillRows: m.prefillRows}, nil
}

// bridgeStepper batch-prefills (recording the rows it was fed) and steps
// identity — the minimal BatchPrefillStepper the embeddings replay needs.
type bridgeStepper struct{ prefillRows *[][]byte }

func (st *bridgeStepper) Step(emb []byte) ([]byte, error) { return emb, nil }

func (st *bridgeStepper) PrefillBatch(embs [][]byte) ([]byte, error) {
	if st.prefillRows != nil {
		*st.prefillRows = append((*st.prefillRows)[:0], embs...)
	}
	return embs[len(embs)-1], nil
}

// bridgeVisionModel layers the engine.VisionTokenModel method set (BY SHAPE)
// over the plain fake — the composed model's exact bridging situation.
type bridgeVisionModel struct{ bridgePlainModel }

func (bridgeVisionModel) AcceptsImageInput() bool        { return true }
func (bridgeVisionModel) ImagePlaceholderTokenID() int32 { return 7 }
func (bridgeVisionModel) ImagePlaceholderBlock(softTokens int) string {
	block := "<img>"
	for range softTokens {
		block += "<tok>"
	}
	return block + "</img>"
}
func (m bridgeVisionModel) ProjectImage(image []byte) ([]byte, int, error) {
	return []byte{1, 2, 3, 4}, 2, nil
}
func (m bridgeVisionModel) TokenEmbeddingsWithFeatures(ids []int32, imageFeatures, audioFeatures, videoFeatures []byte) ([][]byte, error) {
	rows := make([][]byte, len(ids))
	for i, id := range ids {
		rows[i], _ = m.Embed(id)
	}
	return rows, nil
}

// TestComposedTextModel_VisionForwarding_Good pins the bridge's model half: a
// composed model implementing the vision method set BY SHAPE forwards through
// composedTextModel — capability, placeholder id/block, and projection all
// reach the engine unchanged.
func TestComposedTextModel_VisionForwarding_Good(t *testing.T) {
	m := &composedTextModel{sm: bridgeVisionModel{bridgePlainModel{vocab: 16, dModel: 1}}}
	if !m.AcceptsImageInput() {
		t.Fatal("vision-capable composed model must report AcceptsImageInput")
	}
	if got := m.ImagePlaceholderTokenID(); got != 7 {
		t.Fatalf("ImagePlaceholderTokenID = %d, want 7 (forwarded)", got)
	}
	if got := m.ImagePlaceholderBlock(2); got != "<img><tok><tok></img>" {
		t.Fatalf("ImagePlaceholderBlock = %q, want the forwarded block", got)
	}
	features, soft, err := m.ProjectImage([]byte{9})
	if err != nil || soft != 2 || len(features) != 4 {
		t.Fatalf("ProjectImage = (%v, %d, %v), want forwarded (4 bytes, 2, nil)", features, soft, err)
	}
	rows, err := m.TokenEmbeddingsWithFeatures([]int32{1, 2}, features, nil, nil)
	if err != nil || len(rows) != 2 {
		t.Fatalf("TokenEmbeddingsWithFeatures = (%d rows, %v), want 2 rows", len(rows), err)
	}
}

// TestComposedTextModel_VisionForwarding_Bad pins the text-only side: a
// composed model WITHOUT the vision shape answers false/zero/empty through the
// same forwards, and the projection entries refuse with an error instead of
// inventing features.
func TestComposedTextModel_VisionForwarding_Bad(t *testing.T) {
	m := &composedTextModel{sm: bridgePlainModel{vocab: 16, dModel: 1}}
	if m.AcceptsImageInput() {
		t.Fatal("text-only composed model must not report AcceptsImageInput")
	}
	if got := m.ImagePlaceholderTokenID(); got != 0 {
		t.Fatalf("ImagePlaceholderTokenID = %d, want 0 (counts as no placeholder)", got)
	}
	if got := m.ImagePlaceholderBlock(3); got != "" {
		t.Fatalf("ImagePlaceholderBlock = %q, want empty", got)
	}
	if _, _, err := m.ProjectImage([]byte{1}); err == nil {
		t.Fatal("ProjectImage without a vision tower must refuse")
	}
	if _, err := m.TokenEmbeddingsWithFeatures([]int32{1}, nil, nil, nil); err == nil {
		t.Fatal("TokenEmbeddingsWithFeatures without a vision tower must refuse")
	}
	var nilModel *composedTextModel
	if nilModel.AcceptsImageInput() {
		t.Fatal("nil bridge must answer false, not panic")
	}
}

// TestComposedEngineSession_PrefillTokenEmbeddings_Good pins the session half:
// stored rows replay through the composed batch prefill (NOT re-embedded ids —
// the rows are deliberately shifted so the generated walk proves which path
// ran), and Pos reports the multimodal prompt's length.
func TestComposedEngineSession_PrefillTokenEmbeddings_Good(t *testing.T) {
	var fed [][]byte
	sm := bridgePlainModel{vocab: 16, dModel: 1, prefillRows: &fed}
	s := &composedEngineSession{sm: sm, arch: "qwen3_5"}
	ids := []int32{0, 1, 2}
	// Rows encode id+5, unlike what Embed(id) would produce — if the generate
	// re-embedded ids the walk would start at 3; from the rows it starts at 8.
	rows := make([][]byte, len(ids))
	for i, id := range ids {
		row := make([]byte, 2)
		row[0], row[1] = bridgeBF16(float32(id) + 5)
		rows[i] = row
	}
	if err := s.PrefillTokenEmbeddings(ids, rows); err != nil {
		t.Fatalf("PrefillTokenEmbeddings: %v", err)
	}
	if got := s.Pos(); got != len(ids) {
		t.Fatalf("Pos = %d, want %d", got, len(ids))
	}
	got, err := s.GenerateFromCacheEach(2, -1, nil)
	if err != nil {
		t.Fatalf("GenerateFromCacheEach over rows: %v", err)
	}
	if want := []int32{8, 9}; len(got) != 2 || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("rows walk = %v, want %v (the shifted rows, not re-embedded ids)", got, want)
	}
	if len(fed) != len(rows) {
		t.Fatalf("PrefillBatch saw %d rows, want %d (the stored rows, one call)", len(fed), len(rows))
	}
}

// TestComposedEngineSession_PrefillTokenEmbeddings_Bad pins the refusals a
// multimodal turn imposes on the composed session: mismatched/empty prefill,
// continuity appends, -state capture, and the sleep lane all refuse while
// rows are held (an image turn is a stateless turn — rows are not
// token-replayable).
func TestComposedEngineSession_PrefillTokenEmbeddings_Bad(t *testing.T) {
	s := &composedEngineSession{sm: bridgePlainModel{vocab: 16, dModel: 1}, arch: "qwen3_5", numLayers: 1}
	if err := s.PrefillTokenEmbeddings([]int32{1, 2}, [][]byte{{0, 0}}); err == nil {
		t.Fatal("id/row count mismatch must refuse")
	}
	if err := s.PrefillTokenEmbeddings(nil, nil); err == nil {
		t.Fatal("empty multimodal prefill must refuse")
	}
	if err := s.PrefillTokenEmbeddings([]int32{1}, [][]byte{{0, 0}}); err != nil {
		t.Fatalf("valid prefill refused: %v", err)
	}
	if err := s.AppendTokens([]int32{3}); err == nil {
		t.Fatal("continuity append over multimodal rows must refuse")
	}
	if _, err := s.CaptureKVWithOptions(kv.CaptureOptions{}); err == nil {
		t.Fatal("-state capture over multimodal rows must refuse")
	}
	if err := s.RangeKVBlocks(4, kv.CaptureOptions{}, func(kv.Block) (bool, error) { return true, nil }); err == nil {
		t.Fatal("sleep-lane blocks over multimodal rows must refuse")
	}
}
