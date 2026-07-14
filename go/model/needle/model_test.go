// SPDX-Licence-Identifier: EUPL-1.2

package needle

import (
	"math"
	"slices"
	"testing"

	coreio "dappco.re/go/io"
)

// snapshotDir is the local Needle checkpoint. Tests that need the real weights
// skip when it is absent (e.g. CI), so the suite stays green without the model
// while giving a real end-to-end receipt on a machine that has it.
const snapshotDir = "/Users/snider/.cache/huggingface/hub/models--Cactus-Compute--needle/snapshots/5f89b4307696d669c3df1d38ae057e6e1728b107"

// oracleQuery/oracleTools are the exact prompt the PyTorch reference was run on;
// the fixtures below are that run's output (see the lane report).
const (
	oracleQuery = "What is the weather in San Francisco?"
	oracleTools = `[{"name":"get_weather","description":"Get current weather for a city.","parameters":{"location":{"type":"string","description":"City name.","required":true}}}]`
)

var (
	oracleEncIDs = []int{4279, 743, 302, 1149, 362, 711, 327, 1295, 1075, 378, 275, 8047, 8105, 5, 356, 294, 264, 358, 8062, 1331, 265, 283, 264, 618, 407, 1149, 345, 289, 2082, 284, 318, 282, 506, 282, 298, 264, 315, 265, 283, 264, 2523, 417, 284, 301, 262, 312, 434}
	oracleGenIDs = []int{4, 356, 294, 264, 358, 8062, 1331, 265, 393, 282, 506, 264, 8074, 327, 1295, 1075, 378, 275, 8047, 503}
	oracleOutput = ` [{"name":"get_weather","arguments":{"location":"San Francisco"}}]`
)

func loadTestModel(t *testing.T) *Model {
	t.Helper()
	if !coreio.Local.Exists(snapshotDir + "/model.safetensors") {
		t.Skip("needle checkpoint not present at " + snapshotDir)
	}
	m, err := Load(snapshotDir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	return m
}

func close32(a, b, tol float32) bool { return float32(math.Abs(float64(a-b))) <= tol }

// TestModel_Generate_ToolCall is the headline receipt: from the real weights, the
// reference greedily generates the exact coherent function call the PyTorch
// oracle produced.
func TestModel_Generate_ToolCall(t *testing.T) {
	m := loadTestModel(t)
	got := m.Generate(oracleQuery, oracleTools, 64)
	if got != oracleOutput {
		t.Fatalf("Generate mismatch:\n got: %q\nwant: %q", got, oracleOutput)
	}
	t.Logf("VERBATIM GENERATION: %s", got)
}

// TestModel_generateIDs_TokenParity pins exact encoder-input and generated token
// ids against the oracle — a stricter check than the decoded string.
func TestModel_generateIDs_TokenParity(t *testing.T) {
	m := loadTestModel(t)
	enc, gen := m.generateIDs(oracleQuery, oracleTools, 64)
	if !slices.Equal(enc, oracleEncIDs) {
		t.Fatalf("encoder ids mismatch:\n got: %v\nwant: %v", enc, oracleEncIDs)
	}
	if !slices.Equal(gen, oracleGenIDs) {
		t.Fatalf("generated ids mismatch:\n got: %v\nwant: %v", gen, oracleGenIDs)
	}
}

// TestModel_encode_EncoderHidden checks the encoder pass numerically against the
// oracle's encoder output (first token, last token, and the whole-tensor sum),
// isolating encoder correctness from the decoder.
func TestModel_encode_EncoderHidden(t *testing.T) {
	m := loadTestModel(t)
	h := m.encode(oracleEncIDs)
	if len(h) != len(oracleEncIDs)*m.cfg.HiddenSize {
		t.Fatalf("encoder hidden len = %d, want %d", len(h), len(oracleEncIDs)*m.cfg.HiddenSize)
	}
	wantT0 := []float32{-1.44818, 0.02975, 0.43704, 0.51372, -1.47645}
	wantTlast := []float32{-0.62455, 0.66252, 0.34196, 0.43071, 0.47612}
	hidden := m.cfg.HiddenSize
	last := (len(oracleEncIDs) - 1) * hidden
	for i := range wantT0 {
		if !close32(h[i], wantT0[i], 0.02) {
			t.Errorf("enc_hid[0][%d] = %.5f, want %.5f", i, h[i], wantT0[i])
		}
		if !close32(h[last+i], wantTlast[i], 0.02) {
			t.Errorf("enc_hid[last][%d] = %.5f, want %.5f", i, h[last+i], wantTlast[i])
		}
	}
	var sum float32
	for _, v := range h {
		sum += v
	}
	if !close32(sum, -730.317, 2.0) {
		t.Errorf("encoder hidden sum = %.3f, want ~-730.317", sum)
	}
}

// TestModel_decode_CrossAttentionStep checks the decoder-with-cross-attention
// step: the decoder hidden (post-norm) after one step over [EOS], against the
// oracle. This exercises masked self-attn + cross-attn + gates end to end.
func TestModel_decode_CrossAttentionStep(t *testing.T) {
	m := loadTestModel(t)
	encoderHidden := m.encode(oracleEncIDs)
	dh := m.decode([]int{m.cfg.EosTokenID}, encoderHidden, len(oracleEncIDs))
	want := []float32{-0.01168, 0.01785, -0.14802, -0.12168, 0.00175}
	for i := range want {
		if !close32(dh[i], want[i], 0.02) {
			t.Errorf("dec_hid[step0][%d] = %.5f, want %.5f", i, dh[i], want[i])
		}
	}
	// First-token argmax must be <tool_call> (id 4).
	if got := argmax(m.logits(dh[:m.cfg.HiddenSize])); got != 4 {
		t.Errorf("first-token argmax = %d, want 4 (<tool_call>)", got)
	}
}
