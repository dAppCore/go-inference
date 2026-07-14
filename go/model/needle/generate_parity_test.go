// SPDX-Licence-Identifier: EUPL-1.2

package needle

import "testing"

// TestGenerate_Generalises runs the reference on tool-use prompts beyond the
// primary fixture, each checked against its PyTorch-oracle output. Matching more
// than one prompt shows the port is the architecture, not a memorised fixture.
func TestGenerate_Generalises(t *testing.T) {
	m := loadTestModel(t)
	cases := []struct {
		name, query, tools, want string
	}{
		{
			name:  "send_email",
			query: "Send an email to john@example.com saying hello",
			tools: `[{"name":"send_email","description":"Send an email to a recipient.","parameters":{"to":{"type":"string","description":"The recipient email address.","required":true},"body":{"type":"string","description":"The email body text.","required":true}}}]`,
			want:  ` [{"name":"send_email","arguments":{"to":"john@example.com","body":"hello"}}]`,
		},
		{
			name:  "get_stock_price",
			query: "Get the current stock price of AAPL",
			tools: `[{"name":"get_stock_price","description":"Get the current stock price.","parameters":{"symbol":{"type":"string","description":"Ticker symbol.","required":true}}}]`,
			want:  ` [{"name":"get_stock_price","arguments":{"symbol":"AAPL"}}]`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := m.Generate(tc.query, tc.tools, 64); got != tc.want {
				t.Fatalf("Generate(%s):\n got: %q\nwant: %q", tc.name, got, tc.want)
			}
		})
	}
}

// TestGenerate_FirstTokenLogitParity is the gold oracle receipt: the reference's
// first-step logits (decoder over [EOS]) must match the PyTorch oracle in both
// the top-5 token ranking and the raw logit magnitudes (within f32-vs-bf16
// tolerance), not merely the argmax.
func TestGenerate_FirstTokenLogitParity(t *testing.T) {
	m := loadTestModel(t)
	encoderHidden := m.encode(oracleEncIDs)
	dh := m.decode([]int{m.cfg.EosTokenID}, encoderHidden, len(oracleEncIDs))
	logits := m.logits(dh[:m.cfg.HiddenSize])

	wantIDs := []int{4, 294, 8063, 302, 264}
	wantVals := []float32{1.22699, -14.80001, -16.13503, -17.17564, -17.1861}

	gotIDs, gotVals := top5(logits)
	for i := range wantIDs {
		if gotIDs[i] != wantIDs[i] {
			t.Errorf("top5 id[%d] = %d, want %d (full got %v)", i, gotIDs[i], wantIDs[i], gotIDs)
		}
		if !close32(gotVals[i], wantVals[i], 0.05) {
			t.Errorf("top5 logit[%d] = %.5f, want %.5f", i, gotVals[i], wantVals[i])
		}
	}
}

// top5 returns the five highest-scoring ids and their logits, descending.
func top5(v []float32) ([]int, []float32) {
	ids := make([]int, 5)
	vals := make([]float32, 5)
	for i := range 5 {
		vals[i] = -1e30
	}
	for id, x := range v {
		for r := range 5 {
			if x > vals[r] {
				copy(ids[r+1:], ids[r:4])
				copy(vals[r+1:], vals[r:4])
				ids[r] = id
				vals[r] = x
				break
			}
		}
	}
	return ids, vals
}
