// SPDX-Licence-Identifier: EUPL-1.2

package needle

import (
	"slices"
	"testing"

	coreio "dappco.re/go/io"
)

func loadTestTokenizer(t *testing.T) *spmTokenizer {
	t.Helper()
	path := snapshotDir + "/tokenizer.model"
	if !coreio.Local.Exists(path) {
		t.Skip("needle tokenizer.model not present")
	}
	tok, err := loadSPM(path, DefaultConfig())
	if err != nil {
		t.Fatalf("loadSPM: %v", err)
	}
	return tok
}

// TestSPM_encode_QueryParity checks the BPE encoder reproduces the exact
// SentencePiece token ids for the reference query (the first 13 ids of the
// oracle's encoder input) — the tokenizer's ground-truth receipt.
func TestSPM_encode_QueryParity(t *testing.T) {
	tok := loadTestTokenizer(t)
	got := tok.encode(oracleQuery)
	want := oracleEncIDs[:13] // query tokens, before the <tools> separator
	if !slices.Equal(got, want) {
		t.Fatalf("encode(query) mismatch:\n got: %v\nwant: %v", got, want)
	}
}

// TestSPM_encode_ToolsParity checks the tools-JSON string tokenises to the ids
// after the <tools> separator, confirming independent per-string normalisation.
func TestSPM_encode_ToolsParity(t *testing.T) {
	tok := loadTestTokenizer(t)
	got := tok.encode(oracleTools)
	want := oracleEncIDs[14:] // everything after id 5 (<tools>)
	if !slices.Equal(got, want) {
		t.Fatalf("encode(tools) mismatch:\n got: %v\nwant: %v", got, want)
	}
}

// TestSPM_decode_RoundTrip confirms decoding the query's ids returns the original
// text (the leading dummy-prefix space stripped).
func TestSPM_decode_RoundTrip(t *testing.T) {
	tok := loadTestTokenizer(t)
	if got := tok.decode(oracleEncIDs[:13]); got != oracleQuery {
		t.Fatalf("decode round-trip: got %q, want %q", got, oracleQuery)
	}
}

// TestSPM_decode_StripsToolCallMarker confirms the user-defined <tool_call> token
// (id 4) decodes to its literal surface form, which Generate then trims.
func TestSPM_decode_StripsToolCallMarker(t *testing.T) {
	tok := loadTestTokenizer(t)
	if got := tok.decode([]int{4}); got != "<tool_call>" {
		t.Fatalf("decode(<tool_call>) = %q, want %q", got, "<tool_call>")
	}
}
