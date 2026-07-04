// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

// TestEmbedTokensQuant4BitFastPath proves the 4-bit nibble fast path + per-group affine hoist in
// EmbedTokensQuant is byte-identical to the general extractAffineCode path. Same code value, same
// (s·code+b)·scale fp order — so every output byte must match the inline general-path reference.
func TestEmbedTokensQuant4BitFastPath(t *testing.T) {
	const dModel, groupSize, bits, vocab = 256, 32, 4, 4
	groups := dModel / groupSize
	rowPacked := dModel * bits / 8
	rowSB := groups * bf16Size
	packed := make([]byte, vocab*rowPacked)
	scales := make([]byte, vocab*rowSB)
	biases := make([]byte, vocab*rowSB)
	for i := range packed {
		packed[i] = byte(i*37 + 11)
	}
	for i := range scales { // keep exponents modest so values stay finite (NaN would still match, but be tidy)
		scales[i] = byte(i*53 + 7)
		biases[i] = byte(i*29 + 3)
	}
	scale := float32(1.5)
	tokens := []int32{0, 1, 2, 3}
	got, err := EmbedTokensQuant(packed, scales, biases, tokens, vocab, dModel, groupSize, bits, scale)
	if err != nil {
		t.Fatalf("EmbedTokensQuant: %v", err)
	}
	for ti, tok := range tokens {
		pRow := packed[int(tok)*rowPacked : (int(tok)+1)*rowPacked]
		sRow := scales[int(tok)*rowSB : (int(tok)+1)*rowSB]
		bRow := biases[int(tok)*rowSB : (int(tok)+1)*rowSB]
		for c := 0; c < dModel; c++ {
			code := extractAffineCode(pRow, c*bits, bits)
			g := c / groupSize
			s := bf16ToF32(sRow[g*bf16Size], sRow[g*bf16Size+1])
			b := bf16ToF32(bRow[g*bf16Size], bRow[g*bf16Size+1])
			h := f32ToBF16((s*float32(code) + b) * scale)
			if got[ti][c*bf16Size] != byte(h) || got[ti][c*bf16Size+1] != byte(h>>8) {
				t.Fatalf("tok %d elem %d: fast (%d,%d) != general (%d,%d)", tok, c,
					got[ti][c*bf16Size], got[ti][c*bf16Size+1], byte(h), byte(h>>8))
			}
		}
	}
	t.Logf("✓ 4-bit fast path == general path over %d tokens × %d elems", len(tokens), dModel)
}
