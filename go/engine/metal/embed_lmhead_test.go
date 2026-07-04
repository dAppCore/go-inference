// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"math"
	"os"
	"testing"
	"unsafe"
)

// argmaxBF16 returns the index of the largest of n bf16 logits.
func argmaxBF16(logits []byte, n int) int {
	best, bestV := 0, float32(math.Inf(-1))
	for i := 0; i < n; i++ {
		if v := bf16ToF32(logits[i*bf16Size], logits[i*bf16Size+1]); v > bestV {
			best, bestV = i, v
		}
	}
	return best
}

// TestEmbedTokens gates the input embedding: each token's gathered row times sqrt(hidden).
// Checked against the table read independently (proves the right row is gathered and the
// scale applied), plus identity at scale 1 and an out-of-range rejection.
func TestEmbedTokens(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const vocab, dModel = 50, 256
	scale := float32(math.Sqrt(float64(dModel)))
	tbl := make([]float32, vocab*dModel)
	for k := 0; k < vocab; k++ {
		for j := 0; j < dModel; j++ {
			tbl[k*dModel+j] = float32((k*7+j)%17-8) * 0.05 // distinct per (row,col)
		}
	}
	table := toBF16Bytes(tbl)
	ids := []int32{0, 7, 49, 23, 7}

	got, err := EmbedTokensBF16(table, ids, vocab, dModel, scale)
	if err != nil {
		t.Fatalf("EmbedTokensBF16: %v", err)
	}
	if len(got) != len(ids) {
		t.Fatalf("got %d embeddings, want %d", len(got), len(ids))
	}
	rowBytes := dModel * bf16Size
	for i, tok := range ids {
		row := table[int(tok)*rowBytes : (int(tok)+1)*rowBytes]
		for j := 0; j < dModel; j++ {
			want := f32ToBF16(bf16ToF32(row[j*bf16Size], row[j*bf16Size+1]) * scale)
			lo, hi := got[i][j*bf16Size], got[i][j*bf16Size+1]
			if lo != byte(want) || hi != byte(want>>8) {
				t.Fatalf("token %d elem %d: got bf16 %02x%02x, want %04x", tok, j, hi, lo, want)
			}
		}
	}

	// scale 1 → identity gather (embedding == table row).
	id1, err := EmbedTokensBF16(table, []int32{7}, vocab, dModel, 1)
	if err != nil {
		t.Fatalf("EmbedTokensBF16 scale1: %v", err)
	}
	eqBytes(t, "embed scale1 == table row", id1[0], table[7*rowBytes:8*rowBytes])

	if _, err := EmbedTokensBF16(table, []int32{int32(vocab)}, vocab, dModel, scale); err == nil {
		t.Fatal("expected EmbedTokensBF16 to reject an out-of-range token id")
	}
	t.Logf("embed: %d tokens gathered + scaled by √%d ≡ table rows; identity at scale 1; out-of-range rejected", len(ids), dModel)
}

// TestLMHead gates the output head. Without the cap it is exactly final-RMSNorm →
// output projection (the proven ops). With the cap it equals the soft-cap formula applied
// to those raw logits, the capped logits are bounded by ±softCap, and the argmax is
// unchanged (the cap is monotonic).
func TestLMHead(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, vocab = 256, 1000
	const eps, softCap = float32(1e-6), float32(30)
	mk := func(n, salt int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32((i*salt+11)%97-48) * 0.02
		}
		return s
	}
	hidden := toBF16Bytes(mk(dModel, 31))
	finalNormW := toBF16Bytes(mk(dModel, 7))
	outWeight := toBF16Bytes(mk(vocab*dModel, 53))

	// (a) no cap ≡ final-norm → projection.
	gotRaw, err := LMHeadBF16(hidden, finalNormW, outWeight, dModel, vocab, eps, 0)
	if err != nil {
		t.Fatalf("LMHeadBF16 no-cap: %v", err)
	}
	normed, err := RMSNormBF16(hidden, finalNormW, 1, dModel, eps)
	if err != nil {
		t.Fatalf("RMSNormBF16: %v", err)
	}
	refRaw, err := MatVecBF16(outWeight, normed, vocab, dModel)
	if err != nil {
		t.Fatalf("MatVecBF16: %v", err)
	}
	eqBytes(t, "LMHead no-cap == norm+proj", gotRaw, refRaw)

	// (b) cap ≡ softCap·tanh(raw/softCap), bounded, argmax preserved.
	gotCap, err := LMHeadBF16(hidden, finalNormW, outWeight, dModel, vocab, eps, softCap)
	if err != nil {
		t.Fatalf("LMHeadBF16 cap: %v", err)
	}
	wantCap := make([]byte, len(refRaw))
	for i := 0; i < vocab; i++ {
		v := bf16ToF32(refRaw[i*bf16Size], refRaw[i*bf16Size+1])
		h := f32ToBF16(softCap * float32(math.Tanh(float64(v/softCap))))
		wantCap[i*bf16Size] = byte(h)
		wantCap[i*bf16Size+1] = byte(h >> 8)
	}
	eqBytes(t, "LMHead cap == softcap formula", gotCap, wantCap)

	for i := 0; i < vocab; i++ {
		v := bf16ToF32(gotCap[i*bf16Size], gotCap[i*bf16Size+1])
		if v > softCap || v < -softCap {
			t.Fatalf("capped logit %d = %.4f exceeds ±%.0f", i, v, softCap)
		}
	}
	if a, b := argmaxBF16(gotRaw, vocab), argmaxBF16(gotCap, vocab); a != b {
		t.Fatalf("soft-cap changed the argmax: raw %d vs capped %d (must be monotonic)", a, b)
	}
	t.Logf("lm_head: no-cap ≡ final-norm→projection; cap ≡ softCap·tanh(·/softCap), bounded ±%.0f, argmax preserved", softCap)
}

func TestLMHeadBF16CachesResidentWeights(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const dModel, vocab = 64, 128
	hidden := toBF16Bytes(syntheticFloat32(dModel, 31))
	finalNormW := toBF16Bytes(syntheticFloat32(dModel, 7))
	outWeight := toBF16Bytes(syntheticFloat32(vocab*dModel, 53))

	if _, err := LMHeadBF16(hidden, finalNormW, outWeight, dModel, vocab, 1e-6, 0); err != nil {
		t.Fatalf("LMHeadBF16: %v", err)
	}

	key := func(b []byte) uintptr { return uintptr(unsafe.Pointer(&b[0])) }
	residentBufMu.Lock()
	got := len(residentBufs)
	_, hasNorm := residentBufs[key(finalNormW)]
	_, hasHead := residentBufs[key(outWeight)]
	residentBufMu.Unlock()
	if !hasNorm || !hasHead {
		t.Fatalf("LMHeadBF16 did not keep fixed weights resident (finalNorm=%v head=%v resident=%d want>=2)", hasNorm, hasHead, got)
	}
}

func TestLMHeadBF16AllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, vocab = 64, 128
	hidden := toBF16Bytes(syntheticFloat32(dModel, 31))
	finalNormW := toBF16Bytes(syntheticFloat32(dModel, 7))
	outWeight := toBF16Bytes(syntheticFloat32(vocab*dModel, 53))
	if _, err := LMHeadBF16(hidden, finalNormW, outWeight, dModel, vocab, 1e-6, 0); err != nil {
		t.Fatalf("LMHeadBF16 warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := LMHeadBF16(hidden, finalNormW, outWeight, dModel, vocab, 1e-6, 0); err != nil {
			t.Fatalf("LMHeadBF16: %v", err)
		}
	})
	if allocs > 35 {
		t.Fatalf("LMHeadBF16 allocations = %.0f, want <= 35", allocs)
	}
}

func TestLMHeadBF16IntoReusesOutputBackingAndBypassesScratchOutput(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const dModel, vocab = 64, 128
	hidden := toBF16Bytes(syntheticFloat32(dModel, 31))
	finalNormW := toBF16Bytes(syntheticFloat32(dModel, 7))
	outWeight := toBF16Bytes(syntheticFloat32(vocab*dModel, 53))
	want, err := LMHeadBF16(hidden, finalNormW, outWeight, dModel, vocab, 1e-6, 0)
	if err != nil {
		t.Fatalf("LMHeadBF16 reference: %v", err)
	}
	out := bytes.Repeat([]byte{0xa5}, vocab*bf16Size)

	scratch, err := getQMVBF16Scratch(vocab, dModel)
	if err != nil {
		t.Fatalf("getQMVBF16Scratch: %v", err)
	}
	sentinel := bytes.Repeat([]byte{0x4c}, len(scratch.out.bytes))
	copy(scratch.out.bytes, sentinel)
	putQMVBF16Scratch(scratch)

	got, err := LMHeadBF16Into(out, hidden, finalNormW, outWeight, dModel, vocab, 1e-6, 0)
	if err != nil {
		t.Fatalf("LMHeadBF16Into: %v", err)
	}
	if len(got) != len(want) || &got[0] != &out[0] {
		t.Fatal("LMHeadBF16Into did not reuse caller-owned output backing")
	}
	eqBytes(t, "LMHeadBF16Into", got, want)

	scratch, err = getQMVBF16Scratch(vocab, dModel)
	if err != nil {
		t.Fatalf("getQMVBF16Scratch after call: %v", err)
	}
	defer putQMVBF16Scratch(scratch)
	if !bytes.Equal(scratch.out.bytes, sentinel) {
		t.Fatal("LMHeadBF16Into wrote through pooled scratch output instead of caller output")
	}
}
