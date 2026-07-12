// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

// audioAttentionWeightsFixture builds a self-consistent Conformer attention weight set for a
// [hidden, H heads x D head-dim, P positions] geometry — the same shape audio_attention_bench_test.go
// warms up, reused here for the correctness/error-path suite.
func audioAttentionWeightsFixture(hidden, H, D, P int) *AudioAttentionWeights {
	hd := H * D
	return &AudioAttentionWeights{
		QProj:         toBF16Bytes(syntheticFloat32(hd*hidden, 3)),
		KProj:         toBF16Bytes(syntheticFloat32(hd*hidden, 5)),
		VProj:         toBF16Bytes(syntheticFloat32(hd*hidden, 7)),
		Post:          toBF16Bytes(syntheticFloat32(hidden*hd, 9)),
		RelativeKProj: toBF16Bytes(syntheticFloat32(hd*hidden, 11)),
		QScalePerDim:  syntheticFloat32(D, 13),
		PosEmbed:      syntheticFloat32(P*hidden, 15),
		PosCount:      P,
	}
}

func audioAttentionCfgFixture(hidden, H, D, chunk, past, future int) AudioConfig {
	return AudioConfig{
		Hidden: hidden, NumHeads: H, HeadDim: D, ChunkSize: chunk,
		PastHorizon: past, FutureHorizon: future,
		KScale: 0.5, LogitCap: 50, InvalidLogit: -1e9,
	}
}

// TestAudioAttention_AudioAttention_Good cross-checks the bf16 entry point against
// AudioAttentionF32 fed the SAME (bf16-rounded) values — both drive audioAttentionCore, so
// they must agree up to the bf16 in/out rounding at the projections.
func TestAudioAttention_AudioAttention_Good(t *testing.T) {
	requireNativeRuntime(t)

	const hid, H, D, chunk, past, future, T = 16, 2, 8, 4, 2, 1, 6
	P := past + 1
	w := audioAttentionWeightsFixture(hid, H, D, P)
	cfg := audioAttentionCfgFixture(hid, H, D, chunk, past, future)
	xf := bf16Round(syntheticFloat32(T*hid, 17))
	x := toBF16Bytes(xf)

	got, err := AudioAttention(x, w, cfg)
	if err != nil {
		t.Fatalf("AudioAttention: %v", err)
	}
	wantF32, err := AudioAttentionF32(xf, w, cfg, nil)
	if err != nil {
		t.Fatalf("AudioAttentionF32 reference: %v", err)
	}
	relL2, cos := relL2Cos(bf16Floats(got), wantF32)
	if relL2 > 5e-2 || cos < 0.999 {
		t.Fatalf("AudioAttention rel-L2/cos vs AudioAttentionF32 = %.3e/%.6f", relL2, cos)
	}
}

// TestAudioAttention_AudioAttention_Bad exercises the length guards the projection chain
// (clippedMatRowsBF16 -> MatRowsBF16) surfaces for malformed input/weights.
func TestAudioAttention_AudioAttention_Bad(t *testing.T) {
	requireNativeRuntime(t)

	const hid, H, D, chunk, past, future, T = 16, 2, 8, 4, 2, 1, 6
	P := past + 1
	cfg := audioAttentionCfgFixture(hid, H, D, chunk, past, future)

	t.Run("x length not a multiple of hidden frames", func(t *testing.T) {
		w := audioAttentionWeightsFixture(hid, H, D, P)
		x := toBF16Bytes(syntheticFloat32(T*hid, 17))
		x = x[:len(x)-1] // truncate by one byte: T*hid*2-1 is not T'*hid*2 for any integer T'
		if _, err := AudioAttention(x, w, cfg); err == nil {
			t.Fatal("expected AudioAttention to reject a misaligned input length")
		}
	})

	t.Run("QProj weight length mismatch", func(t *testing.T) {
		w := audioAttentionWeightsFixture(hid, H, D, P)
		w.QProj = w.QProj[:len(w.QProj)-2]
		x := toBF16Bytes(syntheticFloat32(T*hid, 17))
		if _, err := AudioAttention(x, w, cfg); err == nil {
			t.Fatal("expected AudioAttention to reject a truncated QProj weight")
		}
	})
}

// TestAudioAttention_AudioAttention_Ugly pins the ChunkSize=1/PastHorizon=0/FutureHorizon=0
// degenerate window: the context per query collapses to exactly the query's own frame, so
// softmax over the single unmasked position is always 1 regardless of the attention logits —
// the whole Q/K/relative-position/soft-cap machinery becomes a no-op and AudioAttention must
// reduce to Post(VProj(x)) exactly.
func TestAudioAttention_AudioAttention_Ugly(t *testing.T) {
	requireNativeRuntime(t)

	const hid, H, D, T = 16, 2, 8, 5
	const chunk, past, future = 1, 0, 0
	P := past + 1
	w := audioAttentionWeightsFixture(hid, H, D, P)
	cfg := audioAttentionCfgFixture(hid, H, D, chunk, past, future)
	x := toBF16Bytes(syntheticFloat32(T*hid, 17))

	got, err := AudioAttention(x, w, cfg)
	if err != nil {
		t.Fatalf("AudioAttention: %v", err)
	}
	vProjected, err := MatRowsBF16(w.VProj, x, T, H*D, hid)
	if err != nil {
		t.Fatalf("VProj reference: %v", err)
	}
	want, err := MatRowsBF16(w.Post, vProjected, T, hid, H*D)
	if err != nil {
		t.Fatalf("Post reference: %v", err)
	}
	if cos := cosineBF16(got, want); cos < 0.999 {
		t.Fatalf("degenerate single-frame window: AudioAttention cosine=%.6f vs Post(VProj(x)), want ~1", cos)
	}
}

// TestAudioAttention_AudioAttentionF32_Good is the f32 entry point's own cross-check against
// AudioAttention at the same (bf16-rounded) values.
func TestAudioAttention_AudioAttentionF32_Good(t *testing.T) {
	requireNativeRuntime(t)

	const hid, H, D, chunk, past, future, T = 16, 2, 8, 4, 2, 1, 6
	P := past + 1
	w := audioAttentionWeightsFixture(hid, H, D, P)
	cfg := audioAttentionCfgFixture(hid, H, D, chunk, past, future)
	xf := bf16Round(syntheticFloat32(T*hid, 19))
	x := toBF16Bytes(xf)

	got, err := AudioAttentionF32(xf, w, cfg, nil)
	if err != nil {
		t.Fatalf("AudioAttentionF32: %v", err)
	}
	want, err := AudioAttention(x, w, cfg)
	if err != nil {
		t.Fatalf("AudioAttention reference: %v", err)
	}
	relL2, cos := relL2Cos(got, bf16Floats(want))
	if relL2 > 5e-2 || cos < 0.999 {
		t.Fatalf("AudioAttentionF32 rel-L2/cos vs AudioAttention = %.3e/%.6f", relL2, cos)
	}
}

// TestAudioAttention_AudioAttentionF32_Bad mirrors the bf16 entry point's length guards, this
// time through the fp32 mixed-dtype matmul (matF32MixedNT) the f32 path drives.
func TestAudioAttention_AudioAttentionF32_Bad(t *testing.T) {
	requireNativeRuntime(t)

	const hid, H, D, chunk, past, future, T = 16, 2, 8, 4, 2, 1, 6
	P := past + 1
	cfg := audioAttentionCfgFixture(hid, H, D, chunk, past, future)

	t.Run("x length not a multiple of hidden frames", func(t *testing.T) {
		w := audioAttentionWeightsFixture(hid, H, D, P)
		xf := syntheticFloat32(T*hid, 17)
		xf = xf[:len(xf)-1]
		if _, err := AudioAttentionF32(xf, w, cfg, nil); err == nil {
			t.Fatal("expected AudioAttentionF32 to reject a misaligned input length")
		}
	})

	t.Run("VProj weight length mismatch", func(t *testing.T) {
		w := audioAttentionWeightsFixture(hid, H, D, P)
		w.VProj = w.VProj[:len(w.VProj)-2]
		xf := syntheticFloat32(T*hid, 17)
		if _, err := AudioAttentionF32(xf, w, cfg, nil); err == nil {
			t.Fatal("expected AudioAttentionF32 to reject a truncated VProj weight")
		}
	})
}

// TestAudioAttention_AudioAttentionF32_Ugly is the fp32 sibling of the degenerate
// ChunkSize=1/PastHorizon=0/FutureHorizon=0 single-frame window collapse.
func TestAudioAttention_AudioAttentionF32_Ugly(t *testing.T) {
	requireNativeRuntime(t)

	const hid, H, D, T = 16, 2, 8, 5
	const chunk, past, future = 1, 0, 0
	P := past + 1
	w := audioAttentionWeightsFixture(hid, H, D, P)
	cfg := audioAttentionCfgFixture(hid, H, D, chunk, past, future)
	xf := syntheticFloat32(T*hid, 17)

	got, err := AudioAttentionF32(xf, w, cfg, nil)
	if err != nil {
		t.Fatalf("AudioAttentionF32: %v", err)
	}
	vProjected, err := clippedMatF32(xf, w.VProj, T, H*D, hid, w.VClip)
	if err != nil {
		t.Fatalf("VProj reference: %v", err)
	}
	want, err := clippedMatF32(vProjected, w.Post, T, hid, H*D, w.PostClip)
	if err != nil {
		t.Fatalf("Post reference: %v", err)
	}
	relL2, cos := relL2Cos(got, want)
	if relL2 > 1e-4 || cos < 0.9999 {
		t.Fatalf("degenerate single-frame window: AudioAttentionF32 rel-L2/cos = %.3e/%.6f vs Post(VProj(x)), want ~exact", relL2, cos)
	}
}

// TestAudioAttention_AudioContextSizeOf_Good pins audioContextSizeOf's arithmetic — the context
// window every chunk/block helper below sizes against.
func TestAudioAttention_AudioContextSizeOf_Good(t *testing.T) {
	cfg := AudioConfig{ChunkSize: 4, PastHorizon: 2, FutureHorizon: 1}
	if got, want := audioContextSizeOf(cfg), 7; got != want {
		t.Fatalf("audioContextSizeOf = %d, want %d", got, want)
	}
}

// TestAudioAttention_AudioBlockContextF32_Good proves the unfolded [nB,ctx,H,D] windows carry
// the right source frame at each context slot, and zero-pad past the sequence boundary.
func TestAudioAttention_AudioBlockContextF32_Good(t *testing.T) {
	const T, H, D, chunk, past, future = 4, 1, 2, 2, 1, 1
	ctx := chunk + past + future
	nB := (T + chunk - 1) / chunk
	x := make([]float32, T*H*D)
	for t := range T {
		x[t*D+0], x[t*D+1] = float32(t+1), float32(-(t + 1))
	}
	out := audioBlockContextF32(x, T, H, D, nB, chunk, past, future)
	if len(out) != nB*ctx*H*D {
		t.Fatalf("len(out) = %d, want %d", len(out), nB*ctx*H*D)
	}
	// block 0, ctx slot 0 is padded (original time -1); slot 1 is frame 0.
	if out[0] != 0 || out[1] != 0 {
		t.Fatalf("block 0 ctx 0 (padding) = (%v,%v), want zero", out[0], out[1])
	}
	got0, got1 := out[(1*H+0)*D+0], out[(1*H+0)*D+1]
	if got0 != 1 || got1 != -1 {
		t.Fatalf("block 0 ctx 1 (frame 0) = (%v,%v), want (1,-1)", got0, got1)
	}
}

// TestAudioAttention_AudioBlockedMask_Bad proves out-of-sequence query/key combinations are
// masked out — the "Bad" (disallowed) attention edges the softmax must never see.
func TestAudioAttention_AudioBlockedMask_Bad(t *testing.T) {
	const seqLen, chunk, past, future = 3, 2, 1, 1
	ctx := chunk + past + future
	nB := (seqLen + chunk - 1) / chunk
	mask := audioBlockedMask(seqLen, nB, chunk, ctx, past, future, nil)
	// block 1 (queries 2..3), the LAST block only has query index 0 in-sequence (seqLen=3);
	// query index 1 (t=3) is past the sequence end and must be masked at every key.
	base := (1*chunk + 1) * ctx
	for j := range ctx {
		if mask[base+j] {
			t.Fatalf("mask[block=1][query=1(out-of-seq)][%d] = true, want false (query beyond seqLen)", j)
		}
	}
}

// TestAudioAttention_AudioRelShiftF32_Ugly pins the Transformer-XL relative shift at ctx==P (no
// growth from the padding fold): each output row reads one step further into the next input row
// than the last, the diagonal-shift signature relShift exists to produce — the boundary case
// where the fold still introduces a padding zero despite ctx==P.
func TestAudioAttention_AudioRelShiftF32_Ugly(t *testing.T) {
	const H, nB, chunk, P = 1, 1, 2, 2
	ctx := P
	x := []float32{1, 2, 3, 4} // [chunk=2, P=2]: row 0 = [1,2], row 1 = [3,4]
	out := audioRelShiftF32(x, H, nB, chunk, P, ctx)
	if len(out) != H*nB*chunk*ctx {
		t.Fatalf("len(out) = %d, want %d", len(out), H*nB*chunk*ctx)
	}
	// padP = ctx+1 = 3. Folded stream (row-major, width padP, P real cols then one zero pad):
	// [1,2,0, 3,4,0]. out[i,c] = folded[i*ctx+c]: row 0 reads folded[0:2] = [1,2]; row 1 reads
	// folded[2:4] = [0,3] — the pad from row 0's tail leaks into row 1's first column, then row
	// 1's own first real value shifts into the second column. That leak is the mechanism.
	want := []float32{1, 2, 0, 3}
	for i, v := range want {
		if out[i] != v {
			t.Fatalf("audioRelShiftF32[%d] = %v, want %v (full out %v)", i, out[i], v, out)
		}
	}
}
