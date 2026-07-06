// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math/rand"
	"sort"
	"testing"

	"dappco.re/go/inference/model"
)

func bf16FromF32s(vals []float32) []byte {
	out := make([]byte, len(vals)*bf16Size)
	for i, v := range vals {
		bits := f32ToBF16(v)
		out[i*bf16Size], out[i*bf16Size+1] = byte(bits), byte(bits>>8)
	}
	return out
}

// TestHostTopKCandidatesBF16 pins the selector against a sort-based reference
// on random logits: the returned ids are exactly the k largest (as a set —
// order is unspecified), values are the ORIGINAL bf16 bit patterns, and
// suppressed ids never appear.
func TestHostTopKCandidatesBF16(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	const vocab, k = 5000, 40
	f := make([]float32, vocab)
	for i := range f {
		f[i] = float32(rng.NormFloat64() * 4)
	}
	logits := bf16FromF32s(f)
	suppress := []int32{7, 4090, 12}

	var vals [headSampleTopKMaxK * bf16Size]byte
	var ids [headSampleTopKMaxK]int32
	gotVals, gotIDs := hostTopKCandidatesBF16(logits, vocab, k, suppress, vals[:], ids[:])
	if len(gotIDs) != k || len(gotVals) != k*bf16Size {
		t.Fatalf("got %d ids / %d val bytes, want %d / %d", len(gotIDs), len(gotVals), k, k*bf16Size)
	}

	// reference: decode the SAME bf16 values, sort ids by value desc, take k
	dec := make([]float32, vocab)
	for i := range vocab {
		dec[i] = bf16ToF32(logits[i*bf16Size], logits[i*bf16Size+1])
	}
	ref := make([]int32, 0, vocab)
	for i := range int32(vocab) {
		if !tokenSuppressed(int(i), suppress) {
			ref = append(ref, i)
		}
	}
	sort.SliceStable(ref, func(a, b int) bool { return dec[ref[a]] > dec[ref[b]] })
	kth := dec[ref[k-1]] // ties at the boundary make the exact id-set ambiguous — compare by value
	got := map[int32]bool{}
	for i, id := range gotIDs {
		if tokenSuppressed(int(id), suppress) {
			t.Fatalf("suppressed id %d selected", id)
		}
		if got[id] {
			t.Fatalf("duplicate id %d", id)
		}
		got[id] = true
		if dec[id] < kth {
			t.Fatalf("id %d value %f below the k-th largest %f", id, dec[id], kth)
		}
		if gotVals[i*bf16Size] != logits[id*bf16Size] || gotVals[i*bf16Size+1] != logits[id*bf16Size+1] {
			t.Fatalf("candidate %d value bytes differ from the original logits", i)
		}
	}
}

// TestHostTopKCandidatesBF16SmallVocab pins the k>vocab clamp: every
// unsuppressed id is returned once.
func TestHostTopKCandidatesBF16SmallVocab(t *testing.T) {
	logits := bf16FromF32s([]float32{1, 3, 2})
	var vals [headSampleTopKMaxK * bf16Size]byte
	var ids [headSampleTopKMaxK]int32
	_, gotIDs := hostTopKCandidatesBF16(logits, 3, 40, nil, vals[:], ids[:])
	if len(gotIDs) != 3 {
		t.Fatalf("got %d ids, want 3", len(gotIDs))
	}
}

// TestSampleHostTopKBF16 pins the sampler contract: same seed → same token,
// the token is inside the true top-k, and suppression holds.
func TestSampleHostTopKBF16(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	const vocab = 9000
	f := make([]float32, vocab)
	for i := range f {
		f[i] = float32(rng.NormFloat64())
	}
	f[123] = 30 // dominant unsuppressed candidate
	f[77] = 31  // dominant but suppressed
	logits := bf16FromF32s(f)
	params := model.SampleParams{Temperature: 0.8, TopK: 8, TopP: 0.95, SuppressTokens: []int32{77}}

	a, err := sampleHostTopKBF16(logits, vocab, model.NewSampler(9), params)
	if err != nil {
		t.Fatalf("sample: %v", err)
	}
	b, err := sampleHostTopKBF16(logits, vocab, model.NewSampler(9), params)
	if err != nil {
		t.Fatalf("sample: %v", err)
	}
	if a != b {
		t.Fatalf("same seed diverged: %d vs %d", a, b)
	}
	if a == 77 {
		t.Fatal("sampled a suppressed token")
	}
	if a != 123 {
		// with logit 30 vs N(0,1) noise, the softmax mass at temp 0.8 is
		// overwhelmingly on 123 — anything else means the selection broke
		t.Fatalf("sampled %d, want the dominant candidate 123", a)
	}
}

// TestHostTopKSamplePreferred pins the routing predicate: TopK 0/1 and tiny
// vocabs stay off the host lane; TopK>1 at real vocab prefers it.
func TestHostTopKSamplePreferred(t *testing.T) {
	if hostTopKSamplePreferred(model.SampleParams{TopK: 0}, 262144) {
		t.Fatal("TopK=0 must not prefer host select")
	}
	if hostTopKSamplePreferred(model.SampleParams{TopK: 1}, 262144) {
		t.Fatal("TopK=1 must not prefer host select")
	}
	if hostTopKSamplePreferred(model.SampleParams{TopK: 40}, 32) {
		t.Fatal("fixture vocab must not prefer host select")
	}
	if !hostTopKSamplePreferred(model.SampleParams{TopK: 40}, 262144) {
		t.Fatal("TopK=40 at 262k vocab must prefer host select")
	}
	if hostTopKSamplePreferred(model.SampleParams{TopK: headSampleTopKMaxK + 1}, 262144) {
		t.Fatal("TopK beyond the max stays on the generic host sampler")
	}
}
