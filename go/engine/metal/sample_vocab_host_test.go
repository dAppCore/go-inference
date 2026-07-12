// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/inference/model"
)

// sample_vocab_host_test.go pins the FULL-VOCAB host sampler arms of the session
// pick path — sampleVocabBF16 and its streaming / logits-rank callees. These only
// engage when vocab > headSampleTopKMaxK (64), so every existing sampled test on the
// vocab-64 MTP fixture routes to sampleSmallVocabBF16 instead and leaves the streaming
// (pure-temperature) and logits-rank (TopP/MinP without host-TopK) branches dark.
//
// The sampler is stochastic, but a single dominant logit makes the outcome
// deterministic: with one token at logit 40 and the rest at -10 the peak carries
// >1-1e-16 of the mass, so sampler.Draw() ∈ [0,1) selects it on every seed. Suppressing
// the peak and planting a second, lower peak lets the same trick pin the suppress-skip
// branches to the surviving peak. No GPU, no metallib — sampleVocabBF16 is pure host
// arithmetic over logit bytes.

const sampleHostVocab = 128 // > headSampleTopKMaxK ⇒ the full-vocab lane

// sharpLogits returns vocab bf16 logits with a single dominant peak at `peak`
// (logit 40) and every other id at -10.
func sharpLogits(peak int) []byte {
	f := make([]float32, sampleHostVocab)
	for i := range f {
		f[i] = -10
	}
	f[peak] = 40
	return toBF16Bytes(f)
}

// twoPeakLogits returns logits with a dominant peak at `hi` (logit 40) and a
// second, lower peak at `lo` (logit 38) — used to pin suppress-skip: suppress hi
// and lo must win.
func twoPeakLogits(hi, lo int) []byte {
	f := make([]float32, sampleHostVocab)
	for i := range f {
		f[i] = -10
	}
	f[hi] = 40
	f[lo] = 38
	return toBF16Bytes(f)
}

func TestSampleVocabBF16FullVocabHostPaths(t *testing.T) {
	const peak, second = 5, 71
	for _, tc := range []struct {
		name   string
		logits []byte
		params model.SampleParams
		want   int32
	}{
		// !rankFilter (pure temperature) ⇒ the vocab-order streaming lane.
		{"streamingNoSuppress", sharpLogits(peak), model.SampleParams{Temperature: 0.8}, peak},
		{"streamingSuppress", twoPeakLogits(peak, second), model.SampleParams{Temperature: 0.8, SuppressTokens: []int32{peak}}, second},
		// rankFilter via TopP<1 (no host-TopK, TopK==0) ⇒ the logits-rank prefix lane.
		{"topPNoSuppress", sharpLogits(peak), model.SampleParams{Temperature: 0.9, TopP: 0.9}, peak},
		{"topPSuppress", twoPeakLogits(peak, second), model.SampleParams{Temperature: 0.9, TopP: 0.9, SuppressTokens: []int32{peak}}, second},
		// rankFilter via MinP>0 ⇒ the logits-rank MinP keep lane.
		{"minPNoSuppress", sharpLogits(peak), model.SampleParams{Temperature: 0.9, MinP: 0.1}, peak},
		{"minPSuppress", twoPeakLogits(peak, second), model.SampleParams{Temperature: 0.9, MinP: 0.1, SuppressTokens: []int32{peak}}, second},
		// TopP AND MinP together ⇒ both keep filters chained on the rank prefix.
		{"topPMinPNoSuppress", sharpLogits(peak), model.SampleParams{Temperature: 0.9, TopP: 0.95, MinP: 0.05}, peak},
		// TopK>64 dodges host-TopK (which gates on TopK<=64), so a TopK+TopP combo
		// reaches the logits-rank TopK branch and its sampleOrderTopPKeepLogits keep.
		{"topKTopPNoSuppress", sharpLogits(peak), model.SampleParams{Temperature: 0.9, TopK: 100, TopP: 0.9}, peak},
		{"topKTopPMinPNoSuppress", sharpLogits(peak), model.SampleParams{Temperature: 0.9, TopK: 100, TopP: 0.9, MinP: 0.05}, peak},
		{"topKTopPSuppress", twoPeakLogits(peak, second), model.SampleParams{Temperature: 0.9, TopK: 100, TopP: 0.9, SuppressTokens: []int32{peak}}, second},
		{"topKNoSuppress", sharpLogits(peak), model.SampleParams{Temperature: 0.9, TopK: 100}, peak},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := &ArchSession{}
			// Two seeds: a dominant peak must win regardless of the draw.
			for _, seed := range []uint64{1, 999} {
				got, err := s.sampleVocabBF16(tc.logits, sampleHostVocab, model.NewSampler(seed), tc.params)
				if err != nil {
					t.Fatalf("seed %d: sampleVocabBF16: %v", seed, err)
				}
				if got != tc.want {
					t.Fatalf("seed %d: sampled %d, want dominant token %d", seed, got, tc.want)
				}
			}
		})
	}
}

func TestSampleVocabBF16FullVocabAllSuppressedErrors(t *testing.T) {
	suppress := make([]int32, sampleHostVocab)
	for i := range suppress {
		suppress[i] = int32(i)
	}
	s := &ArchSession{}
	if _, err := s.sampleVocabBF16(sharpLogits(5), sampleHostVocab, model.NewSampler(1), model.SampleParams{Temperature: 0.8, SuppressTokens: suppress}); err == nil {
		t.Fatal("all-suppressed vocab: expected error, got nil")
	}
}

func TestSampleVocabBF16FullVocabGreedyAndTopK1(t *testing.T) {
	const peak = 33
	s := &ArchSession{}
	// Temperature 0 (sampledGreedyParamsEligible) ⇒ argmax, no draw consumed.
	got, err := s.sampleVocabBF16(sharpLogits(peak), sampleHostVocab, model.NewSampler(1), model.SampleParams{})
	if err != nil {
		t.Fatalf("greedy sampleVocabBF16: %v", err)
	}
	if got != peak {
		t.Fatalf("greedy sampled %d, want argmax %d", got, peak)
	}
	// TopK==1 ⇒ argmax with a single draw consumed.
	got, err = s.sampleVocabBF16(sharpLogits(peak), sampleHostVocab, model.NewSampler(1), model.SampleParams{Temperature: 0.8, TopK: 1})
	if err != nil {
		t.Fatalf("topK1 sampleVocabBF16: %v", err)
	}
	if got != peak {
		t.Fatalf("topK1 sampled %d, want argmax %d", got, peak)
	}
}

// TestSampleVocabBF16FullVocabNucleusExcludesSuppressed pins the suppress path on a
// FLAT distribution (no dominant peak): the sampled token must never be a suppressed id.
func TestSampleVocabBF16FullVocabNucleusExcludesSuppressed(t *testing.T) {
	flat := make([]float32, sampleHostVocab)
	logits := toBF16Bytes(flat)
	suppress := []int32{0, 1, 2, 3, 4, 5, 6, 7}
	s := &ArchSession{}
	for _, seed := range []uint64{1, 2, 3, 7, 42, 99} {
		got, err := s.sampleVocabBF16(logits, sampleHostVocab, model.NewSampler(seed), model.SampleParams{Temperature: 1.0, SuppressTokens: suppress})
		if err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}
		if nativeTokenInSet(got, suppress) {
			t.Fatalf("seed %d: sampled suppressed token %d", seed, got)
		}
		if got < 0 || int(got) >= sampleHostVocab {
			t.Fatalf("seed %d: sampled %d out of vocab", seed, got)
		}
	}
}
