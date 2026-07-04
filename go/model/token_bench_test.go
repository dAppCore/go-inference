// SPDX-Licence-Identifier: EUPL-1.2

package model

import "testing"

// The token-loop benches baseline the contract's generation overhead (AX-11): the
// per-token Embed/Head/append/seq-growth allocations the loop itself owns, over the
// deterministic counterModel (its Embed/Head are minimal makes, so the numbers reflect
// LOOP cost, not a backend's compute). The two paths are the comparison that matters:
// whole-sequence rebuilds the seq each step (O(n²) work, growing seq allocs) while the
// incremental session steps O(1) — the alloc shape of each is the baseline.

const (
	benchPromptLen = 8
	benchMaxNew    = 64
)

func benchPrompt(n int) []int32 {
	ids := make([]int32, n)
	for i := range ids {
		ids[i] = int32(i % 16)
	}
	return ids
}

// BenchmarkGenerate_WholeSeq — the fallback path: re-embeds the running sequence and
// rebuilds the KV cache each step. The growing-seq + per-step DecodeForward allocs.
func BenchmarkGenerate_WholeSeq(b *testing.B) {
	m := counterModel{vocab: 256000, dModel: 2048}
	prompt := benchPrompt(benchPromptLen)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Generate(m, prompt, benchMaxNew, -1); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkGenerate_Stepwise — the incremental path: a persistent-cache session stepped
// one token at a time, O(1)/token. The per-token Embed + Head + append allocs only (no
// seq regrowth), the alloc floor of the contract loop a real backend builds on.
func BenchmarkGenerate_Stepwise(b *testing.B) {
	m := sessionCounterModel{counterModel: counterModel{vocab: 256000, dModel: 2048}}
	prompt := benchPrompt(benchPromptLen)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Generate(m, prompt, benchMaxNew, -1); err != nil {
			b.Fatal(err)
		}
	}
}

type benchDirectGenerateStepper struct{}

func (benchDirectGenerateStepper) Step([]byte) ([]byte, error) {
	return nil, nil
}

func (benchDirectGenerateStepper) Generate(_ []int32, maxNew, _ int) ([]int32, error) {
	gen := make([]int32, maxNew)
	for i := range gen {
		gen[i] = int32((i + 1) % 16)
	}
	return gen, nil
}

type benchDirectGenerateModel struct{ counterModel }

func (benchDirectGenerateModel) OpenSession() (DecodeStepper, error) {
	return benchDirectGenerateStepper{}, nil
}

// BenchmarkGenerate_DirectSessionGenerate is the optional engine fast path:
// a session can generate greedily itself, so the shared contract avoids the
// generic Step+Head logits loop.
func BenchmarkGenerate_DirectSessionGenerate(b *testing.B) {
	m := benchDirectGenerateModel{counterModel: counterModel{vocab: 256000, dModel: 2048}}
	prompt := benchPrompt(benchPromptLen)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Generate(m, prompt, benchMaxNew, -1); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkEmbed — the per-token input bookend in isolation: one dModel-sized bf16 make.
// The smallest repeated allocation in the generation loop.
func BenchmarkEmbed(b *testing.B) {
	m := counterModel{vocab: 256000, dModel: 2048}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := m.Embed(int32(i % m.vocab)); err != nil {
			b.Fatal(err)
		}
	}
}
