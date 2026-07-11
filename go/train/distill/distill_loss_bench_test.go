// SPDX-Licence-Identifier: EUPL-1.2

package distill

import "testing"

// makeBenchLogits builds a batch×seq×vocab logit tensor filled with a
// deterministic, non-degenerate pattern (so the softmax has real spread).
func makeBenchLogits(batch, seq, vocab int, seed float32) Logits {
	out := make(Logits, batch)
	for b := 0; b < batch; b++ {
		row := make([][]float32, seq)
		for s := 0; s < seq; s++ {
			cell := make([]float32, vocab)
			for v := 0; v < vocab; v++ {
				cell[v] = seed + float32((v*7+s*3+b)%13)*0.1
			}
			row[s] = cell
		}
		out[b] = row
	}
	return out
}

// BenchmarkBatchLoss exercises the per-token log-softmax + KL/soft-CE fold on
// a batch×seq×vocab tensor shaped like a distillation step (sized down so the
// bench runs fast). A warm-up call primes the vocab-sized sync.Pool scratch
// buffers so the measured loop reflects the intended steady state — every
// subsequent BatchLoss lifts pre-sized buffers off the pool rather than paying
// three vocab-sized makes per call.
func BenchmarkBatchLoss(b *testing.B) {
	teacher := makeBenchLogits(4, 16, 512, 0.5)
	student := makeBenchLogits(4, 16, 512, 0.3)
	cfg := Config{Loss: LossKL, Temperature: 2}

	if _, err := BatchLoss(teacher, student, nil, cfg); err != nil {
		b.Fatalf("warm-up BatchLoss: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := BatchLoss(teacher, student, nil, cfg); err != nil {
			b.Fatalf("BatchLoss: %v", err)
		}
	}
}
