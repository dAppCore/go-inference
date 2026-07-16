// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math/rand"
	"os"
	"testing"
	"time"
	"unsafe"

	"github.com/tmc/apple/metal"
)

// lthn_topk_reduce_test.go validates the reduction-shaped top-k pair (#23) against the shipping
// insertion-sort pair at the KERNEL level: same logits, same params, same draw → same picked token —
// across temperatures, top-p, repeat penalty with history, suppression, and a ragged tail vocab. It
// also re-runs the falsification measurement the host-preference rule rests on: the old pair's
// 17-33ms was measured per-CALL; here both pairs run in ONE command buffer each and report their GPU
// execution span, which is the number that matters inside the chained sampled tail.

// topkReduceHarness owns the shared device buffers for one vocab shape.
type topkReduceHarness struct {
	logits, valsOld, idxOld, valsNew, idxNew, out, params, suppress, history metal.MTLBuffer
	vocab, topK                                                              int
}

func newTopKReduceHarness(t testing.TB, vocab, topK int, logitsF32 []float32, suppress, history []int32) *topkReduceHarness {
	t.Helper()
	if err := ensureInit(); err != nil {
		t.Skipf("metal unavailable: %v", err)
	}
	lb := make([]byte, vocab*bf16Size)
	for i, v := range logitsF32 {
		h := f32ToBF16(v)
		lb[i*2] = byte(h)
		lb[i*2+1] = byte(h >> 8)
	}
	newBuf := func(n int) metal.MTLBuffer {
		return device.NewBufferWithLengthOptions(uint(n), metal.MTLResourceStorageModeShared)
	}
	fill := func(b metal.MTLBuffer, data []byte) {
		copy(unsafe.Slice((*byte)(b.Contents()), len(data)), data)
	}
	oldTiles := (vocab + bf16LogitsArgmaxRowsPerTile - 1) / bf16LogitsArgmaxRowsPerTile
	newTiles := topKReduceTileCount(vocab)
	h := &topkReduceHarness{
		logits:  newBuf(len(lb)),
		valsOld: newBuf(oldTiles * topK * 4),
		idxOld:  newBuf(oldTiles * topK * 4),
		valsNew: newBuf(newTiles * topK * 4),
		idxNew:  newBuf(newTiles * topK * 4),
		out:     newBuf(4),
		params:  newBuf(int(unsafe.Sizeof(topKSampleKernelParams{}))),
		vocab:   vocab,
		topK:    topK,
	}
	fill(h.logits, lb)
	if len(suppress) > 0 {
		h.suppress = newBuf(len(suppress) * 4)
		fill(h.suppress, unsafe.Slice((*byte)(unsafe.Pointer(&suppress[0])), len(suppress)*4))
	}
	if len(history) > 0 {
		h.history = newBuf(len(history) * 4)
		fill(h.history, unsafe.Slice((*byte)(unsafe.Pointer(&history[0])), len(history)*4))
	}
	return h
}

func (h *topkReduceHarness) run(t testing.TB, reduce bool, temp, topP, minP, draw, penalty float32, suppressCount, historyCount int) (int32, time.Duration) {
	t.Helper()
	*(*topKSampleKernelParams)(h.params.Contents()) = topKSampleKernelParams{
		n:           int32(map[bool]int{true: topKReduceTileCount(h.vocab), false: ((h.vocab + bf16LogitsArgmaxRowsPerTile - 1) / bf16LogitsArgmaxRowsPerTile) * h.topK}[reduce]),
		topK:        int32(h.topK),
		temperature: temp,
		topP:        topP,
		minP:        minP,
		draw:        draw,
	}
	*(*int32)(h.out.Contents()) = -7 // sentinel
	var gpu time.Duration
	var encErr error
	withAutoreleasePool(func() {
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		if reduce {
			if encErr = encBF16LogitsTopKReduceBF16(enc, h.logits, h.valsNew, h.idxNew, h.suppress, h.history, h.vocab, suppressCount, historyCount, h.topK, penalty, 0); encErr == nil {
				encErr = encTopKReduceMergeSampleF32(enc, h.valsNew, h.idxNew, h.out, h.params)
			}
		} else {
			if encErr = encBF16LogitsTopKTilesBF16(enc, h.logits, h.valsOld, h.idxOld, h.suppress, h.history, h.vocab, suppressCount, historyCount, h.topK, penalty, 0); encErr == nil {
				encErr = encTopKMergeSampleF32(enc, h.valsOld, h.idxOld, h.out, h.params)
			}
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		gpu = time.Duration(float64(cb.GPUEndTime()-cb.GPUStartTime()) * float64(time.Second))
	})
	if encErr != nil {
		t.Fatalf("encode (reduce=%v): %v", reduce, encErr)
	}
	return *(*int32)(h.out.Contents()), gpu
}

// TestTopKReducePickMatchesInsertionPair pins pick parity: over random real-vocab logits and a
// sweep of sampling shapes (greedy-ish cold temp, the gemma4 default shape, top-p tails, repeat
// penalty with history, suppression, ragged vocab), the reduction pair and the shipping insertion
// pair return the SAME token for the same draw.
func TestTopKReducePickMatchesInsertionPair(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("metal unavailable: %v", err)
	}
	if !topKReduceSampleUsable(64, 262144) {
		t.Fatal("reduce pipelines unavailable despite a fresh metallib — kernel name or library resolution broke")
	}
	type shape struct {
		name                  string
		temp, topP, minP, pen float32
		suppress, history     []int32
	}
	shapes := []shape{
		{name: "gemma4-defaults", temp: 1.0, topP: 0.95},
		{name: "cold", temp: 0.2},
		{name: "hot-topp", temp: 1.3, topP: 0.7},
		{name: "minp", temp: 1.0, minP: 0.05},
		{name: "penalty", temp: 1.0, topP: 0.95, pen: 1.3, history: []int32{17, 99, 1044, 200000}},
		{name: "suppressed", temp: 0.9, suppress: []int32{int32(0 * 2654435761 % 262144), int32(3 * 2654435761 % 262144), 5, 6, 7, 8}},
	}
	for _, vocab := range []int{262144, 200005} { // aligned + ragged tail
		rng := rand.New(rand.NewSource(11))
		logits := make([]float32, vocab)
		for i := range logits {
			logits[i] = float32(rng.NormFloat64() * 3)
		}
		// A TIE-FREE top region: 100 bf16-exact distinct values (step 0.25 at this magnitude)
		// spread across the vocab, all above the noise ceiling. bf16 has ~65k distinct codes, so
		// at 262k a noisy k-boundary ALWAYS carries ties, and the two implementations order ties
		// differently — both validly (equal values, equal weights: the pick distribution is
		// identical; only which tied ID a given draw lands on differs). Pick-for-pick parity is
		// therefore asserted where it is actually guaranteed: a tie-free top-64 boundary.
		for i := 0; i < 100; i++ {
			logits[(i*2654435761)%vocab] = 60 - 0.25*float32(i)
		}
		for _, sh := range shapes {
			h := newTopKReduceHarness(t, vocab, 64, logits, sh.suppress, sh.history)
			pen := sh.pen
			if pen == 0 {
				pen = 1
			}
			for seed := 0; seed < 6; seed++ {
				draw := float32(seed) * 0.161 // spread across [0,1)
				oldTok, _ := h.run(t, false, sh.temp, sh.topP, sh.minP, draw, pen, len(sh.suppress), len(sh.history))
				newTok, _ := h.run(t, true, sh.temp, sh.topP, sh.minP, draw, pen, len(sh.suppress), len(sh.history))
				if oldTok != newTok {
					t.Fatalf("vocab %d %s draw %.3f: reduce pick %d != insertion pick %d", vocab, sh.name, draw, newTok, oldTok)
				}
				if newTok < 0 || int(newTok) >= vocab {
					t.Fatalf("vocab %d %s: pick %d out of range", vocab, sh.name, newTok)
				}
			}
		}
	}
}

// TestTopKReduceGPUSpanBeatsInsertion re-runs the falsification measurement in the shape that
// matters (ONE command buffer, GPU execution span): the reduction pair must run the 262k-vocab
// k=64 pick at least 5x faster than the insertion pair it replaces, and under 2ms absolute — the
// budget that makes an in-chain device pick worthwhile. Logs both spans for the record.
func TestTopKReduceGPUSpanBeatsInsertion(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("metal unavailable: %v", err)
	}
	if !topKReduceSampleUsable(64, 262144) {
		t.Fatal("reduce pipelines unavailable despite a fresh metallib — kernel name or library resolution broke")
	}
	const vocab = 262144
	rng := rand.New(rand.NewSource(7))
	logits := make([]float32, vocab)
	for i := range logits {
		logits[i] = float32(rng.NormFloat64() * 3)
	}
	h := newTopKReduceHarness(t, vocab, 64, logits, nil, nil)
	warm := func(reduce bool) time.Duration {
		var best time.Duration
		for i := 0; i < 5; i++ {
			_, gpu := h.run(t, reduce, 1.0, 0.95, 0, 0.33, 1, 0, 0)
			if i == 0 || gpu < best {
				best = gpu
			}
		}
		return best
	}
	oldSpan := warm(false)
	newSpan := warm(true)
	t.Logf("GPU span, one CB, 262k vocab k=64: insertion pair %s, reduction pair %s (×%.1f)",
		oldSpan.Round(time.Microsecond), newSpan.Round(time.Microsecond), float64(oldSpan)/float64(newSpan))
	if newSpan >= oldSpan/5 {
		t.Fatalf("reduction pair %s not ≥5x faster than insertion %s", newSpan, oldSpan)
	}
	if newSpan > 2*time.Millisecond {
		t.Fatalf("reduction pair %s exceeds the 2ms in-chain budget", newSpan)
	}
}
