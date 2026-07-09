// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"testing"
)

// sdpaSingleCellParityCase runs one paged SDPA decode through both dispatch
// shapes — the single-cell P1-final fast path and the P1+P2 two-pass — on
// identical inputs, returning both outputs. wantEngage says whether the fast
// path is EXPECTED to take the fast-path arm (single non-empty page, one split
// window); engagement is asserted in both directions so the compare cannot go
// vacuous.
func sdpaSingleCellParityCase(t *testing.T, nHeads, nKVHeads, headDim int, pageLens []int, wantEngage bool) ([]byte, []byte) {
	t.Helper()
	q := toBF16Bytes(syntheticFloat32(nHeads*headDim, 7))
	keyPages := make([][]byte, len(pageLens))
	valuePages := make([][]byte, len(pageLens))
	for i, l := range pageLens {
		keyPages[i] = toBF16Bytes(syntheticFloat32(nKVHeads*l*headDim, 11+i))
		valuePages[i] = toBF16Bytes(syntheticFloat32(nKVHeads*l*headDim, 13+i))
	}
	scale := float32(1.0 / 16.0)

	run := func(fast bool) []byte {
		wasDisabled := sdpaSingleCellDisabled
		sdpaSingleCellDisabled = !fast
		defer func() { sdpaSingleCellDisabled = wasDisabled }()
		before := sdpaSingleCellDispatches.Load()

		out, err := SDPAPagedBF16(q, keyPages, valuePages, nHeads, nKVHeads, headDim, scale)
		if err != nil {
			t.Fatalf("SDPAPagedBF16 (fast=%v): %v", fast, err)
		}
		engaged := sdpaSingleCellDispatches.Load() > before
		if fast && wantEngage && !engaged {
			t.Fatal("single-cell fast path did not engage on a single-cell plan")
		}
		if fast && !wantEngage && engaged {
			t.Fatal("single-cell fast path engaged on a multi-cell plan")
		}
		if !fast && engaged {
			t.Fatal("single-cell fast path engaged while disabled — the A/B is vacuous")
		}
		return out
	}

	twoPass := run(false)
	fast := run(true)
	return twoPass, fast
}

// TestSdpaPaged_SingleCellP1Final_MatchesTwoPass gates the single-cell fast
// path (#340): with one split window of one page, pass 2's log-sum-exp merge
// of a single cell is an identity rescale, so pass 1 writing bf16(acc/S)
// directly must be BYTE-IDENTICAL to the two-pass output — same operands, same
// one division, same bf16 round. The sweep covers both gemma4 head dims and
// GQA, the split-rows boundary (256 engages, 257 must not), multi-page plans
// (must not engage), and an empty leading page (a legitimate cache state that
// still leaves one cell).
func TestSdpaPaged_SingleCellP1Final_MatchesTwoPass(t *testing.T) {
	requireNativeRuntime(t)
	if !gpuHasGeluKernel() {
		t.Skip("custom kernel library (lthn_kernels.metallib) not loaded — run `task metallib:kernels`")
	}
	if _, err := sdpaPagedP1FinalPipeline(); err != nil {
		t.Skipf("single-cell P1 variant unavailable: %v", err)
	}

	cases := []struct {
		name            string
		nHeads, nKV, hd int
		pageLens        []int
		wantEngage      bool
	}{
		{"one row", 8, 2, 256, []int{1}, true},
		{"short page", 8, 2, 256, []int{10}, true},
		{"gqa head 512", 16, 4, 512, []int{33}, true},
		{"split boundary 256", 8, 2, 256, []int{256}, true},
		{"split boundary 257", 8, 2, 256, []int{257}, false},
		{"two pages", 8, 2, 256, []int{10, 10}, false},
	}
	for _, c := range cases {
		twoPass, fast := sdpaSingleCellParityCase(t, c.nHeads, c.nKV, c.hd, c.pageLens, c.wantEngage)
		if !bytes.Equal(twoPass, fast) {
			for i := 0; i+1 < len(twoPass); i += 2 {
				if twoPass[i] != fast[i] || twoPass[i+1] != fast[i+1] {
					t.Logf("%s: first diff at elem %d: two-pass % x (%.9g) fast % x (%.9g)", c.name, i/2,
						twoPass[i:i+2], bf16ToF32(twoPass[i], twoPass[i+1]), fast[i:i+2], bf16ToF32(fast[i], fast[i+1]))
					break
				}
			}
			t.Fatalf("%s: single-cell fast path != two-pass (cosine=%.7f)", c.name, cosineBF16(fast, twoPass))
		}
	}
	t.Logf("single-cell P1-final matches the two-pass output byte-for-byte across %d shapes", len(cases))
}
