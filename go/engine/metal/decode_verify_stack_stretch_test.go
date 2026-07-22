// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"os"
	"testing"
)

// TestVerifyStackStretchInterleaveParity reproduces the re-engagement boundary
// at fixture speed: verify blocks, then a plain chained-step stretch on the
// same session, then more verify blocks — the lane's hiddens AND final cache
// bytes must stay byte-identical to the force-disabled session.
func TestVerifyStackStretchInterleaveParity(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	prevDisabled := verifyStackICBDisabled
	verifyStackICBDisabled = false // opt-in lane force-armed: this parity gates the mechanism itself
	t.Cleanup(func() { verifyStackICBDisabled = prevDisabled })
	pre := [][]int32{{1, 5, 3, 2, 7}, {4, 9, 6, 8, 2}}
	stretch := []int32{3, 7, 1, 9, 4, 6}
	post := [][]int32{{2, 6, 4, 1, 8}, {5, 1, 3, 7, 2}, {9, 2, 6, 4, 1}}

	run := func(disable bool) ([][][]byte, [][]byte) {
		t.Helper()
		verifyStackICBDisabledForTest = disable
		defer func() { verifyStackICBDisabledForTest = false }()
		sess := newVerifyStackHybridQuantFixture(t)
		var hiddens [][][]byte
		pos := 0
		verify := func(blocks [][]int32) {
			sess.state.verifyFoldSmallK = true
			defer func() { sess.state.verifyFoldSmallK = false }()
			for b, ids := range blocks {
				sess.pos = pos
				hs, ok, err := sess.verifyBatchedHiddens(ids)
				if err != nil {
					t.Fatalf("disable=%v block %d: %v", disable, b, err)
				}
				if !ok {
					t.Fatalf("disable=%v block %d: declined", disable, b)
				}
				cp := make([][]byte, len(hs))
				for i, h := range hs {
					cp[i] = append([]byte(nil), h...)
				}
				hiddens = append(hiddens, cp)
				pos += len(ids)
			}
		}
		verify(pre)
		sess.pos = pos
		for _, id := range stretch {
			if _, err := sess.stepID(id); err != nil {
				t.Fatalf("disable=%v stretch step: %v", disable, err)
			}
		}
		pos = sess.pos
		verify(post)
		return hiddens, verifyStackCacheBytes(t, sess, pos)
	}

	want, wantKV := run(true)
	base := verifyStackReplays.Load()
	got, gotKV := run(false)
	if verifyStackReplays.Load() == base {
		t.Fatal("stack lane never replayed — the parity exercises nothing")
	}
	for b := range want {
		for i := range want[b] {
			if !bytes.Equal(want[b][i], got[b][i]) {
				t.Fatalf("block %d row %d hiddens diverge after the stretch interleave", b, i)
			}
		}
	}
	for i := range wantKV {
		if !bytes.Equal(wantKV[i], gotKV[i]) {
			t.Fatalf("cache snapshot %d diverges after the stretch interleave", i)
		}
	}
}
