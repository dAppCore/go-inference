// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"
	"time"

	"dappco.re/go/inference/model"
)

// TestHeadSoftcapArgmaxIdentity proves the greedy decode is TOKEN-identical with the head's final-logit
// softcap applied vs skipped, and measures the skipped loop's cost. The softcap is monotonic, so it never
// changes the argmax — at each of N realistic positions this argmaxes head(hidden, apply) and head(hidden,
// skip) and asserts the same token. Token-identical, NOT byte-identical: a bf16 tie can flip, and skip keeps
// the higher RAW logit (the more faithful token). It decodes forward on the skip token (the production path).
func TestHeadSoftcapArgmaxIdentity(t *testing.T) {
	dir := os.Getenv("NATIVE_BENCH_DIR")
	if dir == "" {
		t.Skip("set NATIVE_BENCH_DIR to a real gemma4 checkpoint dir")
	}
	sess, err := LoadDir(dir, 256)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	defer func() { _ = sess.Close() }()
	prompt := []int32{2, 1841, 689, 573, 6182, 576}
	step := func(id int32) []byte {
		emb, e := sess.embed(id)
		if e != nil {
			t.Fatalf("embed: %v", e)
		}
		h, e := sess.StepWithID(id, emb)
		if e != nil {
			t.Fatalf("step: %v", e)
		}
		return h
	}
	var hidden []byte
	for _, id := range prompt {
		hidden = step(id)
	}
	var applyNs, skipNs time.Duration
	const N = 48
	for i := 0; i < N; i++ {
		t0 := time.Now()
		la, e := sess.head(hidden, false) // apply softcap
		applyNs += time.Since(t0)
		if e != nil {
			t.Fatalf("head apply: %v", e)
		}
		t1 := time.Now()
		ls, e := sess.head(hidden, true) // skip softcap (greedy path)
		skipNs += time.Since(t1)
		if e != nil {
			t.Fatalf("head skip: %v", e)
		}
		na, e := model.Greedy(la, sess.arch.Vocab)
		if e != nil {
			t.Fatalf("greedy apply: %v", e)
		}
		ns, e := model.Greedy(ls, sess.arch.Vocab)
		if e != nil {
			t.Fatalf("greedy skip: %v", e)
		}
		if na != ns {
			t.Fatalf("TOKEN-identity broken at position %d: apply→%d skip→%d", i, na, ns)
		}
		hidden = step(int32(ns))
	}
	t.Logf("✓ softcap skip TOKEN-identical over %d realistic positions", N)
	t.Logf("  head apply %v/call, skip %v/call → softcap host loop ≈ %v/token", applyNs/N, skipNs/N, (applyNs-skipNs)/N)
}
