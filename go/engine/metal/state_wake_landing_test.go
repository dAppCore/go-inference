// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"context"
	"os"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/decode/tokenizer"
	"dappco.re/go/inference/kv"
)

// The -state lane's #1846 question (handover item: "adopt canonical landing IF
// the wobble shows"): a slept-and-woken session appends its next turn in a
// SMALL batch, while the stateless equivalent lands the same rows inside one
// whole prefill — different tile positions, the #1845/#1846 wobble class, and
// the q8 store amplifies it into token flips. The capture→restore round trip
// itself is byte-exact and is NOT the mechanism; the append is.

// TestStateWakeLanding_Q8 (synthetic, in-suite): pins the mechanism and the
// fix on the q8 fixture — a woken session's append lands canonically, making
// its cache byte-identical to an UNBROKEN session that appended the same turn
// (sleep/wake must not change the conversation), and shows the appended rows'
// batched-vs-canonical landing difference against the stateless whole.
func TestStateWakeLanding_Q8(t *testing.T) {
	requireNativeRuntime(t)
	kvQ8ICBForTest = true
	t.Cleanup(func() { kvQ8ICBForTest = false })

	prefix := []int32{1, 2, 3, 4, 5}
	suffix := []int32{6, 7, 8}

	// Unbroken continuity: prefill the prefix, append the suffix in place.
	unbroken := newKVQ8ICBFixtureLen(t, 256)
	defer unbroken.Close()
	if !unbroken.state.icb.hasKVQ8() {
		t.Fatal("fixture did not arm the q8 KV store")
	}
	if err := unbroken.PrefillTokens(prefix); err != nil {
		t.Fatalf("unbroken PrefillTokens: %v", err)
	}
	if err := unbroken.AppendTokens(suffix); err != nil {
		t.Fatalf("unbroken AppendTokens: %v", err)
	}
	unbrokenTokens, err := unbroken.GenerateFromCache(6, -1)
	if err != nil {
		t.Fatalf("unbroken GenerateFromCache: %v", err)
	}

	// Slept and woken: prefill, capture, restore into a fresh session, append.
	sleeper := newKVQ8ICBFixtureLen(t, 256)
	defer sleeper.Close()
	if err := sleeper.PrefillTokens(prefix); err != nil {
		t.Fatalf("sleeper PrefillTokens: %v", err)
	}
	snap, err := sleeper.CaptureKVWithOptions(kv.CaptureOptions{})
	if err != nil {
		t.Fatalf("CaptureKVWithOptions: %v", err)
	}
	woken := newKVQ8ICBFixtureLen(t, 256)
	defer woken.Close()
	if err := woken.RestoreFromKV(context.Background(), snap); err != nil {
		t.Fatalf("RestoreFromKV: %v", err)
	}
	if err := woken.AppendTokens(suffix); err != nil {
		t.Fatalf("woken AppendTokens: %v", err)
	}
	wokenTokens, err := woken.GenerateFromCache(6, -1)
	if err != nil {
		t.Fatalf("woken GenerateFromCache: %v", err)
	}

	// The -state contract: sleeping and waking must not change what the
	// conversation would have said.
	if len(unbrokenTokens) != len(wokenTokens) {
		t.Fatalf("token counts differ: unbroken=%d woken=%d", len(unbrokenTokens), len(wokenTokens))
	}
	for i := range unbrokenTokens {
		if unbrokenTokens[i] != wokenTokens[i] {
			t.Fatalf("woken diverges from unbroken at %d: %d vs %d (tokens %v vs %v)",
				i, wokenTokens[i], unbrokenTokens[i], wokenTokens, unbrokenTokens)
		}
	}

	// Diagnostic: the appended rows vs the stateless whole prefill — the
	// tile-position wobble the q8 store amplifies (logged, not gated; the
	// synthetic scale may absorb it).
	whole := newKVQ8ICBFixtureLen(t, 256)
	defer whole.Close()
	all := append(append([]int32(nil), prefix...), suffix...)
	if err := whole.PrefillTokens(all); err != nil {
		t.Fatalf("whole PrefillTokens: %v", err)
	}
	wholeTokens, err := whole.GenerateFromCache(6, -1)
	if err != nil {
		t.Fatalf("whole GenerateFromCache: %v", err)
	}
	t.Logf("stateless whole=%v · unbroken append=%v · woken append=%v", wholeTokens, unbrokenTokens, wokenTokens)
}

// TestProbeStateWakeParity (LTHN_PROBE_MODEL-gated) is the real-model decider
// for handover item 3: does the woken-append flow flip tokens against the
// stateless whole prefill on a q8-armed checkpoint, and does canonical landing
// on the woken session restore parity — the same instrument shape as
// TestProbePromptReuseParity, driven through the sleep/wake path.
//
//	LTHN_PROBE_MODEL=<snapshot dir> go test -tags metal_runtime \
//	  ./engine/metal/ -run TestProbeStateWakeParity -v
func TestProbeStateWakeParity(t *testing.T) {
	requireNativeRuntime(t)
	dir := os.Getenv("LTHN_PROBE_MODEL")
	if dir == "" {
		t.Skip("LTHN_PROBE_MODEL not set")
	}
	maxLen := 2048
	tm, err := LoadTokenModelDir(dir, maxLen)
	if err != nil {
		t.Fatalf("LoadTokenModelDir: %v", err)
	}
	ntm := tm.(*NativeTokenModel)
	defer ntm.Close()

	tok, err := tokenizer.LoadTokenizer(core.PathJoin(dir, "tokenizer.json"))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	prefix := tok.Encode("Write the integers from 1 to 800 separated by single spaces.")
	suffix := tok.Encode(" Output only the numbers, nothing else. Begin now:")
	gen := 48

	open := func() *ArchSession {
		stepper, oerr := ntm.OpenSession()
		if oerr != nil {
			t.Fatalf("OpenSession: %v", oerr)
		}
		s := stepper.(*ArchSession)
		t.Cleanup(func() { s.Close() })
		return s
	}

	// Stateless reference: one whole prefill.
	stateless := open()
	all := append(append([]int32(nil), prefix...), suffix...)
	if err := stateless.PrefillTokens(all); err != nil {
		t.Fatalf("stateless PrefillTokens: %v", err)
	}
	want, err := stateless.GenerateFromCache(gen, -1)
	if err != nil {
		t.Fatalf("stateless GenerateFromCache: %v", err)
	}

	wakeArm := func(label string, canonical bool) []int32 {
		src := open()
		if err := src.PrefillTokens(prefix); err != nil {
			t.Fatalf("%s PrefillTokens: %v", label, err)
		}
		snap, cerr := src.CaptureKVWithOptions(kv.CaptureOptions{})
		if cerr != nil {
			t.Fatalf("%s CaptureKVWithOptions: %v", label, cerr)
		}
		woken := open()
		if err := woken.RestoreFromKV(context.Background(), snap); err != nil {
			t.Fatalf("%s RestoreFromKV: %v", label, err)
		}
		if canonical {
			woken.SetReuseCanonicalLanding(true)
		}
		if err := woken.AppendTokens(suffix); err != nil {
			t.Fatalf("%s AppendTokens: %v", label, err)
		}
		got, gerr := woken.GenerateFromCache(gen, -1)
		if gerr != nil {
			t.Fatalf("%s GenerateFromCache: %v", label, gerr)
		}
		return got
	}

	diverge := func(a, b []int32) int {
		n := len(a)
		if len(b) < n {
			n = len(b)
		}
		for i := 0; i < n; i++ {
			if a[i] != b[i] {
				return i
			}
		}
		if len(a) != len(b) {
			return n
		}
		return -1
	}

	// Arm 1 (diagnostic): default batched append after wake vs stateless.
	batched := wakeArm("batched", false)
	if i := diverge(want, batched); i >= 0 {
		t.Logf("batched wake DIVERGES from stateless at token %d (the q8 wobble, item-3 confirmed)", i)
	} else {
		t.Logf("batched wake matched stateless over %d tokens on this prompt", gen)
	}

	// Arm 2 (the gate): canonical landing on the woken session → parity.
	canonical := wakeArm("canonical", true)
	if i := diverge(want, canonical); i >= 0 {
		t.Errorf("canonical wake diverges from stateless at token %d — the fix does not hold", i)
	} else {
		t.Logf("canonical wake: TOKEN PARITY with stateless over %d tokens", gen)
	}
}
