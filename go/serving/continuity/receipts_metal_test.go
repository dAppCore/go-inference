// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

// This file is the tiered-KV lane's measurement harness, not a gate. It drives
// the REAL shipped continuity.Manager on a real gemma-4 checkpoint via the
// metal engine and prints the receipt tables the design memo
// (docs/design-tiered-kv.md) points at: TTFT for re-prefill vs store-wake vs
// resident, and the RAM-store vs file-store I/O tax. It is opt-in (needs the
// Apple GPU + a downloaded snapshot) and never asserts timings — it reports.
//
//	GO_INFERENCE_SMOKE_MODEL=~/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/<hash> \
//	MLX_METALLIB_PATH=/Users/snider/Code/core/go-inference/build/dist/lib/mlx.metallib \
//	go test -tags metal_runtime -run TestTieredKVReceipts ./serving/continuity/ -v -timeout 20m
package continuity_test

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"dappco.re/go/inference"
	"dappco.re/go/inference/model/state"
	"dappco.re/go/inference/model/state/filestore"
	"dappco.re/go/inference/serving/continuity"

	_ "dappco.re/go/inference/engine/metal"  // registers the "metal" backend
	_ "dappco.re/go/inference/model/builtin" // registers the built-in arches
)

// reps is the sample count per measurement; the harness reports the min (the
// least-noisy floor) and mean.
const reps = 3

func TestTieredKVReceipts(t *testing.T) {
	modelPath := os.Getenv("GO_INFERENCE_SMOKE_MODEL")
	if modelPath == "" {
		t.Skip("set GO_INFERENCE_SMOKE_MODEL to a gemma-4 snapshot dir to run the tiered-KV receipt harness")
	}
	ctx := context.Background()

	res := inference.LoadModel(modelPath, inference.WithBackend("metal"), inference.WithContextLen(4096))
	if !res.OK {
		t.Fatalf("LoadModel(%q): %v", modelPath, res.Value)
	}
	model, ok := res.Value.(inference.TextModel)
	if !ok || model == nil {
		t.Fatalf("LoadModel returned %T, want an inference.TextModel", res.Value)
	}

	// A long prior context so the re-prefill it avoids is genuinely expensive.
	// ~200 repeats of the pangram is a few thousand tokens — well inside the
	// 4096 window with room for the turns.
	filler := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 200)
	system := inference.Message{Role: "system", Content: "You are a concise assistant. Reference material follows.\n" + filler}
	user1 := inference.Message{Role: "user", Content: "In one word, which animal jumps?"}

	noThink := false
	opts := []inference.GenerateOption{
		inference.WithMaxTokens(8),
		inference.WithTemperature(0),
		inference.WithEnableThinking(&noThink),
	}

	// turn measures one continuity turn end to end: the wall from Chat() (which
	// does acquire + wake + prefill synchronously) to the FIRST token is the
	// TTFT; ranging to exhaustion drives decode and the deferred sleep, so the
	// total wall carries the sleep-write cost. reply is the generated text.
	turn := func(mgr *continuity.Manager, msgs []inference.Message) (ttft, total time.Duration, reply string, accepted bool) {
		start := time.Now()
		seq, ok := mgr.Chat(ctx, msgs, opts...)
		if !ok {
			return 0, 0, "", false
		}
		var sb strings.Builder
		got := false
		for tok := range seq {
			if !got {
				ttft = time.Since(start)
				got = true
			}
			sb.WriteString(tok.Text)
		}
		return ttft, time.Since(start), sb.String(), true
	}

	enable := func(st state.Store) *continuity.Manager {
		mgr, err := continuity.EnableWithManager(model, st)
		if err != nil {
			t.Fatalf("EnableWithManager: %v", err)
		}
		return mgr
	}

	// seed runs turn 1 through a manager on st so the conversation is slept into
	// st and returns the generated assistant reply (needed to form turn 2's
	// lookup prefix).
	seed := func(st state.Store) (assistant1 string) {
		mgr := enable(st)
		ttft, total, reply, ok := turn(mgr, []inference.Message{system, user1})
		if !ok {
			t.Fatal("seed turn declined — continuity should accept a fresh [system,user] turn")
		}
		if s := mgr.Stats(); s.FreshConversations != 1 {
			t.Fatalf("seed turn stats = %+v, want one FreshConversation", s)
		}
		t.Logf("seed turn: fresh-prefill TTFT=%s total=%s reply=%q", ttft.Round(time.Millisecond), total.Round(time.Millisecond), strings.TrimSpace(reply))
		return reply
	}

	turn2 := func(assistant1 string) []inference.Message {
		return []inference.Message{
			system, user1,
			{Role: "assistant", Content: assistant1},
			{Role: "user", Content: "Now answer in exactly two words."},
		}
	}

	// sample runs make() -> turn() reps times and returns min/mean TTFT + total.
	// wantWake asserts the path each rep took (StoreWakes vs FreshConversations).
	type stat struct{ minTTFT, meanTTFT, minTotal, meanTotal time.Duration }
	sample := func(name string, make func() (*continuity.Manager, []inference.Message), wantWake bool) stat {
		var st stat
		var sumT, sumTot time.Duration
		for i := 0; i < reps; i++ {
			mgr, msgs := make()
			ttft, total, _, ok := turn(mgr, msgs)
			if !ok {
				t.Fatalf("%s rep %d: turn declined", name, i)
			}
			s := mgr.Stats()
			if wantWake && s.StoreWakes != 1 {
				t.Fatalf("%s rep %d: stats=%+v, want one StoreWake", name, i, s)
			}
			if !wantWake && s.FreshConversations != 1 {
				t.Fatalf("%s rep %d: stats=%+v, want one FreshConversation (re-prefill)", name, i, s)
			}
			if i == 0 || ttft < st.minTTFT {
				st.minTTFT = ttft
			}
			if i == 0 || total < st.minTotal {
				st.minTotal = total
			}
			sumT += ttft
			sumTot += total
		}
		st.meanTTFT = sumT / reps
		st.meanTotal = sumTot / reps
		t.Logf("%-22s TTFT min=%-8s mean=%-8s  TOTAL min=%-8s mean=%-8s", name,
			st.minTTFT.Round(time.Millisecond), st.meanTTFT.Round(time.Millisecond),
			st.minTotal.Round(time.Millisecond), st.meanTotal.Round(time.Millisecond))
		return st
	}

	// --- RAM store lane ---
	ramStore := state.NewInMemoryStore(nil)
	a1RAM := seed(ramStore)
	wakeRAM := sample("wake (RAM store)", func() (*continuity.Manager, []inference.Message) {
		return enable(ramStore), turn2(a1RAM) // fresh manager = evicted; must wake from store
	}, true)

	// --- File store lane ---
	filePath := t.TempDir() + "/receipts.kv"
	fileStore, err := filestore.Create(ctx, filePath)
	if err != nil {
		t.Fatalf("filestore.Create: %v", err)
	}
	defer fileStore.Close()
	a1File := seed(fileStore)
	wakeFile := sample("wake (file store)", func() (*continuity.Manager, []inference.Message) {
		return enable(fileStore), turn2(a1File)
	}, true)

	// --- Re-prefill baseline (continuity with an empty store: full prefill) ---
	reprefill := sample("re-prefill (cold)", func() (*continuity.Manager, []inference.Message) {
		return enable(state.NewInMemoryStore(nil)), turn2(a1RAM) // empty store => fresh full prefill
	}, false)

	// --- Resident (warm) turn: no store hit at all ---
	residentMgr := enable(state.NewInMemoryStore(nil))
	if _, _, r, ok := turn(residentMgr, []inference.Message{system, user1}); !ok {
		t.Fatalf("resident seed declined (reply=%q)", r)
	}
	rTTFT, rTotal, _, ok := turn(residentMgr, turn2(a1RAM))
	if !ok {
		t.Fatal("resident turn 2 declined")
	}
	if s := residentMgr.Stats(); s.ResidentTurns != 1 {
		t.Fatalf("resident turn stats=%+v, want one ResidentTurn", s)
	}
	t.Logf("%-22s TTFT     =%-8s                    TOTAL     =%-8s", "resident (warm)",
		rTTFT.Round(time.Millisecond), rTotal.Round(time.Millisecond))

	// --- Decomposition receipt ---
	t.Logf("")
	t.Logf("=== TIERED-KV RECEIPT (gemma-4-e2b-it-4bit, ctx 4096, %d reps) ===", reps)
	t.Logf("re-prefill avoided (TTFT): re-prefill=%s -> wake=%s  (saved ~%s)",
		reprefill.minTTFT.Round(time.Millisecond), wakeRAM.minTTFT.Round(time.Millisecond),
		(reprefill.minTTFT - wakeRAM.minTTFT).Round(time.Millisecond))
	t.Logf("wake-read tax (TTFT, file-RAM):   %s", (wakeFile.minTTFT - wakeRAM.minTTFT).Round(time.Millisecond))
	fileTail := wakeFile.minTotal - wakeFile.minTTFT
	ramTail := wakeRAM.minTotal - wakeRAM.minTTFT
	t.Logf("sleep-write+decode tail: file=%s ram=%s  (store tax ~%s)",
		fileTail.Round(time.Millisecond), ramTail.Round(time.Millisecond), (fileTail - ramTail).Round(time.Millisecond))
	t.Logf("total-turn tax (file-RAM):        %s", (wakeFile.minTotal - wakeRAM.minTotal).Round(time.Millisecond))
}
