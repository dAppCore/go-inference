// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

// live_gate_metal_test.go is the SERVE SLEEP LANE's live acceptance gate — the
// "live multi-turn serve gate" the composed-lane handover flagged missing. Unlike
// receipts_metal_test.go (which MEASURES the tiered-KV lane and never asserts),
// this file ASSERTS turn-2 correctness: a two-turn conversation driven through the
// real continuity.Manager — turn 1 slept to the store, turn 2 WOKEN from the store
// and appended with only the new turn — must produce the same answer a stateless
// full re-prefill of the whole turn-2 conversation produces. Sleeping and waking
// must not change what the conversation says; a divergence here is a silent wrong
// answer, exactly what the parked note said had no gate.
//
// The gate is about the SERVE SLEEP LANE, not a specific arch — it is
// engine-agnostic and runs on whatever real checkpoint the metal backend can load
// (the gemma ArchSession lane when pointed at a gemma snapshot; the composed lane
// when pointed at a composed hybrid checkpoint). It gates on the model actually
// loading and skips otherwise:
//
//	LTHN_PROBE_MODEL=<snapshot dir>   # or GO_INFERENCE_SMOKE_MODEL
//	MLX_METALLIB_PATH=/Users/snider/Code/core/go-inference/build/dist/lib/mlx.metallib \
//	go test -tags metal_runtime -run TestLiveMultiTurnServeGate ./serving/continuity/ -v -timeout 15m
package continuity_test

import (
	"context"
	"os"
	"strings"
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/inference/model/state"
	"dappco.re/go/inference/serving/continuity"

	_ "dappco.re/go/inference/engine/metal"  // registers the "metal" backend
	_ "dappco.re/go/inference/model/builtin" // registers the built-in arches
)

// liveGateModelPath resolves the checkpoint dir the gate loads: LTHN_PROBE_MODEL
// (the probe-lane convention) first, then GO_INFERENCE_SMOKE_MODEL (the serve
// harness convention) so the gate runs under either.
func liveGateModelPath() string {
	if p := os.Getenv("LTHN_PROBE_MODEL"); p != "" {
		return p
	}
	return os.Getenv("GO_INFERENCE_SMOKE_MODEL")
}

func TestLiveMultiTurnServeGate(t *testing.T) {
	modelPath := liveGateModelPath()
	if modelPath == "" {
		t.Skip("set LTHN_PROBE_MODEL (or GO_INFERENCE_SMOKE_MODEL) to a checkpoint dir to run the live multi-turn serve gate")
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
	arch := ""
	if r, ok := model.(interface{ Info() inference.ModelInfo }); ok {
		arch = r.Info().Architecture
	}
	t.Logf("live serve sleep gate exercising arch %q from %s", arch, modelPath)

	// Greedy so the woken and stateless lanes are deterministically comparable.
	opts := []inference.GenerateOption{inference.WithMaxTokens(24), inference.WithTemperature(0)}

	// A long prior context so the re-prefill the wake avoids is genuine and the
	// woken state carries real content across the turn boundary.
	filler := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 64)
	system := inference.Message{Role: "system", Content: "You are a concise assistant. Reference material follows.\n" + filler}
	user1 := inference.Message{Role: "user", Content: "In one word, which animal jumps?"}
	user2 := inference.Message{Role: "user", Content: "Now name the colour of that animal, one word only."}

	// turn drives one continuity turn to exhaustion (so the deferred sleep runs)
	// and returns the generated reply plus whether the lane accepted the request.
	turn := func(mgr *continuity.Manager, msgs []inference.Message) (string, bool) {
		seq, ok := mgr.Chat(ctx, msgs, opts...)
		if !ok {
			return "", false
		}
		var sb strings.Builder
		for tok := range seq {
			sb.WriteString(tok.Text)
		}
		return sb.String(), true
	}

	enable := func(st state.Store) *continuity.Manager {
		mgr, err := continuity.EnableWithManager(model, st)
		if err != nil {
			t.Fatalf("EnableWithManager: %v", err)
		}
		return mgr
	}

	// --- Seed turn 1 into the store and capture its reply (turn 2's prefix). ---
	store := state.NewInMemoryStore(nil)
	reply1, ok := turn(enable(store), []inference.Message{system, user1})
	if !ok {
		t.Fatal("seed turn 1 declined — continuity must accept a fresh [system,user] turn")
	}
	t.Logf("turn 1 (fresh prefill) reply: %q", strings.TrimSpace(reply1))

	turn2Msgs := []inference.Message{
		system, user1,
		{Role: "assistant", Content: reply1},
		user2,
	}

	// --- Reference lane: the SAME turn-2 conversation, full stateless prefill on a
	//     fresh manager over an EMPTY store (no state to wake, so a full re-prefill). ---
	refMgr := enable(state.NewInMemoryStore(nil))
	refReply, ok := turn(refMgr, turn2Msgs)
	if !ok {
		t.Fatal("reference turn 2 declined")
	}
	if s := refMgr.Stats(); s.FreshConversations != 1 {
		t.Fatalf("reference turn 2 stats = %+v, want one FreshConversation (full re-prefill)", s)
	}

	// --- Continuity lane: turn 2 on a FRESH manager over the SEEDED store — it must
	//     WAKE turn 1's slept state and append ONLY the new turn (wake -> append ->
	//     sleep across the two turns). ---
	wakeMgr := enable(store)
	wakeReply, ok := turn(wakeMgr, turn2Msgs)
	if !ok {
		t.Fatal("continuity turn 2 declined — the woken-conversation lane must serve it")
	}
	if s := wakeMgr.Stats(); s.StoreWakes != 1 {
		t.Fatalf("continuity turn 2 stats = %+v, want one StoreWake (turn 1 woken from the store)", s)
	}

	// The gate: waking + appending the new turn must produce the same answer as a
	// full re-prefill of the whole conversation. Anything else is the serve sleep
	// lane silently changing the conversation.
	if strings.TrimSpace(wakeReply) != strings.TrimSpace(refReply) {
		t.Fatalf("SERVE SLEEP LANE turn-2 DIVERGENCE (arch %q):\n  woken:     %q\n  reference: %q",
			arch, strings.TrimSpace(wakeReply), strings.TrimSpace(refReply))
	}
	t.Logf("SERVE SLEEP LANE turn-2 PARITY (arch %q): woken == stateless -> %q", arch, strings.TrimSpace(wakeReply))
}
