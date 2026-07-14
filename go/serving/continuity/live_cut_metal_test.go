// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

// live_cut_metal_test.go is the CANCEL LANE's live acceptance gate — the twin of
// live_gate_metal_test.go (the sleep lane). It drives a real continuity.Manager
// on a real checkpoint through the metal engine and ASSERTS the cut-turn contract
// end to end: a turn cancelled mid-stream (as a serving/scheduler Cancel would)
// must be discarded cleanly (one CutTurn counted, no wedge), and the follow-up
// turn — re-sending the client's partial transcript — must produce the same
// answer an uncancelled equivalent conversation produces. This is the live proof
// of evict-and-refresh: the cut leaves the conversation in a clean, re-derivable
// state, never a drifted cache. A divergence here would be the cancel lane
// silently poisoning the next turn's context.
//
//	LTHN_PROBE_MODEL=<snapshot dir>   # or GO_INFERENCE_SMOKE_MODEL
//	MLX_METALLIB_PATH=/Users/snider/Code/core/go-inference/build/dist/lib/mlx.metallib \
//	go test -tags metal_runtime -run TestLiveCutTurnServeGate ./serving/continuity/ -v -timeout 15m
package continuity_test

import (
	"context"
	"strings"
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/inference/model/state"
	"dappco.re/go/inference/serving/continuity"

	_ "dappco.re/go/inference/engine/metal"  // registers the "metal" backend
	_ "dappco.re/go/inference/model/builtin" // registers the built-in arches
)

func TestLiveCutTurnServeGate(t *testing.T) {
	modelPath := liveGateModelPath()
	if modelPath == "" {
		t.Skip("set LTHN_PROBE_MODEL (or GO_INFERENCE_SMOKE_MODEL) to a checkpoint dir to run the live cut-turn serve gate")
	}

	res := inference.LoadModel(modelPath, inference.WithBackend("metal"), inference.WithContextLen(4096))
	if !res.OK {
		t.Fatalf("LoadModel(%q): %v", modelPath, res.Value)
	}
	model, ok := res.Value.(inference.TextModel)
	if !ok || model == nil {
		t.Fatalf("LoadModel returned %T, want an inference.TextModel", res.Value)
	}

	// Greedy so the cut-then-continue and reference lanes are deterministically
	// comparable. A counting prompt for turn 1 guarantees the reply runs well past
	// the cut point, so the cancel reliably lands mid-generation.
	opts := []inference.GenerateOption{inference.WithMaxTokens(24), inference.WithTemperature(0)}
	filler := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 64)
	system := inference.Message{Role: "system", Content: "You are a concise assistant. Reference material follows.\n" + filler}
	user1 := inference.Message{Role: "user", Content: "Count from one to twenty in words, comma separated."}
	user2 := inference.Message{Role: "user", Content: "Now name the colour of the fox, one word only."}

	enable := func(st state.Store) *continuity.Manager {
		mgr, err := continuity.EnableWithManager(model, st)
		if err != nil {
			t.Fatalf("EnableWithManager: %v", err)
		}
		return mgr
	}

	// driveTurn ranges a turn to exhaustion on a fresh (uncancelled) ctx and
	// returns its reply — the follow-up and reference lanes both use it.
	driveTurn := func(mgr *continuity.Manager, msgs []inference.Message) (string, bool) {
		seq, ok := mgr.Chat(context.Background(), msgs, opts...)
		if !ok {
			return "", false
		}
		var sb strings.Builder
		for tk := range seq {
			sb.WriteString(tk.Text)
		}
		return sb.String(), true
	}

	// --- Turn 1: cut mid-stream after cutAfter tokens. ---
	const cutAfter = 4
	store := state.NewInMemoryStore(nil)
	cutMgr := enable(store)
	turnCtx, cancel := context.WithCancel(context.Background())
	defer cancel() // idempotent with the in-loop cut; satisfies the vet leak check on early-return paths
	seq1, ok := cutMgr.Chat(turnCtx, []inference.Message{system, user1}, opts...)
	if !ok {
		t.Fatal("cut turn 1 declined — continuity must accept a fresh [system,user] turn")
	}
	var partial strings.Builder
	got := 0
	for tk := range seq1 {
		partial.WriteString(tk.Text)
		got++
		if got == cutAfter {
			cancel() // scheduler Cancel lands here, mid-generation
		}
	}
	if s := cutMgr.Stats(); s.CutTurns != 1 {
		t.Fatalf("cut turn stats = %+v, want exactly one CutTurn (the cut must have landed mid-stream — got %d tokens)", s, got)
	}
	partialReply := strings.TrimSpace(partial.String())
	t.Logf("turn 1 cut after %d tokens (client received %d total); partial reply: %q", cutAfter, got, partialReply)

	turn2Msgs := []inference.Message{
		system, user1,
		{Role: "assistant", Content: partialReply}, // exactly what the client received
		user2,
	}

	// --- Turn 2 on the cut manager: the cut slept nothing, so this re-prefills the
	//     client transcript fresh (evict-and-refresh), never waking a drifted cache. ---
	contReply, ok := driveTurn(cutMgr, turn2Msgs)
	if !ok {
		t.Fatal("follow-up turn after a cut declined — the cut must not wedge the manager")
	}
	if s := cutMgr.Stats(); s.StoreWakes != 0 {
		t.Fatalf("follow-up stats = %+v, want no StoreWake (a cut persists nothing to wake)", s)
	}

	// --- Reference: the SAME turn-2 transcript on a fresh manager over an empty
	//     store — a full stateless re-prefill, the uncancelled equivalent. ---
	refReply, ok := driveTurn(enable(state.NewInMemoryStore(nil)), turn2Msgs)
	if !ok {
		t.Fatal("reference turn declined")
	}

	// The gate: the cut-then-continue follow-up must say exactly what the
	// uncancelled equivalent says. Anything else is the cancel lane leaking a
	// phantom token from the discarded partial into the next turn's context.
	if strings.TrimSpace(contReply) != strings.TrimSpace(refReply) {
		t.Fatalf("CUT-TURN FOLLOW-UP DIVERGENCE:\n  cut-then-continue: %q\n  reference:         %q",
			strings.TrimSpace(contReply), strings.TrimSpace(refReply))
	}
	t.Logf("CUT-TURN FOLLOW-UP PARITY: cut-then-continue == uncancelled equivalent -> %q", strings.TrimSpace(contReply))
}
