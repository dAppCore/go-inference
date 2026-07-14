// SPDX-Licence-Identifier: EUPL-1.2

package scheduler

import (
	"context"
	"testing"

	"dappco.re/go/inference"
)

// The serial and batch dispatchers must re-arm ScheduledRequest.EnableThinking
// onto the base model exactly as the interleave lane does (the SamplerConfig
// fold cannot carry it). Found live by the SDK-examples fleet: a serial-
// scheduled serve ignored chat_template_kwargs {"enable_thinking": false} —
// the thought channel ate the whole token budget and content came back empty.

// TestSerialThinkingOverride pins the serial worker's re-arm: a Scheduled
// request carrying the override must deliver it to base.Chat.
func TestSerialThinkingOverride(t *testing.T) {
	off := false
	sim := newSimLaneSet()
	model := &thinkingRecordingModel{cbChatCapableModel: cbChatCapableModel{cbCapableModel{sim: sim, available: true, chatSeq: []inference.Token{{ID: 7}}}}}
	sched, err := New(model, Config{Mode: ModeSerial, MaxConcurrent: 1, MaxQueue: 4, StreamBuffer: 4})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer sched.CloseEngine()
	_, ch, err := sched.Schedule(context.Background(), inference.ScheduledRequest{
		ID:             "think-serial",
		Messages:       []inference.Message{{Role: "user", Content: "hi"}},
		Sampler:        inference.SamplerConfig{MaxTokens: 1},
		EnableThinking: &off,
	})
	if err != nil {
		t.Fatalf("Schedule: %v", err)
	}
	collectStream(ch)
	model.mu.Lock()
	got, called := model.chatThinking, model.chatCalled
	model.mu.Unlock()
	if !called {
		t.Fatal("serial dispatch never reached base.Chat")
	}
	if got == nil || *got {
		t.Fatalf("base.Chat received EnableThinking=%v, want the re-armed false override", got)
	}
}

// TestBatchThinkingOverride pins the same re-arm on the batch dispatcher.
func TestBatchThinkingOverride(t *testing.T) {
	off := false
	sim := newSimLaneSet()
	model := &thinkingRecordingModel{cbChatCapableModel: cbChatCapableModel{cbCapableModel{sim: sim, available: true, chatSeq: []inference.Token{{ID: 7}}}}}
	sched, err := New(model, Config{Mode: ModeBatch, MaxConcurrent: 1, MaxQueue: 4, StreamBuffer: 4})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer sched.CloseEngine()
	_, ch, err := sched.Schedule(context.Background(), inference.ScheduledRequest{
		ID:             "think-batch",
		Messages:       []inference.Message{{Role: "user", Content: "hi"}},
		Sampler:        inference.SamplerConfig{MaxTokens: 1},
		EnableThinking: &off,
	})
	if err != nil {
		t.Fatalf("Schedule: %v", err)
	}
	collectStream(ch)
	model.mu.Lock()
	got, called := model.chatThinking, model.chatCalled
	model.mu.Unlock()
	if !called {
		t.Fatal("batch dispatch never reached base.Chat")
	}
	if got == nil || *got {
		t.Fatalf("base.Chat received EnableThinking=%v, want the re-armed false override", got)
	}
}
