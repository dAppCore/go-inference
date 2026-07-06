// SPDX-Licence-Identifier: EUPL-1.2

// Tests for the agent-memory durable-state contracts in agent_memory.go —
// Ref/WakeRequest/WakeResult/SleepRequest/SleepResult are plain data with
// no behaviour of their own, so the meaningful contract to prove is the
// Session/Forker interface pair a runtime implements to plug into it, plus
// the AgentMemory* aliases staying interchangeable with their canonical
// types.

package state

import (
	"context"
	"testing"

	core "dappco.re/go"
)

// fakeForker and fakeSession are a minimal Forker/Session pair standing in
// for a real engine runtime (go-mlx, llama.cpp, etc).
type fakeForker struct{}

func (fakeForker) ForkState(_ context.Context, req WakeRequest) (Session, *WakeResult, error) {
	return fakeSession{}, &WakeResult{
		Entry:        Ref{URI: req.IndexURI + "/entry"},
		PrefixTokens: 12,
		Labels:       map[string]string{"backend": "fake"},
	}, nil
}

type fakeSession struct{}

func (fakeSession) WakeState(_ context.Context, req WakeRequest) (*WakeResult, error) {
	return &WakeResult{Entry: Ref{URI: req.EntryURI}, PrefixTokens: 12}, nil
}

func (fakeSession) SleepState(_ context.Context, req SleepRequest) (*SleepResult, error) {
	return &SleepResult{Entry: Ref{URI: req.EntryURI}, TokenCount: 12}, nil
}

// fakeFailingForker always fails ForkState — proves the contract when a
// runtime cannot restore the requested prefix (missing state, corrupt
// index, incompatible model).
type fakeFailingForker struct{ err error }

func (f fakeFailingForker) ForkState(_ context.Context, _ WakeRequest) (Session, *WakeResult, error) {
	return nil, nil, f.err
}

func TestForker_ForkState_Good(t *testing.T) {
	var forker Forker = fakeForker{}

	session, wake, err := forker.ForkState(context.Background(), WakeRequest{
		Store:    NewInMemoryStore(nil),
		IndexURI: "state://index",
		Model:    ModelIdentity{ID: "tiny"},
	})
	if err != nil {
		t.Fatalf("ForkState() error = %v", err)
	}
	if session == nil || wake == nil || wake.Entry.URI != "state://index/entry" {
		t.Fatalf("ForkState() = %#v, %#v; want session and wake report", session, wake)
	}

	sleep, err := session.SleepState(context.Background(), SleepRequest{EntryURI: "state://entry"})
	if err != nil {
		t.Fatalf("SleepState() error = %v", err)
	}
	if sleep.Entry.URI != "state://entry" || sleep.TokenCount != 12 {
		t.Fatalf("SleepState() = %#v, want entry token count", sleep)
	}
}

// TestForker_ForkState_Bad proves a failing Forker returns a nil
// Session/WakeResult alongside the propagated error — a caller must check
// the error before touching either return value.
func TestForker_ForkState_Bad(t *testing.T) {
	forkErr := core.NewError("no such state prefix")
	var forker Forker = fakeFailingForker{err: forkErr}

	session, wake, err := forker.ForkState(context.Background(), WakeRequest{IndexURI: "state://missing"})
	if !core.Is(err, forkErr) {
		t.Fatalf("ForkState() error = %v, want %v", err, forkErr)
	}
	if session != nil || wake != nil {
		t.Fatalf("ForkState() = %#v, %#v; want nil session and wake report on error", session, wake)
	}
}

// TestSession_AgentMemorySession_Ugly proves the AgentMemory* aliases are
// the same underlying types as their canonical counterparts — a value
// built from one name satisfies an interface built from the other, and
// the two remain assignable in both directions.
func TestSession_AgentMemorySession_Ugly(t *testing.T) {
	var canonical Session = fakeSession{}
	var aliased AgentMemorySession = canonical
	if aliased == nil {
		t.Fatal("AgentMemorySession assignment from Session = nil, want the same interface value")
	}

	req := AgentMemoryWakeRequest{EntryURI: "state://alias/entry"}
	var plain WakeRequest = req
	if plain.EntryURI != "state://alias/entry" {
		t.Fatalf("WakeRequest(AgentMemoryWakeRequest) = %+v, want the alias's fields verbatim", plain)
	}

	result := WakeResult{Entry: Ref{URI: "state://alias/result"}}
	var aliasedResult AgentMemoryWakeResult = result
	if aliasedResult.Entry.URI != "state://alias/result" {
		t.Fatalf("AgentMemoryWakeResult(WakeResult) = %+v, want the canonical value verbatim", aliasedResult)
	}
}
