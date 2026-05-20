// SPDX-Licence-Identifier: EUPL-1.2

package state

import (
	"context"
	"testing"

	core "dappco.re/go"
)

func TestState_InMemoryStore_Good(t *testing.T) {
	store := NewInMemoryStore(map[int]string{7: "chunk seven"})

	text, err := store.Get(context.Background(), 7)
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if text != "chunk seven" {
		t.Fatalf("Get() = %q, want chunk seven", text)
	}
	chunk, err := Resolve(context.Background(), store, 7)
	if err != nil {
		t.Fatalf("Resolve() error = %v", err)
	}
	if chunk.Ref.ChunkID != 7 || !chunk.Ref.HasFrameOffset || chunk.Ref.FrameOffset != 7 || chunk.Ref.Codec != CodecMemory {
		t.Fatalf("chunk ref = %#v", chunk.Ref)
	}
}

func TestState_InMemoryStore_Bad(t *testing.T) {
	store := NewInMemoryStore(nil)

	_, err := store.Get(context.Background(), 42)

	if !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("missing chunk error = %v, want ErrChunkNotFound", err)
	}
}

func TestState_BinaryStore_Good(t *testing.T) {
	store := NewInMemoryStore(nil)
	payload := []byte{0, 1, 2, 255}

	ref, err := store.PutBytes(context.Background(), payload, PutOptions{URI: "state://binary/1"})
	if err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}
	payload[1] = 99

	chunk, err := ResolveBytes(context.Background(), store, ref.ChunkID)
	if err != nil {
		t.Fatalf("ResolveBytes() error = %v", err)
	}
	if chunk.Ref.ChunkID != ref.ChunkID || len(chunk.Data) != 4 || chunk.Data[1] != 1 || chunk.Data[3] != 255 {
		t.Fatalf("ResolveBytes() chunk = %+v, want copied binary payload", chunk)
	}
	chunk.Data[2] = 88
	again, err := ResolveBytes(context.Background(), store, ref.ChunkID)
	if err != nil {
		t.Fatalf("ResolveBytes(second) error = %v", err)
	}
	if again.Data[2] != 2 {
		t.Fatalf("ResolveBytes() returned aliased data = %v", again.Data)
	}
	byURI, err := ResolveURI(context.Background(), store, "state://binary/1")
	if err != nil {
		t.Fatalf("ResolveURI(binary) error = %v", err)
	}
	if len(byURI.Data) != 4 || byURI.Data[0] != 0 {
		t.Fatalf("ResolveURI(binary) chunk = %+v, want binary data", byURI)
	}
}

func TestState_WakeSleepForkContracts_Good(t *testing.T) {
	model := fakeForker{}

	session, wake, err := model.ForkState(context.Background(), WakeRequest{
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

type fakeForker struct{}

func (fakeForker) ForkState(_ context.Context, req WakeRequest) (Session, *WakeResult, error) {
	session := fakeSession{}
	return session, &WakeResult{
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
