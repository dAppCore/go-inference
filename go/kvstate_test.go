// SPDX-Licence-Identifier: EUPL-1.2

package inference

import (
	"context"
	"iter"
	"testing"

	"dappco.re/go/inference/kv"
)

type kvStateModel struct {
	captured  string
	opts      KVSnapshotCaptureOptions
	restored  *kv.Snapshot
	warmed    string
	cleared   bool
	chunksLen int
}

func (m *kvStateModel) CaptureKV(_ context.Context, prompt string, opts KVSnapshotCaptureOptions) (*kv.Snapshot, error) {
	m.captured, m.opts = prompt, opts
	return &kv.Snapshot{Architecture: "gemma4", SeqLen: 3}, nil
}

func (m *kvStateModel) CaptureKVChunks(_ context.Context, chunks iter.Seq[string], opts KVSnapshotCaptureOptions) (*kv.Snapshot, error) {
	m.opts = opts
	for range chunks {
		m.chunksLen++
	}
	return &kv.Snapshot{Architecture: "gemma4", SeqLen: m.chunksLen}, nil
}

func (m *kvStateModel) RestoreFromKV(_ context.Context, snapshot *kv.Snapshot) error {
	m.restored = snapshot
	return nil
}

func (m *kvStateModel) WarmPromptCache(_ context.Context, prompt string) error {
	m.warmed = prompt
	return nil
}

func (m *kvStateModel) WarmPromptCacheChunks(_ context.Context, chunks iter.Seq[string]) error {
	for range chunks {
		m.chunksLen++
	}
	return nil
}

func (m *kvStateModel) ClearPromptCache() { m.cleared = true }

func TestKVState_KVSnapshotter_Good(t *testing.T) {
	model := &kvStateModel{}
	var probe any = model

	s, ok := probe.(KVSnapshotter)
	checkTrue(t, ok)

	snap, err := s.CaptureKV(context.Background(), "hello", KVSnapshotCaptureOptions{RawKVOnly: true, BlockStartToken: 7})
	if err != nil {
		t.Fatalf("CaptureKV: %v", err)
	}
	checkEqual(t, "hello", model.captured)
	checkTrue(t, model.opts.RawKVOnly)
	checkEqual(t, 7, model.opts.BlockStartToken)
	checkEqual(t, "gemma4", snap.Architecture)
}

func TestKVState_KVChunkSnapshotter_Good(t *testing.T) {
	model := &kvStateModel{}
	var probe any = model

	s, ok := probe.(KVChunkSnapshotter)
	checkTrue(t, ok)

	chunks := func(yield func(string) bool) {
		for _, c := range []string{"a", "b"} {
			if !yield(c) {
				return
			}
		}
	}
	snap, err := s.CaptureKVChunks(context.Background(), chunks, KVSnapshotCaptureOptions{})
	if err != nil {
		t.Fatalf("CaptureKVChunks: %v", err)
	}
	checkEqual(t, 2, snap.SeqLen)
}

func TestKVState_KVRestorer_Good(t *testing.T) {
	model := &kvStateModel{}
	var probe any = model

	r, ok := probe.(KVRestorer)
	checkTrue(t, ok)

	want := &kv.Snapshot{Architecture: "gemma4", SeqLen: 9}
	if err := r.RestoreFromKV(context.Background(), want); err != nil {
		t.Fatalf("RestoreFromKV: %v", err)
	}
	checkEqual(t, want, model.restored)
}

func TestKVState_PromptCacheWarmer_Good(t *testing.T) {
	model := &kvStateModel{}
	var probe any = model

	w, ok := probe.(PromptCacheWarmer)
	checkTrue(t, ok)

	if err := w.WarmPromptCache(context.Background(), "warm me"); err != nil {
		t.Fatalf("WarmPromptCache: %v", err)
	}
	checkEqual(t, "warm me", model.warmed)
}

func TestKVState_PromptCacheClearer_Good(t *testing.T) {
	model := &kvStateModel{}
	var probe any = model

	c, ok := probe.(PromptCacheClearer)
	checkTrue(t, ok)

	c.ClearPromptCache()
	checkTrue(t, model.cleared)
}

func TestKVState_Probe_UglyNonImplementer(t *testing.T) {
	var probe any = struct{}{}

	_, snapOK := probe.(KVSnapshotter)
	_, restoreOK := probe.(KVRestorer)

	checkFalse(t, snapOK)
	checkFalse(t, restoreOK)
}
