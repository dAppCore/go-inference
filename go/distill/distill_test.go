// SPDX-Licence-Identifier: EUPL-1.2

package distill

import (
	"context"
	"testing"

	"dappco.re/go/inference/dataset"
)

// --- NewMemoryLogitCache ---

// Good: NewMemoryLogitCache returns a working, empty cache — a miss on
// any key returns (nil, false, nil), and two constructed caches never
// share state.
func TestNewMemoryLogitCache_Good(t *testing.T) {
	a := NewMemoryLogitCache()
	b := NewMemoryLogitCache()
	if a == nil || b == nil {
		t.Fatal("NewMemoryLogitCache() = nil, want a usable cache")
	}
	if err := a.PutTeacherLogits(context.Background(), "k", Logits{{{1}}}); err != nil {
		t.Fatalf("PutTeacherLogits() error = %v", err)
	}
	if _, ok, err := b.GetTeacherLogits(context.Background(), "k"); ok || err != nil {
		t.Fatalf("independently constructed cache saw a's write: ok=%v err=%v", ok, err)
	}
}

// --- MemoryLogitCache.GetTeacherLogits ---

// Good: a value stored via Put is returned by Get as an independent
// clone — mutating the returned slice never reaches the cached copy.
func TestMemoryLogitCache_GetTeacherLogits_Good(t *testing.T) {
	c := NewMemoryLogitCache()
	want := Logits{{{1, 2, 3}}}
	if err := c.PutTeacherLogits(context.Background(), "k", want); err != nil {
		t.Fatalf("PutTeacherLogits() error = %v", err)
	}
	got, ok, err := c.GetTeacherLogits(context.Background(), "k")
	if err != nil || !ok {
		t.Fatalf("GetTeacherLogits() = ok %v err %v, want true,nil", ok, err)
	}
	got[0][0][0] = 99
	again, _, _ := c.GetTeacherLogits(context.Background(), "k")
	if again[0][0][0] != 1 {
		t.Fatalf("GetTeacherLogits() leaked aliasing: cached value mutated to %v", again[0][0][0])
	}
}

// Bad: Get on a nil *MemoryLogitCache receiver returns the safe zero
// result rather than panicking.
func TestMemoryLogitCache_GetTeacherLogits_Bad(t *testing.T) {
	var c *MemoryLogitCache
	logits, ok, err := c.GetTeacherLogits(context.Background(), "k")
	if logits != nil || ok || err != nil {
		t.Fatalf("nil-receiver GetTeacherLogits() = %v,%v,%v, want nil,false,nil", logits, ok, err)
	}
}

// Ugly: a miss on a live but empty cache (zero-value &MemoryLogitCache{},
// bypassing the constructor, so the internal map is nil) returns
// (nil, false, nil) without panicking on the nil map read.
func TestMemoryLogitCache_GetTeacherLogits_Ugly(t *testing.T) {
	c := &MemoryLogitCache{}
	logits, ok, err := c.GetTeacherLogits(context.Background(), "missing")
	if logits != nil || ok || err != nil {
		t.Fatalf("miss on nil-map cache = %v,%v,%v, want nil,false,nil", logits, ok, err)
	}
}

// --- MemoryLogitCache.PutTeacherLogits ---

// Good: Put stores a defensive clone — mutating the source after Put
// never reaches the cached copy.
func TestMemoryLogitCache_PutTeacherLogits_Good(t *testing.T) {
	c := NewMemoryLogitCache()
	source := Logits{{{1, 2}}}
	if err := c.PutTeacherLogits(context.Background(), "k", source); err != nil {
		t.Fatalf("PutTeacherLogits() error = %v", err)
	}
	source[0][0][0] = 42
	got, ok, _ := c.GetTeacherLogits(context.Background(), "k")
	if !ok || got[0][0][0] != 1 {
		t.Fatalf("PutTeacherLogits() aliased the source: got %v, want first cell 1", got)
	}
}

// Bad: Put on a nil *MemoryLogitCache receiver is a no-op that returns a
// nil error rather than panicking.
func TestMemoryLogitCache_PutTeacherLogits_Bad(t *testing.T) {
	var c *MemoryLogitCache
	if err := c.PutTeacherLogits(context.Background(), "k", Logits{{{1}}}); err != nil {
		t.Fatalf("nil-receiver PutTeacherLogits() error = %v, want nil", err)
	}
}

// Ugly: Put on a zero-value &MemoryLogitCache{} (nil internal map) lazily
// initialises storage, and a second Put under the same key overwrites
// rather than appending.
func TestMemoryLogitCache_PutTeacherLogits_Ugly(t *testing.T) {
	c := &MemoryLogitCache{}
	if err := c.PutTeacherLogits(context.Background(), "k", Logits{{{1}}}); err != nil {
		t.Fatalf("PutTeacherLogits() on nil-map cache error = %v", err)
	}
	if err := c.PutTeacherLogits(context.Background(), "k", Logits{{{2}}}); err != nil {
		t.Fatalf("second PutTeacherLogits() error = %v", err)
	}
	got, ok, _ := c.GetTeacherLogits(context.Background(), "k")
	if !ok || len(got) != 1 || got[0][0][0] != 2 {
		t.Fatalf("second Put did not overwrite: got %v, want a single-entry [[[2]]]", got)
	}
}

// --- CollectSamples ---

// Good: with no cap (maxSamples <= 0), CollectSamples drains the whole
// dataset and returns every sample.
func TestCollectSamples_Good(t *testing.T) {
	ds := dataset.NewSliceDataset([]dataset.Sample{
		{Text: "a"}, {Text: "b"}, {Text: "c"},
	})
	got, err := CollectSamples(context.Background(), ds, 0)
	if err != nil {
		t.Fatalf("CollectSamples() error = %v", err)
	}
	if len(got) != 3 || got[0].Text != "a" || got[2].Text != "c" {
		t.Fatalf("CollectSamples() = %+v, want 3 samples a,b,c", got)
	}
}

// Bad: a nil dataset returns the sentinel error rather than panicking.
func TestCollectSamples_Bad(t *testing.T) {
	if _, err := CollectSamples(context.Background(), nil, 0); err != errDatasetNil {
		t.Fatalf("CollectSamples(nil ds) error = %v, want errDatasetNil", err)
	}
}

// Ugly: a positive maxSamples truncates before the dataset is exhausted,
// and a cancelled context is honoured even when samples remain.
func TestCollectSamples_Ugly(t *testing.T) {
	ds := dataset.NewSliceDataset([]dataset.Sample{
		{Text: "a"}, {Text: "b"}, {Text: "c"},
	})
	got, err := CollectSamples(context.Background(), ds, 2)
	if err != nil {
		t.Fatalf("CollectSamples() error = %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("CollectSamples(maxSamples=2) len = %d, want 2", len(got))
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	ds2 := dataset.NewSliceDataset([]dataset.Sample{{Text: "a"}})
	if _, err := CollectSamples(ctx, ds2, 0); err == nil {
		t.Fatal("CollectSamples() with a cancelled context: expected error, got nil")
	}
}
