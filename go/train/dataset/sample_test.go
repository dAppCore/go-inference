// SPDX-Licence-Identifier: EUPL-1.2

package dataset

import (
	"testing"

	"dappco.re/go/inference"
)

// --- Func.Next ---

func TestSample_Func_Next_Good(t *testing.T) {
	want := Sample{Prompt: "p", Response: "r", Format: "prompt_response"}
	calls := 0
	fn := Func(func() (Sample, bool, error) {
		calls++
		return want, true, nil
	})

	got, ok, err := fn.Next()
	if err != nil {
		t.Fatalf("Func.Next() error = %v", err)
	}
	if !ok {
		t.Fatal("Func.Next() ok = false, want true")
	}
	if got.Prompt != want.Prompt || got.Response != want.Response || got.Format != want.Format {
		t.Fatalf("Func.Next() = %+v, want %+v", got, want)
	}
	if calls != 1 {
		t.Fatalf("wrapped fn called %d times, want 1", calls)
	}
}

func TestSample_Func_Next_Bad(t *testing.T) {
	var fn Func // nil function value
	if _, _, err := fn.Next(); err == nil {
		t.Fatal("Func.Next() on nil func: expected error, got nil")
	}
}

// Ugly: the wrapped function reports exhaustion. Next must surface (zero,
// false, nil) without inventing an error, and repeated calls on the same
// exhausted closure must stay (zero, false, nil) — Func is a pure
// pass-through with no state of its own.
func TestSample_Func_Next_Ugly(t *testing.T) {
	fn := Func(func() (Sample, bool, error) {
		return Sample{}, false, nil
	})
	got, ok, err := fn.Next()
	if err != nil {
		t.Fatalf("Func.Next() error = %v", err)
	}
	if ok {
		t.Fatalf("Func.Next() ok = true on exhausted func, want false (got %+v)", got)
	}
	if _, ok, err := fn.Next(); ok || err != nil {
		t.Fatalf("Func.Next() second call = ok %v err %v, want false,nil", ok, err)
	}
}

// --- NewSliceDataset ---

// Good: NewSliceDataset clones the slice header, so reassigning entries in
// the source slice after construction cannot reach into the dataset's
// iteration.
func TestSample_NewSliceDataset_Good(t *testing.T) {
	source := []Sample{
		{Text: "a"},
		{Prompt: "p", Response: "r"},
	}
	ds := NewSliceDataset(source)

	source[0] = Sample{Text: "mutated"}
	source[1] = Sample{Response: "mutated"}

	first, ok, err := ds.Next()
	if err != nil {
		t.Fatalf("Next() error = %v", err)
	}
	if !ok || first.Text != "a" {
		t.Fatalf("NewSliceDataset clone failed: first = %+v ok=%v, want original Text 'a'", first, ok)
	}
}

// Bad: NewSliceDataset is total — the degenerate input is a nil slice; the
// constructed dataset must be usable and immediately exhausted, never nil
// and never a panic on first Next.
func TestSample_NewSliceDataset_Bad(t *testing.T) {
	ds := NewSliceDataset(nil)
	if ds == nil {
		t.Fatal("NewSliceDataset(nil) = nil, want a usable empty dataset")
	}
	if _, ok, err := ds.Next(); ok || err != nil {
		t.Fatalf("NewSliceDataset(nil).Next() = ok %v err %v, want false,nil", ok, err)
	}
}

// Ugly: an empty (zero-length, non-nil) source yields a dataset that is
// immediately exhausted, and Reset on it stays a no-op error-free.
func TestSample_NewSliceDataset_Ugly(t *testing.T) {
	ds := NewSliceDataset([]Sample{})
	if _, ok, err := ds.Next(); ok || err != nil {
		t.Fatalf("empty NewSliceDataset.Next() = ok %v err %v, want false,nil", ok, err)
	}
	if err := ds.Reset(); err != nil {
		t.Fatalf("empty NewSliceDataset.Reset() error = %v", err)
	}
}

// --- SliceDataset.Next ---

// Good: sequential Next calls yield each record in order, then exhaust with
// (zero, false, nil). Reassigning source entries after construction cannot
// reach the iteration (clone-isolation).
func TestSample_SliceDataset_Next_Good(t *testing.T) {
	source := []Sample{
		{Text: "a"},
		{Prompt: "p", Response: "r"},
	}
	ds := NewSliceDataset(source)
	source[0] = Sample{Text: "mutated"}
	source[1] = Sample{Response: "mutated"}

	first, ok, err := ds.Next()
	if err != nil {
		t.Fatalf("Next() error = %v", err)
	}
	if !ok || first.Text != "a" {
		t.Fatalf("Next()[0] = %+v ok=%v, want original Text 'a'", first, ok)
	}
	second, ok, err := ds.Next()
	if err != nil {
		t.Fatalf("Next() error = %v", err)
	}
	if !ok || second.Response != "r" {
		t.Fatalf("Next()[1] = %+v ok=%v, want original Response 'r'", second, ok)
	}
	if _, ok, err := ds.Next(); ok || err != nil {
		t.Fatalf("Next() after end = ok %v err %v, want false,nil", ok, err)
	}
}

// Bad: Next on a nil *SliceDataset returns the sentinel error and ok=false
// rather than panicking.
func TestSample_SliceDataset_Next_Bad(t *testing.T) {
	var ds *SliceDataset
	if _, ok, err := ds.Next(); err == nil || ok {
		t.Fatalf("nil SliceDataset.Next() = ok %v err %v, want false + error", ok, err)
	}
}

// Ugly: Next past the end is idempotent — every further Next keeps
// returning (zero, false, nil) without advancing or erroring, and the zero
// value carries no leaked fields from the last real record.
func TestSample_SliceDataset_Next_Ugly(t *testing.T) {
	ds := NewSliceDataset([]Sample{{Text: "only"}})
	if _, ok, _ := ds.Next(); !ok {
		t.Fatal("first Next() ok = false, want the single record")
	}
	for i := 0; i < 3; i++ {
		got, ok, err := ds.Next()
		if ok || err != nil {
			t.Fatalf("Next() past end #%d = ok %v err %v, want false,nil", i, ok, err)
		}
		if got.Text != "" || got.Prompt != "" || got.Response != "" || got.Format != "" || got.Labels != nil {
			t.Fatalf("Next() past end #%d returned non-zero sample %+v", i, got)
		}
	}
}

// --- SliceDataset.Reset ---

// Good: Reset rewinds so a second full pass yields the same records.
func TestSample_SliceDataset_Reset_Good(t *testing.T) {
	ds := NewSliceDataset([]Sample{{Text: "row0"}, {Text: "row1"}})
	drain := func() []Sample {
		var out []Sample
		for {
			s, ok, err := ds.Next()
			if err != nil {
				t.Fatalf("Next() error = %v", err)
			}
			if !ok {
				return out
			}
			out = append(out, s)
		}
	}
	first := drain()
	if len(first) != 2 {
		t.Fatalf("first pass len = %d, want 2", len(first))
	}
	if err := ds.Reset(); err != nil {
		t.Fatalf("Reset() error = %v", err)
	}
	second := drain()
	if len(second) != 2 || second[0].Text != "row0" || second[1].Text != "row1" {
		t.Fatalf("second pass after Reset = %+v, want identical replay", second)
	}
}

// Bad: Reset on a nil *SliceDataset returns the sentinel error rather than
// panicking.
func TestSample_SliceDataset_Reset_Bad(t *testing.T) {
	var ds *SliceDataset
	if err := ds.Reset(); err == nil {
		t.Fatal("nil SliceDataset.Reset(): expected error, got nil")
	}
}

// Ugly: Reset is safe at the boundaries — before any Next (no-op), and
// twice in a row, both leave the cursor at the start with a faithful first
// record.
func TestSample_SliceDataset_Reset_Ugly(t *testing.T) {
	ds := NewSliceDataset([]Sample{{Text: "head"}, {Text: "tail"}})
	if err := ds.Reset(); err != nil {
		t.Fatalf("Reset() before first Next error = %v", err)
	}
	if err := ds.Reset(); err != nil {
		t.Fatalf("second consecutive Reset() error = %v", err)
	}
	got, ok, err := ds.Next()
	if err != nil || !ok || got.Text != "head" {
		t.Fatalf("after boundary Resets, Next() = %+v ok=%v err=%v, want head", got, ok, err)
	}
}

// --- CloneSample ---

// Good: CloneSample deep-copies Labels and Messages — mutating the clone's
// map or message slice never reaches the source.
func TestSample_CloneSample_Good(t *testing.T) {
	src := Sample{
		Prompt:   "p",
		Response: "r",
		Messages: []inference.Message{{Role: "user", Content: "hi"}},
		Labels:   map[string]string{"split": "train"},
	}
	clone := CloneSample(src)
	clone.Labels["split"] = "test"
	clone.Messages[0].Content = "changed"

	if src.Labels["split"] != "train" {
		t.Fatalf("CloneSample aliased Labels: src split = %q, want 'train'", src.Labels["split"])
	}
	if src.Messages[0].Content != "hi" {
		t.Fatalf("CloneSample aliased Messages: src content = %q, want 'hi'", src.Messages[0].Content)
	}
	if clone.Prompt != "p" || clone.Response != "r" {
		t.Fatalf("CloneSample dropped scalar fields: %+v", clone)
	}
}

// Bad: CloneSample is total (no error channel), so the adversarial input is
// the fully zero Sample. The clone must equal the zero value exactly — no
// map or slice materialised, no scalar invented.
func TestSample_CloneSample_Bad(t *testing.T) {
	clone := CloneSample(Sample{})
	if clone.Text != "" || clone.Prompt != "" || clone.Response != "" || clone.Format != "" {
		t.Fatalf("CloneSample(zero) = %+v, want the zero Sample", clone)
	}
	if clone.Labels != nil || clone.Messages != nil {
		t.Fatalf("CloneSample(zero) = %+v, want no map/slice materialised", clone)
	}
}

// Ugly: a Sample with no Labels clones to a nil Labels map (not an empty
// allocated one) while non-Labels fields survive; an explicitly empty
// (non-nil) Labels also collapses to nil.
func TestSample_CloneSample_Ugly(t *testing.T) {
	clone := CloneSample(Sample{Text: "no labels"})
	if clone.Labels != nil {
		t.Fatalf("CloneSample(no labels).Labels = %v, want nil", clone.Labels)
	}
	if clone.Text != "no labels" {
		t.Fatalf("CloneSample dropped Text: %+v", clone)
	}
	emptied := CloneSample(Sample{Text: "empty labels", Labels: map[string]string{}})
	if emptied.Labels != nil {
		t.Fatalf("CloneSample(empty labels).Labels = %v, want nil", emptied.Labels)
	}
}

// --- CloneSamples ---

func TestSample_CloneSamples_Good(t *testing.T) {
	source := []Sample{{Text: "a", Labels: map[string]string{"k": "v"}}}
	out := CloneSamples(source)
	if len(out) != 1 {
		t.Fatalf("CloneSamples len = %d, want 1", len(out))
	}
	out[0].Labels["k"] = "changed"
	if source[0].Labels["k"] != "v" {
		t.Fatalf("CloneSamples aliased Labels: source k = %q, want 'v'", source[0].Labels["k"])
	}
}

// Bad: CloneSamples has no error channel, so the degenerate input is the
// empty slice — it must return nil (not a zero-length non-nil slice). A
// slice of zero-value Samples clones element-for-element without inventing
// Labels maps.
func TestSample_CloneSamples_Bad(t *testing.T) {
	if got := CloneSamples([]Sample{}); got != nil {
		t.Fatalf("CloneSamples(empty) = %v, want nil", got)
	}
	out := CloneSamples([]Sample{{}, {}})
	if len(out) != 2 {
		t.Fatalf("CloneSamples([2 empties]) len = %d, want 2", len(out))
	}
	for i, s := range out {
		if s.Text != "" || s.Prompt != "" || s.Response != "" || s.Format != "" || s.Labels != nil {
			t.Fatalf("CloneSamples element %d = %+v, want zero Sample", i, s)
		}
	}
}

// Ugly: nil input returns nil (not a zero-length non-nil slice).
func TestSample_CloneSamples_Ugly(t *testing.T) {
	if got := CloneSamples(nil); got != nil {
		t.Fatalf("CloneSamples(nil) = %v, want nil", got)
	}
}
