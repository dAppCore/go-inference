// SPDX-Licence-Identifier: EUPL-1.2

package inference

import "testing"

// bareTextModel is a minimal TextModel used only to identify the innermost
// model after unwrapping; its methods are never called by BaseTextModel.
type bareTextModel struct {
	TextModel
	tag string
}

// oneWrapper wraps a TextModel and re-exposes it via Unwrap.
type oneWrapper struct {
	TextModel
	inner TextModel
}

func (w oneWrapper) Unwrap() TextModel { return w.inner }

// selfWrapper returns itself from Unwrap — the pathological cycle BaseTextModel
// must not spin on.
type selfWrapper struct{ TextModel }

func (w selfWrapper) Unwrap() TextModel { return w }

// TestBaseTextModel_Good_Undecorated returns an undecorated model unchanged.
func TestBaseTextModel_Good_Undecorated(t *testing.T) {
	base := &bareTextModel{tag: "base"}
	if got := BaseTextModel(base); got != base {
		t.Fatalf("BaseTextModel(undecorated) = %v, want the same model", got)
	}
}

// TestBaseTextModel_Good_NestedWrappers unwraps a two-deep decorator chain.
func TestBaseTextModel_Good_NestedWrappers(t *testing.T) {
	base := &bareTextModel{tag: "base"}
	inner := oneWrapper{inner: base}
	outer := oneWrapper{inner: inner}
	if got := BaseTextModel(outer); got != TextModel(base) {
		t.Fatalf("BaseTextModel(nested) did not reach the base model: %v", got)
	}
}

// TestBaseTextModel_Ugly_Cycle returns rather than looping on a self-wrapper.
func TestBaseTextModel_Ugly_Cycle(t *testing.T) {
	w := selfWrapper{}
	if got := BaseTextModel(w); got != TextModel(w) {
		t.Fatalf("BaseTextModel(cycle) = %v, want the wrapper itself (bounded)", got)
	}
}

// TestBaseTextModel_Ugly_Nil tolerates a nil model.
func TestBaseTextModel_Ugly_Nil(t *testing.T) {
	if got := BaseTextModel(nil); got != nil {
		t.Fatalf("BaseTextModel(nil) = %v, want nil", got)
	}
}
