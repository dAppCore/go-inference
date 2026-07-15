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

// visionCapableBase implements VisionModel alongside TextModel — the
// capability a hiding wrapper (below) must not make unreachable.
type visionCapableBase struct {
	TextModel
	accepts bool
}

func (v *visionCapableBase) AcceptsImages() bool { return v.accepts }

// hidingWrapper wraps a TextModel and exposes Unwrap but — unlike the real
// welfareTextModel/policyTextModel/profileModel decorators — forwards NO
// optional capability. It is the shape of a NEW decorator whose author never
// wrote the AcceptsImages/AcceptsAudio boilerplate: As must still find the
// capability by walking Unwrap, where a direct type assertion cannot.
type hidingWrapper struct {
	TextModel
}

func (w hidingWrapper) Unwrap() TextModel { return w.TextModel }

// TestAs_Good_Direct finds a capability the model implements itself, with no
// unwrap needed.
func TestAs_Good_Direct(t *testing.T) {
	base := &visionCapableBase{accepts: true}
	vision, ok := As[VisionModel](base)
	if !ok {
		t.Fatal("As[VisionModel](base) ok=false, want true")
	}
	if !vision.AcceptsImages() {
		t.Fatal("As[VisionModel](base).AcceptsImages() = false, want true")
	}
}

// TestAs_Good_ThroughHidingWrapper is the regression the mechanism exists to
// prevent: a wrapper that never forwards AcceptsImages must not silently hide
// a vision-capable base model from the capability gate — the bug class a
// per-wrapper forward pattern leaves open for every wrapper that forgets one.
func TestAs_Good_ThroughHidingWrapper(t *testing.T) {
	base := &visionCapableBase{accepts: true}
	wrapped := hidingWrapper{TextModel: base}

	// A direct type assertion against the wrapper fails — proving the bug
	// class is real for any capability a wrapper doesn't forward by hand, and
	// that this test would prove nothing if it didn't.
	if _, ok := TextModel(wrapped).(VisionModel); ok {
		t.Fatal("hidingWrapper unexpectedly satisfies VisionModel directly; the regression this test guards is untestable")
	}

	vision, ok := As[VisionModel](wrapped)
	if !ok {
		t.Fatal("As[VisionModel] did not walk the Unwrap chain to the capable base model")
	}
	if !vision.AcceptsImages() {
		t.Fatal("As[VisionModel] found the wrong model — AcceptsImages() = false, want true")
	}
}

// TestAs_Good_NestedWrappers walks a two-deep decorator chain, mirroring
// TestBaseTextModel_Good_NestedWrappers for the capability-probe form.
func TestAs_Good_NestedWrappers(t *testing.T) {
	base := &visionCapableBase{accepts: true}
	inner := hidingWrapper{TextModel: base}
	outer := hidingWrapper{TextModel: inner}
	vision, ok := As[VisionModel](outer)
	if !ok || !vision.AcceptsImages() {
		t.Fatalf("As[VisionModel](nested) = (%v, %v), want a capable model and true", vision, ok)
	}
}

// TestAs_Bad_NotImplementedAnywhere reports false when no model in the chain
// implements the requested capability — no partial or zero-value match.
func TestAs_Bad_NotImplementedAnywhere(t *testing.T) {
	wrapped := hidingWrapper{TextModel: &bareTextModel{tag: "base"}}
	if _, ok := As[VisionModel](wrapped); ok {
		t.Fatal("As[VisionModel] found a capability that no model in the chain implements")
	}
}

// TestAs_Ugly_Cycle returns promptly rather than spinning on a self-wrapper,
// mirroring TestBaseTextModel_Ugly_Cycle for the capability-probe form.
func TestAs_Ugly_Cycle(t *testing.T) {
	w := selfWrapper{}
	if _, ok := As[VisionModel](w); ok {
		t.Fatal("As[VisionModel](cycle) ok=true, want false — selfWrapper implements no capability")
	}
}

// TestAs_Ugly_Nil tolerates a nil model.
func TestAs_Ugly_Nil(t *testing.T) {
	if _, ok := As[VisionModel](nil); ok {
		t.Fatal("As[VisionModel](nil) ok=true, want false")
	}
}
