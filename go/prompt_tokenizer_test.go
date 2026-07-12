// SPDX-Licence-Identifier: EUPL-1.2

package inference

import (
	"testing"

	core "dappco.re/go"
)

// tokenizerCapableBase implements PromptTokenizer alongside TextModel — the
// capability a hiding wrapper (the welfare/policy decorators' shape) must not
// make unreachable to the cross-conversation prefix-sharing probe.
type tokenizerCapableBase struct {
	TextModel
	ids []int32
	err error
}

func (b *tokenizerCapableBase) Tokenize(string) ([]int32, error) { return b.ids, b.err }

// TestAs_PromptTokenizer_Good_Direct finds the tokeniser on the model itself.
func TestAs_PromptTokenizer_Good_Direct(t *testing.T) {
	base := &tokenizerCapableBase{ids: []int32{1, 2, 3}}
	tk, ok := As[PromptTokenizer](base)
	if !ok {
		t.Fatal("As[PromptTokenizer](base) ok=false, want true")
	}
	got, err := tk.Tokenize("hi")
	if err != nil {
		t.Fatalf("Tokenize returned error: %v", err)
	}
	if len(got) != 3 || got[0] != 1 {
		t.Fatalf("Tokenize = %v, want [1 2 3]", got)
	}
}

// TestAs_PromptTokenizer_Good_ThroughHidingWrapper is the regression the whole
// capability exists to survive: a serving decorator that forwards NO optional
// method must not hide the base model's tokeniser from the prefix-sharing
// probe — a direct assertion fails, As must walk Unwrap and find it.
func TestAs_PromptTokenizer_Good_ThroughHidingWrapper(t *testing.T) {
	base := &tokenizerCapableBase{ids: []int32{7, 8}}
	wrapped := hidingWrapper{TextModel: base}

	if _, ok := TextModel(wrapped).(PromptTokenizer); ok {
		t.Fatal("hidingWrapper unexpectedly satisfies PromptTokenizer directly; the regression this test guards is untestable")
	}

	tk, ok := As[PromptTokenizer](wrapped)
	if !ok {
		t.Fatal("As[PromptTokenizer] did not walk the Unwrap chain to the tokeniser-capable base model")
	}
	got, _ := tk.Tokenize("hi")
	if len(got) != 2 || got[1] != 8 {
		t.Fatalf("As[PromptTokenizer] found the wrong model — Tokenize = %v, want [7 8]", got)
	}
}

// TestAs_PromptTokenizer_Ugly_Error surfaces a tokeniser that reports failure
// (a model with no usable tokeniser) rather than swallowing it — the caller
// falls back to a fresh prefill on this error, never a wrong graft.
func TestAs_PromptTokenizer_Ugly_Error(t *testing.T) {
	base := &tokenizerCapableBase{err: core.NewError("no tokenizer")}
	tk, ok := As[PromptTokenizer](hidingWrapper{TextModel: base})
	if !ok {
		t.Fatal("As[PromptTokenizer] must still find a tokeniser that errors")
	}
	if _, err := tk.Tokenize("hi"); err == nil {
		t.Fatal("Tokenize error was swallowed; the caller cannot fall back safely")
	}
}

// TestAs_PromptTokenizer_Bad_NotImplemented reports false when no model in the
// chain tokenises — the probe declines and the serving layer stays on the
// fresh-prefill path.
func TestAs_PromptTokenizer_Bad_NotImplemented(t *testing.T) {
	if _, ok := As[PromptTokenizer](hidingWrapper{TextModel: &bareTextModel{tag: "base"}}); ok {
		t.Fatal("As[PromptTokenizer] found a tokeniser no model in the chain implements")
	}
}
