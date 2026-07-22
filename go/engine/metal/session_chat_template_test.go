// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"reflect"
	"testing"

	"dappco.re/go/inference/engine"
)

// TestSessionTextModel_DeclaredChatTemplate_ChatML proves the real serve wrap
// (sessionTextModel, built by metalBackend.LoadModel for every composed/hybrid
// checkpoint) declares the ChatML dialect for a qwen-family model_type — the
// exact chatMLChatTemplate() this package renders through — rather than the
// composed.ChatMLDialect check living unreached above the wrap.
func TestSessionTextModel_DeclaredChatTemplate_ChatML(t *testing.T) {
	for _, mt := range []string{"qwen3_5", "qwen3_5_moe", "qwen3_6", "qwen3_6_moe", "qwen3_next"} {
		m := &sessionTextModel{modelType: mt}
		got, ok := m.DeclaredChatTemplate()
		if !ok {
			t.Fatalf("DeclaredChatTemplate(%q) ok = false, want true", mt)
		}
		if want := chatMLChatTemplate(); !reflect.DeepEqual(got, want) {
			t.Fatalf("DeclaredChatTemplate(%q) = %+v, want the ChatML template %+v", mt, got, want)
		}
	}
}

// TestSessionTextModel_DeclaredChatTemplate_GemmaFallback proves a non-qwen
// composed model_type (the generic "composed"/"hybrid" ids, or any future
// non-qwen family) keeps the gemma fallback rather than ChatML — the declared
// template still wins (ok=true), it just isn't the ChatML dialect.
func TestSessionTextModel_DeclaredChatTemplate_GemmaFallback(t *testing.T) {
	for _, mt := range []string{"composed", "hybrid", "gemma4", ""} {
		m := &sessionTextModel{modelType: mt}
		got, ok := m.DeclaredChatTemplate()
		if !ok {
			t.Fatalf("DeclaredChatTemplate(%q) ok = false, want true", mt)
		}
		if got.Open == "<|im_start|>" {
			t.Fatalf("DeclaredChatTemplate(%q) = %+v, leaked ChatML for a non-qwen composed arch", mt, got)
		}
		want := engine.GemmaChatTemplate(engine.DetectTurnTokens(nil), false)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("DeclaredChatTemplate(%q) = %+v, want the gemma fallback %+v", mt, got, want)
		}
	}
}

// TestSessionTextModel_DeclaredChatTemplate_PlainCompletion proves the standalone-recurrent archs
// (rwkv7, mamba2) declare the plain-completion dialect — no bracket markers, no role labels — rather
// than leaking ChatML or the gemma fallback's fabricated <start_of_turn>/<end_of_turn> markers, neither
// of which either checkpoint's vocabulary carries.
func TestSessionTextModel_DeclaredChatTemplate_PlainCompletion(t *testing.T) {
	for _, mt := range []string{"rwkv7", "mamba2"} {
		m := &sessionTextModel{modelType: mt}
		got, ok := m.DeclaredChatTemplate()
		if !ok {
			t.Fatalf("DeclaredChatTemplate(%q) ok = false, want true", mt)
		}
		if want := plainCompletionChatTemplate(); !reflect.DeepEqual(got, want) {
			t.Fatalf("DeclaredChatTemplate(%q) = %+v, want the plain-completion template %+v", mt, got, want)
		}
		if got.Open != "" || got.Close != "" {
			t.Fatalf("DeclaredChatTemplate(%q) = %+v, want no bracket markers (neither checkpoint's vocab carries any)", mt, got)
		}
	}
}
