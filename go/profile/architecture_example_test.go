// SPDX-Licence-Identifier: EUPL-1.2

// Runnable usage examples for the architecture-registry surface — the single
// home the loader, memory planner, gguf/hf readers, and LoRA setup all read
// through instead of name-branching on a model family. Each Example mirrors a
// real call site: canonicalise a config signal to an internal id, map a
// HuggingFace class name, ask a metadata-only feature question, or canonicalise
// a checkpoint weight name. Output is pinned to deterministic strings/bools and
// the stable registry order (never a map or %+v dump) so the examples compile
// as assertions.

package profile_test

import (
	"fmt"

	prof "dappco.re/go/inference/profile"
)

// NormalizeArchitecture canonicalises a model_type identifier to the stable id
// the registry dispatches on: it lowercases, trims, folds '-'/'.' to '_', then
// maps known aliases. It is the single source of truth the memory, gguf, model,
// and minimax packages share.
func ExampleNormalizeArchitecture() {
	fmt.Println(prof.NormalizeArchitecture("Qwen3.6"))
	// Output: qwen3_6
}

// A dash-and-caps alias folds and maps to the canonical id in one pass.
func ExampleNormalizeArchitecture_alias() {
	fmt.Println(prof.NormalizeArchitecture("MiniMax-M2"))
	// Output: minimax_m2
}

// An unrecognised value is returned in its normalised form rather than
// guessed at — the loader then reports an unknown architecture honestly.
func ExampleNormalizeArchitecture_unknown() {
	fmt.Println(prof.NormalizeArchitecture("Some-New.Arch"))
	// Output: some_new_arch
}

// A non-ASCII or over-length value takes the heap-stable Lower+Replace
// fallback; the canonicalisation semantics stay identical to the ASCII path.
func ExampleNormalizeArchitecture_nonASCII() {
	fmt.Println(prof.NormalizeArchitecture("Café-Gemma3"))
	// Output: café_gemma3
}

// ArchitectureFromTransformersName maps a HuggingFace architectures class name
// to its canonical go-mlx id. The multimodal Gemma-4 wrapper loads via the base
// gemma4 family, not the text-only tower.
func ExampleArchitectureFromTransformersName() {
	fmt.Println(prof.ArchitectureFromTransformersName("Qwen3MoeForCausalLM"))
	fmt.Println(prof.ArchitectureFromTransformersName("Gemma4ForConditionalGeneration"))
	// Output:
	// qwen3_moe
	// gemma4
}

// A class name that matches no known family returns the empty string, so the
// resolver falls through to its next signal.
func ExampleArchitectureFromTransformersName_unknown() {
	fmt.Printf("%q\n", prof.ArchitectureFromTransformersName("SomethingForCausalLM"))
	// Output: ""
}

// ArchitectureID is the full resolver: it accepts a config model_type or a
// Transformers class name in any casing/separator form and returns the
// canonical id, the form the registry index is keyed on.
func ExampleArchitectureID() {
	fmt.Println(prof.ArchitectureID("Gemma4ForConditionalGeneration"))
	fmt.Println(prof.ArchitectureID("qwen-3.5"))
	// Output:
	// gemma4
	// qwen3_6
}

// ChatTemplateName returns the metadata-only chat-template id advertised for an
// architecture. The Gemma-4 family advertises its own template; a bare qwen id
// falls back to the family default.
func ExampleChatTemplateName() {
	fmt.Println(prof.ChatTemplateName("Gemma4ForConditionalGeneration"))
	fmt.Println(prof.ChatTemplateName("qwen3_6_moe"))
	// Output:
	// gemma4
	// qwen
}

// DefaultThinkingEnabled reports whether an architecture renders its chat
// prompt with reasoning on by default — true for the Gemma-4 family, false for
// families that do not. It is the single home both the metal generation path
// and the serve adapter read, so the two never disagree.
func ExampleDefaultThinkingEnabled() {
	fmt.Println(prof.DefaultThinkingEnabled("gemma4"))
	fmt.Println(prof.DefaultThinkingEnabled("qwen3"))
	// Output:
	// true
	// false
}

// AttachedOnlyArchitecture reports whether a family can only load attached to a
// target (an MTP assistant drafter), never standalone. The loader reads this to
// reject a standalone load instead of name-branching on the architecture.
func ExampleAttachedOnlyArchitecture() {
	fmt.Println(prof.AttachedOnlyArchitecture("gemma4_assistant"))
	fmt.Println(prof.AttachedOnlyArchitecture("gemma4"))
	// Output:
	// true
	// false
}

// IsGemma4TargetArchitecture reports whether an architecture is a Gemma-4
// target that can own prompts, LoRA adapters, and fused packs. The attached
// drafter is deliberately excluded even though it is a Gemma-4 family member.
func ExampleIsGemma4TargetArchitecture() {
	fmt.Println(prof.IsGemma4TargetArchitecture("Gemma4ForConditionalGeneration"))
	fmt.Println(prof.IsGemma4TargetArchitecture("gemma4_assistant"))
	// Output:
	// true
	// false
}

// DefaultLoRATargets returns the registered narrow default LoRA target set for
// a family — the keys applied when a caller requests a LoRA without explicit
// targets. The result is a copy; an unknown family yields nil rather than a
// guess.
func ExampleDefaultLoRATargets() {
	fmt.Println(prof.DefaultLoRATargets("gemma4"))
	fmt.Println(prof.DefaultLoRATargets("nonexistent_family") == nil)
	// Output:
	// [q_proj v_proj o_proj]
	// true
}

// LoRATargetPath canonicalises a LoRA target key into the projection path
// adapter metadata uses; SafeLoRATarget reports whether that target is safe to
// enable by default (resolves to a known path not in the family's opt-in
// extended set).
func ExampleLoRATargetPath() {
	path, ok := prof.LoRATargetPath("gemma4", "gate_proj")
	fmt.Println(path, ok)
	fmt.Println(prof.SafeLoRATarget("gemma4", "gate_proj"))
	fmt.Println(prof.SafeLoRATarget("gemma4", "router.proj"))
	// Output:
	// mlp.gate_proj true
	// true
	// false
}

// CanonicalWeightName strips the family's declared checkpoint wrapper prefixes,
// drops non-text helper tensors (ok=false), and re-roots text tensors under
// "model.". An architecture with no weight rules passes the name through
// unchanged.
func ExampleCanonicalWeightName() {
	name, ok := prof.CanonicalWeightName("gemma4", "language_model.model.layers.0.self_attn.q_proj.weight")
	fmt.Println(name, ok)
	_, ok = prof.CanonicalWeightName("gemma4", "model.vision_tower.patch_embedding.weight")
	fmt.Println(ok)
	// Output:
	// model.layers.0.self_attn.q_proj.weight true
	// false
}

// ArchitectureIDs lists every built-in architecture id in registry order — the
// list a capability report or a `--list-architectures` surface enumerates.
func ExampleArchitectureIDs() {
	ids := prof.ArchitectureIDs()
	fmt.Println(ids[0], ids[1], ids[2])
	// Output: gemma2 gemma3 gemma3_text
}

// BuiltinArchitectureProfiles returns the full metadata-only feature target
// list as defensive deep clones — the registry the loader, memory planner, and
// gguf/hf readers read through. The first entries follow the stable registry
// order.
func ExampleBuiltinArchitectureProfiles() {
	profiles := prof.BuiltinArchitectureProfiles()
	fmt.Println(profiles[0].ID, profiles[0].Family)
	// Output: gemma2 gemma
}

// LookupArchitectureProfile resolves a config model_type or a Transformers
// class name to a defensive deep-clone of the built-in profile; external
// callers may mutate the result without touching the shared registry.
func ExampleLookupArchitectureProfile() {
	p, ok := prof.LookupArchitectureProfile("Qwen3MoeForCausalLM")
	fmt.Println(ok, p.ID, p.MoE)
	// Output: true qwen3_moe true
}

// LookupArchitectureProfileRef resolves an architecture name to a pointer into
// the immutable registry — the hot-path lookup that skips the per-call clone.
// The returned pointer must not be mutated.
func ExampleLookupArchitectureProfileRef() {
	ref, ok := prof.LookupArchitectureProfileRef("gemma4")
	fmt.Println(ok, ref.ID, ref.DefaultThinking)
	// Output: true gemma4 true
}

// IsGemma4LargeVariant reports whether Gemma-4 prompt rendering should take the
// large-variant suppressor path — true for a gemma4 target with at least 16
// attention heads, false for smaller head counts or non-target families.
func ExampleIsGemma4LargeVariant() {
	fmt.Println(prof.IsGemma4LargeVariant("Gemma4ForConditionalGeneration", 16))
	fmt.Println(prof.IsGemma4LargeVariant("gemma4_text", 8))
	// Output:
	// true
	// false
}

// SafeLoRATarget reports whether a LoRA target is safe to enable by default for
// a family — it resolves to a known projection path that is not in the family's
// opt-in extended set. The attention projections are safe; the router is opt-in.
func ExampleSafeLoRATarget() {
	fmt.Println(prof.SafeLoRATarget("gemma4", "q_proj"))
	fmt.Println(prof.SafeLoRATarget("gemma4", "router.proj"))
	// Output:
	// true
	// false
}

// TrimWeightWrapperPrefix removes one of a family's declared checkpoint wrapper
// prefixes, reporting whether one matched. An unknown architecture is a no-op.
func ExampleTrimWeightWrapperPrefix() {
	trimmed, ok := prof.TrimWeightWrapperPrefix("gemma4", "language_model.model.layers.0")
	fmt.Println(trimmed, ok)
	// Output: layers.0 true
}
