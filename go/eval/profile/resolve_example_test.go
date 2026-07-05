// SPDX-Licence-Identifier: EUPL-1.2

// Runnable usage examples for ResolveArchitecture — the single home for the
// config-probe → registered-id resolution ORDER (top-level model_type, then a
// declared text tower, then the architectures fallback) and the two family
// refinements that used to live as name-branches in the metal loader. Each
// Example mirrors how the loader feeds the three config signals in and reads
// back one canonical id; the empty result is the honest "unrecognised" path.

package profile_test

import (
	"fmt"

	prof "dappco.re/go/inference/eval/profile"
)

// A Gemma-4 multimodal wrapper whose text_config names its text tower resolves
// to that tower, so the loader dispatches on the text id without name-branching
// on "gemma4". This is the documented headline case.
func ExampleResolveArchitecture() {
	id := prof.ResolveArchitecture("gemma4", "gemma4_text", []string{"Gemma4ForConditionalGeneration"})
	fmt.Println(id)
	// Output: gemma4_text
}

// With no top-level model_type, a declared text tower is canonicalised and
// returned — the qwen3.5 text tower folds to the qwen3_6 id.
func ExampleResolveArchitecture_textTowerFallback() {
	id := prof.ResolveArchitecture("", "qwen3_5_text", []string{"Qwen3_5ForConditionalGeneration"})
	fmt.Println(id)
	// Output: qwen3_6
}

// With neither a model_type nor a text tower, the first architectures class
// name that maps to a known family wins.
func ExampleResolveArchitecture_architecturesFallback() {
	id := prof.ResolveArchitecture("", "", []string{"MistralForCausalLM"})
	fmt.Println(id)
	// Output: mistral
}

// A BERT encoder and a BERT cross-encoder differ only in the architectures
// class list: a sequence-classification head refines the base encoder id to the
// rerank sibling registered in the same family.
func ExampleResolveArchitecture_rerankRefinement() {
	plain := prof.ResolveArchitecture("bert", "", []string{"BertModel"})
	rerank := prof.ResolveArchitecture("bert", "", []string{"BertForSequenceClassification"})
	fmt.Println(plain, rerank)
	// Output: bert bert_rerank
}

// When none of the three signals name a recognised architecture the result is
// the empty string, so the loader reports an unknown model rather than guessing.
func ExampleResolveArchitecture_unrecognised() {
	fmt.Printf("%q\n", prof.ResolveArchitecture("", "", nil))
	// Output: ""
}
