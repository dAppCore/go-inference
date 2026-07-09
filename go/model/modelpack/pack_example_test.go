// SPDX-Licence-Identifier: EUPL-1.2

// Runnable usage examples for the pack public API. Each example calls the
// real symbol and pins its observable contract via an Output comment, so
// these double as executable documentation and coverage.

package modelpack

import core "dappco.re/go"

// ApplyOptions reduces functional options onto a ModelPackConfig. With no
// options it returns the default: chat template required, no quant or
// context expectations.
func ExampleApplyOptions() {
	cfg := ApplyOptions(nil)
	core.Println(core.Sprintf("bits=%d ctx=%d chat=%v", cfg.ExpectedQuantBits, cfg.MaxContextLength, cfg.RequireChatTemplate))
	// Output: bits=0 ctx=0 chat=true
}

// WithPackQuantization records an expected quantization width.
func ExampleWithPackQuantization() {
	cfg := ApplyOptions([]ModelPackOption{WithPackQuantization(4)})
	core.Println(core.Sprintf("bits=%d", cfg.ExpectedQuantBits))
	// Output: bits=4
}

// WithPackMaxContextLength caps the declared context a pack may advertise.
func ExampleWithPackMaxContextLength() {
	cfg := ApplyOptions([]ModelPackOption{WithPackMaxContextLength(8192)})
	core.Println(core.Sprintf("max=%d", cfg.MaxContextLength))
	// Output: max=8192
}

// WithPackRequireChatTemplate toggles whether a chat template is mandatory.
func ExampleWithPackRequireChatTemplate() {
	cfg := ApplyOptions([]ModelPackOption{WithPackRequireChatTemplate(false)})
	core.Println(core.Sprintf("required=%v", cfg.RequireChatTemplate))
	// Output: required=false
}

// AddIssue appends a validation finding; Valid keys off the OK flag.
func ExampleModelPack_AddIssue() {
	var p ModelPack
	p.AddIssue(ModelPackIssueError, ModelPackIssueMissingWeights, "no safetensors or gguf", "/models/foo")
	core.Println(core.Sprintf("issues=%d code=%s", len(p.Issues), p.Issues[0].Code))
	// Output: issues=1 code=missing_weights
}

// Valid reports the OK flag — the authoritative validity gate.
func ExampleModelPack_Valid() {
	valid := ModelPack{OK: true}
	invalid := ModelPack{OK: false}
	core.Println(core.Sprintf("%v %v", valid.Valid(), invalid.Valid()))
	// Output: true false
}

// HasIssue checks for a specific machine-readable code.
func ExampleModelPack_HasIssue() {
	var p ModelPack
	p.AddIssue(ModelPackIssueError, ModelPackIssueContextTooLarge, "ctx 200000 > 131072", "")
	core.Println(core.Sprintf("%v %v", p.HasIssue(ModelPackIssueContextTooLarge), p.HasIssue(ModelPackIssueInvalidGGUF)))
	// Output: true false
}

// HasErrorIssue reports whether any finding is error-severity; warnings
// alone do not trip it.
func ExampleModelPack_HasErrorIssue() {
	var p ModelPack
	p.AddIssue(ModelPackIssueWarning, ModelPackIssueMissingChatTemplate, "advisory", "")
	core.Println(core.Sprintf("warn-only=%v", p.HasErrorIssue()))
	p.AddIssue(ModelPackIssueError, ModelPackIssueMissingConfig, "fatal", "")
	core.Println(core.Sprintf("with-error=%v", p.HasErrorIssue()))
	// Output:
	// warn-only=false
	// with-error=true
}

// IssueSummary joins error-severity codes with ", "; warnings are skipped
// and an empty (or warning-only) pack summarises as "unknown".
func ExampleModelPack_IssueSummary() {
	var p ModelPack
	p.AddIssue(ModelPackIssueError, ModelPackIssueMissingConfig, "c", "")
	p.AddIssue(ModelPackIssueWarning, ModelPackIssueMissingChatTemplate, "w", "")
	p.AddIssue(ModelPackIssueError, ModelPackIssueUnsupportedRuntime, "r", "")
	core.Println(p.IssueSummary())
	core.Println(ModelPack{}.IssueSummary())
	// Output:
	// missing_config, unsupported_runtime
	// unknown
}
