// SPDX-Licence-Identifier: EUPL-1.2

// Coverage tests for the pack utilities — option apply + reduction
// (ApplyOptions, WithPack*), issue accumulation (AddIssue), and the
// validity / summary predicates (Valid, HasIssue, HasErrorIssue,
// IssueSummary). The package is pure data-shaping with no I/O, so every
// test drives the real symbol with concrete args and asserts on the
// returned value.
//
// Good = happy / present path, Bad = the negative outcome, Ugly = the
// empty / nil-slice edge. IssueSummary carries the only branch structure
// worth pinning: empty, all-warnings, single-error, multi-error.

package modelpack

import "testing"

// --- ApplyOptions ---

func TestPack_ApplyOptions_Good(t *testing.T) {
	cfg := ApplyOptions([]ModelPackOption{
		WithPackQuantization(4),
		WithPackMaxContextLength(131072),
		WithPackRequireChatTemplate(false),
	})
	if cfg.ExpectedQuantBits != 4 {
		t.Fatalf("ExpectedQuantBits = %d, want 4", cfg.ExpectedQuantBits)
	}
	if cfg.MaxContextLength != 131072 {
		t.Fatalf("MaxContextLength = %d, want 131072", cfg.MaxContextLength)
	}
	if cfg.RequireChatTemplate {
		t.Fatalf("RequireChatTemplate = true, want false (option set it off)")
	}
}

func TestPack_ApplyOptions_Bad(t *testing.T) {
	// Later options for the same field win — a conflicting pair must
	// resolve to the last applied, not the first.
	cfg := ApplyOptions([]ModelPackOption{
		WithPackQuantization(8),
		WithPackQuantization(4),
	})
	if cfg.ExpectedQuantBits != 4 {
		t.Fatalf("ExpectedQuantBits = %d, want 4 (last option wins)", cfg.ExpectedQuantBits)
	}
}

func TestPack_ApplyOptions_Ugly(t *testing.T) {
	// Zero options must take the fast-path default: RequireChatTemplate
	// true, everything else zero. nil and empty slice behave the same.
	for _, opts := range [][]ModelPackOption{nil, {}} {
		cfg := ApplyOptions(opts)
		if !cfg.RequireChatTemplate {
			t.Fatalf("RequireChatTemplate = false, want true by default (opts=%v)", opts)
		}
		if cfg.ExpectedQuantBits != 0 {
			t.Fatalf("ExpectedQuantBits = %d, want 0 (opts=%v)", cfg.ExpectedQuantBits, opts)
		}
		if cfg.MaxContextLength != 0 {
			t.Fatalf("MaxContextLength = %d, want 0 (opts=%v)", cfg.MaxContextLength, opts)
		}
	}
}

// --- WithPackQuantization ---

func TestPack_WithPackQuantization_Good(t *testing.T) {
	var cfg ModelPackConfig
	WithPackQuantization(8)(&cfg)
	if cfg.ExpectedQuantBits != 8 {
		t.Fatalf("ExpectedQuantBits = %d, want 8", cfg.ExpectedQuantBits)
	}
}

func TestPack_WithPackQuantization_Bad(t *testing.T) {
	// A zero width is a legitimate "no expectation" — the option must
	// faithfully record it rather than coerce to a default.
	cfg := ModelPackConfig{ExpectedQuantBits: 4}
	WithPackQuantization(0)(&cfg)
	if cfg.ExpectedQuantBits != 0 {
		t.Fatalf("ExpectedQuantBits = %d, want 0 (option records the value verbatim)", cfg.ExpectedQuantBits)
	}
}

func TestPack_WithPackQuantization_Ugly(t *testing.T) {
	// Negative bits is nonsensical but the option is a pure setter — it
	// stores what it is given without validation.
	var cfg ModelPackConfig
	WithPackQuantization(-1)(&cfg)
	if cfg.ExpectedQuantBits != -1 {
		t.Fatalf("ExpectedQuantBits = %d, want -1 (no clamping)", cfg.ExpectedQuantBits)
	}
}

// --- WithPackMaxContextLength ---

func TestPack_WithPackMaxContextLength_Good(t *testing.T) {
	var cfg ModelPackConfig
	WithPackMaxContextLength(8192)(&cfg)
	if cfg.MaxContextLength != 8192 {
		t.Fatalf("MaxContextLength = %d, want 8192", cfg.MaxContextLength)
	}
}

func TestPack_WithPackMaxContextLength_Bad(t *testing.T) {
	// Zero means "no cap"; the option must store it, overwriting a prior
	// limit rather than ignoring the call.
	cfg := ModelPackConfig{MaxContextLength: 4096}
	WithPackMaxContextLength(0)(&cfg)
	if cfg.MaxContextLength != 0 {
		t.Fatalf("MaxContextLength = %d, want 0 (cleared the prior cap)", cfg.MaxContextLength)
	}
}

func TestPack_WithPackMaxContextLength_Ugly(t *testing.T) {
	var cfg ModelPackConfig
	WithPackMaxContextLength(-100)(&cfg)
	if cfg.MaxContextLength != -100 {
		t.Fatalf("MaxContextLength = %d, want -100 (no clamping)", cfg.MaxContextLength)
	}
}

// --- WithPackRequireChatTemplate ---

func TestPack_WithPackRequireChatTemplate_Good(t *testing.T) {
	// Default cfg has RequireChatTemplate false; the option flips it on.
	var cfg ModelPackConfig
	WithPackRequireChatTemplate(true)(&cfg)
	if !cfg.RequireChatTemplate {
		t.Fatalf("RequireChatTemplate = false, want true")
	}
}

func TestPack_WithPackRequireChatTemplate_Bad(t *testing.T) {
	// Must be able to turn the requirement OFF against a config that
	// already has it on (the ApplyOptions default).
	cfg := ModelPackConfig{RequireChatTemplate: true}
	WithPackRequireChatTemplate(false)(&cfg)
	if cfg.RequireChatTemplate {
		t.Fatalf("RequireChatTemplate = true, want false")
	}
}

func TestPack_WithPackRequireChatTemplate_Ugly(t *testing.T) {
	// Applying the same value twice is idempotent — the second call must
	// not toggle it back.
	var cfg ModelPackConfig
	WithPackRequireChatTemplate(true)(&cfg)
	WithPackRequireChatTemplate(true)(&cfg)
	if !cfg.RequireChatTemplate {
		t.Fatalf("RequireChatTemplate = false, want true (idempotent set)")
	}
}

// --- AddIssue ---

func TestPack_AddIssue_Good(t *testing.T) {
	var p ModelPack
	p.AddIssue(ModelPackIssueError, ModelPackIssueMissingConfig, "config missing", "/m/config.json")
	if len(p.Issues) != 1 {
		t.Fatalf("len(Issues) = %d, want 1", len(p.Issues))
	}
	got := p.Issues[0]
	if got.Severity != ModelPackIssueError {
		t.Fatalf("Severity = %q, want %q", got.Severity, ModelPackIssueError)
	}
	if got.Code != ModelPackIssueMissingConfig {
		t.Fatalf("Code = %q, want %q", got.Code, ModelPackIssueMissingConfig)
	}
	if got.Message != "config missing" {
		t.Fatalf("Message = %q, want %q", got.Message, "config missing")
	}
	if got.Path != "/m/config.json" {
		t.Fatalf("Path = %q, want %q", got.Path, "/m/config.json")
	}
}

func TestPack_AddIssue_Bad(t *testing.T) {
	// Repeated AddIssue must accumulate, not replace — the slice grows
	// and preserves order.
	var p ModelPack
	p.AddIssue(ModelPackIssueWarning, ModelPackIssueMissingChatTemplate, "first", "")
	p.AddIssue(ModelPackIssueError, ModelPackIssueMissingWeights, "second", "/m")
	if len(p.Issues) != 2 {
		t.Fatalf("len(Issues) = %d, want 2", len(p.Issues))
	}
	if p.Issues[0].Code != ModelPackIssueMissingChatTemplate {
		t.Fatalf("Issues[0].Code = %q, want %q", p.Issues[0].Code, ModelPackIssueMissingChatTemplate)
	}
	if p.Issues[1].Code != ModelPackIssueMissingWeights {
		t.Fatalf("Issues[1].Code = %q, want %q", p.Issues[1].Code, ModelPackIssueMissingWeights)
	}
}

func TestPack_AddIssue_Ugly(t *testing.T) {
	// Empty message and empty path are valid inputs — AddIssue stores the
	// record verbatim without substituting placeholders.
	var p ModelPack
	p.AddIssue(ModelPackIssueWarning, ModelPackIssueMissingArchitecture, "", "")
	if len(p.Issues) != 1 {
		t.Fatalf("len(Issues) = %d, want 1", len(p.Issues))
	}
	if p.Issues[0].Message != "" {
		t.Fatalf("Message = %q, want empty", p.Issues[0].Message)
	}
	if p.Issues[0].Path != "" {
		t.Fatalf("Path = %q, want empty", p.Issues[0].Path)
	}
}

// --- Valid ---

func TestPack_Valid_Good(t *testing.T) {
	p := ModelPack{OK: true}
	if !p.Valid() {
		t.Fatalf("Valid() = false, want true when OK is set")
	}
}

func TestPack_Valid_Bad(t *testing.T) {
	p := ModelPack{OK: false}
	if p.Valid() {
		t.Fatalf("Valid() = true, want false when OK is clear")
	}
}

func TestPack_Valid_Ugly(t *testing.T) {
	// Valid reflects only OK, never the presence of issues — a pack with
	// error issues but OK=true still reports valid (the flag is the
	// source of truth, the issue list is advisory).
	var p ModelPack
	if p.Valid() {
		t.Fatalf("Valid() = true on zero value, want false")
	}
	p.AddIssue(ModelPackIssueError, ModelPackIssueMissingWeights, "x", "")
	p.OK = true
	if !p.Valid() {
		t.Fatalf("Valid() = false, want true — Valid keys off OK, not Issues")
	}
}

// --- HasIssue ---

func TestPack_HasIssue_Good(t *testing.T) {
	var p ModelPack
	p.AddIssue(ModelPackIssueError, ModelPackIssueContextTooLarge, "ctx", "")
	if !p.HasIssue(ModelPackIssueContextTooLarge) {
		t.Fatalf("HasIssue(ContextTooLarge) = false, want true")
	}
}

func TestPack_HasIssue_Bad(t *testing.T) {
	// Present issues but not the queried code — must report false.
	var p ModelPack
	p.AddIssue(ModelPackIssueError, ModelPackIssueMissingConfig, "x", "")
	p.AddIssue(ModelPackIssueWarning, ModelPackIssueMissingChatTemplate, "y", "")
	if p.HasIssue(ModelPackIssueInvalidGGUF) {
		t.Fatalf("HasIssue(InvalidGGUF) = true, want false (code not present)")
	}
}

func TestPack_HasIssue_Ugly(t *testing.T) {
	// Empty issue list: any query is false, including the zero-value code.
	var p ModelPack
	if p.HasIssue(ModelPackIssueMissingConfig) {
		t.Fatalf("HasIssue on empty pack = true, want false")
	}
	if p.HasIssue("") {
		t.Fatalf("HasIssue(\"\") on empty pack = true, want false")
	}
}

// --- HasErrorIssue ---

func TestPack_HasErrorIssue_Good(t *testing.T) {
	var p ModelPack
	p.AddIssue(ModelPackIssueWarning, ModelPackIssueQuantizationMismatch, "warn", "")
	p.AddIssue(ModelPackIssueError, ModelPackIssueMissingWeights, "err", "")
	if !p.HasErrorIssue() {
		t.Fatalf("HasErrorIssue() = false, want true (one error present)")
	}
}

func TestPack_HasErrorIssue_Bad(t *testing.T) {
	// Issues present but all warning-severity — must report false.
	var p ModelPack
	p.AddIssue(ModelPackIssueWarning, ModelPackIssueMissingChatTemplate, "w1", "")
	p.AddIssue(ModelPackIssueWarning, ModelPackIssueQuantizationMismatch, "w2", "")
	if p.HasErrorIssue() {
		t.Fatalf("HasErrorIssue() = true, want false (warnings only)")
	}
}

func TestPack_HasErrorIssue_Ugly(t *testing.T) {
	var p ModelPack
	if p.HasErrorIssue() {
		t.Fatalf("HasErrorIssue() = true on empty pack, want false")
	}
}

// --- IssueSummary — four branches: empty, all-warnings, single, multi ---

func TestPack_IssueSummary_Good(t *testing.T) {
	// Multi-error path: comma-space separated, error codes only, in order.
	var p ModelPack
	p.AddIssue(ModelPackIssueError, ModelPackIssueMissingConfig, "c", "")
	p.AddIssue(ModelPackIssueWarning, ModelPackIssueMissingChatTemplate, "w", "")
	p.AddIssue(ModelPackIssueError, ModelPackIssueContextTooLarge, "ctx", "")
	got := p.IssueSummary()
	want := string(ModelPackIssueMissingConfig) + ", " + string(ModelPackIssueContextTooLarge)
	if got != want {
		t.Fatalf("IssueSummary() = %q, want %q (errors only, warnings skipped)", got, want)
	}
}

func TestPack_IssueSummary_Bad(t *testing.T) {
	// Issues present but every one is a warning → no error codes →
	// "unknown" via the count==0 branch (distinct from the len==0 branch).
	var p ModelPack
	p.AddIssue(ModelPackIssueWarning, ModelPackIssueMissingChatTemplate, "w1", "")
	p.AddIssue(ModelPackIssueWarning, ModelPackIssueQuantizationMismatch, "w2", "")
	if got := p.IssueSummary(); got != "unknown" {
		t.Fatalf("IssueSummary() = %q, want %q (all warnings)", got, "unknown")
	}
}

func TestPack_IssueSummary_Single(t *testing.T) {
	// Single error → no separator emitted.
	var p ModelPack
	p.AddIssue(ModelPackIssueError, ModelPackIssueInvalidConfig, "bad json", "")
	got := p.IssueSummary()
	want := string(ModelPackIssueInvalidConfig)
	if got != want {
		t.Fatalf("IssueSummary() = %q, want %q (single error, no separator)", got, want)
	}
}

func TestPack_IssueSummary_Ugly(t *testing.T) {
	// Empty issue list → "unknown" via the len==0 fast path.
	var p ModelPack
	if got := p.IssueSummary(); got != "unknown" {
		t.Fatalf("IssueSummary() = %q, want %q (no issues)", got, "unknown")
	}
}
