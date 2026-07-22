// SPDX-Licence-Identifier: EUPL-1.2

package whisper

import "testing"

func realGenerationConfig(t *testing.T) *GenerationConfig {
	t.Helper()
	g, err := LoadGenerationConfig("testdata")
	if err != nil {
		t.Fatalf("LoadGenerationConfig: %v", err)
	}
	return g
}

// TestLoadGenerationConfig_Good parses the REAL generation_config.json shipped with openai/whisper-tiny.
func TestLoadGenerationConfig_Good(t *testing.T) {
	g := realGenerationConfig(t)
	if g.DecoderStartTokenID != 50258 {
		t.Fatalf("DecoderStartTokenID = %d, want 50258 (<|startoftranscript|>)", g.DecoderStartTokenID)
	}
	if g.EOSTokenID != 50257 {
		t.Fatalf("EOSTokenID = %d, want 50257 (<|endoftext|>)", g.EOSTokenID)
	}
	if g.NoTimestampsTokenID != 50363 {
		t.Fatalf("NoTimestampsTokenID = %d, want 50363", g.NoTimestampsTokenID)
	}
	if g.MaxLength != 448 {
		t.Fatalf("MaxLength = %d, want 448", g.MaxLength)
	}
	if len(g.LangToID) < 90 {
		t.Fatalf("len(LangToID) = %d, want ~99", len(g.LangToID))
	}
	if id, ok := g.TaskToID["transcribe"]; !ok || id != 50359 {
		t.Fatalf("TaskToID[transcribe] = %d, ok=%v; want 50359", id, ok)
	}
	if len(g.SuppressTokens) == 0 || len(g.BeginSuppressTokens) == 0 {
		t.Fatal("suppress_tokens/begin_suppress_tokens must not be empty on the real checkpoint")
	}
}

func TestLoadGenerationConfig_Bad(t *testing.T) {
	if _, err := LoadGenerationConfig(t.TempDir()); err == nil {
		t.Fatal("LoadGenerationConfig accepted a directory with no generation_config.json")
	}
}

// TestLoadGenerationConfig_Ugly proves a syntactically valid but task-token-empty document is refused
// (missing decoder_start_token_id/lang_to_id/task_to_id) rather than silently zero-valued.
func TestLoadGenerationConfig_Ugly(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "generation_config.json", `{"eos_token_id": 1}`)
	if _, err := LoadGenerationConfig(dir); err == nil {
		t.Fatal("LoadGenerationConfig accepted a document with no task-token machinery")
	}
}

// TestLoadGenerationConfig_MissingTranscribe_Bad proves task_to_id specifically needing a "transcribe"
// entry is checked (translate-only would otherwise load silently and fail much later, mid-decode).
func TestLoadGenerationConfig_MissingTranscribe_Bad(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "generation_config.json", `{"decoder_start_token_id":1,"lang_to_id":{"<|en|>":2},"task_to_id":{"translate":3}}`)
	if _, err := LoadGenerationConfig(dir); err == nil {
		t.Fatal("LoadGenerationConfig accepted task_to_id with no \"transcribe\" entry")
	}
}

func TestGenerationConfig_LanguageTokenID_Good(t *testing.T) {
	g := realGenerationConfig(t)
	if id, ok := g.LanguageTokenID("en"); !ok || id != 50259 {
		t.Fatalf("LanguageTokenID(\"en\") = %d, ok=%v; want 50259", id, ok)
	}
}

// TestGenerationConfig_LanguageTokenID_Bad proves the bracketed form also resolves — "<|en|>" and "en"
// must be equivalent inputs (the --language flag accepts either, per BuildInitTokens' doc comment).
func TestGenerationConfig_LanguageTokenID_Bad(t *testing.T) {
	g := realGenerationConfig(t)
	id1, ok1 := g.LanguageTokenID("en")
	id2, ok2 := g.LanguageTokenID("<|en|>")
	if !ok1 || !ok2 || id1 != id2 {
		t.Fatalf("LanguageTokenID(\"en\")=%d/%v and LanguageTokenID(\"<|en|>\")=%d/%v must agree", id1, ok1, id2, ok2)
	}
}

func TestGenerationConfig_LanguageTokenID_Ugly(t *testing.T) {
	g := realGenerationConfig(t)
	if _, ok := g.LanguageTokenID("not-a-real-language"); ok {
		t.Fatal("LanguageTokenID accepted an unknown code")
	}
}

func TestGenerationConfig_LanguageCode_Good(t *testing.T) {
	g := realGenerationConfig(t)
	if code := g.LanguageCode(50259); code != "en" {
		t.Fatalf("LanguageCode(50259) = %q, want \"en\"", code)
	}
}

func TestGenerationConfig_LanguageCode_Bad(t *testing.T) {
	g := realGenerationConfig(t)
	if code := g.LanguageCode(-1); code != "" {
		t.Fatalf("LanguageCode(-1) = %q, want \"\" for an id that is not a language token", code)
	}
}

func TestFromBracketedToken_Good(t *testing.T) {
	if got := fromBracketedToken("<|en|>"); got != "en" {
		t.Fatalf("fromBracketedToken(\"<|en|>\") = %q, want \"en\"", got)
	}
}

// TestFromBracketedToken_Ugly proves an unwrapped string passes through unchanged (not truncated or
// panicking on the short-string bound).
func TestFromBracketedToken_Ugly(t *testing.T) {
	for _, s := range []string{"en", "", "<|", "|>"} {
		if got := fromBracketedToken(s); got != s {
			t.Fatalf("fromBracketedToken(%q) = %q, want unchanged", s, got)
		}
	}
}
