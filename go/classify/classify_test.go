package classify

import (
	"context"
	"iter"
	"testing"

	"dappco.re/go"
	"dappco.re/go/inference"
)

func TestMapTokenToDomain(t *testing.T) {
	tests := []struct {
		token string
		want  string
	}{
		{"technical", "technical"},
		{"Technical", "technical"},
		{"tech", "technical"},
		{"creative", "creative"},
		{"Creative", "creative"},
		{"cre", "creative"},
		{"ethical", "ethical"},
		{"Ethical", "ethical"},
		{"eth", "ethical"},
		{"casual", "casual"},
		{"Casual", "casual"},
		{"cas", "casual"},
		{"unknown", "unknown"},
		{"", "unknown"},
		{"foo", "unknown"},
		// Verify prefix collision fix: these must NOT match any domain
		{"castle", "unknown"},
		{"cascade", "unknown"},
		{"credential", "unknown"},
		{"creature", "unknown"},
	}
	for _, tt := range tests {
		t.Run(tt.token, func(t *testing.T) {
			got := mapTokenToDomain(tt.token)
			if got != tt.want {
				t.Errorf("mapTokenToDomain(%q) = %q, want %q", tt.token, got, tt.want)
			}
		})
	}
}

// mockModel satisfies inference.TextModel for testing.
type mockModel struct {
	classifyFunc func(ctx context.Context, prompts []string, opts ...inference.GenerateOption) ([]inference.ClassifyResult, error)
}

func (m *mockModel) Generate(_ context.Context, _ string, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {}
}

func (m *mockModel) Chat(_ context.Context, _ []inference.Message, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {}
}

func (m *mockModel) Classify(ctx context.Context, prompts []string, opts ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	return m.classifyFunc(ctx, prompts, opts...)
}

func (m *mockModel) BatchGenerate(_ context.Context, _ []string, _ ...inference.GenerateOption) ([]inference.BatchResult, error) {
	return nil, nil
}

func (m *mockModel) ModelType() string                  { return "mock" }
func (m *mockModel) Info() inference.ModelInfo          { return inference.ModelInfo{} }
func (m *mockModel) Metrics() inference.GenerateMetrics { return inference.GenerateMetrics{} }
func (m *mockModel) Err() error                         { return nil }
func (m *mockModel) Close() error                       { return nil }

func TestClassifyCorpus_Basic(t *testing.T) {
	model := &mockModel{
		classifyFunc: func(_ context.Context, prompts []string, _ ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
			results := make([]inference.ClassifyResult, len(prompts))
			for i := range prompts {
				results[i] = inference.ClassifyResult{Token: inference.Token{Text: "technical"}}
			}
			return results, nil
		},
	}

	input := core.NewReader(
		`{"seed_id":"1","domain":"general","prompt":"Delete the file"}` + "\n" +
			`{"seed_id":"2","domain":"science","prompt":"Explain gravity"}` + "\n",
	)
	output := core.NewBuffer()

	stats, err := ClassifyCorpus(context.Background(), model, input, output, WithBatchSize(16))
	if err != nil {
		t.Fatalf("ClassifyCorpus returned error: %v", err)
	}
	if stats.Total != 2 {
		t.Errorf("Total = %d, want 2", stats.Total)
	}
	if stats.Skipped != 0 {
		t.Errorf("Skipped = %d, want 0", stats.Skipped)
	}

	lines := core.Split(core.Trim(output.String()), "\n")
	if len(lines) != 2 {
		t.Fatalf("output lines = %d, want 2", len(lines))
	}

	for i, line := range lines {
		var record map[string]any
		if r := core.JSONUnmarshal([]byte(line), &record); !r.OK {
			t.Fatalf("line %d: unmarshal: %v", i, r.Value)
		}
		if record["domain_1b"] != "technical" {
			t.Errorf("line %d: domain_1b = %v, want %q", i, record["domain_1b"], "technical")
		}
		// original domain field must be preserved
		if _, ok := record["domain"]; !ok {
			t.Errorf("line %d: original domain field missing", i)
		}
	}
}

func TestClassifyCorpus_SkipsMalformed(t *testing.T) {
	model := &mockModel{
		classifyFunc: func(_ context.Context, prompts []string, _ ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
			results := make([]inference.ClassifyResult, len(prompts))
			for i := range prompts {
				results[i] = inference.ClassifyResult{Token: inference.Token{Text: "technical"}}
			}
			return results, nil
		},
	}

	input := core.NewReader(
		"not valid json\n" +
			`{"seed_id":"1","domain":"general","prompt":"Hello world"}` + "\n" +
			`{"seed_id":"2","domain":"general"}` + "\n",
	)
	output := core.NewBuffer()

	stats, err := ClassifyCorpus(context.Background(), model, input, output)
	if err != nil {
		t.Fatalf("ClassifyCorpus returned error: %v", err)
	}
	if stats.Total != 1 {
		t.Errorf("Total = %d, want 1", stats.Total)
	}
	if stats.Skipped != 2 {
		t.Errorf("Skipped = %d, want 2", stats.Skipped)
	}
}

func TestClassifyCorpus_DomainMapping(t *testing.T) {
	model := &mockModel{
		classifyFunc: func(_ context.Context, prompts []string, _ ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
			results := make([]inference.ClassifyResult, len(prompts))
			for i, p := range prompts {
				if core.Contains(p, "Delete") {
					results[i] = inference.ClassifyResult{Token: inference.Token{Text: "technical"}}
				} else {
					results[i] = inference.ClassifyResult{Token: inference.Token{Text: "ethical"}}
				}
			}
			return results, nil
		},
	}

	input := core.NewReader(
		`{"prompt":"Delete the file now"}` + "\n" +
			`{"prompt":"Is it right to lie?"}` + "\n",
	)
	output := core.NewBuffer()

	stats, err := ClassifyCorpus(context.Background(), model, input, output, WithBatchSize(16))
	if err != nil {
		t.Fatalf("ClassifyCorpus returned error: %v", err)
	}
	if stats.ByDomain["technical"] != 1 {
		t.Errorf("ByDomain[technical] = %d, want 1", stats.ByDomain["technical"])
	}
	if stats.ByDomain["ethical"] != 1 {
		t.Errorf("ByDomain[ethical] = %d, want 1", stats.ByDomain["ethical"])
	}
}

func TestClassifyCorpus_ResultCountMismatch(t *testing.T) {
	model := &mockModel{
		classifyFunc: func(_ context.Context, prompts []string, _ ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
			if len(prompts) == 0 {
				return nil, nil
			}
			return []inference.ClassifyResult{{Token: inference.Token{Text: "technical"}}}, nil
		},
	}

	input := core.NewReader(
		`{"prompt":"Delete the file now"}` + "\n" +
			`{"prompt":"Create the repo"}` + "\n",
	)

	output := core.NewBuffer()
	stats, err := ClassifyCorpus(context.Background(), model, input, output, WithBatchSize(16))
	if err == nil {
		t.Fatal("ClassifyCorpus returned nil error, want mismatch failure")
	}
	if stats.Total != 0 {
		t.Errorf("Total = %d, want 0", stats.Total)
	}
	if output.Len() != 0 {
		t.Errorf("output len = %d, want 0", output.Len())
	}
}

// --- AX-7 canonical triplets ---

func TestClassify_WithBatchSize_Good(t *testing.T) {
	called := false
	noPanicForAudit(t, func() {
		called = true
		cfg := defaultClassifyConfig()
		WithBatchSize(2)(&cfg)
		if cfg.batchSize != 2 {
			t.Fatalf("got %d", cfg.batchSize)
		}
	})
	if !called {
		t.Fatal("WithBatchSize was not exercised")
	}
}

func TestClassify_WithBatchSize_Bad(t *testing.T) {
	called := false
	noPanicForAudit(t, func() {
		called = true
		cfg := defaultClassifyConfig()
		WithBatchSize(0)(&cfg)
		if cfg.batchSize != 0 {
			t.Fatalf("got %d", cfg.batchSize)
		}
	})
	if !called {
		t.Fatal("WithBatchSize was not exercised")
	}
}

func TestClassify_WithBatchSize_Ugly(t *testing.T) {
	called := false
	noPanicForAudit(t, func() {
		called = true
		cfg := defaultClassifyConfig()
		WithBatchSize(-1)(&cfg)
		if cfg.batchSize != -1 {
			t.Fatalf("got %d", cfg.batchSize)
		}
	})
	if !called {
		t.Fatal("WithBatchSize was not exercised")
	}
}

func TestClassify_WithPromptField_Good(t *testing.T) {
	called := false
	noPanicForAudit(t, func() {
		called = true
		cfg := defaultClassifyConfig()
		WithPromptField("text")(&cfg)
		if cfg.promptField != "text" {
			t.Fatalf("got %q", cfg.promptField)
		}
	})
	if !called {
		t.Fatal("WithPromptField was not exercised")
	}
}

func TestClassify_WithPromptField_Bad(t *testing.T) {
	called := false
	noPanicForAudit(t, func() {
		called = true
		cfg := defaultClassifyConfig()
		WithPromptField("")(&cfg)
		if cfg.promptField != "" {
			t.Fatalf("got %q", cfg.promptField)
		}
	})
	if !called {
		t.Fatal("WithPromptField was not exercised")
	}
}

func TestClassify_WithPromptField_Ugly(t *testing.T) {
	called := false
	noPanicForAudit(t, func() {
		called = true
		cfg := defaultClassifyConfig()
		WithPromptField("nested.prompt")(&cfg)
		if cfg.promptField != "nested.prompt" {
			t.Fatalf("got %q", cfg.promptField)
		}
	})
	if !called {
		t.Fatal("WithPromptField was not exercised")
	}
}

func TestClassify_WithPromptTemplate_Good(t *testing.T) {
	called := false
	noPanicForAudit(t, func() {
		called = true
		cfg := defaultClassifyConfig()
		WithPromptTemplate("Classify: %s")(&cfg)
		if cfg.promptTemplate != "Classify: %s" {
			t.Fatalf("got %q", cfg.promptTemplate)
		}
	})
	if !called {
		t.Fatal("WithPromptTemplate was not exercised")
	}
}

func TestClassify_WithPromptTemplate_Bad(t *testing.T) {
	called := false
	noPanicForAudit(t, func() {
		called = true
		cfg := defaultClassifyConfig()
		WithPromptTemplate("")(&cfg)
		if cfg.promptTemplate != "" {
			t.Fatalf("got %q", cfg.promptTemplate)
		}
	})
	if !called {
		t.Fatal("WithPromptTemplate was not exercised")
	}
}

func TestClassify_WithPromptTemplate_Ugly(t *testing.T) {
	called := false
	noPanicForAudit(t, func() {
		called = true
		cfg := defaultClassifyConfig()
		WithPromptTemplate("[%s]")(&cfg)
		if cfg.promptTemplate != "[%s]" {
			t.Fatalf("got %q", cfg.promptTemplate)
		}
	})
	if !called {
		t.Fatal("WithPromptTemplate was not exercised")
	}
}

func TestClassify_ClassifyCorpus_Good(t *testing.T) {
	called := false
	noPanicForAudit(t, func() {
		called = true
		model := &mockModel{classifyFunc: func(_ context.Context, prompts []string, _ ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
			results := make([]inference.ClassifyResult, len(prompts))
			for i := range prompts {
				results[i] = inference.ClassifyResult{Token: inference.Token{Text: "technical"}}
			}
			return results, nil
		}}
		input := core.NewBufferString(`{"prompt":"Delete the file"}` + "\n")
		stats, err := ClassifyCorpus(context.Background(), model, input, core.NewBuffer())
		if err != nil || stats.Total != 1 {
			t.Fatalf("stats=%+v err=%v", stats, err)
		}
	})
	if !called {
		t.Fatal("ClassifyCorpus was not exercised")
	}
}

func TestClassify_ClassifyCorpus_Bad(t *testing.T) {
	called := false
	noPanicForAudit(t, func() {
		called = true
		model := &mockModel{classifyFunc: func(_ context.Context, prompts []string, _ ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
			return make([]inference.ClassifyResult, len(prompts)), nil
		}}
		input := core.NewBufferString("not-json\n")
		stats, err := ClassifyCorpus(context.Background(), model, input, core.NewBuffer())
		if err != nil || stats.Skipped != 1 {
			t.Fatalf("stats=%+v err=%v", stats, err)
		}
	})
	if !called {
		t.Fatal("ClassifyCorpus was not exercised")
	}
}

func TestClassify_ClassifyCorpus_Ugly(t *testing.T) {
	called := false
	noPanicForAudit(t, func() {
		called = true
		model := &mockModel{classifyFunc: func(_ context.Context, prompts []string, _ ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
			return make([]inference.ClassifyResult, len(prompts)), nil
		}}
		input := core.NewBufferString("")
		stats, err := ClassifyCorpus(context.Background(), model, input, core.NewBuffer())
		if err != nil || stats.Total != 0 {
			t.Fatalf("stats=%+v err=%v", stats, err)
		}
	})
	if !called {
		t.Fatal("ClassifyCorpus was not exercised")
	}
}
