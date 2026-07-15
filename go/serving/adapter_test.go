// SPDX-Licence-Identifier: EUPL-1.2

package serving

import (
	"context"
	"iter"

	"dappco.re/go"
	"dappco.re/go/inference"
)

// mockTextModel implements inference.TextModel for testing the InferenceAdapter.
type mockTextModel struct {
	tokens    []inference.Token // tokens to yield
	err       error             // error to return from Err()
	closed    bool
	modelType string
}

func (m *mockTextModel) Generate(_ context.Context, _ string, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		for _, tok := range m.tokens {
			if !yield(tok) {
				return
			}
		}
	}
}

func (m *mockTextModel) Chat(_ context.Context, _ []inference.Message, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		for _, tok := range m.tokens {
			if !yield(tok) {
				return
			}
		}
	}
}

func (m *mockTextModel) Classify(_ context.Context, _ []string, _ ...inference.GenerateOption) core.Result {
	panic("Classify not used by adapter")
}

func (m *mockTextModel) BatchGenerate(_ context.Context, _ []string, _ ...inference.GenerateOption) core.Result {
	panic("BatchGenerate not used by adapter")
}

func (m *mockTextModel) ModelType() string                  { return m.modelType }
func (m *mockTextModel) Info() inference.ModelInfo          { return inference.ModelInfo{} }
func (m *mockTextModel) Metrics() inference.GenerateMetrics { return inference.GenerateMetrics{} }
func (m *mockTextModel) Err() core.Result                   { return core.ResultOf(nil, m.err) }
func (m *mockTextModel) Close() core.Result                 { m.closed = true; return core.Ok(nil) }

// --- Tests ---

func TestInferenceAdapter_Generate_Good(t *core.T) {
	mock := &mockTextModel{
		tokens: []inference.Token{
			{ID: 1, Text: "Hello"},
			{ID: 2, Text: " "},
			{ID: 3, Text: "world"},
		},
	}
	adapter := NewInferenceAdapter(mock, "test")

	r := adapter.Generate(context.Background(), "prompt", GenOpts{})
	requireResultOK(t, r)
	result := r.Value.(Result)
	core.AssertEqual(t, "Hello world", result.Text)
	core.AssertNotNil(t, result.Metrics)
}

func TestInferenceAdapter_Generate_Empty_Good(t *core.T) {
	mock := &mockTextModel{tokens: nil}
	adapter := NewInferenceAdapter(mock, "test")

	r := adapter.Generate(context.Background(), "prompt", GenOpts{})
	requireResultOK(t, r)
	result := r.Value.(Result)
	core.AssertEqual(t, "", result.Text)
	core.AssertNotNil(t, result.Metrics)
}

func TestInferenceAdapter_Generate_ModelError_Bad(t *core.T) {
	mock := &mockTextModel{
		tokens: []inference.Token{
			{ID: 1, Text: "partial"},
		},
		err: core.NewError("out of memory"),
	}
	adapter := NewInferenceAdapter(mock, "test")

	r := adapter.Generate(context.Background(), "prompt", GenOpts{})
	assertResultError(t, r)
	core.AssertContains(t, r.Error(), "out of memory")
}

func TestInferenceAdapter_GenerateStream_Good(t *core.T) {
	mock := &mockTextModel{
		tokens: []inference.Token{
			{ID: 1, Text: "one"},
			{ID: 2, Text: "two"},
			{ID: 3, Text: "three"},
		},
	}
	adapter := NewInferenceAdapter(mock, "test")

	var collected []string
	rStream := adapter.GenerateStream(context.Background(), "prompt", GenOpts{}, func(token string) error {
		collected = append(collected, token)
		return nil
	})
	requireResultOK(t, rStream)
	core.AssertEqual(t, []string{"one", "two", "three"}, collected)
}

func TestInferenceAdapter_GenerateStream_CallbackError_Bad(t *core.T) {
	callbackErr := core.NewError("client disconnected")
	mock := &mockTextModel{
		tokens: []inference.Token{
			{ID: 1, Text: "one"},
			{ID: 2, Text: "two"},
			{ID: 3, Text: "three"},
		},
	}
	adapter := NewInferenceAdapter(mock, "test")

	count := 0
	rStream := adapter.GenerateStream(context.Background(), "prompt", GenOpts{}, func(token string) error {
		count++
		if count >= 2 {
			return callbackErr
		}
		return nil
	})
	assertResultError(t, rStream, "client disconnected")
	core.AssertEqual(t, 2, count, "callback should have been called exactly twice")
}

func TestInferenceAdapterContextCancellationBadScenario(t *core.T) {
	ctx, cancel := context.WithCancel(context.Background())

	// Create a mock that respects context cancellation.
	mock := &mockTextModel{}
	mock.tokens = nil // no tokens; the mock Generate just returns empty
	// Simulate context cancel causing model error.
	cancel()
	mock.err = ctx.Err()

	adapter := NewInferenceAdapter(mock, "test")
	r := adapter.Generate(ctx, "prompt", GenOpts{})
	assertResultError(t, r)
	core.AssertContains(t, r.Error(), context.Canceled.Error())
}

func TestInferenceAdapter_Chat_Good(t *core.T) {
	mock := &mockTextModel{
		tokens: []inference.Token{
			{ID: 1, Text: "Hi"},
			{ID: 2, Text: " there"},
		},
	}
	adapter := NewInferenceAdapter(mock, "test")

	messages := []Message{
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi"},
		{Role: "user", Content: "How are you?"},
	}
	r := adapter.Chat(context.Background(), messages, GenOpts{})
	requireResultOK(t, r)
	result := r.Value.(Result)
	core.AssertEqual(t, "Hi there", result.Text)
	core.AssertNotNil(t, result.Metrics)
}

func TestInferenceAdapter_ChatStream_Good(t *core.T) {
	mock := &mockTextModel{
		tokens: []inference.Token{
			{ID: 1, Text: "reply"},
			{ID: 2, Text: "!"},
		},
	}
	adapter := NewInferenceAdapter(mock, "test")

	messages := []Message{{Role: "user", Content: "test"}}
	var collected []string
	rStream := adapter.ChatStream(context.Background(), messages, GenOpts{}, func(token string) error {
		collected = append(collected, token)
		return nil
	})
	requireResultOK(t, rStream)
	core.AssertEqual(t, []string{"reply", "!"}, collected)
}

func TestInferenceAdapter_StopSequences_Good(t *core.T) {
	mock := &mockTextModel{
		tokens: []inference.Token{
			{ID: 1, Text: "hello "},
			{ID: 2, Text: "STOP world"},
			{ID: 3, Text: "ignored"},
		},
	}
	adapter := NewInferenceAdapter(mock, "test")

	r := adapter.Generate(context.Background(), "prompt", GenOpts{StopSequences: []string{"STOP"}})
	requireResultOK(t, r)
	result := r.Value.(Result)
	core.AssertEqual(t, "hello ", result.Text)

	var collected []string
	rStream := adapter.GenerateStream(context.Background(), "prompt", GenOpts{StopSequences: []string{"STOP"}}, func(token string) error {
		collected = append(collected, token)
		return nil
	})
	requireResultOK(t, rStream)
	core.AssertEqual(t, []string{"hello "}, collected)
}

func TestInferenceAdapterConvertOptsGoodScenario(t *core.T) {
	// Non-zero values should produce options.
	opts := convertOpts(GenOpts{Temperature: 0.7, MaxTokens: 512, Model: "ignored"})
	core.AssertLen(t, opts, 2)

	// Zero values should produce no options.
	opts = convertOpts(GenOpts{})
	core.AssertLen(t, opts, 0)

	// Only temperature set.
	opts = convertOpts(GenOpts{Temperature: 0.5})
	core.AssertLen(t, opts, 1)

	// Only max tokens set.
	opts = convertOpts(GenOpts{MaxTokens: 100})
	core.AssertLen(t, opts, 1)
}

func TestInferenceAdapterConvertOptsNewFieldsGoodScenario(t *core.T) {
	// TopK only.
	opts := convertOpts(GenOpts{TopK: 40})
	core.AssertLen(t, opts, 1)

	// TopP only.
	opts = convertOpts(GenOpts{TopP: 0.9})
	core.AssertLen(t, opts, 1)

	// RepeatPenalty only.
	opts = convertOpts(GenOpts{RepeatPenalty: 1.1})
	core.AssertLen(t, opts, 1)

	// All new fields set together.
	opts = convertOpts(GenOpts{TopK: 40, TopP: 0.9, RepeatPenalty: 1.1})
	core.AssertLen(t, opts, 3)

	// All fields set (Temperature + MaxTokens + TopK + TopP + RepeatPenalty).
	opts = convertOpts(GenOpts{
		Temperature:   0.7,
		MaxTokens:     512,
		TopK:          40,
		TopP:          0.9,
		RepeatPenalty: 1.1,
	})
	core.AssertLen(t, opts, 5)

	// Zero TopK/TopP/RepeatPenalty should not produce options.
	opts = convertOpts(GenOpts{Temperature: 0.5, TopK: 0, TopP: 0, RepeatPenalty: 0})
	core.AssertLen(t, opts, 1) // only Temperature
}

func TestInferenceAdapterMessageAliasGoodScenario(t *core.T) {
	// serving.Message and inference.Message are the same type — verify interchangeability.
	mlMsg := Message{Role: "user", Content: "Hello"}
	inferMsg := inference.Message{Role: "user", Content: "Hello"}
	core.AssertEqual(t, mlMsg, inferMsg)

	// Can assign directly without conversion.
	var msgs []inference.Message
	msgs = append(msgs, mlMsg)
	core.AssertEqual(t, "user", msgs[0].Role)
	core.AssertEqual(t, "Hello", msgs[0].Content)
}

func TestInferenceAdapterNameAndAvailableGoodScenario(t *core.T) {
	mock := &mockTextModel{}
	adapter := NewInferenceAdapter(mock, "mlx")

	core.AssertEqual(t, "mlx", adapter.Name())
	core.AssertTrue(t, adapter.Available())
}

func TestInferenceAdapter_Close_Good(t *core.T) {
	mock := &mockTextModel{}
	adapter := NewInferenceAdapter(mock, "test")

	requireResultOK(t, adapter.Close())
	core.AssertTrue(t, mock.closed)
}

func TestInferenceAdapter_Model_Good(t *core.T) {
	mock := &mockTextModel{modelType: "qwen3"}
	adapter := NewInferenceAdapter(mock, "test")

	core.AssertEqual(t, "qwen3", adapter.Model().ModelType())
}

// --- v0.9.0 shape triplets ---

func TestAdapter_NewInferenceAdapter_Good(t *core.T) {
	mock := &mockTextModel{modelType: "adapter-good"}
	adapter := NewInferenceAdapter(mock, "adapter-good")
	core.AssertNotNil(t, adapter)
	core.AssertEqual(t, "adapter-good", adapter.Name())
	core.AssertEqual(t, mock, adapter.Model())
}

func TestAdapter_NewInferenceAdapter_Bad(t *core.T) {
	adapter := NewInferenceAdapter(nil, "")
	core.AssertNotNil(t, adapter)
	core.AssertEqual(t, "", adapter.Name())
	core.AssertNil(t, adapter.Model())
}

func TestAdapter_NewInferenceAdapter_Ugly(t *core.T) {
	mock := &mockTextModel{tokens: []inference.Token{{Text: "edge"}}}
	adapter := NewInferenceAdapter(mock, "edge")
	r := adapter.Generate(context.Background(), "prompt", GenOpts{})
	requireResultOK(t, r)
	result := r.Value.(Result)
	core.AssertEqual(t, "edge", result.Text)
}

func TestAdapter_InferenceAdapter_Generate_Good(t *core.T) {
	mock := &mockTextModel{tokens: []inference.Token{{Text: "hello"}, {Text: " world"}}}
	adapter := NewInferenceAdapter(mock, "gen")
	r := adapter.Generate(context.Background(), "prompt", GenOpts{})
	requireResultOK(t, r)
	result := r.Value.(Result)
	core.AssertEqual(t, "hello world", result.Text)
}

func TestAdapter_InferenceAdapter_Generate_Bad(t *core.T) {
	mock := &mockTextModel{tokens: []inference.Token{{Text: "partial"}}, err: core.NewError("generate failed")}
	adapter := NewInferenceAdapter(mock, "gen")
	r := adapter.Generate(context.Background(), "prompt", GenOpts{})
	assertResultError(t, r, "generate failed")
}

func TestAdapter_InferenceAdapter_Generate_Ugly(t *core.T) {
	mock := &mockTextModel{tokens: []inference.Token{{Text: "before"}, {Text: "STOP after"}}}
	adapter := NewInferenceAdapter(mock, "gen")
	r := adapter.Generate(context.Background(), "prompt", GenOpts{StopSequences: []string{"STOP"}})
	requireResultOK(t, r)
	result := r.Value.(Result)
	core.AssertEqual(t, "before", result.Text)
}

func TestAdapter_InferenceAdapter_Chat_Good(t *core.T) {
	mock := &mockTextModel{tokens: []inference.Token{{Text: "reply"}}}
	adapter := NewInferenceAdapter(mock, "chat")
	r := adapter.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}}, GenOpts{})
	requireResultOK(t, r)
	result := r.Value.(Result)
	core.AssertEqual(t, "reply", result.Text)
}

func TestAdapter_InferenceAdapter_Chat_Bad(t *core.T) {
	mock := &mockTextModel{err: core.NewError("chat failed")}
	adapter := NewInferenceAdapter(mock, "chat")
	r := adapter.Chat(context.Background(), nil, GenOpts{})
	assertResultError(t, r, "chat failed")
}

func TestAdapter_InferenceAdapter_Chat_Ugly(t *core.T) {
	mock := &mockTextModel{tokens: []inference.Token{{Text: "first"}, {Text: "END ignored"}}}
	adapter := NewInferenceAdapter(mock, "chat")
	r := adapter.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}}, GenOpts{StopSequences: []string{"END"}})
	requireResultOK(t, r)
	result := r.Value.(Result)
	core.AssertEqual(t, "first", result.Text)
}

func TestAdapter_InferenceAdapter_GenerateStream_Good(t *core.T) {
	mock := &mockTextModel{tokens: []inference.Token{{Text: "a"}, {Text: "b"}}}
	adapter := NewInferenceAdapter(mock, "stream")
	var got []string
	rStream := adapter.GenerateStream(context.Background(), "prompt", GenOpts{}, func(token string) error {
		got = append(got, token)
		return nil
	})
	requireResultOK(t, rStream)
	core.AssertEqual(t, []string{"a", "b"}, got)
}

func TestAdapter_InferenceAdapter_GenerateStream_Bad(t *core.T) {
	stopErr := core.NewError("callback stopped")
	mock := &mockTextModel{tokens: []inference.Token{{Text: "a"}, {Text: "b"}}}
	adapter := NewInferenceAdapter(mock, "stream")
	rStream := adapter.GenerateStream(context.Background(), "prompt", GenOpts{}, func(token string) error {
		if token == "b" {
			return stopErr
		}
		return nil
	})
	assertResultError(t, rStream, "callback stopped")
}

func TestAdapter_InferenceAdapter_GenerateStream_Ugly(t *core.T) {
	mock := &mockTextModel{tokens: []inference.Token{{Text: "one"}, {Text: "STOP two"}}}
	adapter := NewInferenceAdapter(mock, "stream")
	var got []string
	rStream := adapter.GenerateStream(context.Background(), "prompt", GenOpts{StopSequences: []string{"STOP"}}, func(token string) error {
		got = append(got, token)
		return nil
	})
	requireResultOK(t, rStream)
	core.AssertEqual(t, []string{"one"}, got)
}

func TestAdapter_InferenceAdapter_ChatStream_Good(t *core.T) {
	mock := &mockTextModel{tokens: []inference.Token{{Text: "x"}, {Text: "y"}}}
	adapter := NewInferenceAdapter(mock, "chat-stream")
	var got []string
	rStream := adapter.ChatStream(context.Background(), []Message{{Role: "user", Content: "hi"}}, GenOpts{}, func(token string) error {
		got = append(got, token)
		return nil
	})
	requireResultOK(t, rStream)
	core.AssertEqual(t, []string{"x", "y"}, got)
}

func TestAdapter_InferenceAdapter_ChatStream_Bad(t *core.T) {
	stopErr := core.NewError("chat callback stopped")
	mock := &mockTextModel{tokens: []inference.Token{{Text: "x"}}}
	adapter := NewInferenceAdapter(mock, "chat-stream")
	rStream := adapter.ChatStream(context.Background(), nil, GenOpts{}, func(string) error { return stopErr })
	assertResultError(t, rStream, "chat callback stopped")
}

func TestAdapter_InferenceAdapter_ChatStream_Ugly(t *core.T) {
	mock := &mockTextModel{tokens: []inference.Token{{Text: "ok"}, {Text: "CUT ignored"}}}
	adapter := NewInferenceAdapter(mock, "chat-stream")
	var got []string
	rStream := adapter.ChatStream(context.Background(), nil, GenOpts{StopSequences: []string{"CUT"}}, func(token string) error {
		got = append(got, token)
		return nil
	})
	requireResultOK(t, rStream)
	core.AssertEqual(t, []string{"ok"}, got)
}

func TestAdapter_InferenceAdapter_Name_Good(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	adapter := NewInferenceAdapter(&mockTextModel{}, "named")
	core.AssertEqual(t, "named", adapter.Name())
}

func TestAdapter_InferenceAdapter_Name_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	adapter := NewInferenceAdapter(&mockTextModel{}, "")
	core.AssertEqual(t, "", adapter.Name())
}

func TestAdapter_InferenceAdapter_Name_Ugly(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	adapter := NewInferenceAdapter(&mockTextModel{}, "name with spaces")
	core.AssertEqual(t, "name with spaces", adapter.Name())
}

func TestAdapter_InferenceAdapter_Available_Good(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	adapter := NewInferenceAdapter(&mockTextModel{}, "available")
	core.AssertTrue(t, adapter.Available())
}

func TestAdapter_InferenceAdapter_Available_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	adapter := NewInferenceAdapter(nil, "available")
	core.AssertTrue(t, adapter.Available())
}

func TestAdapter_InferenceAdapter_Available_Ugly(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	adapter := NewInferenceAdapter(&mockTextModel{}, "")
	core.AssertTrue(t, adapter.Available())
}

func TestAdapter_InferenceAdapter_Close_Good(t *core.T) {
	mock := &mockTextModel{}
	adapter := NewInferenceAdapter(mock, "close")
	requireResultOK(t, adapter.Close())
	core.AssertTrue(t, mock.closed)
}

func TestAdapter_InferenceAdapter_Close_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	adapter := NewInferenceAdapter(&mockTextModel{}, "close")
	assertResultOK(t, adapter.Close())
}

func TestAdapter_InferenceAdapter_Close_Ugly(t *core.T) {
	mock := &mockTextModel{}
	adapter := NewInferenceAdapter(mock, "close")
	requireResultOK(t, adapter.Close())
	requireResultOK(t, adapter.Close())
	core.AssertTrue(t, mock.closed)
}

func TestAdapter_InferenceAdapter_Model_Good(t *core.T) {
	mock := &mockTextModel{modelType: "model"}
	adapter := NewInferenceAdapter(mock, "model")
	core.AssertEqual(t, mock, adapter.Model())
}

func TestAdapter_InferenceAdapter_Model_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	adapter := NewInferenceAdapter(nil, "model")
	core.AssertNil(t, adapter.Model())
}

func TestAdapter_InferenceAdapter_Model_Ugly(t *core.T) {
	mock := &mockTextModel{}
	adapter := NewInferenceAdapter(mock, "")
	core.AssertEqual(t, mock, adapter.Model())
}

func TestAdapter_InferenceAdapter_Capabilities_Good(t *core.T) {
	model := &mockTextModel{modelType: "qwen3"}
	adapter := NewInferenceAdapter(model, "mlx")
	report := adapter.Capabilities()
	core.AssertTrue(t, report.Available)
	core.AssertEqual(t, "mlx", report.Runtime.Backend)
	core.AssertTrue(t, report.Supports(inference.CapabilityGenerate))
}

func TestAdapter_InferenceAdapter_Capabilities_Bad(t *core.T) {
	adapter := NewInferenceAdapter(nil, "missing")
	report := adapter.Capabilities()
	core.AssertFalse(t, report.Available)
	core.AssertEqual(t, "missing", report.Runtime.Backend)
	core.AssertFalse(t, report.Supports(inference.CapabilityGenerate))
}

func TestAdapter_InferenceAdapter_Capabilities_Ugly(t *core.T) {
	var adapter *InferenceAdapter
	report := adapter.Capabilities()
	core.AssertFalse(t, report.Available)
	core.AssertEqual(t, "", report.Runtime.Backend)
	core.AssertEmpty(t, report.Capabilities)
}

func TestAdapter_InferenceAdapter_InspectAttention_Good(t *core.T) {
	adapter := NewInferenceAdapter(&mockTextModel{}, "plain")
	r := adapter.InspectAttention(context.Background(), "prompt")
	assertResultError(t, r, "does not support attention")
}

func TestAdapter_InferenceAdapter_InspectAttention_Bad(t *core.T) {
	adapter := NewInferenceAdapter(&mockTextModel{}, "bad")
	r := adapter.InspectAttention(context.Background(), "")
	assertResultError(t, r, "does not support attention")
}

func TestAdapter_InferenceAdapter_InspectAttention_Ugly(t *core.T) {
	adapter := NewInferenceAdapter(&mockTextModel{}, "unicode")
	r := adapter.InspectAttention(context.Background(), "λ")
	assertResultError(t, r, "does not support attention")
}

// TestInferenceAdapter_ConvertOptsReasoningControls_Good pins the GenOpts →
// GenerateOption pass-through for the reasoning + sampling controls the
// config reconcile added: EnableThinking, ThinkingBudget, MinP, Seed.
func TestInferenceAdapter_ConvertOptsReasoningControls_Good(t *core.T) {
	off := false
	seed := uint64(42)
	opts := convertOpts(GenOpts{MinP: 0.05, Seed: &seed, EnableThinking: &off, ThinkingBudget: 256})

	cfg := inference.ApplyGenerateOpts(opts)
	if cfg.MinP != 0.05 {
		t.Fatalf("MinP = %v, want 0.05", cfg.MinP)
	}
	if !cfg.SeedSet || cfg.Seed != 42 {
		t.Fatalf("Seed = %d set=%v, want 42 set", cfg.Seed, cfg.SeedSet)
	}
	if cfg.EnableThinking == nil || *cfg.EnableThinking {
		t.Fatalf("EnableThinking = %v, want &false", cfg.EnableThinking)
	}
	if cfg.ThinkingBudget != 256 {
		t.Fatalf("ThinkingBudget = %d, want 256", cfg.ThinkingBudget)
	}
}
