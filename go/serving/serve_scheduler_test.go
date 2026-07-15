// SPDX-Licence-Identifier: EUPL-1.2

package serving

import (
	"bytes"
	"context"
	"iter"
	"net/http"
	"net/http/httptest"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/serving/compat"
	openai "dappco.re/go/inference/serving/provider/openai"
	"dappco.re/go/inference/serving/scheduler"
)

// fakeChatModel is a minimal inference.TextModel whose Generate/Chat yield a
// fixed token list — enough to drive a chat-completions request end to end
// through the scheduler and back out as a 200.
type fakeChatModel struct {
	tokens []inference.Token
}

func (m *fakeChatModel) streamTokens(ctx context.Context) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		for _, token := range m.tokens {
			if ctx.Err() != nil {
				return
			}
			if !yield(token) {
				return
			}
		}
	}
}

func (m *fakeChatModel) Generate(ctx context.Context, _ string, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.streamTokens(ctx)
}

func (m *fakeChatModel) Chat(ctx context.Context, _ []inference.Message, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.streamTokens(ctx)
}

func (m *fakeChatModel) Classify(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok([]inference.ClassifyResult(nil))
}

func (m *fakeChatModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok([]inference.BatchResult(nil))
}

func (m *fakeChatModel) ModelType() string { return "fake" }
func (m *fakeChatModel) Info() inference.ModelInfo {
	return inference.ModelInfo{Architecture: "gemma3"}
}
func (m *fakeChatModel) Metrics() inference.GenerateMetrics { return inference.GenerateMetrics{} }
func (m *fakeChatModel) Err() core.Result                   { return core.Ok(nil) }
func (m *fakeChatModel) Close() core.Result                 { return core.Ok(nil) }

func TestParseSchedulerMode_Good(t *testing.T) {
	for _, want := range []scheduler.Mode{scheduler.ModeSerial, scheduler.ModeBatch, scheduler.ModeInterleave} {
		got, err := parseSchedulerMode(string(want))
		if err != nil || got != want {
			t.Fatalf("parseSchedulerMode(%q) = %q/%v, want %q/nil", want, got, err, want)
		}
	}
	// Case + whitespace tolerant.
	if got, err := parseSchedulerMode("  SERIAL "); err != nil || got != scheduler.ModeSerial {
		t.Fatalf("parseSchedulerMode(padded) = %q/%v, want serial", got, err)
	}
}

func TestParseSchedulerMode_Bad(t *testing.T) {
	if _, err := parseSchedulerMode("roundrobin"); err == nil {
		t.Fatal("parseSchedulerMode(unknown) error = nil, want fail-closed error")
	}
}

// TestSchedulerResolver_RoutesThroughScheduler stands up the real compat mux
// over a scheduler-wrapping resolver, fires a chat-completions request, and
// proves the request routed through the scheduler by reading its Stats counter.
func TestSchedulerResolver_RoutesThroughScheduler(t *testing.T) {
	base := openai.ResolverFunc(func(context.Context, string) (inference.TextModel, error) {
		return &fakeChatModel{tokens: []inference.Token{{Text: "hello"}, {Text: " world"}}}, nil
	})
	sched := newSchedulerResolver(base, schedulerServeConfig(scheduler.ModeSerial, 0))
	defer sched.close()

	mux := compat.NewMuxWithAdmin(sched, compat.AdminConfig{})
	rec := httptest.NewRecorder()
	body := `{"model":"whatever","messages":[{"role":"user","content":"hi"}]}`
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, openai.DefaultChatCompletionsPath, bytes.NewReader([]byte(body))))

	if rec.Code != http.StatusOK {
		t.Fatalf("chat completions through scheduler = %d, want 200 (body: %s)", rec.Code, rec.Body.String())
	}
	cur := sched.current()
	if cur == nil {
		t.Fatal("scheduler was never constructed for the resolved model")
	}
	if s := cur.Stats(); s.Submitted < 1 {
		t.Fatalf("scheduler Stats.Submitted = %d after a routed request, want >= 1", s.Submitted)
	}
}

// TestSchedulerResolver_Unset_NoWrapper proves that WITHOUT the wrapper (the
// flag-unset serve path) the resolver hands the model straight to the mux — no
// scheduler is constructed. This is the resolver-seam witness for serve.go's
// byte-untouched request path when -scheduler is empty.
func TestSchedulerResolver_Unset_NoWrapper(t *testing.T) {
	model := &fakeChatModel{tokens: []inference.Token{{Text: "x"}}}
	base := openai.ResolverFunc(func(context.Context, string) (inference.TextModel, error) {
		return model, nil
	})

	// Bare resolver (unset path): the resolved model is the plain model itself.
	got, err := base.ResolveModel(context.Background(), "m")
	if err != nil {
		t.Fatalf("bare ResolveModel error = %v", err)
	}
	if _, isScheduler := got.(*scheduler.Model); isScheduler {
		t.Fatal("bare resolver returned a *scheduler.Model — a scheduler was built with the flag unset")
	}
	if got != inference.TextModel(model) {
		t.Fatal("bare resolver did not return the underlying model unchanged")
	}

	// Wrapped resolver (flag set): the resolved model IS a scheduler.
	sched := newSchedulerResolver(base, schedulerServeConfig(scheduler.ModeSerial, 0))
	defer sched.close()
	wrapped, err := sched.ResolveModel(context.Background(), "m")
	if err != nil {
		t.Fatalf("wrapped ResolveModel error = %v", err)
	}
	if _, isScheduler := wrapped.(*scheduler.Model); !isScheduler {
		t.Fatal("wrapped resolver did not return a *scheduler.Model")
	}

	// Multi-model twin of the same contract (#35): a multiModelResolver that
	// never called setScheduler (the -models-config path with -scheduler
	// unset) must ALSO hand back the plain model — ensureResident's scheduler
	// branch is gated on schedCfg != nil, so no scheduler package type is
	// constructed at all.
	mmLoader, _, _ := countingLoader()
	mm := mustResolver(t, []ModelSpec{{ID: "m", Path: "/m/m"}}, MultiModelOptions{})
	mm.setLoader(mmLoader)
	mmGot, err := mm.ResolveModel(context.Background(), "m")
	if err != nil {
		t.Fatalf("multi-model bare ResolveModel error = %v", err)
	}
	if _, isScheduler := mmGot.(*scheduler.Model); isScheduler {
		t.Fatal("multi-model resolver returned a *scheduler.Model — a scheduler was built with -scheduler unset")
	}
}
