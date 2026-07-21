// SPDX-Licence-Identifier: EUPL-1.2

package generate

import (
	"context"
	"iter"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// fakeRetainedTokenStream yields n (at least 1) dummy tokens — enough for
// RunGenerate's "produced only N tokens" floor without a real model.
func fakeRetainedTokenStream(n int) iter.Seq[inference.Token] {
	if n <= 0 {
		n = 1
	}
	return func(yield func(inference.Token) bool) {
		for i := range n {
			if !yield(inference.Token{ID: int32(i), Text: "x"}) {
				return
			}
		}
	}
}

// retainedFakeTextModel is a Chat-capable fake inference.TextModel that ALSO
// implements retainedChatModel. buildCount counts "arch decode state"
// constructions observed through either seam — Chat's fresh-session-per-call
// default, or OpenRetainedSession's one-shared-session path — the proxy
// runBasicGenerate's own session lifecycle drives against a real engine.
type retainedFakeTextModel struct {
	inference.TextModel
	acceptImages bool

	buildCount    int
	chatCalls     int
	openRetained  int
	closeRetained int
}

var (
	_ inference.TextModel   = (*retainedFakeTextModel)(nil)
	_ inference.VisionModel = (*retainedFakeTextModel)(nil)
	_ retainedChatModel     = (*retainedFakeTextModel)(nil)
)

func (f *retainedFakeTextModel) AcceptsImages() bool { return f.acceptImages }

func (f *retainedFakeTextModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	f.buildCount++
	f.chatCalls++
	cfg := inference.ApplyGenerateOpts(opts)
	return fakeRetainedTokenStream(cfg.MaxTokens)
}

func (f *retainedFakeTextModel) OpenRetainedSession() (any, error) {
	f.buildCount++
	f.openRetained++
	return new(int), nil
}

func (f *retainedFakeTextModel) CloseRetainedSession(rs any) error {
	f.closeRetained++
	return nil
}

func (f *retainedFakeTextModel) ChatRetained(ctx context.Context, rs any, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	cfg := inference.ApplyGenerateOpts(opts)
	return fakeRetainedTokenStream(cfg.MaxTokens)
}

func (f *retainedFakeTextModel) Err() core.Result                   { return core.Ok(nil) }
func (f *retainedFakeTextModel) Close() core.Result                 { return core.Ok(nil) }
func (f *retainedFakeTextModel) Metrics() inference.GenerateMetrics { return inference.GenerateMetrics{} }

// retainedFakeBackend registers retainedFakeTextModel behind a unique test-
// only backend name (routed to via inference.WithBackend), so RunGenerate's
// load reaches it without touching the real "metal"/"rocm" registrations.
// Register makes it globally visible for the rest of the test binary, and
// inference.Default() falls back to ANY available backend when none of
// metal/rocm/llama_cpp is registered (true in this portable test binary) — so
// LoadModel only succeeds for stubModelPath, the path every test in this file
// uses; any other path (e.g. TestLoadTextModel_MissingDir_Bad's t.TempDir())
// fails exactly as it would with no fake registered at all, keeping this
// backend from silently adopting unrelated tests' loads.
type retainedFakeBackend struct{ model inference.TextModel }

func (b *retainedFakeBackend) Name() string    { return "generate-test-retained" }
func (b *retainedFakeBackend) Available() bool { return true }
func (b *retainedFakeBackend) LoadModel(path string, opts ...inference.LoadOption) core.Result {
	if path != stubModelPath {
		return core.Fail(core.NewError("retainedFakeBackend: no model at " + path))
	}
	return core.Ok(b.model)
}

// TestRunGenerate_RetainedSession_OneBuild_Good proves the fix: a plain
// (non-multimodal) generate run against a model implementing
// retainedChatModel shares ONE arch decode-state build across the kernel-warm
// pass and the timed pass — where a Chat-only model pays for two (see the
// _Bad sibling below).
func TestRunGenerate_RetainedSession_OneBuild_Good(t *testing.T) {
	fake := &retainedFakeTextModel{}
	inference.Register(&retainedFakeBackend{model: fake})

	err := RunGenerate(context.Background(), Config{
		ModelPath:   stubModelPath,
		Prompt:      "hello",
		MaxTokens:   16,
		Out:         core.NewBuffer(),
		Log:         core.NewBuffer(),
		LoadOptions: []inference.LoadOption{inference.WithBackend("generate-test-retained")},
	})
	if err != nil {
		t.Fatalf("RunGenerate: %v", err)
	}
	if fake.buildCount != 1 {
		t.Fatalf("arch decode state builds = %d, want 1 (warm + timed pass sharing one retained session)", fake.buildCount)
	}
	if fake.chatCalls != 0 {
		t.Fatalf("Chat called %d times, want 0 — both turns should route through the retained session", fake.chatCalls)
	}
	if fake.openRetained != 1 || fake.closeRetained != 1 {
		t.Fatalf("OpenRetainedSession/CloseRetainedSession = %d/%d, want 1/1", fake.openRetained, fake.closeRetained)
	}
}

// TestRunGenerate_RetainedSession_MultimodalSkipsIt_Ugly proves an image turn
// stays on Chat even when the loaded model implements retainedChatModel — the
// retained-session path carries no multimodal input, so runBasicGenerate must
// not route a vision turn through it.
func TestRunGenerate_RetainedSession_MultimodalSkipsIt_Ugly(t *testing.T) {
	fake := &retainedFakeTextModel{acceptImages: true}
	inference.Register(&retainedFakeBackend{model: fake})

	url := "data:image/png;base64," + core.Base64Encode(pngMagic)
	err := RunGenerate(context.Background(), Config{
		ModelPath:    stubModelPath,
		Prompt:       "describe this",
		MaxTokens:    16,
		ImageSources: []string{url},
		Out:          core.NewBuffer(),
		Log:          core.NewBuffer(),
		LoadOptions:  []inference.LoadOption{inference.WithBackend("generate-test-retained")},
	})
	if err != nil {
		t.Fatalf("RunGenerate: %v", err)
	}
	if fake.openRetained != 0 {
		t.Fatalf("OpenRetainedSession called %d times, want 0 — a multimodal turn must not use the retained-session path", fake.openRetained)
	}
	if fake.chatCalls != 2 {
		t.Fatalf("Chat called %d times, want 2 (warm + timed) — multimodal turns keep the original per-call session", fake.chatCalls)
	}
}

// chatOnlyFakeTextModel implements Chat but NOT retainedChatModel — pins the
// fallback for a backend that has not adopted the capability (today's MTP
// speculative wrapper, or any future backend): generate keeps its original
// two-build-per-run behaviour rather than failing or silently dropping a turn.
type chatOnlyFakeTextModel struct {
	inference.TextModel
	chatCalls int
}

var _ inference.TextModel = (*chatOnlyFakeTextModel)(nil)

func (f *chatOnlyFakeTextModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	f.chatCalls++
	cfg := inference.ApplyGenerateOpts(opts)
	return fakeRetainedTokenStream(cfg.MaxTokens)
}

func (f *chatOnlyFakeTextModel) Err() core.Result                   { return core.Ok(nil) }
func (f *chatOnlyFakeTextModel) Close() core.Result                 { return core.Ok(nil) }
func (f *chatOnlyFakeTextModel) Metrics() inference.GenerateMetrics { return inference.GenerateMetrics{} }

// chatOnlyFakeBackend is path-gated on stubModelPath for the same reason as
// retainedFakeBackend above — see its comment.
type chatOnlyFakeBackend struct{ model inference.TextModel }

func (b *chatOnlyFakeBackend) Name() string    { return "generate-test-chatonly" }
func (b *chatOnlyFakeBackend) Available() bool { return true }
func (b *chatOnlyFakeBackend) LoadModel(path string, opts ...inference.LoadOption) core.Result {
	if path != stubModelPath {
		return core.Fail(core.NewError("chatOnlyFakeBackend: no model at " + path))
	}
	return core.Ok(b.model)
}

// TestRunGenerate_ChatOnlyModel_FallsBackToTwoCalls_Bad pins the graceful
// decline: a model without the retained-session capability keeps today's
// fresh-Chat-per-turn behaviour (two calls, warm + timed) — proving the fix
// does not require every backend to adopt the new seam.
func TestRunGenerate_ChatOnlyModel_FallsBackToTwoCalls_Bad(t *testing.T) {
	fake := &chatOnlyFakeTextModel{}
	inference.Register(&chatOnlyFakeBackend{model: fake})

	err := RunGenerate(context.Background(), Config{
		ModelPath:   stubModelPath,
		Prompt:      "hello",
		MaxTokens:   16,
		Out:         core.NewBuffer(),
		Log:         core.NewBuffer(),
		LoadOptions: []inference.LoadOption{inference.WithBackend("generate-test-chatonly")},
	})
	if err != nil {
		t.Fatalf("RunGenerate: %v", err)
	}
	if fake.chatCalls != 2 {
		t.Fatalf("Chat called %d times, want 2 (warm + timed, unchanged fallback behaviour)", fake.chatCalls)
	}
}
