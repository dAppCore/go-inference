package mlservice

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/eval/score"
	"dappco.re/go/inference/serving"
)

func newServiceForTest(t *core.T, opts Options) *Service {
	t.Helper()
	factory := NewService(opts)
	r := factory(core.New())
	requireResultOK(t, r)
	raw := r.Value
	svc, ok := raw.(*Service)
	core.RequireTrue(t, ok)
	return svc
}

func TestService_NewService_Good(t *core.T) {
	factory := NewService(Options{DefaultBackend: "stub"})
	r := factory(core.New())
	requireResultOK(t, r)
	raw := r.Value
	svc := raw.(*Service)
	core.AssertNotNil(t, svc)
	core.AssertEqual(t, "stub", svc.Options().DefaultBackend)
}

func TestService_NewService_Bad(t *core.T) {
	factory := NewService(Options{})
	r := factory(core.New())
	requireResultOK(t, r)
	raw := r.Value
	svc := raw.(*Service)
	core.AssertNotNil(t, svc)
	core.AssertEqual(t, 4, svc.Options().Concurrency)
}

func TestService_NewService_Ugly(t *core.T) {
	factory := NewService(Options{Suites: "heuristic", Concurrency: 1})
	r := factory(core.New())
	requireResultOK(t, r)
	raw := r.Value
	svc := raw.(*Service)
	core.AssertNotNil(t, svc)
	core.AssertEqual(t, "heuristic", svc.Options().Suites)
}

func TestService_RegisterCore_Good(t *core.T) {
	// RegisterCore wires the service into Core via core.WithService; the
	// returned Result is the registration success marker (Value=nil). The
	// constructed Service is accessible via the Core service registry, not
	// via the Result. To verify default Options shape, invoke the factory
	// directly.
	requireResultOK(t, RegisterCore(core.New()))
	r := NewService(Options{})(core.New())
	requireResultOK(t, r)
	svc := r.Value.(*Service)
	core.AssertEqual(t, 4, svc.Options().Concurrency)
}

func TestService_RegisterCore_Bad(t *core.T) {
	requireResultOK(t, RegisterCore(core.New()))
	r := NewService(Options{})(core.New())
	requireResultOK(t, r)
	svc := r.Value.(*Service)
	core.AssertEqual(t, "all", svc.Options().Suites)
}

func TestService_RegisterCore_Ugly(t *core.T) {
	requireResultOK(t, RegisterCore(core.New()))
	r := NewService(Options{})(core.New())
	requireResultOK(t, r)
	svc := r.Value.(*Service)
	core.AssertEmpty(t, svc.Backends())
}

func TestService_Service_OnStartup_Good(t *core.T) {
	svc := newServiceForTest(t, Options{OllamaURL: "http://127.0.0.1", JudgeURL: "http://127.0.0.1", JudgeModel: "judge"})
	err := svc.OnStartup(context.Background())
	requireResultOK(t, err)
	core.AssertNotNil(t, svc.Engine())
}

func TestService_Service_OnStartup_Bad(t *core.T) {
	svc := newServiceForTest(t, Options{})
	err := svc.OnStartup(context.Background())
	requireResultOK(t, err)
	core.AssertNil(t, svc.Engine())
}

func TestService_Service_OnStartup_Ugly(t *core.T) {
	svc := newServiceForTest(t, Options{OllamaURL: "http://127.0.0.1"})
	err := svc.OnStartup(context.Background())
	requireResultOK(t, err)
	core.AssertNotNil(t, svc.Backend("ollama"))
}

func TestService_Service_OnShutdown_Good(t *core.T) {
	svc := newServiceForTest(t, Options{})
	err := svc.OnShutdown(context.Background())
	assertResultOK(t, err)
	core.AssertNotNil(t, svc)
}

func TestService_Service_OnShutdown_Bad(t *core.T) {
	svc := newServiceForTest(t, Options{Concurrency: 2})
	err := svc.OnShutdown(context.Background())
	assertResultOK(t, err)
	core.AssertEqual(t, 2, svc.Options().Concurrency)
}

func TestService_Service_OnShutdown_Ugly(t *core.T) {
	svc := newServiceForTest(t, Options{Suites: "all"})
	err := svc.OnShutdown(context.Background())
	assertResultOK(t, err)
	core.AssertEqual(t, "all", svc.Options().Suites)
}

func TestService_Service_RegisterBackend_Good(t *core.T) {
	svc := newServiceForTest(t, Options{})
	svc.RegisterBackend("stub", &testBackend{name: "stub", available: true})
	core.AssertNotNil(t, svc.Backend("stub"))
	core.AssertContains(t, svc.Backends(), "stub")
}

func TestService_Service_RegisterBackend_Bad(t *core.T) {
	svc := newServiceForTest(t, Options{})
	svc.RegisterBackend("nil", nil)
	core.AssertNil(t, svc.Backend("nil"))
	core.AssertContains(t, svc.Backends(), "nil")
}

func TestService_Service_RegisterBackend_Ugly(t *core.T) {
	svc := newServiceForTest(t, Options{})
	svc.RegisterBackend("stub", &testBackend{name: "old"})
	svc.RegisterBackend("stub", &testBackend{name: "new"})
	core.AssertEqual(t, "new", svc.Backend("stub").Name())
}

func TestService_Service_Backend_Good(t *core.T) {
	svc := newServiceForTest(t, Options{})
	svc.RegisterBackend("stub", &testBackend{name: "stub"})
	core.AssertEqual(t, "stub", svc.Backend("stub").Name())
}

func TestService_Service_Backend_Bad(t *core.T) {
	svc := newServiceForTest(t, Options{})
	backend := svc.Backend("missing")
	core.AssertNil(t, backend)
}

func TestService_Service_Backend_Ugly(t *core.T) {
	svc := newServiceForTest(t, Options{})
	svc.RegisterBackend("", &testBackend{name: "blank"})
	core.AssertEqual(t, "blank", svc.Backend("").Name())
}

func TestService_Service_DefaultBackend_Good(t *core.T) {
	svc := newServiceForTest(t, Options{DefaultBackend: "stub"})
	svc.RegisterBackend("stub", &testBackend{name: "stub"})
	core.AssertEqual(t, "stub", svc.DefaultBackend().Name())
}

func TestService_Service_DefaultBackend_Bad(t *core.T) {
	svc := newServiceForTest(t, Options{})
	backend := svc.DefaultBackend()
	core.AssertNil(t, backend)
}

func TestService_Service_DefaultBackend_Ugly(t *core.T) {
	svc := newServiceForTest(t, Options{})
	svc.RegisterBackend("ollama", &testBackend{name: "ollama"})
	core.AssertEqual(t, "ollama", svc.DefaultBackend().Name())
}

func TestService_Service_Backends_Good(t *core.T) {
	svc := newServiceForTest(t, Options{})
	svc.RegisterBackend("b", &testBackend{name: "b"})
	svc.RegisterBackend("a", &testBackend{name: "a"})
	core.AssertEqual(t, []string{"a", "b"}, svc.Backends())
}

func TestService_Service_Backends_Bad(t *core.T) {
	svc := newServiceForTest(t, Options{})
	names := svc.Backends()
	core.AssertEmpty(t, names)
	core.AssertEqual(t, 0, len(names))
}

func TestService_Service_Backends_Ugly(t *core.T) {
	svc := newServiceForTest(t, Options{})
	svc.RegisterBackend("", &testBackend{})
	core.AssertEqual(t, []string{""}, svc.Backends())
}

func TestService_Service_BackendsIter_Good(t *core.T) {
	svc := newServiceForTest(t, Options{})
	svc.RegisterBackend("stub", &testBackend{})
	var names []string
	for name := range svc.BackendsIter() {
		names = append(names, name)
	}
	core.AssertContains(t, names, "stub")
}

func TestService_Service_BackendsIter_Bad(t *core.T) {
	svc := newServiceForTest(t, Options{})
	count := 0
	for range svc.BackendsIter() {
		count++
	}
	core.AssertEqual(t, 0, count)
}

func TestService_Service_BackendsIter_Ugly(t *core.T) {
	svc := newServiceForTest(t, Options{})
	svc.RegisterBackend("one", &testBackend{})
	count := 0
	for range svc.BackendsIter() {
		count++
		break
	}
	core.AssertEqual(t, 1, count)
}

func TestService_Service_BackendCapabilities_Good(t *core.T) {
	svc := newServiceForTest(t, Options{})
	svc.RegisterBackend("local", &testBackend{name: "local", available: true})
	report, ok := svc.BackendCapabilities("local")
	core.AssertTrue(t, ok)
	core.AssertEqual(t, "local", report.Runtime.Backend)
	core.AssertTrue(t, report.Supports(inference.CapabilityGenerate))
}

func TestService_Service_BackendCapabilities_Bad(t *core.T) {
	svc := newServiceForTest(t, Options{})
	report, ok := svc.BackendCapabilities("missing")
	core.AssertFalse(t, ok)
	core.AssertEqual(t, inference.CapabilityReport{}, report)
}

func TestService_Service_BackendCapabilities_Ugly(t *core.T) {
	svc := newServiceForTest(t, Options{})
	svc.RegisterBackend("nil", nil)
	report, ok := svc.BackendCapabilities("nil")
	core.AssertFalse(t, ok)
	core.AssertEqual(t, inference.CapabilityReport{}, report)
}

func TestService_Service_Judge_Good(t *core.T) {
	svc := newServiceForTest(t, Options{JudgeURL: "http://127.0.0.1", JudgeModel: "judge"})
	requireResultOK(t, svc.OnStartup(context.Background()))
	core.AssertNotNil(t, svc.Judge())
}

func TestService_Service_Judge_Bad(t *core.T) {
	svc := newServiceForTest(t, Options{})
	judge := svc.Judge()
	core.AssertNil(t, judge)
}

func TestService_Service_Judge_Ugly(t *core.T) {
	svc := newServiceForTest(t, Options{})
	svc.judge = score.NewJudge(&testBackend{})
	core.AssertNotNil(t, svc.Judge())
}

func TestService_Service_Engine_Good(t *core.T) {
	svc := newServiceForTest(t, Options{JudgeURL: "http://127.0.0.1", JudgeModel: "judge"})
	requireResultOK(t, svc.OnStartup(context.Background()))
	core.AssertNotNil(t, svc.Engine())
}

func TestService_Service_Engine_Bad(t *core.T) {
	svc := newServiceForTest(t, Options{})
	engine := svc.Engine()
	core.AssertNil(t, engine)
}

func TestService_Service_Engine_Ugly(t *core.T) {
	svc := newServiceForTest(t, Options{})
	svc.engine = score.NewEngine(score.NewJudge(&testBackend{result: serving.Result{Text: `{"sovereignty":1}`}}), 1, "heuristic")
	core.AssertNotNil(t, svc.Engine())
}

func TestService_Service_Generate_Good(t *core.T) {
	svc := newServiceForTest(t, Options{DefaultBackend: "stub"})
	svc.RegisterBackend("stub", &testBackend{result: serving.Result{Text: "generated"}})
	r := svc.Generate(context.Background(), "", "prompt", serving.GenOpts{})
	requireResultOK(t, r)
	result := r.Value.(serving.Result)
	core.AssertEqual(t, "generated", result.Text)
}

func TestService_Service_Generate_Bad(t *core.T) {
	svc := newServiceForTest(t, Options{})
	r := svc.Generate(context.Background(), "missing", "prompt", serving.GenOpts{})
	assertResultError(t, r)
}

func TestService_Service_Generate_Ugly(t *core.T) {
	svc := newServiceForTest(t, Options{})
	svc.RegisterBackend("err", &testBackend{err: core.AnError})
	r := svc.Generate(context.Background(), "err", "prompt", serving.GenOpts{})
	assertResultError(t, r)
}

func TestService_Service_ScoreResponses_Good(t *core.T) {
	svc := newServiceForTest(t, Options{})
	svc.engine = score.NewEngine(score.NewJudge(&testBackend{result: serving.Result{Text: `{"sovereignty":1,"ethical_depth":1,"creative_expression":1,"self_concept":1}`}}), 1, "heuristic")
	r := svc.ScoreResponses(context.Background(), []score.Response{{ID: "r1", Model: "m", Response: "substantial response text"}})
	requireResultOK(t, r)
	out := r.Value.(map[string][]score.PromptScore)
	core.AssertNotNil(t, out)
}

func TestService_Service_ScoreResponses_Bad(t *core.T) {
	svc := newServiceForTest(t, Options{})
	r := svc.ScoreResponses(context.Background(), nil)
	assertResultError(t, r)
}

func TestService_Service_ScoreResponses_Ugly(t *core.T) {
	svc := newServiceForTest(t, Options{})
	svc.engine = score.NewEngine(score.NewJudge(&testBackend{}), 1, "heuristic")
	r := svc.ScoreResponses(context.Background(), nil)
	requireResultOK(t, r)
	out := r.Value.(map[string][]score.PromptScore)
	core.AssertEmpty(t, out)
}
