package mcp

import (
	"context"
	"net/http"
	"net/http/httptest"

	core "dappco.re/go"
	"dappco.re/go/inference"
	mlpkg "dappco.re/go/inference/mlservice"
	"dappco.re/go/inference/serving"
	coremcp "dappco.re/go/mcp/pkg/mcp"
)

func newCoreMCPService() *coremcp.Service {
	svc, err := coremcp.New(coremcp.Options{})
	if err != nil {
		panic(err)
	}
	return svc
}

func newMCPTestService() *mlpkg.Service {
	factory := mlpkg.NewService(mlpkg.Options{DefaultBackend: "test"})
	r := factory(core.New())
	if !r.OK {
		panic(r.Error())
	}
	svc := r.Value.(*mlpkg.Service)
	svc.RegisterBackend("test", &stubBackend{name: "test"})
	return svc
}

type stubBackend struct {
	name string
}

func (b *stubBackend) Name() string    { return b.name }
func (b *stubBackend) Available() bool { return true }
func (b *stubBackend) Generate(_ context.Context, prompt string, _ serving.GenOpts) core.Result {
	return core.Ok(serving.Result{Text: "response: " + prompt})
}
func (b *stubBackend) Chat(_ context.Context, messages []serving.Message, _ serving.GenOpts) core.Result {
	return core.Ok(serving.Result{Text: "chat response"})
}

// failingBackend is a serving.Backend whose Generate/Chat always fail. It
// exercises the error-propagation branches of mlGenerate and mlProbe, which
// stubBackend (always OK) cannot reach.
type failingBackend struct {
	name string
}

func (b *failingBackend) Name() string    { return b.name }
func (b *failingBackend) Available() bool { return true }
func (b *failingBackend) Generate(_ context.Context, _ string, _ serving.GenOpts) core.Result {
	return core.Fail(core.E("test.failingBackend.Generate", "generate always fails", nil))
}
func (b *failingBackend) Chat(_ context.Context, _ []serving.Message, _ serving.GenOpts) core.Result {
	return core.Fail(core.E("test.failingBackend.Chat", "chat always fails", nil))
}

// newMCPTestServiceNoBackend builds a service with no backends registered at
// all (including no usable default), so Service.Generate fails at the "no
// backend available" guard — the source of mlGenerate's "generate" failure
// branch, distinct from the empty-prompt validation failure.
func newMCPTestServiceNoBackend() *mlpkg.Service {
	factory := mlpkg.NewService(mlpkg.Options{DefaultBackend: "missing"})
	r := factory(core.New())
	if !r.OK {
		panic(r.Error())
	}
	return r.Value.(*mlpkg.Service)
}

// newMCPTestServiceWithJudge builds a service with both a "test" generate
// backend and a judge backend wired via Service.OnStartup, pointed at
// judgeURL (an httptest server) rather than a live LLM judge.
func newMCPTestServiceWithJudge(t *core.T, judgeURL string) *mlpkg.Service {
	t.Helper()
	factory := mlpkg.NewService(mlpkg.Options{
		DefaultBackend: "test",
		JudgeURL:       judgeURL,
		JudgeModel:     "judge-model",
	})
	r := factory(core.New())
	if !r.OK {
		t.Fatalf("newMCPTestServiceWithJudge: factory: %s", r.Error())
	}
	svc := r.Value.(*mlpkg.Service)
	svc.RegisterBackend("test", &stubBackend{name: "test"})
	if r := svc.OnStartup(context.Background()); !r.OK {
		t.Fatalf("newMCPTestServiceWithJudge: OnStartup: %s", r.Error())
	}
	return svc
}

// judgeChatMessage, judgeChatChoice and judgeChatResponse mirror the
// OpenAI-compatible chat-completion wire shape serving.HTTPBackend parses.
// Kept local to the test so it needn't reach into serving's unexported
// types (mirrors the pattern in score/judge_test.go's mockJudgeServer).
type judgeChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}
type judgeChatChoice struct {
	Message judgeChatMessage `json:"message"`
}
type judgeChatResponse struct {
	Choices []judgeChatChoice `json:"choices"`
}

// newJudgeServer starts an httptest server that answers any request with a
// single chat-completion choice whose content is reply, for driving
// score.Judge.ScoreSemantic through mlScore's "semantic" suite hermetically.
func newJudgeServer(t *core.T, reply string) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		resp := judgeChatResponse{Choices: []judgeChatChoice{{Message: judgeChatMessage{Role: "assistant", Content: reply}}}}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(core.JSONMarshalString(resp)))
	}))
	t.Cleanup(server.Close)
	return server
}

// newStatusInfluxServer starts an httptest server that answers every
// InfluxDB v3 query_sql request with the same fixed rowsJSON body, driving
// mlStatus's datapipe.NewInfluxClient + mlservice.PrintStatus round trip
// hermetically (no real InfluxDB, no localhost default port).
func newStatusInfluxServer(t *core.T, rowsJSON string) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(rowsJSON))
	}))
	t.Cleanup(server.Close)
	return server
}

// fakeInferenceBackend is a minimal inference.Backend (the global registry
// type inference.List/Get/Default/Register operate on) — distinct from
// serving.Backend, which is mlservice.Service's per-instance backend map.
// mlBackends walks the former, so exercising its ok/native/capabilities
// branches requires registering one of these rather than a serving.Backend.
type fakeInferenceBackend struct {
	name      string
	available bool
}

func (b *fakeInferenceBackend) Name() string    { return b.name }
func (b *fakeInferenceBackend) Available() bool { return b.available }
func (b *fakeInferenceBackend) LoadModel(_ string, _ ...inference.LoadOption) core.Result {
	return core.Fail(core.E("test.fakeInferenceBackend.LoadModel", "not implemented", nil))
}
func (b *fakeInferenceBackend) Capabilities() inference.CapabilityReport {
	return inference.CapabilityReport{
		Runtime:      inference.RuntimeIdentity{Backend: b.name, NativeRuntime: true},
		Available:    b.available,
		Capabilities: []inference.Capability{inference.SupportedCapability(inference.CapabilityGenerate, inference.CapabilityGroupModel)},
	}
}

// callMCPTool looks up a tool registered by RegisterTools via the coremcp
// REST bridge (Service.Tools()[i].RESTHandler) and invokes it, JSON-encoding
// input the same way a real REST caller would. This drives the tool wrapper
// closures declared inside MLSubsystem.RegisterTools — the seam the coremcp
// package exposes specifically so tool handlers can be exercised without a
// live MCP transport or client.
func callMCPTool(t *core.T, svc *coremcp.Service, name string, input any) (any, error) {
	t.Helper()
	for _, rec := range svc.Tools() {
		if rec.Name == name {
			body := core.AsBytes(core.JSONMarshalString(input))
			return rec.RESTHandler(context.Background(), body)
		}
	}
	t.Fatalf("callMCPTool: tool %q not registered", name)
	return nil, nil
}

func TestSubsystem_NewMLSubsystem_Good(t *core.T) {
	subsystem := NewMLSubsystem(nil)
	core.AssertNotNil(t, subsystem)
	core.AssertEqual(t, "ml", subsystem.Name())
}

func TestSubsystem_NewMLSubsystem_Bad(t *core.T) {
	subsystem := NewMLSubsystem(&mlpkg.Service{})
	core.AssertNotNil(t, subsystem)
	core.AssertNotNil(t, subsystem.logger)
}

func TestSubsystem_NewMLSubsystem_Ugly(t *core.T) {
	subsystem := NewMLSubsystem(nil)
	subsystem.service = nil
	core.AssertNil(t, subsystem.service)
}

func TestSubsystem_MLSubsystem_Name_Good(t *core.T) {
	subsystem := NewMLSubsystem(nil)
	got := subsystem.Name()
	core.AssertEqual(t, "ml", got)
}

func TestSubsystem_MLSubsystem_Name_Bad(t *core.T) {
	subsystem := &MLSubsystem{}
	got := subsystem.Name()
	core.AssertEqual(t, "ml", got)
}

func TestSubsystem_MLSubsystem_Name_Ugly(t *core.T) {
	subsystem := NewMLSubsystem(nil)
	subsystem.logger = nil
	got := subsystem.Name()
	core.AssertEqual(t, "ml", got)
}

func TestSubsystem_MLSubsystem_RegisterTools_Good(t *core.T) {
	subsystem := NewMLSubsystem(nil)
	svc := newCoreMCPService()
	core.AssertNotPanics(t, func() { subsystem.RegisterTools(svc) })
}

func TestSubsystem_MLSubsystem_RegisterTools_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	subsystem := NewMLSubsystem(nil)
	core.AssertPanics(t, func() { subsystem.RegisterTools(nil) })
}

func TestSubsystem_MLSubsystem_RegisterTools_Ugly(t *core.T) {
	subsystem := &MLSubsystem{}
	svc := newCoreMCPService()
	core.AssertNotPanics(t, func() { subsystem.RegisterTools(svc) })
}

func TestSubsystem_MLSubsystem_Shutdown_Good(t *core.T) {
	subsystem := NewMLSubsystem(nil)
	core.AssertNil(t, subsystem.Shutdown(context.Background()))
	core.AssertEqual(t, "ml", subsystem.Name())
}

func TestSubsystem_MLSubsystem_Shutdown_Bad(t *core.T) {
	subsystem := &MLSubsystem{}
	core.AssertNil(t, subsystem.Shutdown(context.Background()))
}

func TestSubsystem_MLSubsystem_Shutdown_Ugly(t *core.T) {
	subsystem := NewMLSubsystem(nil)
	subsystem.logger = nil
	core.AssertNil(t, subsystem.Shutdown(context.Background()))
}

func TestSubsystem_mlGenerate_Good(t *core.T) {
	svc := newMCPTestService()
	subsystem := NewMLSubsystem(svc)
	input := MLGenerateInput{Prompt: "hello"}
	r := subsystem.mlGenerate(context.Background(), nil, input)
	core.AssertTrue(t, r.OK)
}

func TestSubsystem_mlGenerate_Bad(t *core.T) {
	svc := newMCPTestService()
	subsystem := NewMLSubsystem(svc)
	input := MLGenerateInput{Prompt: ""}
	r := subsystem.mlGenerate(context.Background(), nil, input)
	core.AssertFalse(t, r.OK)
}

func TestSubsystem_mlGenerate_Ugly(t *core.T) {
	svc := newMCPTestService()
	subsystem := NewMLSubsystem(svc)
	subsystem.logger = nil
	input := MLGenerateInput{Prompt: "test", Temperature: 0.7, MaxTokens: 100}
	r := subsystem.mlGenerate(context.Background(), nil, input)
	core.AssertTrue(t, r.OK)
}

func TestSubsystem_mlScore_Good(t *core.T) {
	svc := newMCPTestService()
	subsystem := NewMLSubsystem(svc)
	input := MLScoreInput{Prompt: "what is 2+2", Response: "4"}
	r := subsystem.mlScore(context.Background(), nil, input)
	core.AssertTrue(t, r.OK)
}

func TestSubsystem_mlScore_Bad(t *core.T) {
	svc := newMCPTestService()
	subsystem := NewMLSubsystem(svc)
	input := MLScoreInput{Prompt: "", Response: ""}
	r := subsystem.mlScore(context.Background(), nil, input)
	core.AssertFalse(t, r.OK)
}

func TestSubsystem_mlScore_Ugly(t *core.T) {
	svc := newMCPTestService()
	subsystem := NewMLSubsystem(svc)
	subsystem.logger = nil
	input := MLScoreInput{Prompt: "test", Response: "result", Suites: "semantic"}
	r := subsystem.mlScore(context.Background(), nil, input)
	core.AssertFalse(t, r.OK)
}

func TestSubsystem_mlScore_contentRejected(t *core.T) {
	svc := newMCPTestService()
	subsystem := NewMLSubsystem(svc)
	input := MLScoreInput{Prompt: "test", Response: "result", Suites: "content"}
	r := subsystem.mlScore(context.Background(), nil, input)
	core.AssertFalse(t, r.OK)
}

func TestSubsystem_mlProbe_Good(t *core.T) {
	svc := newMCPTestService()
	subsystem := NewMLSubsystem(svc)
	input := MLProbeInput{Backend: "test"}
	r := subsystem.mlProbe(context.Background(), nil, input)
	core.AssertTrue(t, r.OK)
	output := r.Value.(MLProbeOutput)
	core.AssertTrue(t, output.Total > 0)
}

func TestSubsystem_mlProbe_Bad(t *core.T) {
	svc := newMCPTestService()
	subsystem := NewMLSubsystem(svc)
	input := MLProbeInput{Categories: "nonexistent"}
	r := subsystem.mlProbe(context.Background(), nil, input)
	core.AssertTrue(t, r.OK)
	output := r.Value.(MLProbeOutput)
	core.AssertEqual(t, 0, output.Total)
}

func TestSubsystem_mlProbe_Ugly(t *core.T) {
	svc := newMCPTestService()
	subsystem := NewMLSubsystem(svc)
	subsystem.logger = nil
	input := MLProbeInput{}
	r := subsystem.mlProbe(context.Background(), nil, input)
	core.AssertTrue(t, r.OK)
}

func TestSubsystem_mlBackends_Good(t *core.T) {
	svc := newMCPTestService()
	subsystem := NewMLSubsystem(svc)
	r := subsystem.mlBackends(context.Background(), nil, MLBackendsInput{})
	core.AssertTrue(t, r.OK)
	output := r.Value.(MLBackendsOutput)
	core.AssertTrue(t, len(output.Backends) >= 0)
}

func TestSubsystem_mlBackends_Bad(t *core.T) {
	subsystem := NewMLSubsystem(nil)
	subsystem.logger = nil
	r := subsystem.mlBackends(context.Background(), nil, MLBackendsInput{})
	core.AssertTrue(t, r.OK)
}

func TestSubsystem_mlBackends_Ugly(t *core.T) {
	svc := newMCPTestService()
	subsystem := NewMLSubsystem(svc)
	subsystem.logger = nil
	r := subsystem.mlBackends(context.Background(), nil, MLBackendsInput{})
	core.AssertTrue(t, r.OK)
}

func TestSubsystem_capabilityIDStrings_Good(t *core.T) {
	got := capabilityIDStrings([]inference.CapabilityID{"chat", "generate"})
	core.AssertEqual(t, 2, len(got))
	core.AssertEqual(t, "chat", got[0])
	core.AssertEqual(t, "generate", got[1])
}

func TestSubsystem_capabilityIDStrings_Bad(t *core.T) {
	got := capabilityIDStrings(nil)
	core.AssertEqual(t, 0, len(got))
}

func TestSubsystem_capabilityIDStrings_Ugly(t *core.T) {
	got := capabilityIDStrings([]inference.CapabilityID{})
	core.AssertEqual(t, 0, len(got))
	core.AssertNotNil(t, got)
}

// --- mlGenerate: generate-call failure (distinct from empty-prompt Bad) ---

func TestSubsystem_mlGenerate_generateFailure(t *core.T) {
	svc := newMCPTestServiceNoBackend()
	subsystem := NewMLSubsystem(svc)
	input := MLGenerateInput{Prompt: "hello", Backend: "missing"}
	r := subsystem.mlGenerate(context.Background(), nil, input)
	core.AssertFalse(t, r.OK)
	core.AssertContains(t, r.Error(), "generate")
}

// --- mlScore: semantic suite success/failure + multi-suite combination ---

func TestSubsystem_mlScore_semanticSuccess(t *core.T) {
	server := newJudgeServer(t, `{"sovereignty": 8, "ethical_depth": 7, "creative_expression": 6, "self_concept": 5, "reasoning": "solid"}`)
	svc := newMCPTestServiceWithJudge(t, server.URL)
	subsystem := NewMLSubsystem(svc)
	input := MLScoreInput{Prompt: "what is 2+2", Response: "4", Suites: "semantic"}
	r := subsystem.mlScore(context.Background(), nil, input)
	core.AssertTrue(t, r.OK)
	output := r.Value.(MLScoreOutput)
	core.AssertNotNil(t, output.Semantic)
	core.AssertEqual(t, 8, output.Semantic.Sovereignty)
	core.AssertEqual(t, "solid", output.Semantic.Reasoning)
}

func TestSubsystem_mlScore_semanticFailure(t *core.T) {
	server := newJudgeServer(t, "no json here at all")
	svc := newMCPTestServiceWithJudge(t, server.URL)
	subsystem := NewMLSubsystem(svc)
	input := MLScoreInput{Prompt: "what is 2+2", Response: "4", Suites: "semantic"}
	r := subsystem.mlScore(context.Background(), nil, input)
	core.AssertFalse(t, r.OK)
	core.AssertContains(t, r.Error(), "semantic score")
}

func TestSubsystem_mlScore_multiSuite(t *core.T) {
	server := newJudgeServer(t, `{"sovereignty": 3, "ethical_depth": 3, "creative_expression": 3, "self_concept": 3}`)
	svc := newMCPTestServiceWithJudge(t, server.URL)
	subsystem := NewMLSubsystem(svc)
	input := MLScoreInput{Prompt: "hello", Response: "world", Suites: "heuristic, semantic"}
	r := subsystem.mlScore(context.Background(), nil, input)
	core.AssertTrue(t, r.OK)
	output := r.Value.(MLScoreOutput)
	core.AssertNotNil(t, output.Heuristic)
	core.AssertNotNil(t, output.Semantic)
}

// --- mlProbe: backend failure surfaces as an "error: ..." probe response ---

func TestSubsystem_mlProbe_generateFailure(t *core.T) {
	svc := newMCPTestService()
	svc.RegisterBackend("broken", &failingBackend{name: "broken"})
	subsystem := NewMLSubsystem(svc)
	input := MLProbeInput{Backend: "broken", Categories: "arithmetic"}
	r := subsystem.mlProbe(context.Background(), nil, input)
	core.AssertTrue(t, r.OK)
	output := r.Value.(MLProbeOutput)
	core.AssertGreater(t, output.Total, 0)
	for _, item := range output.Results {
		core.AssertContains(t, item.Response, "error:")
	}
}

// --- mlBackends: global inference registry walk (ok/native/capabilities/default) ---

func TestSubsystem_mlBackends_registeredBackend(t *core.T) {
	inference.Register(&fakeInferenceBackend{name: "mcp_test_backend", available: true})

	svc := newMCPTestService()
	subsystem := NewMLSubsystem(svc)
	r := subsystem.mlBackends(context.Background(), nil, MLBackendsInput{})
	core.AssertTrue(t, r.OK)
	output := r.Value.(MLBackendsOutput)

	var found *MLBackendInfo
	for i := range output.Backends {
		if output.Backends[i].Name == "mcp_test_backend" {
			found = &output.Backends[i]
		}
	}
	core.AssertNotNil(t, found)
	core.AssertTrue(t, found.Available)
	core.AssertTrue(t, found.Native)
	core.AssertContains(t, found.Capabilities, "generate")
	core.AssertEqual(t, "mcp_test_backend", output.Default)
}

// --- mlStatus: not previously covered at all (0% baseline) ---

func TestSubsystem_mlStatus_Good(t *core.T) {
	server := newStatusInfluxServer(t, `[{"model":"gemma4","run_id":"r1","status":"training","iteration":10,"total_iters":100,"pct":10}]`)
	svc := newMCPTestService()
	subsystem := NewMLSubsystem(svc)
	input := MLStatusInput{InfluxURL: server.URL, InfluxDB: "lem"}
	r := subsystem.mlStatus(context.Background(), nil, input)
	core.AssertTrue(t, r.OK)
	output := r.Value.(MLStatusOutput)
	core.AssertContains(t, output.Status, "Training:")
	core.AssertContains(t, output.Status, "gemma4")
}

// TestSubsystem_mlStatus_Bad exercises missing/invalid caller config (empty
// InfluxDB, triggering the db=="" default-substitution branch) rather than a
// Result-level failure: mlservice.PrintStatus tolerates unreachable/empty
// query results by design (see mlStatus's own "functions deliberately left
// untested" note), so there is no input that makes mlStatus itself fail.
func TestSubsystem_mlStatus_Bad(t *core.T) {
	server := newStatusInfluxServer(t, `[]`)
	svc := newMCPTestService()
	subsystem := NewMLSubsystem(svc)
	input := MLStatusInput{InfluxURL: server.URL, InfluxDB: ""}
	r := subsystem.mlStatus(context.Background(), nil, input)
	core.AssertTrue(t, r.OK)
	output := r.Value.(MLStatusOutput)
	core.AssertContains(t, output.Status, "(no data)")
}

func TestSubsystem_mlStatus_Ugly(t *core.T) {
	server := newStatusInfluxServer(t, `[]`)
	svc := newMCPTestService()
	subsystem := NewMLSubsystem(svc)
	subsystem.logger = nil
	input := MLStatusInput{InfluxURL: server.URL, InfluxDB: "lem"}
	r := subsystem.mlStatus(context.Background(), nil, input)
	core.AssertTrue(t, r.OK)
	output := r.Value.(MLStatusOutput)
	core.AssertContains(t, output.Status, "Generation:")
}

// --- RegisterTools: drive the registered tool closures via the REST bridge ---
//
// RegisterTools itself only wires 5 closures into the coremcp.Service; the
// closures' bodies (the result.OK check + type assertion) are statements of
// RegisterTools for coverage purposes but only execute when a tool is
// actually invoked. coremcp.Service.Tools()[i].RESTHandler is the seam the
// mcp package exposes precisely for this (also used for its MCP-to-REST
// bridge), so these tests call tools through it rather than standing up a
// live MCP client/transport.

func TestSubsystem_MLSubsystem_RegisterTools_mlGenerateInvocationSuccess(t *core.T) {
	mlSvc := newMCPTestService()
	subsystem := NewMLSubsystem(mlSvc)
	svc := newCoreMCPService()
	subsystem.RegisterTools(svc)

	result, err := callMCPTool(t, svc, "ml_generate", MLGenerateInput{Prompt: "hello"})
	core.AssertNoError(t, err)
	output := result.(MLGenerateOutput)
	core.AssertContains(t, output.Response, "hello")
}

func TestSubsystem_MLSubsystem_RegisterTools_mlGenerateInvocationFailure(t *core.T) {
	mlSvc := newMCPTestService()
	subsystem := NewMLSubsystem(mlSvc)
	svc := newCoreMCPService()
	subsystem.RegisterTools(svc)

	_, err := callMCPTool(t, svc, "ml_generate", MLGenerateInput{Prompt: ""})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "prompt cannot be empty")
}

func TestSubsystem_MLSubsystem_RegisterTools_mlScoreInvocationSuccess(t *core.T) {
	mlSvc := newMCPTestService()
	subsystem := NewMLSubsystem(mlSvc)
	svc := newCoreMCPService()
	subsystem.RegisterTools(svc)

	result, err := callMCPTool(t, svc, "ml_score", MLScoreInput{Prompt: "2+2", Response: "4"})
	core.AssertNoError(t, err)
	output := result.(MLScoreOutput)
	core.AssertNotNil(t, output.Heuristic)
}

func TestSubsystem_MLSubsystem_RegisterTools_mlScoreInvocationFailure(t *core.T) {
	mlSvc := newMCPTestService()
	subsystem := NewMLSubsystem(mlSvc)
	svc := newCoreMCPService()
	subsystem.RegisterTools(svc)

	_, err := callMCPTool(t, svc, "ml_score", MLScoreInput{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "prompt and response cannot be empty")
}

func TestSubsystem_MLSubsystem_RegisterTools_mlProbeInvocation(t *core.T) {
	mlSvc := newMCPTestService()
	subsystem := NewMLSubsystem(mlSvc)
	svc := newCoreMCPService()
	subsystem.RegisterTools(svc)

	result, err := callMCPTool(t, svc, "ml_probe", MLProbeInput{Backend: "test", Categories: "arithmetic"})
	core.AssertNoError(t, err)
	output := result.(MLProbeOutput)
	core.AssertGreater(t, output.Total, 0)
}

func TestSubsystem_MLSubsystem_RegisterTools_mlStatusInvocation(t *core.T) {
	server := newStatusInfluxServer(t, `[]`)
	mlSvc := newMCPTestService()
	subsystem := NewMLSubsystem(mlSvc)
	svc := newCoreMCPService()
	subsystem.RegisterTools(svc)

	result, err := callMCPTool(t, svc, "ml_status", MLStatusInput{InfluxURL: server.URL, InfluxDB: "lem"})
	core.AssertNoError(t, err)
	output := result.(MLStatusOutput)
	core.AssertContains(t, output.Status, "Training:")
}

func TestSubsystem_MLSubsystem_RegisterTools_mlBackendsInvocation(t *core.T) {
	mlSvc := newMCPTestService()
	subsystem := NewMLSubsystem(mlSvc)
	svc := newCoreMCPService()
	subsystem.RegisterTools(svc)

	result, err := callMCPTool(t, svc, "ml_backends", MLBackendsInput{})
	core.AssertNoError(t, err)
	_, ok := result.(MLBackendsOutput)
	core.AssertTrue(t, ok)
}
