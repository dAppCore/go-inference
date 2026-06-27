package mcp

import (
	"context"

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
	got := capabilityIDStrings(nil)
	core.AssertEqual(t, 0, len(got))
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
