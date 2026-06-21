package mcp

import (
	"context"
	"iter"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// --- AX-7 canonical triplets ---

func TestToolsExternal_Buffer_String_Good(t *core.T) {
	var buffer safeBuffer
	buffer.append([]byte("agent"))
	got := buffer.String()

	core.AssertEqual(t, "agent", got)
}

func TestToolsExternal_Buffer_String_Bad(t *core.T) {
	var buffer safeBuffer
	got := buffer.String()
	want := ""

	core.AssertEqual(t, want, got)
	core.AssertEmpty(t, got)
}

func TestToolsExternal_Buffer_String_Ugly(t *core.T) {
	var buffer safeBuffer
	buffer.append([]byte("agent"))
	first := buffer.String()

	core.AssertEqual(t, first, buffer.String())
}

type capabilityBackend struct {
	name string
}

func (backend capabilityBackend) Name() string { return backend.name }

func (backend capabilityBackend) Available() bool { return true }

func (backend capabilityBackend) LoadModel(string, ...inference.LoadOption) core.Result {
	return core.Fail(core.AnError)
}

func (backend capabilityBackend) Capabilities() inference.CapabilityReport {
	return inference.CapabilityReport{
		Runtime:   inference.RuntimeIdentity{Backend: backend.name, NativeRuntime: true},
		Available: true,
		Capabilities: []inference.Capability{
			inference.SupportedCapability(inference.CapabilityGenerate, inference.CapabilityGroupModel),
			inference.SupportedCapability(inference.CapabilityProbeEvents, inference.CapabilityGroupProbe),
		},
	}
}

func TestToolsExternal_mlBackends_Good(t *core.T) {
	name := "ai-capability-test-" + t.Name()
	inference.Register(capabilityBackend{name: name})

	result := (&Service{}).mlBackends(context.Background(), MLBackendsInput{})
	output := result.Value.(MLBackendsOutput)

	var found *MLBackendInfo
	for i := range output.Backends {
		if output.Backends[i].Name == name {
			found = &output.Backends[i]
			break
		}
	}

	core.AssertNotNil(t, found)
	core.AssertTrue(t, found.Available)
	core.AssertTrue(t, found.Native)
	core.AssertContains(t, found.Capabilities, string(inference.CapabilityGenerate))
	core.AssertContains(t, found.Capabilities, string(inference.CapabilityProbeEvents))
}

func TestToolsExternal_MLGenerate_Good_UsesConfiguredInferenceModel(t *core.T) {
	model := &generateModel{}
	serviceResult := New(WithInferenceModel(model, "external-openai", "gpt-test"))
	core.AssertTrue(t, serviceResult.OK)
	service := serviceResult.Value.(*Service)

	result := service.mlGenerate(context.Background(), MLGenerateInput{
		Prompt:      "hello",
		Model:       "gpt-test",
		Temperature: 0.25,
		MaxTokens:   8,
	})
	core.AssertTrue(t, result.OK)

	output := result.Value.(MLGenerateOutput)
	core.AssertEqual(t, "provider answer", output.Response)
	core.AssertEqual(t, "external-openai", output.Backend)
	core.AssertEqual(t, "gpt-test", output.Model)
	core.AssertEqual(t, "hello", model.prompt)
	core.AssertEqual(t, 8, model.cfg.MaxTokens)
	core.AssertEqual(t, float32(0.25), model.cfg.Temperature)
}

type generateModel struct {
	prompt string
	cfg    inference.GenerateConfig
	err    error
}

func (m *generateModel) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	m.prompt = prompt
	m.cfg = inference.ApplyGenerateOpts(opts)
	return func(yield func(inference.Token) bool) {
		yield(inference.Token{Text: "provider answer"})
	}
}

func (m *generateModel) Chat(context.Context, []inference.Message, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(func(inference.Token) bool) {}
}

func (m *generateModel) Classify(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Fail(core.AnError)
}

func (m *generateModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Fail(core.AnError)
}

func (m *generateModel) ModelType() string { return "external" }

func (m *generateModel) Info() inference.ModelInfo {
	return inference.ModelInfo{Architecture: "external"}
}

func (m *generateModel) Metrics() inference.GenerateMetrics { return inference.GenerateMetrics{} }

func (m *generateModel) Err() core.Result {
	if m.err != nil {
		return core.Fail(m.err)
	}
	return core.Ok(nil)
}

func (m *generateModel) Close() core.Result { return core.Ok(nil) }
