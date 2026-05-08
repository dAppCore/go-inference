// SPDX-Licence-Identifier: EUPL-1.2

package inference

import (
	"context"
	"testing"

	core "dappco.re/go"
)

type capabilityModel struct {
	*stubTextModel
	sink    ProbeSink
	adapter AdapterIdentity
}

func (m *capabilityModel) Encode(text string) []int32 {
	return []int32{int32(len(text))}
}

func (m *capabilityModel) Decode(ids []int32) string {
	return core.Sprintf("%d", len(ids))
}

func (m *capabilityModel) ApplyChatTemplate(messages []Message) (string, error) {
	if len(messages) == 0 {
		return "", nil
	}
	return messages[0].Content, nil
}

func (m *capabilityModel) LoadAdapter(path string) (AdapterIdentity, error) {
	m.adapter = AdapterIdentity{Path: path, Format: "lora"}
	return m.adapter, nil
}

func (m *capabilityModel) UnloadAdapter() error {
	m.adapter = AdapterIdentity{}
	return nil
}

func (m *capabilityModel) ActiveAdapter() AdapterIdentity {
	return m.adapter
}

func (m *capabilityModel) CaptureState(context.Context, string, ...GenerateOption) (*StateBundle, error) {
	return &StateBundle{Model: ModelIdentity{Architecture: "stub"}}, nil
}

func (m *capabilityModel) RestoreState(context.Context, *StateBundle) error {
	return nil
}

func (m *capabilityModel) SetProbeSink(sink ProbeSink) {
	m.sink = sink
}

func (m *capabilityModel) Benchmark(context.Context, BenchConfig) (*BenchReport, error) {
	return &BenchReport{Model: ModelIdentity{Architecture: "stub"}}, nil
}

func (m *capabilityModel) PlanModelFit(context.Context, ModelIdentity, uint64) (*ModelFitReport, error) {
	return &ModelFitReport{Fits: true}, nil
}

func (m *capabilityModel) TrainSFT(context.Context, DatasetStream, TrainingConfig) (*TrainingResult, error) {
	return &TrainingResult{Adapter: AdapterIdentity{Format: "lora"}}, nil
}

func (m *capabilityModel) Distill(context.Context, DatasetStream, DistillConfig) (*TrainingResult, error) {
	return &TrainingResult{Model: ModelIdentity{Architecture: "student"}}, nil
}

func (m *capabilityModel) TrainGRPO(context.Context, DatasetStream, GRPOConfig) (*TrainingResult, error) {
	return &TrainingResult{Metrics: TrainingMetrics{Step: 1}}, nil
}

func TestCapabilityInterfaces(t *testing.T) {
	model := &capabilityModel{stubTextModel: &stubTextModel{}}

	_, ok := any(model).(TokenizerModel)
	checkTrue(t, ok)
	_, ok = any(model).(AdapterModel)
	checkTrue(t, ok)
	_, ok = any(model).(StatefulModel)
	checkTrue(t, ok)
	_, ok = any(model).(ProbeableModel)
	checkTrue(t, ok)
	_, ok = any(model).(BenchableModel)
	checkTrue(t, ok)
	_, ok = any(model).(ModelFitPlanner)
	checkTrue(t, ok)
	_, ok = any(model).(SFTTrainer)
	checkTrue(t, ok)
	_, ok = any(model).(DistillTrainer)
	checkTrue(t, ok)
	_, ok = any(model).(GRPOTrainer)
	checkTrue(t, ok)
}

func TestCapability_TokenizerModel_Good(t *testing.T) {
	model := &capabilityModel{}
	tokenizer := any(model).(TokenizerModel)

	ids := tokenizer.Encode("hello")
	text := tokenizer.Decode([]int32{1, 2, 3})
	prompt, err := tokenizer.ApplyChatTemplate([]Message{{Role: "user", Content: "hi"}})

	checkNoError(t, err)
	checkEqual(t, []int32{5}, ids)
	checkEqual(t, "3", text)
	checkEqual(t, "hi", prompt)
}

func TestCapability_AdapterModel_Good(t *testing.T) {
	model := &capabilityModel{}
	adapter := any(model).(AdapterModel)

	identity, err := adapter.LoadAdapter("/tmp/adapter.safetensors")
	checkNoError(t, err)
	checkEqual(t, "/tmp/adapter.safetensors", identity.Path)
	checkEqual(t, "lora", adapter.ActiveAdapter().Format)

	checkNoError(t, adapter.UnloadAdapter())
	checkEqual(t, AdapterIdentity{}, adapter.ActiveAdapter())
}

func TestCapability_StateAndProbe_Ugly_MinimalModel(t *testing.T) {
	model := &capabilityModel{}
	stateful := any(model).(StatefulModel)
	probeable := any(model).(ProbeableModel)

	bundle, err := stateful.CaptureState(context.Background(), "prompt")
	checkNoError(t, err)
	checkEqual(t, "stub", bundle.Model.Architecture)

	probeable.SetProbeSink(ProbeSinkFunc(func(ProbeEvent) {}))
	checkNotNil(t, model.sink)
}
