// SPDX-Licence-Identifier: EUPL-1.2

package agent

import (
	"iter"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestWorkflow_NewModelWorkflow_Good(t *core.T) {
	model := &workflowModel{workflowTextOnlyModel: workflowTextOnlyModel{modelType: "mlx"}}

	result := NewModelWorkflow(model)

	core.AssertTrue(t, result.OK)
	workflow := result.Value.(*ModelWorkflow)
	core.AssertEqual(t, "mlx", workflow.Model().ModelType())
}

func TestWorkflow_NewModelWorkflow_Bad(t *core.T) {
	result := NewModelWorkflow(nil)

	core.AssertFalse(t, result.OK)
	core.AssertContains(t, result.Error(), "model is required")
}

func TestWorkflow_NewModelWorkflow_Ugly(t *core.T) {
	model := &workflowModel{}

	result := NewModelWorkflow(model)

	core.AssertTrue(t, result.OK)
	workflow := result.Value.(*ModelWorkflow)
	core.AssertEqual(t, model, workflow.Model())
}

func TestWorkflow_ModelWorkflow_Run_Good(t *core.T) {
	model := &workflowModel{
		evalReport:  &inference.EvalReport{Metrics: inference.EvalMetrics{Samples: 2, Tokens: 12, Perplexity: 3.5}},
		benchReport: &inference.BenchReport{PromptTokens: 8, GeneratedTokens: 4, DecodeTokensPerSec: 42},
		trainResult: &inference.TrainingResult{Metrics: inference.TrainingMetrics{Step: 3, Loss: 0.25}},
	}
	workflow := mustModelWorkflow(t, model)
	dataset := &workflowDataset{samples: []inference.DatasetSample{{Text: "one"}}}
	sink := inference.ProbeSinkFunc(func(inference.ProbeEvent) {})

	eval := workflow.Run(core.Background(), ModelWorkflowRequest{
		Operation: ModelWorkflowEvaluate,
		Dataset:   dataset,
		Eval:      inference.EvalConfig{MaxSamples: 1},
		ProbeSink: sink,
	})
	bench := workflow.Run(core.Background(), ModelWorkflowRequest{
		Operation: ModelWorkflowBenchmark,
		Bench:     inference.BenchConfig{Prompts: []string{"hello"}, MeasuredRuns: 1},
	})
	train := workflow.Run(core.Background(), ModelWorkflowRequest{
		Operation: ModelWorkflowTrainSFT,
		Dataset:   dataset,
		Training:  inference.TrainingConfig{BatchSize: 1},
	})
	distill := workflow.Run(core.Background(), ModelWorkflowRequest{
		Operation: ModelWorkflowDistill,
		Dataset:   dataset,
		Distill:   inference.DistillConfig{Temperature: 2},
	})
	grpo := workflow.Run(core.Background(), ModelWorkflowRequest{
		Operation: ModelWorkflowGRPO,
		Dataset:   dataset,
		GRPO:      inference.GRPOConfig{GroupSize: 2},
	})

	core.AssertTrue(t, eval.OK)
	core.AssertTrue(t, bench.OK)
	core.AssertTrue(t, train.OK)
	core.AssertTrue(t, distill.OK)
	core.AssertTrue(t, grpo.OK)
	core.AssertEqual(t, 1, model.evalCalls)
	core.AssertEqual(t, 1, model.benchCalls)
	core.AssertEqual(t, 1, model.sftCalls)
	core.AssertEqual(t, 1, model.distillCalls)
	core.AssertEqual(t, 1, model.grpoCalls)
	core.AssertNotNil(t, model.probeSink)

	evalResult := eval.Value.(ModelWorkflowResult)
	benchResult := bench.Value.(ModelWorkflowResult)
	trainResult := train.Value.(ModelWorkflowResult)
	core.AssertEqual(t, float64(3.5), evalResult.Eval.Metrics.Perplexity)
	core.AssertEqual(t, 42.0, benchResult.Bench.DecodeTokensPerSec)
	core.AssertEqual(t, 3, trainResult.Training.Metrics.Step)
}

func TestWorkflow_ModelWorkflow_Run_Bad(t *core.T) {
	workflow := mustModelWorkflow(t, &workflowTextOnlyModel{})
	dataset := &workflowDataset{samples: []inference.DatasetSample{{Text: "one"}}}

	result := workflow.Run(core.Background(), ModelWorkflowRequest{
		Operation: ModelWorkflowEvaluate,
		Dataset:   dataset,
	})
	core.AssertFalse(t, result.OK)
	core.AssertContains(t, result.Error(), "does not support evaluation")

	// Every other operation rejects a model that implements none of the
	// optional workflow interfaces, the same way Evaluate does above.
	bench := workflow.Run(core.Background(), ModelWorkflowRequest{Operation: ModelWorkflowBenchmark})
	sft := workflow.Run(core.Background(), ModelWorkflowRequest{Operation: ModelWorkflowTrainSFT, Dataset: dataset})
	distill := workflow.Run(core.Background(), ModelWorkflowRequest{Operation: ModelWorkflowDistill, Dataset: dataset})
	grpo := workflow.Run(core.Background(), ModelWorkflowRequest{Operation: ModelWorkflowGRPO, Dataset: dataset})

	core.AssertFalse(t, bench.OK)
	core.AssertContains(t, bench.Error(), "does not support benchmarking")
	core.AssertFalse(t, sft.OK)
	core.AssertContains(t, sft.Error(), "does not support SFT training")
	core.AssertFalse(t, distill.OK)
	core.AssertContains(t, distill.Error(), "does not support distillation")
	core.AssertFalse(t, grpo.OK)
	core.AssertContains(t, grpo.Error(), "does not support GRPO")

	// A ProbeSink on a model that isn't ProbeableModel is rejected before
	// the operation switch even runs.
	probeRejected := workflow.Run(core.Background(), ModelWorkflowRequest{
		Operation: ModelWorkflowEvaluate,
		Dataset:   dataset,
		ProbeSink: inference.ProbeSinkFunc(func(inference.ProbeEvent) {}),
	})
	core.AssertFalse(t, probeRejected.OK)
	core.AssertContains(t, probeRejected.Error(), "does not support probe events")
}

func TestWorkflow_ModelWorkflow_Run_Ugly(t *core.T) {
	workflow := mustModelWorkflow(t, &workflowModel{})

	missingOperation := workflow.Run(core.Background(), ModelWorkflowRequest{})
	missingDataset := workflow.Run(core.Background(), ModelWorkflowRequest{Operation: ModelWorkflowTrainSFT})
	missingEvalDataset := workflow.Run(core.Background(), ModelWorkflowRequest{Operation: ModelWorkflowEvaluate})
	missingDistillDataset := workflow.Run(core.Background(), ModelWorkflowRequest{Operation: ModelWorkflowDistill})
	missingGRPODataset := workflow.Run(core.Background(), ModelWorkflowRequest{Operation: ModelWorkflowGRPO})
	unsupportedOp := workflow.Run(core.Background(), ModelWorkflowRequest{Operation: ModelWorkflowOperation("unknown")})

	core.AssertFalse(t, missingOperation.OK)
	core.AssertContains(t, missingOperation.Error(), "operation is required")
	core.AssertFalse(t, missingDataset.OK)
	core.AssertContains(t, missingDataset.Error(), "dataset is required")
	core.AssertFalse(t, missingEvalDataset.OK)
	core.AssertContains(t, missingEvalDataset.Error(), "dataset is required")
	core.AssertFalse(t, missingDistillDataset.OK)
	core.AssertContains(t, missingDistillDataset.Error(), "dataset is required")
	core.AssertFalse(t, missingGRPODataset.OK)
	core.AssertContains(t, missingGRPODataset.Error(), "dataset is required")
	core.AssertFalse(t, unsupportedOp.OK)
	core.AssertContains(t, unsupportedOp.Error(), "unsupported operation")

	// A nil model workflow (or nil receiver) is rejected before the
	// operation switch runs at all.
	var nilWorkflow *ModelWorkflow
	nilResult := nilWorkflow.Run(core.Background(), ModelWorkflowRequest{Operation: ModelWorkflowEvaluate})
	core.AssertFalse(t, nilResult.OK)
	core.AssertContains(t, nilResult.Error(), "model is required")

	// Every operation wraps a real error surfaced by the underlying model
	// rather than swallowing it.
	dataset := &workflowDataset{samples: []inference.DatasetSample{{Text: "one"}}}
	failing := &workflowModel{
		evalErr:    core.NewError("eval boom"),
		benchErr:   core.NewError("bench boom"),
		sftErr:     core.NewError("sft boom"),
		distillErr: core.NewError("distill boom"),
		grpoErr:    core.NewError("grpo boom"),
	}
	failingWorkflow := mustModelWorkflow(t, failing)

	evalFail := failingWorkflow.Run(core.Background(), ModelWorkflowRequest{Operation: ModelWorkflowEvaluate, Dataset: dataset})
	benchFail := failingWorkflow.Run(core.Background(), ModelWorkflowRequest{Operation: ModelWorkflowBenchmark})
	sftFail := failingWorkflow.Run(core.Background(), ModelWorkflowRequest{Operation: ModelWorkflowTrainSFT, Dataset: dataset})
	distillFail := failingWorkflow.Run(core.Background(), ModelWorkflowRequest{Operation: ModelWorkflowDistill, Dataset: dataset})
	grpoFail := failingWorkflow.Run(core.Background(), ModelWorkflowRequest{Operation: ModelWorkflowGRPO, Dataset: dataset})

	core.AssertFalse(t, evalFail.OK)
	core.AssertContains(t, evalFail.Error(), "evaluate dataset")
	core.AssertFalse(t, benchFail.OK)
	core.AssertContains(t, benchFail.Error(), "benchmark model")
	core.AssertFalse(t, sftFail.OK)
	core.AssertContains(t, sftFail.Error(), "train SFT")
	core.AssertFalse(t, distillFail.OK)
	core.AssertContains(t, distillFail.Error(), "distill model")
	core.AssertFalse(t, grpoFail.OK)
	core.AssertContains(t, grpoFail.Error(), "train GRPO")
}

func TestWorkflow_ModelWorkflow_Model_Good(t *core.T) {
	model := &workflowTextOnlyModel{modelType: "plain"}
	workflow := mustModelWorkflow(t, model)
	got := workflow.Model()
	core.AssertEqual(t, model, got)
}

func TestWorkflow_ModelWorkflow_Model_Bad(t *core.T) {
	var workflow *ModelWorkflow
	got := workflow.Model()
	core.AssertNil(t, got)
}

func TestWorkflow_ModelWorkflow_Model_Ugly(t *core.T) {
	workflow := &ModelWorkflow{}
	got := workflow.Model()
	core.AssertNil(t, got)
}

func TestWorkflow_ModelWorkflow_Capabilities_Good(t *core.T) {
	workflow := mustModelWorkflow(t, &workflowModel{workflowTextOnlyModel: workflowTextOnlyModel{modelType: "native"}})

	report := workflow.Capabilities()

	core.AssertTrue(t, report.Supports(inference.CapabilityEvaluation))
	core.AssertTrue(t, report.Supports(inference.CapabilityBenchmark))
	core.AssertTrue(t, report.Supports(inference.CapabilityLoRATraining))
	core.AssertTrue(t, report.Supports(inference.CapabilityDistillation))
	core.AssertTrue(t, report.Supports(inference.CapabilityGRPO))
}

func TestWorkflow_ModelWorkflow_Capabilities_Bad(t *core.T) {
	var workflow *ModelWorkflow

	report := workflow.Capabilities()

	core.AssertEqual(t, inference.CapabilityReport{}, report)
}

func TestWorkflow_ModelWorkflow_Capabilities_Ugly(t *core.T) {
	workflow := mustModelWorkflow(t, &workflowTextOnlyModel{modelType: "plain"})

	report := workflow.Capabilities()

	core.AssertTrue(t, report.Supports(inference.CapabilityGenerate))
	core.AssertFalse(t, report.Supports(inference.CapabilityEvaluation))
}

func mustModelWorkflow(t *core.T, model inference.TextModel) *ModelWorkflow {
	t.Helper()
	result := NewModelWorkflow(model)
	if !result.OK {
		t.Fatalf("NewModelWorkflow() error = %s", result.Error())
	}
	return result.Value.(*ModelWorkflow)
}

type workflowDataset struct {
	samples []inference.DatasetSample
	index   int
}

func (d *workflowDataset) Next() (inference.DatasetSample, bool, error) {
	if d.index >= len(d.samples) {
		return inference.DatasetSample{}, false, nil
	}
	sample := d.samples[d.index]
	d.index++
	return sample, true, nil
}

type workflowTextOnlyModel struct {
	modelType string
	err       error
}

func (m *workflowTextOnlyModel) Generate(_ core.Context, _ string, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(func(inference.Token) bool) {}
}

func (m *workflowTextOnlyModel) Chat(_ core.Context, _ []inference.Message, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(func(inference.Token) bool) {}
}

func (m *workflowTextOnlyModel) Classify(core.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Fail(core.NewError("not implemented"))
}

func (m *workflowTextOnlyModel) BatchGenerate(core.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Fail(core.NewError("not implemented"))
}

func (m *workflowTextOnlyModel) ModelType() string { return m.modelType }

func (m *workflowTextOnlyModel) Info() inference.ModelInfo {
	return inference.ModelInfo{Architecture: m.modelType}
}

func (m *workflowTextOnlyModel) Metrics() inference.GenerateMetrics {
	return inference.GenerateMetrics{}
}

func (m *workflowTextOnlyModel) Err() core.Result { return core.ResultOf(nil, m.err) }

func (m *workflowTextOnlyModel) Close() core.Result { return core.Ok(nil) }

type workflowModel struct {
	workflowTextOnlyModel

	evalReport  *inference.EvalReport
	benchReport *inference.BenchReport
	trainResult *inference.TrainingResult
	probeSink   inference.ProbeSink

	evalCalls    int
	benchCalls   int
	sftCalls     int
	distillCalls int
	grpoCalls    int

	// *Err, when set, is returned by the matching method instead of a
	// zero-value report — exercises Run()'s per-operation error-wrapping.
	evalErr    error
	benchErr   error
	sftErr     error
	distillErr error
	grpoErr    error
}

func (m *workflowModel) SetProbeSink(sink inference.ProbeSink) {
	m.probeSink = sink
}

func (m *workflowModel) Evaluate(_ core.Context, _ inference.DatasetStream, _ inference.EvalConfig) (*inference.EvalReport, error) {
	m.evalCalls++
	if m.evalErr != nil {
		return nil, m.evalErr
	}
	if m.evalReport != nil {
		return m.evalReport, nil
	}
	return &inference.EvalReport{}, nil
}

func (m *workflowModel) Benchmark(_ core.Context, _ inference.BenchConfig) (*inference.BenchReport, error) {
	m.benchCalls++
	if m.benchErr != nil {
		return nil, m.benchErr
	}
	if m.benchReport != nil {
		return m.benchReport, nil
	}
	return &inference.BenchReport{}, nil
}

func (m *workflowModel) TrainSFT(_ core.Context, _ inference.DatasetStream, _ inference.TrainingConfig) (*inference.TrainingResult, error) {
	m.sftCalls++
	if m.sftErr != nil {
		return nil, m.sftErr
	}
	if m.trainResult != nil {
		return m.trainResult, nil
	}
	return &inference.TrainingResult{}, nil
}

func (m *workflowModel) Distill(_ core.Context, _ inference.DatasetStream, _ inference.DistillConfig) (*inference.TrainingResult, error) {
	m.distillCalls++
	if m.distillErr != nil {
		return nil, m.distillErr
	}
	if m.trainResult != nil {
		return m.trainResult, nil
	}
	return &inference.TrainingResult{}, nil
}

func (m *workflowModel) TrainGRPO(_ core.Context, _ inference.DatasetStream, _ inference.GRPOConfig) (*inference.TrainingResult, error) {
	m.grpoCalls++
	if m.grpoErr != nil {
		return nil, m.grpoErr
	}
	if m.trainResult != nil {
		return m.trainResult, nil
	}
	return &inference.TrainingResult{}, nil
}
