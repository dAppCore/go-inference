// SPDX-Licence-Identifier: EUPL-1.2

package agent

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
)

// ModelWorkflowOperation identifies a backend-neutral model workflow.
type ModelWorkflowOperation string

const (
	ModelWorkflowEvaluate  ModelWorkflowOperation = "evaluate"
	ModelWorkflowBenchmark ModelWorkflowOperation = "benchmark"
	ModelWorkflowTrainSFT  ModelWorkflowOperation = "train.sft"
	ModelWorkflowDistill   ModelWorkflowOperation = "distill"
	ModelWorkflowGRPO      ModelWorkflowOperation = "grpo"
)

// ModelWorkflowRequest carries the shared workflow inputs for eval, bench,
// supervised LoRA training, distillation, and GRPO.
type ModelWorkflowRequest struct {
	Operation ModelWorkflowOperation
	Dataset   inference.DatasetStream
	ProbeSink inference.ProbeSink

	Eval     inference.EvalConfig
	Bench    inference.BenchConfig
	Training inference.TrainingConfig
	Distill  inference.DistillConfig
	GRPO     inference.GRPOConfig

	Labels map[string]string
}

// ModelWorkflowResult carries the typed report returned by a workflow run.
type ModelWorkflowResult struct {
	Operation ModelWorkflowOperation
	Eval      *inference.EvalReport
	Bench     *inference.BenchReport
	Training  *inference.TrainingResult
	Labels    map[string]string
}

// ModelWorkflow delegates go-ml workflow orchestration to shared inference
// contracts implemented by native or remote model backends.
type ModelWorkflow struct {
	model inference.TextModel
}

// NewModelWorkflow creates a workflow façade around an inference model.
func NewModelWorkflow(model inference.TextModel) core.Result {
	if model == nil {
		return core.Fail(core.E("ml.NewModelWorkflow", "model is required", nil))
	}
	return core.Ok(&ModelWorkflow{model: model})
}

// Model returns the underlying inference model for advanced callers.
func (w *ModelWorkflow) Model() inference.TextModel {
	if w == nil {
		return nil
	}
	return w.model
}

// Capabilities returns the workflow model's shared capability report.
func (w *ModelWorkflow) Capabilities() inference.CapabilityReport {
	if w == nil || w.model == nil {
		return inference.CapabilityReport{}
	}
	report, ok := inference.CapabilitiesOf(w.model)
	if ok {
		return report
	}
	return inference.TextModelCapabilities(inference.RuntimeIdentity{Backend: w.model.ModelType()}, w.model)
}

// Run executes one backend-neutral model workflow.
func (w *ModelWorkflow) Run(ctx core.Context, request ModelWorkflowRequest) core.Result {
	if w == nil || w.model == nil {
		return core.Fail(core.E("ml.ModelWorkflow.Run", "model is required", nil))
	}
	if request.Operation == "" {
		return core.Fail(core.E("ml.ModelWorkflow.Run", "operation is required", nil))
	}
	if request.ProbeSink != nil {
		probeable, ok := w.model.(inference.ProbeableModel)
		if !ok {
			return core.Fail(core.E("ml.ModelWorkflow.Run", "model does not support probe events", nil))
		}
		probeable.SetProbeSink(request.ProbeSink)
	}

	switch request.Operation {
	case ModelWorkflowEvaluate:
		return w.evaluate(ctx, request)
	case ModelWorkflowBenchmark:
		return w.benchmark(ctx, request)
	case ModelWorkflowTrainSFT:
		return w.trainSFT(ctx, request)
	case ModelWorkflowDistill:
		return w.distill(ctx, request)
	case ModelWorkflowGRPO:
		return w.grpo(ctx, request)
	default:
		return core.Fail(core.E("ml.ModelWorkflow.Run", core.Sprintf("unsupported operation %q", request.Operation), nil))
	}
}

func (w *ModelWorkflow) evaluate(ctx core.Context, request ModelWorkflowRequest) core.Result {
	if request.Dataset == nil {
		return core.Fail(core.E("ml.ModelWorkflow.Evaluate", "dataset is required", nil))
	}
	evaluator, ok := w.model.(inference.Evaluator)
	if !ok {
		return core.Fail(core.E("ml.ModelWorkflow.Evaluate", "model does not support evaluation", nil))
	}
	report, err := evaluator.Evaluate(ctx, request.Dataset, request.Eval)
	if err != nil {
		return core.Fail(core.E("ml.ModelWorkflow.Evaluate", "evaluate dataset", err))
	}
	return core.Ok(ModelWorkflowResult{
		Operation: request.Operation,
		Eval:      report,
		Labels:    core.MapClone(request.Labels),
	})
}

func (w *ModelWorkflow) benchmark(ctx core.Context, request ModelWorkflowRequest) core.Result {
	benchable, ok := w.model.(inference.BenchableModel)
	if !ok {
		return core.Fail(core.E("ml.ModelWorkflow.Benchmark", "model does not support benchmarking", nil))
	}
	report, err := benchable.Benchmark(ctx, request.Bench)
	if err != nil {
		return core.Fail(core.E("ml.ModelWorkflow.Benchmark", "benchmark model", err))
	}
	return core.Ok(ModelWorkflowResult{
		Operation: request.Operation,
		Bench:     report,
		Labels:    core.MapClone(request.Labels),
	})
}

func (w *ModelWorkflow) trainSFT(ctx core.Context, request ModelWorkflowRequest) core.Result {
	if request.Dataset == nil {
		return core.Fail(core.E("ml.ModelWorkflow.TrainSFT", "dataset is required", nil))
	}
	trainer, ok := w.model.(inference.SFTTrainer)
	if !ok {
		return core.Fail(core.E("ml.ModelWorkflow.TrainSFT", "model does not support SFT training", nil))
	}
	report, err := trainer.TrainSFT(ctx, request.Dataset, request.Training)
	if err != nil {
		return core.Fail(core.E("ml.ModelWorkflow.TrainSFT", "train SFT", err))
	}
	return core.Ok(ModelWorkflowResult{
		Operation: request.Operation,
		Training:  report,
		Labels:    core.MapClone(request.Labels),
	})
}

func (w *ModelWorkflow) distill(ctx core.Context, request ModelWorkflowRequest) core.Result {
	if request.Dataset == nil {
		return core.Fail(core.E("ml.ModelWorkflow.Distill", "dataset is required", nil))
	}
	trainer, ok := w.model.(inference.DistillTrainer)
	if !ok {
		return core.Fail(core.E("ml.ModelWorkflow.Distill", "model does not support distillation", nil))
	}
	report, err := trainer.Distill(ctx, request.Dataset, request.Distill)
	if err != nil {
		return core.Fail(core.E("ml.ModelWorkflow.Distill", "distill model", err))
	}
	return core.Ok(ModelWorkflowResult{
		Operation: request.Operation,
		Training:  report,
		Labels:    core.MapClone(request.Labels),
	})
}

func (w *ModelWorkflow) grpo(ctx core.Context, request ModelWorkflowRequest) core.Result {
	if request.Dataset == nil {
		return core.Fail(core.E("ml.ModelWorkflow.GRPO", "dataset is required", nil))
	}
	trainer, ok := w.model.(inference.GRPOTrainer)
	if !ok {
		return core.Fail(core.E("ml.ModelWorkflow.GRPO", "model does not support GRPO training", nil))
	}
	report, err := trainer.TrainGRPO(ctx, request.Dataset, request.GRPO)
	if err != nil {
		return core.Fail(core.E("ml.ModelWorkflow.GRPO", "train GRPO", err))
	}
	return core.Ok(ModelWorkflowResult{
		Operation: request.Operation,
		Training:  report,
		Labels:    core.MapClone(request.Labels),
	})
}
