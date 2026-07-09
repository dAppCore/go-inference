// SPDX-Licence-Identifier: EUPL-1.2

package agent

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
)

func ExampleNewModelWorkflow() {
	result := NewModelWorkflow(&workflowModel{
		workflowTextOnlyModel: workflowTextOnlyModel{modelType: "mlx"},
	})
	workflow := result.Value.(*ModelWorkflow)

	core.Println(workflow.Model().ModelType())
	// Output:
	// mlx
}

func ExampleModelWorkflow_Run() {
	workflow := mustExampleWorkflow()
	result := workflow.Run(core.Background(), ModelWorkflowRequest{
		Operation: ModelWorkflowEvaluate,
		Dataset:   &workflowDataset{samples: []inference.DatasetSample{{Text: "hello"}}},
	})
	report := result.Value.(ModelWorkflowResult).Eval

	core.Println(report.Metrics.Samples)
	// Output:
	// 1
}

func ExampleModelWorkflow_Model() {
	result := NewModelWorkflow(&workflowTextOnlyModel{modelType: "plain"})
	workflow := result.Value.(*ModelWorkflow)
	core.Println(workflow.Model().ModelType())
	// Output:
	// plain
}

func ExampleModelWorkflow_Capabilities() {
	result := NewModelWorkflow(&workflowTextOnlyModel{modelType: "plain"})
	workflow := result.Value.(*ModelWorkflow)
	report := workflow.Capabilities()
	core.Println(report.Supports(inference.CapabilityGenerate))
	// Output:
	// true
}

func mustExampleWorkflow() *ModelWorkflow {
	result := NewModelWorkflow(&workflowModel{
		evalReport: &inference.EvalReport{Metrics: inference.EvalMetrics{Samples: 1}},
	})
	if !result.OK {
		panic(result.Error())
	}
	return result.Value.(*ModelWorkflow)
}
