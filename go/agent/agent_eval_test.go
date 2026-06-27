package agent

import (
	"bufio"
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference/capability"
	"dappco.re/go/inference/score"
	"dappco.re/go/inference/serving"
)

type evalWriteCloser struct{}

func (evalWriteCloser) Write(p []byte) (int, error) { return len(p), nil }
func (evalWriteCloser) Close() error                { return nil }

func contentRunnerScanner() *bufio.Scanner {
	b := core.NewBuilder()
	for range score.ContentProbes {
		_, _ = b.WriteString(`{"response":"runner answer"}`)
		_ = b.WriteByte('\n')
	}
	return bufio.NewScanner(core.NewReader(b.String()))
}

func TestAgentEval_RunCapabilityProbes_Good(t *core.T) {
	backend := &testBackend{result: serving.Result{Text: "4"}, available: true}
	result := RunCapabilityProbes(context.Background(), backend)
	core.AssertEqual(t, len(capability.CapabilityProbes), result.Total)
	core.AssertLen(t, result.Probes, len(capability.CapabilityProbes))
}

func TestAgentEval_RunCapabilityProbes_Bad(t *core.T) {
	backend := &testBackend{err: core.AnError}
	result := RunCapabilityProbes(context.Background(), backend)
	core.AssertEqual(t, len(capability.CapabilityProbes), result.Total)
	core.AssertEqual(t, 0, result.Correct)
}

func TestAgentEval_RunCapabilityProbes_Ugly(t *core.T) {
	backend := &testBackend{result: serving.Result{Text: core.Concat(repeatStr("x", MaxStoredResponseLen), "tail")}}
	result := RunCapabilityProbes(context.Background(), backend)
	core.AssertEqual(t, len(capability.CapabilityProbes), result.Total)
	core.AssertTrue(t, len(result.Probes) > 0)
}

func TestAgentEval_RunCapabilityProbesFull_Good(t *core.T) {
	calls := 0
	result, full := RunCapabilityProbesFull(context.Background(), &testBackend{result: serving.Result{Text: "4"}}, func(_, _ string, _ bool, _ string, _, _ int) { calls++ })
	core.AssertEqual(t, len(capability.CapabilityProbes), result.Total)
	core.AssertLen(t, full, len(capability.CapabilityProbes))
	core.AssertEqual(t, len(capability.CapabilityProbes), calls)
}

func TestAgentEval_RunCapabilityProbesFull_Bad(t *core.T) {
	result, full := RunCapabilityProbesFull(context.Background(), &testBackend{err: core.AnError}, nil)
	core.AssertEqual(t, len(capability.CapabilityProbes), result.Total)
	core.AssertLen(t, full, len(capability.CapabilityProbes))
	core.AssertEqual(t, 0, result.Correct)
}

func TestAgentEval_RunCapabilityProbesFull_Ugly(t *core.T) {
	result, full := RunCapabilityProbesFull(context.Background(), &testBackend{result: serving.Result{Text: ""}}, func(_, _ string, _ bool, _ string, _, _ int) {})
	core.AssertEqual(t, len(capability.CapabilityProbes), result.Total)
	core.AssertLen(t, full, len(capability.CapabilityProbes))
	core.AssertNotNil(t, result.ByCategory)
}

func TestAgentEval_RunContentProbesViaAPI_Good(t *core.T) {
	responses := RunContentProbesViaAPI(context.Background(), &testBackend{result: serving.Result{Text: "content answer"}})
	core.AssertLen(t, responses, len(score.ContentProbes))
	core.AssertEqual(t, score.ContentProbes[0].ID, responses[0].Probe.ID)
}

func TestAgentEval_RunContentProbesViaAPI_Bad(t *core.T) {
	responses := RunContentProbesViaAPI(context.Background(), &testBackend{err: core.AnError})
	core.AssertEmpty(t, responses)
	core.AssertEqual(t, 0, len(responses))
}

func TestAgentEval_RunContentProbesViaAPI_Ugly(t *core.T) {
	responses := RunContentProbesViaAPI(context.Background(), &testBackend{result: serving.Result{Text: "<think>x</think>visible"}})
	core.AssertLen(t, responses, len(score.ContentProbes))
	core.AssertEqual(t, "visible", responses[0].Response)
}

func TestAgentEval_RunContentProbes_Good(t *core.T) {
	responses := RunContentProbes(context.Background(), &testBackend{result: serving.Result{Text: "alias answer"}})
	core.AssertLen(t, responses, len(score.ContentProbes))
	core.AssertEqual(t, score.ContentProbes[0].ID, responses[0].Probe.ID)
}

func TestAgentEval_RunContentProbes_Bad(t *core.T) {
	responses := RunContentProbes(context.Background(), &testBackend{err: core.AnError})
	core.AssertEmpty(t, responses)
	core.AssertEqual(t, 0, len(responses))
}

func TestAgentEval_RunContentProbes_Ugly(t *core.T) {
	responses := RunContentProbes(context.Background(), &testBackend{result: serving.Result{Text: ""}})
	core.AssertLen(t, responses, len(score.ContentProbes))
	core.AssertEqual(t, score.ContentProbes[0].Prompt, responses[0].Response)
}

func TestAgentEval_RunContentProbesViaRunner_Good(t *core.T) {
	responses := RunContentProbesViaRunner(evalWriteCloser{}, contentRunnerScanner())
	core.AssertLen(t, responses, len(score.ContentProbes))
	core.AssertEqual(t, "runner answer", responses[0].Response)
}

func TestAgentEval_RunContentProbesViaRunner_Bad(t *core.T) {
	responses := RunContentProbesViaRunner(evalWriteCloser{}, bufio.NewScanner(core.NewReader("")))
	core.AssertEmpty(t, responses)
	core.AssertEqual(t, 0, len(responses))
}

func TestAgentEval_RunContentProbesViaRunner_Ugly(t *core.T) {
	responses := RunContentProbesViaRunner(evalWriteCloser{}, bufio.NewScanner(core.NewReader(`{"error":"runner failed"}`+"\n")))
	core.AssertEmpty(t, responses)
	core.AssertEqual(t, 0, len(responses))
}
