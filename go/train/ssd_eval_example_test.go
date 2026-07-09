// SPDX-Licence-Identifier: EUPL-1.2

package train_test

import (
	"context"
	"fmt"
	"strings"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/train"
)

// ExampleRunSSDCodeBenchmark samples one candidate per task and delegates
// native test execution to the injected RunTests hook. The "add" solution
// contains a "+" so its tests pass; the "sub" solution does not, so the pass
// rate over the two single-candidate tasks is 0.5. pass@1 equals the pass rate
// when there is one candidate per task.
func ExampleRunSSDCodeBenchmark() {
	report, err := train.RunSSDCodeBenchmark(context.Background(), train.SSDCodeBenchmarkRunner{
		Generate: func(_ context.Context, prompt string, _ inference.GenerateConfig) (string, error) {
			if strings.Contains(prompt, "add") {
				return "```python\ndef add(a, b): return a + b\n```", nil
			}
			return "```python\ndef sub(a, b): return a - b\n```", nil
		},
		RunTests: func(_ context.Context, _ train.SSDCodeBenchmarkSample, candidate train.SSDCodeCandidate) (train.SSDCodeExecution, error) {
			return train.SSDCodeExecution{Passed: strings.Contains(candidate.Solution, "+")}, nil
		},
	}, []train.SSDCodeBenchmarkSample{
		{ID: "add", Prompt: "write add"},
		{ID: "sub", Prompt: "write sub"},
	}, train.SSDCodeBenchmarkConfig{NRepeat: 1})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println("samples:", report.Metrics.Samples)
	fmt.Println("candidates:", report.Metrics.Candidates)
	fmt.Println("passed:", report.Metrics.Passed)
	fmt.Println("pass_rate:", report.Metrics.PassRate)
	fmt.Println("pass@1:", report.Metrics.PassAtK["pass@1"])
	// Output:
	// samples: 2
	// candidates: 2
	// passed: 1
	// pass_rate: 0.5
	// pass@1: 0.5
}

// ExampleLoadSSDLiveCodeBenchV6JSONL loads LiveCodeBench-style task rows and
// filters them to the v6 contest-date window (2025-02-01 inclusive to
// 2025-06-01 exclusive). The January and June rows fall outside the window, so
// only the February and May tasks survive.
func ExampleLoadSSDLiveCodeBenchV6JSONL() {
	raw := strings.Join([]string{
		`{"id":"jan","prompt":"old","contest_date":"2025-01-31"}`,
		`{"id":"feb","prompt":"first v6","contest_date":"2025-02-01"}`,
		`{"id":"may","prompt":"last v6","contest_date":"2025-05-31"}`,
		`{"id":"jun","prompt":"new","contest_date":"2025-06-01"}`,
	}, "\n")

	samples, err := train.LoadSSDLiveCodeBenchV6JSONL(raw)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println("samples:", len(samples))
	fmt.Println("ids:", samples[0].ID, samples[1].ID)
	// Output:
	// samples: 2
	// ids: feb may
}

// ExampleSSDPostProcessCode extracts the final fenced code block from a model
// response and strips the LiveCodeBench markup. An earlier go fence is ignored
// — the last python fence wins — and the <code> tags are removed.
func ExampleSSDPostProcessCode() {
	response := "analysis\n```go\nnot this\n```\nfinal\n```python\n<code>def add(a, b):\n    return a + b</code>\n```\n"
	code, ok := train.SSDPostProcessCode(response)
	fmt.Println("ok:", ok)
	fmt.Println(core.Trim(code))
	// Output:
	// ok: true
	// def add(a, b):
	//     return a + b
}

// ExampleLoadSSDCodeBenchmarkJSONL parses LiveCodeBench-style JSONL rows into
// native code-benchmark samples. The id/prompt/test fields map straight across;
// blank lines are skipped.
func ExampleLoadSSDCodeBenchmarkJSONL() {
	raw := strings.Join([]string{
		`{"id":"q1","prompt":"Write add.","tests":["assert add(1, 2) == 3"]}`,
		`{"id":"q2","prompt":"Write sub.","test":"assert sub(3, 1) == 2"}`,
	}, "\n")

	samples, err := train.LoadSSDCodeBenchmarkJSONL(raw)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println("samples:", len(samples))
	fmt.Println("first id:", samples[0].ID)
	fmt.Println("first test:", samples[0].Tests[0])
	// Output:
	// samples: 2
	// first id: q1
	// first test: assert add(1, 2) == 3
}

// ExampleLoadSSDCodeBenchmarkJSONLFile loads benchmark tasks from a JSONL file
// path. The file is written to a temp dir, then read back into native samples.
func ExampleLoadSSDCodeBenchmarkJSONLFile() {
	dirResult := core.MkdirTemp("", "lcb-example-*")
	if !dirResult.OK {
		fmt.Println("error:", dirResult.Value)
		return
	}
	dir := dirResult.Value.(string)
	defer core.RemoveAll(dir)
	path := core.PathJoin(dir, "lcb.jsonl")
	if w := core.WriteFile(path, []byte(`{"id":"q","prompt":"Write identity.","tests":["assert f(1) == 1"]}`+"\n"), 0o644); !w.OK {
		fmt.Println("write error:", w.Value)
		return
	}

	samples, err := train.LoadSSDCodeBenchmarkJSONLFile(path)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println("samples:", len(samples))
	fmt.Println("id:", samples[0].ID)
	// Output:
	// samples: 1
	// id: q
}

// ExampleLoadSSDLiveCodeBenchV6JSONLFile loads the v6 contest-date subset from a
// JSONL file path. The single in-window row survives the filter.
func ExampleLoadSSDLiveCodeBenchV6JSONLFile() {
	dirResult := core.MkdirTemp("", "lcb-v6-example-*")
	if !dirResult.OK {
		fmt.Println("error:", dirResult.Value)
		return
	}
	dir := dirResult.Value.(string)
	defer core.RemoveAll(dir)
	path := core.PathJoin(dir, "lcb-v6.jsonl")
	if w := core.WriteFile(path, []byte(`{"id":"mar","prompt":"v6 task","contest_date":"2025-03-15"}`+"\n"), 0o644); !w.OK {
		fmt.Println("write error:", w.Value)
		return
	}

	samples, err := train.LoadSSDLiveCodeBenchV6JSONLFile(path)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println("samples:", len(samples))
	fmt.Println("id:", samples[0].ID)
	// Output:
	// samples: 1
	// id: mar
}

// ExampleFilterSSDLiveCodeBenchV6Samples keeps only the samples whose
// contest_date meta falls in the v6 window (Feb–May 2025), dropping the rest.
func ExampleFilterSSDLiveCodeBenchV6Samples() {
	in := []train.SSDCodeBenchmarkSample{
		{ID: "jan", Meta: map[string]string{"contest_date": "2025-01-15"}},
		{ID: "mar", Meta: map[string]string{"contest_date": "2025-03-15"}},
		{ID: "jun", Meta: map[string]string{"contest_date": "2025-06-15"}},
	}
	kept := train.FilterSSDLiveCodeBenchV6Samples(in)
	fmt.Println("kept:", len(kept))
	fmt.Println("id:", kept[0].ID)
	// Output:
	// kept: 1
	// id: mar
}

// ExampleFormatSSDLiveCodeBenchPrompt frames a stdin-style task with the
// read-from-stdin instruction wrapped around the problem statement.
func ExampleFormatSSDLiveCodeBenchPrompt() {
	prompt := train.FormatSSDLiveCodeBenchPrompt(train.SSDCodeBenchmarkSample{
		Prompt: "Add two numbers from stdin.",
		Meta:   map[string]string{"is_stdin": "true"},
	})
	fmt.Println("has stdin framing:", strings.Contains(prompt, "stdin"))
	fmt.Println("has problem:", strings.Contains(prompt, "Add two numbers from stdin."))
	// Output:
	// has stdin framing: true
	// has problem: true
}
