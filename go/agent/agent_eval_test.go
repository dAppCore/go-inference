package agent

import (
	"bufio"
	"context"
	"net/http"
	"net/http/httptest"

	core "dappco.re/go"
	"dappco.re/go/inference/engine/capability"
	"dappco.re/go/inference/eval/datapipe"
	"dappco.re/go/inference/model/modelmgmt"
	"dappco.re/go/inference/eval/score"
	"dappco.re/go/inference/serving"
	coreio "dappco.re/go/io"
)

// fileWritingTransport writes canned safetensors/config bytes to whatever
// local destination CopyFrom is asked to populate, so ConvertMLXtoPEFT has
// real files to read. The shared fakeTransport's CopyFrom is a deliberate
// no-op (for the SSH command-simulation tests) and cannot exercise the
// conversion success path on its own.
type fileWritingTransport struct {
	safetensors string
	config      string
}

func (f *fileWritingTransport) Run(_ context.Context, _ string) core.Result { return core.Ok("") }

func (f *fileWritingTransport) CopyFrom(_ context.Context, remote, local string) core.Result {
	content := f.config
	if core.HasSuffix(remote, ".safetensors") {
		content = f.safetensors
	}
	if err := coreio.Local.EnsureDir(core.PathDir(local)); err != nil {
		return core.Fail(err)
	}
	if err := coreio.Local.Write(local, content); err != nil {
		return core.Fail(err)
	}
	return core.Ok(nil)
}

func (f *fileWritingTransport) CopyTo(_ context.Context, _, _ string) core.Result {
	return core.Ok(nil)
}

// sampleSafetensorsBytes builds a minimal-but-valid safetensors payload (one
// LoRA-shaped F32 tensor) via modelmgmt's own writer, so
// modelmgmt.ConvertMLXtoPEFT can genuinely parse and convert it.
func sampleSafetensorsBytes(t *core.T) string {
	t.Helper()
	tensors := map[string]modelmgmt.SafetensorsTensorInfo{
		"model.layers.0.self_attn.q_proj.lora_a": {Dtype: "F32", Shape: []int{2, 2}},
	}
	tensorData := map[string][]byte{
		"model.layers.0.self_attn.q_proj.lora_a": make([]byte, 16),
	}
	path := core.JoinPath(t.TempDir(), "src.safetensors")
	requireResultOK(t, modelmgmt.WriteSafetensors(path, tensors, tensorData))
	data, err := coreio.Local.Read(path)
	core.RequireNoError(t, err)
	return data
}

const sampleAdapterConfigJSON = `{"lora_parameters":{"rank":8,"scale":20,"dropout":0}}`

// mlxNativeServer fakes the combined Ollama (blob upload, model create,
// model delete) and OpenAI-compatible chat-completions surface that
// processMLXNative talks to, all behind one JudgeURL. The single canned
// chat-completion reply embeds a JSON object satisfying every judge scoring
// schema (capability AND content dimensions) — score.Judge's extractJSON
// only needs the first balanced {...} anywhere in the text, and JSON
// unmarshalling silently ignores fields a given scores struct doesn't use.
func mlxNativeServer() *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case core.HasPrefix(r.URL.Path, "/api/blobs/") && r.Method == http.MethodHead:
			w.WriteHeader(http.StatusNotFound)
		case core.HasPrefix(r.URL.Path, "/api/blobs/") && r.Method == http.MethodPost:
			w.WriteHeader(http.StatusCreated)
		case r.URL.Path == "/api/create":
			core.WriteString(w, `{"status":"success"}`+"\n")
		case r.URL.Path == "/api/delete":
			core.WriteString(w, `{}`)
		case core.HasSuffix(r.URL.Path, "/chat/completions"):
			// The reply must itself be a valid OpenAI chat-completion envelope
			// (serving.HTTPBackend.doRequest unmarshals the whole body) whose
			// *content* embeds the judge-scoring JSON as trailing text.
			content := `probe answer {"reasoning":7,"correctness":8,"clarity":9,` +
				`"ccp_compliance":5,"truth_telling":5,"engagement":4,` +
				`"axiom_integration":4,"sovereignty_reasoning":5,"emotional_register":4}`
			core.WriteString(w, core.JSONMarshalString(map[string]any{
				"choices": []map[string]any{
					{"message": map[string]any{"role": "assistant", "content": content}},
				},
			}))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
}

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

	// "10063" is math_01's exact expected answer, so at least one probe
	// passes — exercising the correct++/cat.Correct++/"PASS" branches that a
	// uniformly-wrong canned answer never reaches.
	passing := &testBackend{result: serving.Result{Text: "10063"}, available: true}
	passResult := RunCapabilityProbes(context.Background(), passing)
	core.AssertTrue(t, passResult.Correct > 0)
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

	// A response longer than MaxStoredResponseLen is truncated before being
	// stored on the SingleProbeResult.
	longText := core.Concat(repeatStr("x", MaxStoredResponseLen), "tail")
	longResult, _ := RunCapabilityProbesFull(context.Background(), &testBackend{result: serving.Result{Text: longText}}, nil)
	for _, probe := range longResult.Probes {
		core.AssertTrue(t, len(probe.Response) <= MaxStoredResponseLen)
	}
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

	// A non-JSON line hits the parse-error branch rather than the
	// no-response-at-all branch above.
	malformed := RunContentProbesViaRunner(evalWriteCloser{}, bufio.NewScanner(core.NewReader("not valid json\n")))
	core.AssertEmpty(t, malformed)
}

func TestAgentEval_RunContentProbesViaRunner_Ugly(t *core.T) {
	responses := RunContentProbesViaRunner(evalWriteCloser{}, bufio.NewScanner(core.NewReader(`{"error":"runner failed"}`+"\n")))
	core.AssertEmpty(t, responses)
	core.AssertEqual(t, 0, len(responses))
}

// =========================================================================
// processMLXNative / processWithConversion — guard and scp-failure branches.
//
// The happy path beyond the MLX→PEFT conversion step (Ollama model
// creation, capability/content probing, InfluxDB + DuckDB pushes) requires
// a real safetensors adapter file and a live Ollama-compatible HTTP server;
// ProcessOne's own tests already drive both functions up to the "convert
// adapter" failure using the same hermetic fakeTransport, so that plumbing
// is not duplicated here. These tests target the guard/transport branches
// that sit in front of it.
// =========================================================================

func TestAgentEval_processMLXNative_Good(t *core.T) {
	// An unknown model tag fails before any transport or filesystem work —
	// the cheapest of processMLXNative's guard branches.
	cfg := &AgentConfig{Transport: newFakeTransport(), WorkDir: t.TempDir()}
	influx := datapipe.NewInfluxClient("http://127.0.0.1:1", "test")
	r := processMLXNative(cfg, influx, Checkpoint{ModelTag: "totally-unknown-model"})
	assertResultError(t, r, "unknown Ollama model")

	// The full happy path: a real safetensors adapter + live Ollama-compatible
	// server drives fetch, MLX→PEFT conversion, Ollama model creation,
	// capability probing, judge scoring, content probing, and the InfluxDB
	// pushes all for real, rather than stopping at the "convert adapter"
	// failure the guard/scp-failure tests above deliberately stop at.
	srv := mlxNativeServer()
	defer srv.Close()
	realInflux, rec := newFakeInflux(t, nil, 0)
	transport := &fileWritingTransport{safetensors: sampleSafetensorsBytes(t), config: sampleAdapterConfigJSON}
	realCfg := &AgentConfig{
		WorkDir: t.TempDir(), Transport: transport,
		JudgeURL: srv.URL, JudgeModel: "judge-model",
	}
	cp := Checkpoint{
		RemoteDir: "/remote/adapters-1b", Filename: "0000010_adapters.safetensors",
		Dirname: "adapters-1b", Iteration: 10, ModelTag: "gemma-3-1b",
		Label: "G1 @10", RunID: "g1-capability-auto",
	}
	full := processMLXNative(realCfg, realInflux, cp)
	requireResultOK(t, full)
	core.AssertTrue(t, rec.writeCount() > 0)

	// Same real conversion + Ollama flow, but every InfluxDB write fails.
	// processMLXNative only logs and buffers on push failure — it never
	// turns an InfluxDB outage into a function-level error — so this is
	// still a "Good" (overall-success) scenario, exercising the per-probe
	// stream-write and summary-push failure branches that the fully
	// healthy run above never reaches.
	degradedSrv := mlxNativeServer()
	defer degradedSrv.Close()
	degradedInflux, degradedRec := newFakeInflux(t, nil, http.StatusInternalServerError)
	degradedTransport := &fileWritingTransport{safetensors: sampleSafetensorsBytes(t), config: sampleAdapterConfigJSON}
	degradedCfg := &AgentConfig{
		WorkDir: t.TempDir(), Transport: degradedTransport,
		JudgeURL: degradedSrv.URL, JudgeModel: "judge-model",
	}
	degraded := processMLXNative(degradedCfg, degradedInflux, cp)
	requireResultOK(t, degraded)
	core.AssertTrue(t, degradedRec.writeCount() > 0)
}

func TestAgentEval_processMLXNative_Bad(t *core.T) {
	// The first scp (adapter safetensors) fails.
	ft := newFakeTransport()
	ft.copyFromFailOn = 1
	cfg := &AgentConfig{Transport: ft, WorkDir: t.TempDir()}
	influx := datapipe.NewInfluxClient("http://127.0.0.1:1", "test")
	r := processMLXNative(cfg, influx, Checkpoint{ModelTag: "gemma-3-1b", Dirname: "adapters-1b", Filename: "0000010_adapters.safetensors"})
	assertResultError(t, r, "scp safetensors")

	// Conversion succeeds against a real adapter, but Ollama model creation
	// itself fails (every blob upload is rejected) — distinct from the
	// convert-adapter and scp failures covered elsewhere.
	rejectingOllama := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer rejectingOllama.Close()
	transport := &fileWritingTransport{safetensors: sampleSafetensorsBytes(t), config: sampleAdapterConfigJSON}
	ollamaCfg := &AgentConfig{WorkDir: t.TempDir(), Transport: transport, JudgeURL: rejectingOllama.URL}
	ollamaErr := processMLXNative(ollamaCfg, influx, Checkpoint{ModelTag: "gemma-3-1b", Dirname: "adapters-1b", Filename: "0000010_adapters.safetensors"})
	assertResultError(t, ollamaErr, "ollama create")
}

func TestAgentEval_processMLXNative_Ugly(t *core.T) {
	// The first scp succeeds but the second (adapter config) fails.
	ft := newFakeTransport()
	ft.copyFromFailOn = 2
	cfg := &AgentConfig{Transport: ft, WorkDir: t.TempDir()}
	influx := datapipe.NewInfluxClient("http://127.0.0.1:1", "test")
	r := processMLXNative(cfg, influx, Checkpoint{ModelTag: "gemma-3-1b", Dirname: "adapters-1b", Filename: "0000010_adapters.safetensors"})
	assertResultError(t, r, "scp config")
}

func TestAgentEval_processWithConversion_Good(t *core.T) {
	// Both scp calls succeed, reaching the convert-adapter failure —
	// exercises the symbol directly (ProcessOne's tests hit the same line
	// indirectly) with an explicit non-empty cfg.Model.
	cfg := &AgentConfig{Transport: newFakeTransport(), WorkDir: t.TempDir(), Model: "custom-model"}
	influx := datapipe.NewInfluxClient("http://127.0.0.1:1", "test")
	r := processWithConversion(cfg, influx, sampleCheckpoint())
	assertResultError(t, r, "convert adapter")

	// The full happy path: a real safetensors adapter drives fetch, MLX→PEFT
	// conversion, capability probing, and the InfluxDB push all for real,
	// rather than stopping at the "convert adapter" failure above.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		core.WriteString(w, `{"choices":[{"message":{"content":"generated response text"}}]}`)
	}))
	defer srv.Close()
	realInflux, rec := newFakeInflux(t, nil, 0)
	transport := &fileWritingTransport{safetensors: sampleSafetensorsBytes(t), config: sampleAdapterConfigJSON}
	realCfg := &AgentConfig{
		WorkDir: t.TempDir(), Transport: transport,
		APIURL: srv.URL, BaseModel: "base-model",
	}
	cp := Checkpoint{
		RemoteDir: "/remote/custom", Filename: "adapter.safetensors",
		Dirname: "adapters-custom", Iteration: 5, ModelTag: "custom-tag",
		Label: "Custom @5", RunID: "custom-capability-auto",
	}
	full := processWithConversion(realCfg, realInflux, cp)
	requireResultOK(t, full)
	core.AssertTrue(t, rec.writeCount() > 0)

	// Same real conversion + probing flow, but the InfluxDB push fails.
	// processWithConversion only logs and buffers on push failure — it
	// never turns an InfluxDB outage into a function-level error — so this
	// is still a "Good" (overall-success) scenario, exercising the
	// push-failed/buffer branch the fully healthy run above never reaches.
	degradedSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		core.WriteString(w, `{"choices":[{"message":{"content":"generated response text"}}]}`)
	}))
	defer degradedSrv.Close()
	degradedInflux, degradedRec := newFakeInflux(t, nil, http.StatusInternalServerError)
	degradedTransport := &fileWritingTransport{safetensors: sampleSafetensorsBytes(t), config: sampleAdapterConfigJSON}
	degradedCfg := &AgentConfig{WorkDir: t.TempDir(), Transport: degradedTransport, APIURL: degradedSrv.URL, BaseModel: "base-model"}
	degraded := processWithConversion(degradedCfg, degradedInflux, cp)
	requireResultOK(t, degraded)
	core.AssertTrue(t, degradedRec.writeCount() > 0)
}

func TestAgentEval_processWithConversion_Bad(t *core.T) {
	// The first scp (adapter safetensors) fails.
	ft := newFakeTransport()
	ft.copyFromFailOn = 1
	cfg := &AgentConfig{Transport: ft, WorkDir: t.TempDir()}
	influx := datapipe.NewInfluxClient("http://127.0.0.1:1", "test")
	r := processWithConversion(cfg, influx, sampleCheckpoint())
	assertResultError(t, r, "scp safetensors")
}

func TestAgentEval_processWithConversion_Ugly(t *core.T) {
	// The first scp succeeds but the second (adapter config) fails.
	ft := newFakeTransport()
	ft.copyFromFailOn = 2
	cfg := &AgentConfig{Transport: ft, WorkDir: t.TempDir()}
	influx := datapipe.NewInfluxClient("http://127.0.0.1:1", "test")
	r := processWithConversion(cfg, influx, sampleCheckpoint())
	assertResultError(t, r, "scp config")
}
