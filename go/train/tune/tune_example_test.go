// SPDX-Licence-Identifier: EUPL-1.2

package tune

import (
	"bytes"
	"context"

	core "dappco.re/go"
)

// ExampleRunTune demonstrates the tune plan for a target with an active MTP
// drafter beside it: RunTune resolves the drafter and reports the plan. This
// example's process registers no concrete engine backend, so — honestly — no
// registered backend exposes inference.SpeculativePairBackend (see the
// package doc): no measurement is run and no profile is written. A process
// that DOES register one (the metal engine does, on darwin/arm64) sweeps
// every requested block and persists the winner instead — see
// TestTune_RunTune_GoodSeamPresent in tune_test.go. The report is written to
// an io.Writer (here a buffer, since the temp model path it names is not
// deterministic across runs) — only the fixed parts of the report are shown.
func ExampleRunTune() {
	baseResult := core.MkdirTemp("", "tune-example-*")
	if !baseResult.OK {
		panic("tempdir failed")
	}
	modelDir := baseResult.Value.(string)
	defer core.RemoveAll(modelDir)

	if r := core.WriteFile(core.PathJoin(modelDir, "config.json"), []byte(`{"model_type":"gemma4"}`), 0o644); !r.OK {
		panic(r.Value)
	}
	assistant := core.PathJoin(modelDir, "assistant")
	if r := core.MkdirAll(assistant, 0o755); !r.OK {
		panic(r.Value)
	}
	if r := core.WriteFile(core.PathJoin(assistant, "config.json"), []byte(`{"model_type":"gemma4"}`), 0o644); !r.OK {
		panic(r.Value)
	}
	if r := core.WriteFile(core.PathJoin(assistant, "model.safetensors"), []byte("weights"), 0o644); !r.OK {
		panic(r.Value)
	}

	var out bytes.Buffer
	err := RunTune(context.Background(), Config{
		ModelPath: modelDir,
		Depths:    "4,5",
		Out:       &out,
	})
	if err != nil {
		panic(err)
	}

	report := out.String()
	core.Println(core.Contains(report, "tune: target "+modelDir))
	core.Println(core.Contains(report, "no registered go-inference engine backend exposes a speculative-pair loader"))
	// Output:
	// true
	// true
}
