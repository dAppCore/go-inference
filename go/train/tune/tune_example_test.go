// SPDX-Licence-Identifier: EUPL-1.2

package tune

import (
	"bytes"
	"context"

	core "dappco.re/go"
)

// ExampleRunTune demonstrates the tune plan for a target with an active MTP
// drafter beside it: RunTune resolves the drafter and reports the plan, but —
// honestly — the MTP sweep itself is blocked on a speculative-pair engine
// seam go-inference does not yet expose (see the package doc), so no
// measurement is run and no profile is written. The report is written to an
// io.Writer (here a buffer, since the temp model path it names is not
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
	core.Println(core.Contains(report, "no registered go-inference engine exposes one yet"))
	// Output:
	// true
	// true
}
