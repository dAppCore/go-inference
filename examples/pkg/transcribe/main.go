// SPDX-Licence-Identifier: EUPL-1.2

// Speech-to-text via the host Whisper encoder-decoder. model/arch/openai/whisper is a pure-CPU float32
// forward pass — like model/arch/bert (see pkg/embed), it never touches the GPU registry the chat/eval
// examples blank-import (no metallib, no engine backend needed), so this example has no
// `_ "dappco.re/go/inference/examples/internal/engine"` line. Whisper is an encoder-decoder, a different
// shape from the decoder-only causal-LM the neutral inference.TextModel contract assumes, so it is
// driven directly rather than through inference.LoadModel (mirroring the CLI: `lem transcribe`, not
// `lem generate`).
//
//	go run ./pkg/transcribe -model ~/.cache/huggingface/hub/models--openai--whisper-tiny/snapshots/<rev> -audio clip.wav
//
// Fetch a snapshot first:
//
//	hf download openai/whisper-tiny
package main

import (
	"flag"
	"fmt"
	"os"

	"dappco.re/go/inference/model/arch/openai/whisper"
)

func main() {
	model := flag.String("model", os.Getenv("LEM_MODEL"), "Whisper checkpoint directory (config.json, tokenizer.json, *.safetensors)")
	audio := flag.String("audio", "", "a 16-bit PCM mono 16 kHz WAV file to transcribe")
	language := flag.String("language", "", "force the source language (\"en\", \"<|en|>\"); empty auto-detects")
	flag.Parse()
	if *model == "" {
		fmt.Fprintln(os.Stderr, "set -model (or LEM_MODEL) to a whisper-tiny snapshot directory")
		os.Exit(2)
	}
	if *audio == "" {
		fmt.Fprintln(os.Stderr, "set -audio to a 16-bit PCM mono 16 kHz WAV file")
		os.Exit(2)
	}

	// whisper.Load and Model.Transcribe return plain Go errors, not core.Result — the host encoder-
	// decoder path never crosses the engine registry that LoadModel/Generate report through.
	m, err := whisper.Load(*model)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load:", err)
		os.Exit(1)
	}

	wavBytes, err := os.ReadFile(*audio)
	if err != nil {
		fmt.Fprintln(os.Stderr, "read audio:", err)
		os.Exit(1)
	}

	result, err := m.Transcribe(wavBytes, whisper.Options{Language: *language})
	if err != nil {
		fmt.Fprintln(os.Stderr, "transcribe:", err)
		os.Exit(1)
	}

	fmt.Printf("language: %s\n", result.Language)
	fmt.Printf("text:     %s\n", result.Text)
}
