// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"flag"
	"io"

	core "dappco.re/go"
	"dappco.re/go/inference/model/arch/openai/whisper"
)

// transcribe.go is thin flag-parsing over dappco.re/go/inference/model/arch/openai/whisper's Load/
// Model.Transcribe — the ASR business logic lives there, not here (mirroring generate.go's doc comment:
// "the serve business logic lives in dappco.re/go/inference/serving, not here"). Whisper is an
// encoder-decoder that never enters model.Assemble (see whisper.Config.Arch's doc comment), so this
// verb calls whisper.Load directly rather than going through the shared generate/serve model-loading
// path — the same "own loader" shape mamba2 would need if it grew a dedicated verb.
//
//	lem transcribe --model ~/models/whisper-tiny clip.wav
//
// ctx carries no cancellation point yet — Model.Transcribe is a single host-CPU forward pass with no
// natural cut point (v1 scope; a later slice could thread it through the decode loop).
func runTranscribeCommand(_ context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("transcribe"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	modelDir := fs.String("model", "", "Whisper checkpoint directory (config.json, tokenizer.json, *.safetensors)")
	language := fs.String("language", "", "force the source language (\"en\" or \"<|en|>\"); empty auto-detects")
	jsonOut := fs.Bool("json", false, "print {\"text\":...,\"language\":...} instead of plain transcript text")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s transcribe --model <whisper-dir> [flags] <audio.wav>\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Transcribe one WAV clip (16-bit PCM, mono, 16 kHz) through a loaded Whisper\n")
		core.WriteString(stderr, "checkpoint: host-f32 greedy decode, a single <=30s window (v1 scope — longer\n")
		core.WriteString(stderr, "audio is a named refusal, not silent truncation).\n")
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Flags:\n")
		printFlagBlock(stderr, fs)
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Examples:\n")
		core.WriteString(stderr, core.Sprintf("  %s transcribe --model ~/models/whisper-tiny clip.wav\n", name))
		core.WriteString(stderr, "    # auto-detect language, transcript to stdout\n")
		core.WriteString(stderr, core.Sprintf("  %s transcribe --model ~/models/whisper-tiny --language en --json clip.wav\n", name))
		core.WriteString(stderr, "    # force English, {\"text\":...,\"language\":...} to stdout\n")
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if *modelDir == "" {
		core.Print(stderr, "%s transcribe: --model is required", cliName())
		fs.Usage()
		return 2
	}
	if fs.NArg() != 1 {
		core.Print(stderr, "%s transcribe: expected exactly one audio WAV path", cliName())
		fs.Usage()
		return 2
	}

	m, err := whisper.Load(*modelDir)
	if err != nil {
		core.Print(stderr, "%s transcribe: %v", cliName(), err)
		return 1
	}
	audioPath := fs.Arg(0)
	read := core.ReadFile(audioPath)
	if !read.OK {
		core.Print(stderr, "%s transcribe: read %s: %s", cliName(), audioPath, read.Error())
		return 1
	}
	wavBytes, ok := read.Value.([]byte)
	if !ok {
		core.Print(stderr, "%s transcribe: read %s returned non-byte data", cliName(), audioPath)
		return 1
	}

	result, err := m.Transcribe(wavBytes, whisper.Options{Language: *language})
	if err != nil {
		core.Print(stderr, "%s transcribe: %v", cliName(), err)
		return 1
	}

	if *jsonOut {
		printJSON(stdout, map[string]string{"text": result.Text, "language": result.Language})
		return 0
	}
	core.WriteString(stdout, result.Text)
	core.WriteString(stdout, "\n")
	return 0
}
