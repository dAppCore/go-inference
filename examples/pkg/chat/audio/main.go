// SPDX-Licence-Identifier: EUPL-1.2

// Audio chat: attach a WAV clip (16-bit PCM mono 16 kHz — see the Audios
// field doc) to a user turn. As with vision, accepting audio is a live
// capability of the LOADED CHECKPOINT, not a family-wide guarantee, so probe
// before sending.
//
//	go run ./pkg/chat/audio -model ~/models/gemma-4-e2b-it-4bit -audio clip.wav
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"dappco.re/go/inference"
	_ "dappco.re/go/inference/examples/internal/engine" // registers the platform engine
)

func main() {
	model := flag.String("model", os.Getenv("LEM_MODEL"), "model snapshot directory (config.json + *.safetensors)")
	audio := flag.String("audio", "", "WAV file, 16-bit PCM mono 16 kHz")
	prompt := flag.String("prompt", "Transcribe this clip in one sentence.", "user message")
	flag.Parse()
	if *model == "" || *audio == "" {
		fmt.Fprintln(os.Stderr, "usage: -model <snapshot dir> -audio <file.wav>")
		os.Exit(2)
	}

	clip, err := os.ReadFile(*audio)
	if err != nil {
		fmt.Fprintln(os.Stderr, "audio:", err)
		os.Exit(1)
	}

	r := inference.LoadModel(*model)
	if !r.OK {
		fmt.Fprintln(os.Stderr, "load:", r.Value)
		os.Exit(1)
	}
	m := r.Value.(inference.TextModel)
	defer m.Close()

	// The optional capability interface: implemented (and true) only when the
	// checkpoint carries an audio head.
	if a, ok := m.(inference.AudioModel); !ok || !a.AcceptsAudio() {
		fmt.Fprintln(os.Stderr, "this checkpoint does not accept audio input")
		os.Exit(1)
	}

	// Audio rides the Message itself; placeholder blocks follow the turn text
	// (the gemma4 audio-after-text convention).
	off := false
	var reply strings.Builder
	for tok := range m.Chat(context.Background(),
		[]inference.Message{{Role: "user", Content: *prompt, Audios: [][]byte{clip}}},
		inference.WithMaxTokens(512),
		inference.WithEnableThinking(&off),
	) {
		reply.WriteString(tok.Text)
	}
	if er := m.Err(); !er.OK {
		fmt.Fprintln(os.Stderr, "generate:", er.Value)
		os.Exit(1)
	}
	fmt.Println(strings.TrimSpace(reply.String()))
}
