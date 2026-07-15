// SPDX-Licence-Identifier: EUPL-1.2

// Video chat: Message.Videos carries the sampled FRAMES of one video —
// encoded PNG/JPEG images, in time order — rather than a video container
// itself; the engine treats each frame as a timestamped vision block. This
// example takes pre-extracted frame files (ffmpeg or similar does the
// sampling) and attaches them in the order given.
//
//	go run ./pkg/chat/video -model ~/models/gemma-4-e2b-it-4bit -frames f1.jpg,f2.jpg,f3.jpg
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
	frames := flag.String("frames", "", "comma-separated PNG/JPEG frame files, in time order")
	prompt := flag.String("prompt", "Describe what happens across these frames.", "user message")
	flag.Parse()
	if *model == "" || *frames == "" {
		fmt.Fprintln(os.Stderr, "usage: -model <snapshot dir> -frames <f1.png,f2.png,...>")
		os.Exit(2)
	}

	var video [][]byte
	for _, path := range strings.Split(*frames, ",") {
		b, err := os.ReadFile(path)
		if err != nil {
			fmt.Fprintln(os.Stderr, "frame:", err)
			os.Exit(1)
		}
		video = append(video, b)
	}

	r := inference.LoadModel(*model)
	if !r.OK {
		fmt.Fprintln(os.Stderr, "load:", r.Value)
		os.Exit(1)
	}
	m := r.Value.(inference.TextModel)
	defer m.Close()

	// Video is a vision capability — same probe as still images.
	if v, ok := m.(inference.VisionModel); !ok || !v.AcceptsImages() {
		fmt.Fprintln(os.Stderr, "this checkpoint does not accept vision input")
		os.Exit(1)
	}

	off := false
	var reply strings.Builder
	for tok := range m.Chat(context.Background(),
		[]inference.Message{{Role: "user", Content: *prompt, Videos: video}},
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
