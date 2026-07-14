// SPDX-Licence-Identifier: EUPL-1.2

// Vision chat: attach an image (PNG/JPEG bytes) to a user turn. Whether the
// LOADED CHECKPOINT accepts images is a live capability probe — the family
// supporting vision does not mean this snapshot shipped the tower — so probe
// before sending, exactly as the serve layer does.
//
//	go run ./pkg/chat/vision -model ~/models/gemma-4-e2b-it-4bit -image cat.png
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
	image := flag.String("image", "", "PNG or JPEG file to describe")
	prompt := flag.String("prompt", "Describe this image in two sentences.", "user message")
	flag.Parse()
	if *model == "" || *image == "" {
		fmt.Fprintln(os.Stderr, "usage: -model <snapshot dir> -image <file.png>")
		os.Exit(2)
	}

	img, err := os.ReadFile(*image)
	if err != nil {
		fmt.Fprintln(os.Stderr, "image:", err)
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
	// checkpoint carries the vision tower.
	if v, ok := m.(inference.VisionModel); !ok || !v.AcceptsImages() {
		fmt.Fprintln(os.Stderr, "this checkpoint does not accept image input")
		os.Exit(1)
	}

	// Images ride the Message itself; the engine splices the vision blocks
	// ahead of the turn text in prompt order.
	off := false
	var reply strings.Builder
	for tok := range m.Chat(context.Background(),
		[]inference.Message{{Role: "user", Content: *prompt, Images: [][]byte{img}}},
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
