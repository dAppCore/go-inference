// SPDX-Licence-Identifier: EUPL-1.2

// Classify plus WithLogits: each prompt gets one forward pass (see
// pkg/eval for the plain top-token version); WithLogits also returns the raw
// vocab-sized logits, so a caller can score confidence instead of only
// reading the sampled token.
//
// TextModel exposes no tokenizer-encode call, so there is no public way to
// turn the strings "positive"/"negative" into vocab ids directly. Instead
// this example BOOTSTRAPS the two candidate ids from a calibration Classify
// call: two unambiguous reviews whose sampled Token.ID *is* the model's own
// id for that label in this exact prompt frame. Those ids then index into
// the logits of the real (ambiguous) reviews to compute a margin — how much
// the model preferred "positive" over "negative", even when it sampled a
// different token.
//
//	go run ./pkg/classify -model ~/models/gemma-4-e2b-it-4bit
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

func sentimentPrompt(review string) string {
	return "Review: " + review + "\nAnswer with exactly one word, positive or negative.\nSentiment:"
}

func main() {
	model := flag.String("model", os.Getenv("LEM_MODEL"), "model snapshot directory (config.json + *.safetensors)")
	flag.Parse()
	if *model == "" {
		fmt.Fprintln(os.Stderr, "set -model (or LEM_MODEL) to a model snapshot directory")
		os.Exit(2)
	}

	r := inference.LoadModel(*model)
	if !r.OK {
		fmt.Fprintln(os.Stderr, "load:", r.Value)
		os.Exit(1)
	}
	m := r.Value.(inference.TextModel)
	defer m.Close()

	// Calibration pass: unambiguous reviews pin down this prompt frame's ids
	// for the two label words.
	calib := m.Classify(context.Background(), []string{
		sentimentPrompt("Wonderful, faultless service from start to finish."),
		sentimentPrompt("Dreadful, rude staff and a filthy waiting room."),
	})
	if !calib.OK {
		fmt.Fprintln(os.Stderr, "calibrate:", calib.Value)
		os.Exit(1)
	}
	labels := calib.Value.([]inference.ClassifyResult)
	posID, negID := labels[0].Token.ID, labels[1].Token.ID
	fmt.Printf("calibrated ids: positive=%d (%q)  negative=%d (%q)\n",
		posID, labels[0].Token.Text, negID, labels[1].Token.Text)

	// The real, ambiguous-ish cases, this time with logits enabled.
	reviews := []string{
		"The lighthouse tour was magical — our guide knew every stone.",
		"Waited an hour in the rain and the lamp room was shut.",
	}
	prompts := make([]string, len(reviews))
	for i, review := range reviews {
		prompts[i] = sentimentPrompt(review)
	}

	cr := m.Classify(context.Background(), prompts, inference.WithLogits())
	if !cr.OK {
		fmt.Fprintln(os.Stderr, "classify:", cr.Value)
		os.Exit(1)
	}
	results := cr.Value.([]inference.ClassifyResult)

	for i, res := range results {
		top := strings.TrimSpace(res.Token.Text)
		margin := "unavailable"
		if int(posID) < len(res.Logits) && int(negID) < len(res.Logits) {
			margin = fmt.Sprintf("%.3f", res.Logits[posID]-res.Logits[negID])
		}
		fmt.Printf("%q -> top=%-8s  positive-negative margin=%s\n", reviews[i], top, margin)
	}
}
