// SPDX-Licence-Identifier: EUPL-1.2

// What loaded: architecture metadata (Info, ModelType) plus the live
// capability probes — VisionModel/AudioModel are per-CHECKPOINT type
// assertions (a vision-capable family can still ship a text-only snapshot),
// and CapabilitiesOf reports the model's full feature surface when it
// implements CapabilityReporter, as the engine's TextModel does.
//
//	go run ./pkg/info -model ~/models/gemma-4-e2b-it-4bit
package main

import (
	"flag"
	"fmt"
	"os"

	"dappco.re/go/inference"
	_ "dappco.re/go/inference/examples/internal/engine" // registers the platform engine
)

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

	info := m.Info()
	fmt.Printf("architecture  %s\n", m.ModelType())
	fmt.Printf("layers        %d\n", info.NumLayers)
	fmt.Printf("hidden size   %d\n", info.HiddenSize)
	fmt.Printf("vocab         %d\n", info.VocabSize)
	fmt.Printf("quant         %d-bit (group %d)\n", info.QuantBits, info.QuantGroup)

	if v, ok := m.(inference.VisionModel); ok {
		fmt.Printf("vision        accepts images: %v\n", v.AcceptsImages())
	} else {
		fmt.Println("vision        not implemented by this model")
	}
	if a, ok := m.(inference.AudioModel); ok {
		fmt.Printf("audio         accepts audio: %v\n", a.AcceptsAudio())
	} else {
		fmt.Println("audio         not implemented by this model")
	}

	if report, ok := inference.CapabilitiesOf(m); ok {
		fmt.Printf("capabilities  (%d reported)\n", len(report.Capabilities))
		for _, capability := range report.Capabilities {
			if capability.Usable() {
				fmt.Printf("  %-24s %s\n", capability.ID, capability.Status)
			}
		}
	}
}
