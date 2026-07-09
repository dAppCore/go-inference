// SPDX-Licence-Identifier: EUPL-1.2

package train_test

import (
	"context"
	"fmt"

	"dappco.re/go/inference/train"
	"dappco.re/go/inference/train/dataset"
)

// ExampleRunSSDModel shows the Model-bound SSD entry: RunSSDModel samples the
// frozen model over the dataset via model.Generate, capturing + (optionally)
// scoring each self-output at birth. echoModel implements no
// inference.PromptCacheWarmer, so this also proves the "no warm capability"
// fallback — the kernel lane is simply absent (never armed here).
func ExampleRunSSDModel() {
	model := &echoModel{}
	ds := dataset.NewSliceDataset([]dataset.Sample{
		{Prompt: "colour"},
		{Prompt: "fruit"},
	})
	cfg := train.SSDConfig{SampleTemperature: 0.7, DisableCapture: true}

	result, err := train.RunSSDModel(context.Background(), model, ds, cfg, nil)
	if err != nil {
		panic(err)
	}
	for _, sample := range result.Samples {
		fmt.Println(sample.Response)
	}
	// Output:
	// echo:colour
	// echo:fruit
}
