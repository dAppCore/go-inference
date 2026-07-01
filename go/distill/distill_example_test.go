// SPDX-Licence-Identifier: EUPL-1.2

package distill_test

import (
	"context"
	"fmt"

	"dappco.re/go/inference/dataset"
	"dappco.re/go/inference/distill"
)

// ExampleNewMemoryLogitCache stores and retrieves teacher logits for a
// cache key, and shows that a miss on an unknown key returns ok=false
// rather than an error.
func ExampleNewMemoryLogitCache() {
	cache := distill.NewMemoryLogitCache()
	ctx := context.Background()

	logits := distill.Logits{{{1, 2, 3}}}
	if err := cache.PutTeacherLogits(ctx, "batch-0", logits); err != nil {
		panic(err)
	}

	got, ok, err := cache.GetTeacherLogits(ctx, "batch-0")
	if err != nil {
		panic(err)
	}
	fmt.Println("hit:", ok, got[0][0])

	_, ok, _ = cache.GetTeacherLogits(ctx, "unknown")
	fmt.Println("miss:", ok)
	// Output:
	// hit: true [1 2 3]
	// miss: false
}

// ExampleCollectSamples pulls a capped number of samples from a dataset
// stream into a plain slice, ready to replay via dataset.NewSliceDataset.
func ExampleCollectSamples() {
	ds := dataset.NewSliceDataset([]dataset.Sample{
		{Text: "row0"},
		{Text: "row1"},
		{Text: "row2"},
	})

	samples, err := distill.CollectSamples(context.Background(), ds, 2)
	if err != nil {
		panic(err)
	}
	for _, s := range samples {
		fmt.Println(s.Text)
	}
	// Output:
	// row0
	// row1
}
