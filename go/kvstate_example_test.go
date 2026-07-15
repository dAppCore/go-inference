// SPDX-Licence-Identifier: EUPL-1.2

package inference_test

import (
	"context"
	"fmt"

	"dappco.re/go/inference"
	"dappco.re/go/inference/kv"
)

type exampleKVModel struct{}

func (exampleKVModel) CaptureKV(_ context.Context, prompt string, _ inference.KVSnapshotCaptureOptions) (*kv.Snapshot, error) {
	return &kv.Snapshot{Architecture: "gemma4", SeqLen: len(prompt)}, nil
}

// KVSnapshotter is the portable conversation-state capture: engines emit
// kv.Snapshot directly, so state moves between engines without a converter.
func ExampleKVSnapshotter() {
	var model any = exampleKVModel{}
	if s, ok := model.(inference.KVSnapshotter); ok {
		snap, _ := s.CaptureKV(context.Background(), "hi", inference.KVSnapshotCaptureOptions{})
		fmt.Printf("%s seq=%d\n", snap.Architecture, snap.SeqLen)
	}
	// Output: gemma4 seq=2
}
