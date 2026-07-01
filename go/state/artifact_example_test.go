// SPDX-Licence-Identifier: EUPL-1.2

package state

import (
	"context"
	"fmt"
)

// ExampleExportArtifact packages an engine-supplied analysis payload into a
// versioned Artifact and archives it via an in-memory Writer.
func ExampleExportArtifact() {
	store := NewInMemoryStore(nil)

	record, err := ExportArtifact(context.Background(), map[string]any{"mean_coherence": 0.91}, ArtifactOptions{
		Model: "gemma3-1b",
		Kind:  "go-mlx/session-state",
		Store: store,
		Put:   PutOptions{URI: "mlx://session/trace-1"},
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(record.Kind, record.ChunkRef.ChunkID > 0)
	// Output:
	// go-mlx/session-state true
}
