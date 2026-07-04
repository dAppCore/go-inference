// SPDX-Licence-Identifier: EUPL-1.2

package session

import (
	"fmt"

	"dappco.re/go/inference/artifact"
	"dappco.re/go/inference/spine"
	"dappco.re/go/inference/state/session/internal/sessionfake"
)

// ExampleSession_ExportArtifacts captures the session's retained KV state
// and exports it as a local artifact record via dappco.re/go/inference/artifact.
// With no Options.KVPath the export stays in memory.
func ExampleSession_ExportArtifacts() {
	sess := New(&sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, spine.ModelInfo{Architecture: "gemma4_text"}, nil)

	record, err := sess.ExportArtifacts(artifact.Options{Model: "gemma4-1b", Prompt: "hello"})
	if err != nil {
		fmt.Println("export error:", err)
		return
	}

	fmt.Println("model:", record.Model)
	fmt.Println("architecture:", record.Snapshot.Architecture)
	fmt.Println("tokens:", record.Snapshot.TokenCount)
	// Output:
	// model: gemma4-1b
	// architecture: gemma4_text
	// tokens: 2
}
