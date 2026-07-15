// SPDX-Licence-Identifier: EUPL-1.2

package spine

import (
	"iter"

	core "dappco.re/go"
)

// PromptChunksToString concatenates a chunk sequence into one prompt
// string for callers that take iter.Seq[string] prompt surfaces.
//
//	prompt := spine.PromptChunksToString(chunks)
func PromptChunksToString(chunks iter.Seq[string]) string {
	if chunks == nil {
		return ""
	}
	builder := core.NewBuilder()
	for chunk := range chunks {
		builder.WriteString(chunk)
	}
	return builder.String()
}
