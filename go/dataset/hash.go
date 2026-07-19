// SPDX-Licence-Identifier: EUPL-1.2

package dataset

import (
	core "dappco.re/go"
)

// contentHashSeparator and contentHashTurnSeparator delimit the
// canonicalised fields ContentHash hashes over. Control bytes that never
// occur in legitimate chat text — a real Prompt/Response/turn containing
// a raw NUL or record-separator byte is not a threat model this
// dedupe-oriented hash defends against.
const (
	contentHashSeparator     = "\x00"
	contentHashTurnSeparator = "\x1e"
)

// ContentHash computes an Item's canonicalised content hash — hex sha256
// over the content's PARSED semantic fields, not its raw bytes. Hashing
// the parsed form (rather than the JSON bytes directly) makes the hash
// immune to incidental JSON differences (key order, whitespace) between
// two encodings of the same semantic content, which the design calls
// for explicitly ("canonicalised content"). KindTrace is the one
// exception: its content is opaque by design, so there is no semantic
// form to parse — its raw bytes are hashed directly.
//
// Returns core.Fail when content does not parse as kind's shape — call
// [ValidateItemContent] first if you need a distinguishable "invalid
// shape" error from a hashing call site.
//
//	r := dataset.ContentHash(dataset.KindPair, content)
//	hash := r.Value.(string)
func ContentHash(kind ItemKind, content []byte) core.Result {
	switch kind {
	case KindPair:
		var pc PairContent
		if r := core.JSONUnmarshal(content, &pc); !r.OK {
			return core.Fail(core.E("dataset.ContentHash", "pair content is not valid JSON", r.Err()))
		}
		return core.Ok(core.SHA256HexString(pc.Prompt + contentHashSeparator + pc.Response))
	case KindMessages:
		var mc MessagesContent
		if r := core.JSONUnmarshal(content, &mc); !r.OK {
			return core.Fail(core.E("dataset.ContentHash", "messages content is not valid JSON", r.Err()))
		}
		var sb core.Builder
		for _, turn := range mc.Messages {
			sb.WriteString(turn.Role)
			sb.WriteString(contentHashSeparator)
			sb.WriteString(turn.Content)
			sb.WriteString(contentHashTurnSeparator)
		}
		return core.Ok(core.SHA256HexString(sb.String()))
	case KindTrace:
		if len(content) == 0 {
			return core.Fail(core.NewError("dataset: trace content must be non-empty"))
		}
		return core.Ok(core.SHA256Hex(content))
	default:
		return core.Fail(core.NewError("dataset: unknown item kind for content hash"))
	}
}

// ManifestHash computes an export manifest's hash — hex sha256 over the
// ordered content hashes, newline-joined. Order matters: the manifest is
// the receipt that names exactly what a training run saw, in the order
// it saw it. An empty hashes slice is a defined edge case (the sha256 of
// the empty string), not an error.
//
//	hash := dataset.ManifestHash([]string{itemA.ContentHash, itemB.ContentHash})
func ManifestHash(orderedHashes []string) string {
	return core.SHA256HexString(core.Join("\n", orderedHashes...))
}
