// SPDX-Licence-Identifier: EUPL-1.2

// mtp-probe measured the composed-loaded Qwen MTP head's speculative acceptance against its composed
// base: a per-token vs block-verify lane comparison, plus a -parity mode diffing composed's batched
// forward against sequential single-token stepping bit-for-bit. #50 retires model/composed, and this
// probe was one of its last two consumers in go-inference (the other: engine/hip/composed_runtime.go —
// see that file's doc comment for the parallel sever). model/composed's LoadSpeculativePairDirs and
// ComposedTokenModel are gone from this binary's dependency graph.
//
// It is NOT converted to the engine/metal factory route. That route (native.LoadSpeculativePair,
// engine/metal/speculative_model.go) hands back a shared inference.TextModel driven by string prompts
// through Generate/Chat streaming iterators — not this probe's raw token-ID
// pair.GenerateSpeculative(ids, maxNew, eosID, temp) call. Converting main()'s acceptance-rate
// comparison would mean rebuilding it around that different calling convention (context.Context,
// option functions, streamed tokens, metrics read back post-hoc via SpeculativeMetricsProvider) — a
// real rewrite, not a same-file sever. Its -parity mode has no public equivalent on the new engine at
// all: ArchSession's batched/sequential forward paths are private to engine/metal, and the exact
// question -parity asked (does a batched forward match a token-by-token reference, bit-for-bit) is now
// a standing, TESTED invariant of the MTP exact-verify lane rather than something a standalone probe
// needs to check by hand — see engine/metal's mtp_exact_lane_test.go (the #55 regression pins) and its
// mtp_session_test.go / mtp_reengage_test.go neighbours.
//
// Live replacements:
//   - measure a real base+draft pair (acceptance, tok/s, MTP metrics):
//     `lem generate -draft <assistant-path> -temp 0 -max-tokens <n> <base-model>` (cli/generate.go's
//     -draft flag — the shipped speculative serve lane; NOT "lem pair", which is stale wording left
//     over in model/composed/register.go's own retired refusal message).
//   - verify exact-lane correctness: `go test ./engine/metal/ -run MTP` (mtp_exact_lane_test.go,
//     mtp_session_test.go, mtp_reengage_test.go).
//
// This file is flagged for DELETION — kept as a pointer stub, not a working tool, so a caller who still
// reaches for `mtp-probe` lands on the real commands instead of a silently-stale binary.
package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Fprintln(os.Stderr, "mtp-probe is retired (#50: model/composed removal) — it no longer measures anything.")
	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintln(os.Stderr, "For real acceptance/tok/s/MTP metrics on a base+draft pair:")
	fmt.Fprintln(os.Stderr, "  lem generate -draft <assistant-path> -temp 0 -max-tokens <n> <base-model>")
	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintln(os.Stderr, "For exact-lane correctness (what -parity used to check by hand):")
	fmt.Fprintln(os.Stderr, "  go test ./engine/metal/ -run MTP   (mtp_exact_lane_test.go, mtp_session_test.go, mtp_reengage_test.go)")
	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintln(os.Stderr, "This binary (go/cmd/mtp-probe) is flagged for deletion.")
	os.Exit(1)
}
