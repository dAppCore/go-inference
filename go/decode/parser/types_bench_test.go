// SPDX-Licence-Identifier: EUPL-1.2

// No CPU-only public surface; skipped.
// types.go declares Hint, Config, Mode, Chunk, Result and the internal
// reasoningMarker / thinkingMarker / toolBlockMarker structs — pure
// type definitions with no runtime functions to benchmark. Benches for
// the consumers of these types live in the per-file benches that
// drive them (builtin_bench_test.go, thinking_bench_test.go,
// registry_bench_test.go, reasoning_bench_test.go, tools_bench_test.go).

package parser
