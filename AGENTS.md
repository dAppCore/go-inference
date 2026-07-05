# Agent Guide

This repository defines `dappco.re/go/inference`, the small contract package
shared by model backends and applications that consume them. Keep it portable:
the root package must compile without native GPU libraries, cgo flags, model
weights, or backend-specific dependencies.

## Repository Shape

The public surface is split across four source files:

- `inference.go` declares the backend registry, model interfaces, generation
  result types, metrics, and the `Default` / `LoadModel` entry points.
- `options.go` declares generation and loading configuration plus option
  helpers used by both consumers and backends.
- `discover.go` scans model directories using the `core/go` filesystem,
  path, string, and JSON wrappers.
- `training.go` declares the optional LoRA training interfaces and the
  `LoadTrainable` adapter around `LoadModel`.

Tests and examples are deliberately file-aware. Public symbols in
`<file>.go` have triplet tests in `<file>_test.go` and usage examples in
`<file>_example_test.go`. Do not create monolithic compliance files, versioned
test files, or `ax7*` files; extend the sibling test or example file instead.

## Core/Go Compliance

Consumer code in this repository uses `dappco.re/go` wrappers instead of the
banned direct stdlib imports for formatting, errors, filesystem access, paths,
JSON, strings, logging, bytes, and process execution. Public production
functions that can fail return `core.Result`; callers branch on `r.OK` and use
`r.Value` only after success. Backend interface methods may still expose Go
`error` values where backend implementations need to satisfy their own local
contracts, but root package loader functions adapt those pairs into
`core.Result`.

Examples use `Println` from `dappco.re/go`, not `fmt.Println`. Tests should
assert behavior directly against the symbol named by the test. A triplet named
`TestOptions_WithMaxTokens_Bad` must invoke `WithMaxTokens` in its own body,
not route through a dispatcher helper.

## Writing Tests, Examples & Benchmarks

Every source file ships three siblings — extend them, never create monolithic
compliance files, versioned test files (`_v2`), or `ax7*` files:

| Sibling of `foo.go` | Holds | Verified by |
|---------------------|-------|-------------|
| `foo_test.go` | one `Test<Symbol>_<Case>` per exported symbol per variant | `task test` |
| `foo_example_test.go` | one `Example<Symbol>` per symbol, with an `// Output:` block | `task test` (runs + diffs the output) |
| `foo_bench_test.go` | one `Benchmark<Symbol>` per hot symbol | `task bench` |

**Tests — name the symbol, exercise it directly.** A test asserts against the
symbol its name claims: `TestOptions_WithMaxTokens_Bad` must call
`WithMaxTokens` in its own body, not route through a dispatcher/table helper.
A test that never names its symbol is fake coverage the audit flags. Write the
AX-7 triplet for each symbol — `_Good` (valid input, happy path), `_Bad`
(invalid input is rejected), `_Ugly` (malformed / boundary / empty). Production
functions that can fail return `core.Result`: the `_Good` test asserts `r.OK`
then reads `r.Value`; the `_Bad`/`_Ugly` tests assert `!r.OK`.

**Examples are compiled documentation.** `func ExampleWithMaxTokens()` ends with
a `// Output:` block so `go test` runs and diffs it — a stale example fails the
build. Print with `Println` from `dappco.re/go`, never `fmt.Println`.

**Benchmarks measure the load path.** Shape:

```go
var sinkResult core.Result // package sink — stops the compiler eliding the call

func BenchmarkDiscover(b *testing.B) {
    dir := writeFixtureModel(b)     // setup OUTSIDE the timed loop
    b.ReportAllocs()
    b.ResetTimer()                  // discount the setup
    for i := 0; i < b.N; i++ {
        sinkResult = Discover(dir)  // assign to the sink so it can't be optimised away
    }
}
```

Read **B/op as hard as allocs/op** — the biggest wins (whole-slice clones,
full-file reads) leave allocs/op flat while B/op screams. allocs/op is only
trustworthy at steady state, so `task bench` runs `-benchtime=20x`; a cold
3-iteration number is inflated by setup.

## Working Locally

Run the Taskfile gates before handing work back (portable lanes need no GPU;
`*:metal` lanes need `task metallib` first):

```sh
task qa            # gofmt check + go vet + portable tests — the pre-handback gate
task test          # portable suite (default tags, runs anywhere)
task test:metal    # engine/metal suite (-tags metal_runtime; needs task metallib)
task cover         # coverage.out + total — must clear the 95% codecov target
task bench         # every benchmark with -benchmem (allocation regressions)
```

`codecov.yml` enforces **95%** on both the project and each patch, measured on
the portable `task cover` profile (the surface a Linux CI compiles; engine/metal
is Darwin-only and covered by `task test:metal`).

For core/go idiom compliance specifically, the audit script is the work
provider — a change is not complete until it reports `verdict: COMPLIANT` with
every counter at zero:

```sh
GOWORK=off go mod tidy
gofmt -l .
bash /Users/snider/Code/core/go/tests/cli/v090-upgrade/audit.sh .
```
