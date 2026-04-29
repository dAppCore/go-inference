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

## Working Locally

Use the same commands as the compliance brief before handing work back:

```sh
GOWORK=off go mod tidy
GOWORK=off go vet ./...
GOWORK=off go test -count=1 ./...
gofmt -l .
bash /Users/snider/Code/core/go/tests/cli/v090-upgrade/audit.sh .
```

The audit script is the work provider for compliance tasks. A change is not
complete until it reports `verdict: COMPLIANT` with every counter at zero.
