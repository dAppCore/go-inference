module dappco.re/go/inference/examples/sdk/go

go 1.23

require dappco.re/go/inference/lemsdk v0.0.0

// The client is GENERATED locally by `task sdk` (build/sdk is gitignored) —
// the replace points this demo at that output; a published SDK would be a
// normal require.
replace dappco.re/go/inference/lemsdk => ../../../build/sdk/go
