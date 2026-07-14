module dappco.re/go/inference/cli

go 1.26.2

// The lem CLI (and its sibling commands) as their own module: go/ stays a
// pure library; cli/ consumes it. Inside the repository the root go.work
// overrides this require with the live ./go tree; standalone builds resolve
// the released module (tag go/v0.12.0).
require dappco.re/go/inference v0.12.0
