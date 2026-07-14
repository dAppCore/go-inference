module dappco.re/go/inference/examples

go 1.26.2

// Standalone consumers resolve the released module (tag go/v0.12.0). Inside
// this repository the root go.work overrides this with the live ./go tree,
// so examples always build against the code you are reading.
require dappco.re/go/inference v0.12.0
