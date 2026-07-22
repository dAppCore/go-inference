module dappco.re/go/inference/examples

go 1.26.2

// Standalone consumers resolve the released module (tag go/v0.12.0). Inside
// this repository the root go.work overrides this with the live ./go tree,
// so examples always build against the code you are reading.
require dappco.re/go/inference v0.12.0

require dappco.re/go v0.11.0

require (
	dappco.re/go/cgo v0.11.2 // indirect
	dappco.re/go/i18n v0.12.1 // indirect
	dappco.re/go/io v0.15.0 // indirect
	dappco.re/go/log v0.13.1 // indirect
	dappco.re/go/process v0.16.1 // indirect
	github.com/ebitengine/purego v0.10.1 // indirect
	github.com/tmc/apple v0.6.12 // indirect
	golang.org/x/sys v0.43.0 // indirect
	golang.org/x/text v0.37.0 // indirect
)
