// SPDX-Licence-Identifier: EUPL-1.2

//go:build !metal_runtime

// Package metaltest holds the compile-time gates for hardware- and
// model-dependent tests. They replace the GO_MLX_RUN_METAL_TESTS /
// GO_MLX_RUN_MODEL_EVAL_TESTS env vars — settings selected by build tags, not a
// process-env control surface. Test files stay un-tagged so they always
// compile (catching compile regressions); only these consts flip, and the test
// helpers skip the hardware body unless the tag is set:
//
//	go test -tags metal_runtime ./...              # hardware kernel tests
//	go test -tags 'metal_runtime model_eval' ./... # + full model-eval runs
package enginegate

// RunMetalTests is false by default — hardware-dependent tests skip. Build with
// -tags metal_runtime to run them.
const RunMetalTests = false
