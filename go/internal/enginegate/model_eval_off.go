// SPDX-Licence-Identifier: EUPL-1.2

//go:build !model_eval

package enginegate

// RunModelEvalTests is false by default — full model-eval tests skip. Build with
// -tags model_eval to run them (they additionally need a model on disk).
const RunModelEvalTests = false
