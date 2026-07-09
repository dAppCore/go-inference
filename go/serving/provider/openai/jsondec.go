// SPDX-Licence-Identifier: EUPL-1.2

// Hand-rolled JSON-decoding adapters for the openai variant-shape
// unmarshallers. The walker primitives now live in jsonenc/ so that
// anthropic + ollama field-dispatch UnmarshalJSON paths can share
// the same byte-pump (lifted from this file in W11-B). The shapes
// this file owns — StopList / EmbeddingInput — both reduce to
// `ParseJSONStringList`, so the helpers here are thin variant-shape
// dispatchers.
//
// Per-call performance unchanged from the W10-M baseline — the
// underlying byte walker is identical.

package openai

import "dappco.re/go/inference/jsonenc"

// parseJSONStringList walks data as either a JSON string (e.g.
// `"END"`) or an array of JSON strings (e.g. `["END","</s>"]`) and
// returns a []string with the inner values unescaped.
//
// Forwards to jsonenc.ParseJSONStringList — kept under the package-
// local name so existing call sites (StopList / EmbeddingInput) need
// no churn.
func parseJSONStringList(data []byte) ([]string, error) {
	return jsonenc.ParseJSONStringList(data)
}
