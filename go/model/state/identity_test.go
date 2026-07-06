// SPDX-Licence-Identifier: EUPL-1.2

// Tests for the portable identity/metadata types in identity.go. These
// are plain JSON-tagged data with no methods of their own, so the
// meaningful contract to prove is the wire shape (omitempty collapsing
// zero values, deprecated fields still round-tripping independently of
// their replacement) and the StateBundle/Bundle alias identity. Every
// type here carries a Labels map field, which makes the whole struct
// non-comparable with == — assertions compare the scalar fields
// individually instead.

package state

import (
	"testing"

	core "dappco.re/go"
)

// TestIdentity_ModelIdentity_JSONRoundTrip proves a fully populated
// ModelIdentity survives a marshal/unmarshal round trip verbatim.
func TestIdentity_ModelIdentity_JSONRoundTrip(t *testing.T) {
	want := ModelIdentity{
		ID:            "gemma4",
		Architecture:  "gemma4_text",
		Hash:          "model-a",
		QuantBits:     4,
		ContextLength: 8192,
		NumLayers:     28,
		Labels:        map[string]string{"scope": "repo"},
	}

	data := core.JSONMarshal(want)
	if !data.OK {
		t.Fatalf("JSONMarshal() error = %v", data.Err())
	}

	var got ModelIdentity
	if r := core.JSONUnmarshal(data.Value.([]byte), &got); !r.OK {
		t.Fatalf("JSONUnmarshal() error = %v", r.Err())
	}
	if got.ID != want.ID || got.Architecture != want.Architecture || got.Hash != want.Hash ||
		got.QuantBits != want.QuantBits || got.ContextLength != want.ContextLength || got.NumLayers != want.NumLayers {
		t.Fatalf("round-tripped ModelIdentity = %+v, want %+v", got, want)
	}
	if got.Labels["scope"] != "repo" {
		t.Fatalf("round-tripped ModelIdentity.Labels = %+v, want scope=repo", got.Labels)
	}
}

// TestIdentity_ModelIdentity_ZeroValue proves every field carries
// `omitempty` — a zero-value ModelIdentity marshals to an empty object,
// not a document full of zeroed keys.
func TestIdentity_ModelIdentity_ZeroValue(t *testing.T) {
	if got := core.JSONMarshalString(ModelIdentity{}); got != "{}" {
		t.Fatalf("JSONMarshalString(zero ModelIdentity) = %s, want {}", got)
	}
}

// TestIdentity_TokenizerIdentity_JSONRoundTrip proves the tokenizer
// control-token IDs (which are legitimately zero for many tokenizers)
// round-trip distinctly from an entirely empty identity.
func TestIdentity_TokenizerIdentity_JSONRoundTrip(t *testing.T) {
	want := TokenizerIdentity{Kind: "bpe", Hash: "tok-a", BOSID: 1, EOSID: 2, PADID: 0}

	data := core.JSONMarshal(want)
	if !data.OK {
		t.Fatalf("JSONMarshal() error = %v", data.Err())
	}
	var got TokenizerIdentity
	if r := core.JSONUnmarshal(data.Value.([]byte), &got); !r.OK {
		t.Fatalf("JSONUnmarshal() error = %v", r.Err())
	}
	if got.Kind != want.Kind || got.Hash != want.Hash || got.BOSID != want.BOSID || got.EOSID != want.EOSID || got.PADID != want.PADID {
		t.Fatalf("round-tripped TokenizerIdentity = %+v, want %+v", got, want)
	}
}

// TestIdentity_AdapterIdentity_JSONRoundTrip proves TargetKeys (the only
// slice field) survives the round trip alongside the scalar fields.
func TestIdentity_AdapterIdentity_JSONRoundTrip(t *testing.T) {
	want := AdapterIdentity{Hash: "adapter-a", Rank: 8, Alpha: 16, TargetKeys: []string{"q_proj", "v_proj"}}

	data := core.JSONMarshal(want)
	if !data.OK {
		t.Fatalf("JSONMarshal() error = %v", data.Err())
	}
	var got AdapterIdentity
	if r := core.JSONUnmarshal(data.Value.([]byte), &got); !r.OK {
		t.Fatalf("JSONUnmarshal() error = %v", r.Err())
	}
	if got.Hash != want.Hash || got.Rank != want.Rank || len(got.TargetKeys) != 2 || got.TargetKeys[1] != "v_proj" {
		t.Fatalf("round-tripped AdapterIdentity = %+v, want %+v", got, want)
	}
}

// TestIdentity_RuntimeIdentity_JSONRoundTrip proves the NativeRuntime bool
// round-trips both true and its (omitted) false zero value.
func TestIdentity_RuntimeIdentity_JSONRoundTrip(t *testing.T) {
	native := RuntimeIdentity{Backend: "metal", NativeRuntime: true}
	if got := core.JSONMarshalString(native); !core.Contains(got, `"native_runtime":true`) {
		t.Fatalf("JSONMarshalString(native) = %s, want native_runtime:true present", got)
	}

	nonNative := RuntimeIdentity{Backend: "metal"}
	if got := core.JSONMarshalString(nonNative); core.Contains(got, "native_runtime") {
		t.Fatalf("JSONMarshalString(non-native) = %s, want native_runtime omitted", got)
	}
}

// TestIdentity_SamplerConfig_JSONRoundTrip proves the stop-token/sequence
// slices round-trip alongside the scalar sampler settings.
func TestIdentity_SamplerConfig_JSONRoundTrip(t *testing.T) {
	want := SamplerConfig{
		MaxTokens:     512,
		Temperature:   0.7,
		TopK:          40,
		TopP:          0.9,
		StopTokens:    []int32{1, 2},
		StopSequences: []string{"\n\n"},
	}

	data := core.JSONMarshal(want)
	if !data.OK {
		t.Fatalf("JSONMarshal() error = %v", data.Err())
	}
	var got SamplerConfig
	if r := core.JSONUnmarshal(data.Value.([]byte), &got); !r.OK {
		t.Fatalf("JSONUnmarshal() error = %v", r.Err())
	}
	if got.MaxTokens != want.MaxTokens || len(got.StopTokens) != 2 || len(got.StopSequences) != 1 || got.StopSequences[0] != "\n\n" {
		t.Fatalf("round-tripped SamplerConfig = %+v, want %+v", got, want)
	}
}

// TestIdentity_StateRef_JSONRoundTrip proves SizeBytes (a uint64) survives
// the round trip without precision loss.
func TestIdentity_StateRef_JSONRoundTrip(t *testing.T) {
	want := StateRef{Kind: "kv", URI: "state://kv/block", SizeBytes: 1 << 40}

	data := core.JSONMarshal(want)
	if !data.OK {
		t.Fatalf("JSONMarshal() error = %v", data.Err())
	}
	var got StateRef
	if r := core.JSONUnmarshal(data.Value.([]byte), &got); !r.OK {
		t.Fatalf("JSONUnmarshal() error = %v", r.Err())
	}
	if got.Kind != want.Kind || got.URI != want.URI || got.SizeBytes != want.SizeBytes {
		t.Fatalf("round-tripped StateRef = %+v, want %+v", got, want)
	}
}

// TestIdentity_Bundle_DeprecatedMemvidRefs proves the deprecated
// MemvidRefs field still round-trips independently of its StateRefs
// replacement — existing archived bundles that only populated MemvidRefs
// must not silently lose data when read back by newer code.
func TestIdentity_Bundle_DeprecatedMemvidRefs(t *testing.T) {
	want := Bundle{
		Version:    "1",
		StateRefs:  []StateRef{{Kind: "kv", URI: "state://kv/new"}},
		MemvidRefs: []StateRef{{Kind: "kv", URI: "state://kv/legacy"}},
	}

	data := core.JSONMarshal(want)
	if !data.OK {
		t.Fatalf("JSONMarshal() error = %v", data.Err())
	}
	var got Bundle
	if r := core.JSONUnmarshal(data.Value.([]byte), &got); !r.OK {
		t.Fatalf("JSONUnmarshal() error = %v", r.Err())
	}
	if len(got.StateRefs) != 1 || got.StateRefs[0].URI != "state://kv/new" {
		t.Fatalf("round-tripped Bundle.StateRefs = %+v, want the new-style ref preserved", got.StateRefs)
	}
	if len(got.MemvidRefs) != 1 || got.MemvidRefs[0].URI != "state://kv/legacy" {
		t.Fatalf("round-tripped Bundle.MemvidRefs = %+v, want the deprecated ref still preserved", got.MemvidRefs)
	}
}

// TestIdentity_StateBundle_Alias proves StateBundle is the same underlying
// type as Bundle — a value built under either name is assignable to and
// from the other with no conversion.
func TestIdentity_StateBundle_Alias(t *testing.T) {
	bundle := Bundle{Version: "1", Model: ModelIdentity{ID: "gemma4"}}

	var aliased StateBundle = bundle
	if aliased.Model.ID != "gemma4" {
		t.Fatalf("StateBundle(Bundle) = %+v, want the canonical value verbatim", aliased)
	}

	var backToCanonical Bundle = aliased
	if backToCanonical.Version != "1" {
		t.Fatalf("Bundle(StateBundle) = %+v, want the alias's value verbatim", backToCanonical)
	}
}
