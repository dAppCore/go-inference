// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	core "dappco.re/go"
)

// ExampleSnapshot_MarshalBinary round-trips a snapshot through the
// encoding.BinaryMarshaler interface — the in-memory serialisation path State
// stores and session caches use. The decoded snapshot recovers the same
// architecture and token count as the source.
func ExampleSnapshot_MarshalBinary() {
	data, err := testSnapshot().MarshalBinary()
	if err != nil {
		core.Println("marshal error:", err)
		return
	}

	var loaded Snapshot
	if err := loaded.UnmarshalBinary(data); err != nil {
		core.Println("unmarshal error:", err)
		return
	}
	core.Println("architecture:", loaded.Architecture)
	core.Println("tokens:", len(loaded.Tokens))
	// Output:
	// architecture: gemma4_text
	// tokens: 2
}

// ExampleSnapshot_MarshalBinary_nativeDtypes round-trips a native-dtype
// snapshot in memory. The float16 key and bfloat16 value dtype tags survive the
// encode/decode (the decoder's dtype-string reader recognises the canonical
// vocabulary), and the raw byte payloads are preserved bit-exact.
func ExampleSnapshot_MarshalBinary_nativeDtypes() {
	source := exampleNativeSnapshot()
	data, err := source.bytesWithOptions(SaveOptions{KVEncoding: EncodingNative})
	if err != nil {
		core.Println("encode error:", err)
		return
	}

	var loaded Snapshot
	if err := loaded.UnmarshalBinary(data); err != nil {
		core.Println("decode error:", err)
		return
	}
	head := loaded.Layers[0].Heads[0]
	core.Println("key dtype:", head.KeyDType)
	core.Println("value dtype:", head.ValueDType)
	core.Println("key bytes preserved:", equalBytes(head.KeyBytes, source.Layers[0].Heads[0].KeyBytes))
	// Output:
	// key dtype: float16
	// value dtype: bfloat16
	// key bytes preserved: true
}

// ExampleSnapshot_Save writes a snapshot to a file path using the default
// (float32) KV encoding and loads it back to confirm the round-trip.
func ExampleSnapshot_Save() {
	dir := core.MkdirTemp("", "kv-save-example-*").Value.(string)
	path := core.PathJoin(dir, "snapshot.kvbin")

	if err := testSnapshot().Save(path); err != nil {
		core.Println("save error:", err)
		return
	}
	loaded, err := Load(path)
	if err != nil {
		core.Println("load error:", err)
		return
	}
	core.Println("architecture:", loaded.Architecture)
	// Output:
	// architecture: gemma4_text
}

// ExampleSnapshot_SaveWithOptions writes a snapshot under an explicit KV
// encoding (Q8) and loads it back to confirm the quantised round-trip recovers
// the architecture.
func ExampleSnapshot_SaveWithOptions() {
	dir := core.MkdirTemp("", "kv-save-opts-example-*").Value.(string)
	path := core.PathJoin(dir, "snapshot-q8.kvbin")

	if err := testSnapshot().SaveWithOptions(path, SaveOptions{KVEncoding: EncodingQ8}); err != nil {
		core.Println("save error:", err)
		return
	}
	loaded, err := Load(path)
	if err != nil {
		core.Println("load error:", err)
		return
	}
	core.Println("architecture:", loaded.Architecture)
	// Output:
	// architecture: gemma4_text
}
