// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	core "dappco.re/go"
)

// ExampleLoad shows the file round-trip: Save writes a snapshot to a path and
// Load reads it back, recovering the architecture.
func ExampleLoad() {
	dir := core.MkdirTemp("", "kv-load-example-*").Value.(string)
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

// ExampleSnapshot_UnmarshalBinary decodes an in-memory binary buffer (produced
// by MarshalBinary) back into a Snapshot, the symmetric read side of the
// encoding.BinaryMarshaler round-trip.
func ExampleSnapshot_UnmarshalBinary() {
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
	core.Println("tokens:", len(loaded.Tokens))
	// Output:
	// tokens: 2
}

// ExampleLoadWithOptions reads a snapshot from a path with explicit decode
// options; the default options decode float32 side slices so the head exposes
// usable key values.
func ExampleLoadWithOptions() {
	dir := core.MkdirTemp("", "kv-lwo-example-*").Value.(string)
	path := core.PathJoin(dir, "snapshot.kvbin")
	if err := testSnapshot().Save(path); err != nil {
		core.Println("save error:", err)
		return
	}

	loaded, err := LoadWithOptions(path, LoadOptions{})
	if err != nil {
		core.Println("load error:", err)
		return
	}
	head, _ := loaded.Head(0, 0)
	core.Println("key values:", len(head.Key))
	// Output:
	// key values: 4
}
