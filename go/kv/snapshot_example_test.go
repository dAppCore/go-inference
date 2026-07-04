// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"math"

	core "dappco.re/go"
)

// exampleNativeSnapshot builds a single-layer native-dtype snapshot whose key
// is float16 and value is bfloat16 — the raw-byte capture shape produced by an
// MLX cache export, used by the round-trip examples below.
func exampleNativeSnapshot() *Snapshot {
	keyBytes := appendUint16LE(nil, float32ToFloat16(1.5))
	keyBytes = appendUint16LE(keyBytes, float32ToFloat16(-2))
	valueBytes := appendUint16LE(nil, uint16(math.Float32bits(0.25)>>16))
	valueBytes = appendUint16LE(valueBytes, uint16(math.Float32bits(-0.75)>>16))
	return &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1},
		TokenOffset:   1,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        1,
		HeadDim:       2,
		NumQueryHeads: 1,
		Layers: []LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []HeadSnapshot{{
				Key:        []float32{1.5, -2},
				KeyDType:   "float16",
				KeyBytes:   keyBytes,
				Value:      []float32{0.25, -0.75},
				ValueDType: "bfloat16",
				ValueBytes: valueBytes,
			}},
		}},
	}
}

// ExampleSnapshot_Head reads a single head out of a snapshot by (layer, head)
// index, returning a clone. An out-of-range head index reports ok=false.
func ExampleSnapshot_Head() {
	snapshot := testSnapshot()

	head, ok := snapshot.Head(0, 0)
	core.Println("ok:", ok, "key len:", len(head.Key))

	_, missing := snapshot.Head(0, 99)
	core.Println("missing head ok:", missing)
	// Output:
	// ok: true key len: 4
	// missing head ok: false
}

// ExampleSnapshot_Clone produces a deep copy: mutating the clone's head data
// leaves the original untouched.
func ExampleSnapshot_Clone() {
	original := testSnapshot()
	clone := original.Clone()
	clone.Layers[0].Heads[0].Key[0] = -999

	core.Println("original intact:", original.Layers[0].Heads[0].Key[0] == 1)
	core.Println("clone mutated:", clone.Layers[0].Heads[0].Key[0] == -999)
	// Output:
	// original intact: true
	// clone mutated: true
}

// ExampleDropFloat32 drops the float32 side slices on a head that also carries
// raw native bytes, freeing the redundant decoded copy while keeping the raw
// payload for serialisation.
func ExampleDropFloat32() {
	snapshot := &Snapshot{Layers: []LayerSnapshot{{
		Heads: []HeadSnapshot{{
			Key:        []float32{1, 2},
			KeyBytes:   []byte{1, 2, 3, 4},
			Value:      []float32{3, 4},
			ValueBytes: []byte{5, 6, 7, 8},
		}},
	}}}

	DropFloat32(snapshot)

	head := snapshot.Layers[0].Heads[0]
	core.Println("float32 dropped:", len(head.Key) == 0 && len(head.Value) == 0)
	core.Println("raw bytes kept:", len(head.KeyBytes) == 4)
	// Output:
	// float32 dropped: true
	// raw bytes kept: true
}

// ExampleResultError converts a failed core.Result into a Go error, the bridge
// between the core.Result IO surface and the error-returning snapshot APIs.
func ExampleResultError() {
	err := ResultError(core.Result{Value: "disk full"})
	core.Println("error:", err)
	// Output:
	// error: disk full
}

// ExampleHashSnapshot computes a stable content-addressed identifier for a
// snapshot; the same snapshot always hashes to the same length-64 hex digest.
func ExampleHashSnapshot() {
	hash, err := HashSnapshot(testSnapshot())
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("hash length:", len(hash))
	// Output: hash length: 64
}
