// SPDX-Licence-Identifier: EUPL-1.2

package safetensors

import (
	"context"
	"testing"

	core "dappco.re/go"
)

// TestWrite_WriteSubset_Good writes a chosen subset of an indexed source
// file to a fresh safetensors container and confirms only the selected
// tensors survive — payloads copied bit-exact, excluded tensors absent.
func TestWrite_WriteSubset_Good(t *testing.T) {
	dir := t.TempDir()
	source := core.PathJoin(dir, "source.safetensors")
	target := core.PathJoin(dir, "attention.safetensors")
	writeRawSafetensors(t, source, map[string][]byte{
		"model.embed_tokens.weight":                  {1, 2, 3, 4},
		"model.layers.0.self_attn.q_proj.weight":     {5, 6, 7, 8},
		"model.layers.0.mlp.down_proj.weight":        {9, 10, 11, 12},
		"model.layers.0.self_attn.q_proj.weight.idx": {13, 14, 15, 16},
	})
	index, err := ReadIndex(source)
	if err != nil {
		t.Fatalf("ReadIndex: %v", err)
	}

	err = WriteSubset(context.Background(), target, []TensorRef{
		index.Tensors["model.embed_tokens.weight"],
		index.Tensors["model.layers.0.self_attn.q_proj.weight"],
	})
	if err != nil {
		t.Fatalf("WriteSubset: %v", err)
	}

	got, err := ReadIndex(target)
	if err != nil {
		t.Fatalf("ReadIndex(target): %v", err)
	}
	if len(got.Names) != 2 {
		t.Fatalf("names = %v, want two tensors", got.Names)
	}
	if _, ok := got.Tensors["model.layers.0.mlp.down_proj.weight"]; ok {
		t.Fatalf("target contains excluded MLP tensor: %v", got.Names)
	}
	assertRawTensorEqual(t, index.Tensors["model.embed_tokens.weight"], got.Tensors["model.embed_tokens.weight"])
	assertRawTensorEqual(t, index.Tensors["model.layers.0.self_attn.q_proj.weight"], got.Tensors["model.layers.0.self_attn.q_proj.weight"])
}

// TestWrite_WriteSubset_Bad drives the up-front validation rejections:
// an empty tensor list, a blank destination path, and a ref carrying an
// empty tensor name. Each must surface its typed sentinel before any
// file is created.
func TestWrite_WriteSubset_Bad(t *testing.T) {
	t.Run("nil_refs", func(t *testing.T) {
		err := WriteSubset(context.Background(), core.PathJoin(t.TempDir(), "empty.safetensors"), nil)
		if err == nil {
			t.Fatal("WriteSubset(nil) error = nil")
		}
	})
	t.Run("empty_path", func(t *testing.T) {
		err := WriteSubset(context.Background(), "  ", []TensorRef{{Name: "x", DType: "F32", ByteLen: 4}})
		if err == nil {
			t.Fatal("WriteSubset(empty path) error = nil")
		}
	})
	t.Run("empty_tensor_name", func(t *testing.T) {
		err := WriteSubset(context.Background(), core.PathJoin(t.TempDir(), "out.safetensors"),
			[]TensorRef{{Name: "  ", DType: "F32", ByteLen: 4}})
		if err == nil {
			t.Fatal("WriteSubset(empty tensor name) error = nil")
		}
	})
}

// TestWrite_WriteSubset_Ugly exercises the awkward-but-legal edges that
// still must round-trip: a context cancelled before the chunk loop runs,
// and a tensor name carrying a double-quote, backslash, newline and a low
// control byte (driving appendJSONString's escape branch + hexNibble and
// the header parser's string-unescape path through one real round-trip).
func TestWrite_WriteSubset_Ugly(t *testing.T) {
	t.Run("context_cancelled", func(t *testing.T) {
		dir := t.TempDir()
		source := core.PathJoin(dir, "source.safetensors")
		target := core.PathJoin(dir, "cancelled.safetensors")
		writeRawSafetensors(t, source, map[string][]byte{"x": {1, 2, 3, 4}})
		index, err := ReadIndex(source)
		if err != nil {
			t.Fatalf("ReadIndex: %v", err)
		}
		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		err = WriteSubset(ctx, target, []TensorRef{index.Tensors["x"]})
		if err == nil {
			t.Fatal("WriteSubset(cancelled) error = nil")
		}
	})

	t.Run("escaped_name", func(t *testing.T) {
		dir := t.TempDir()
		source := core.PathJoin(dir, "source.safetensors")
		target := core.PathJoin(dir, "escaped.safetensors")
		// Name contains a double-quote, backslash, newline and a low
		// control byte (0x01), all of which must survive the
		// encode-then-parse round-trip.
		weird := "weird\"name\\with\nctrl\x01"
		writeRawSafetensors(t, source, map[string][]byte{weird: {9, 8, 7, 6}})
		index, err := ReadIndex(source)
		if err != nil {
			t.Fatalf("ReadIndex(source): %v", err)
		}

		if err := WriteSubset(context.Background(), target, []TensorRef{index.Tensors[weird]}); err != nil {
			t.Fatalf("WriteSubset: %v", err)
		}
		got, err := ReadIndex(target)
		if err != nil {
			t.Fatalf("ReadIndex(target): %v", err)
		}
		if _, ok := got.Tensors[weird]; !ok {
			t.Fatalf("escaped name not round-tripped; got names = %v", got.Names)
		}
		assertRawTensorEqual(t, index.Tensors[weird], got.Tensors[weird])
	})
}

// assertRawTensorEqual reads two refs' raw payloads and fails the test
// unless their bytes match exactly. Used by the WriteSubset round-trip
// assertions to confirm the copied payload is bit-identical to the source.
func assertRawTensorEqual(t *testing.T, want, got TensorRef) {
	t.Helper()
	wantRaw, err := ReadRefRaw(want)
	if err != nil {
		t.Fatalf("ReadRefRaw(want): %v", err)
	}
	gotRaw, err := ReadRefRaw(got)
	if err != nil {
		t.Fatalf("ReadRefRaw(got): %v", err)
	}
	if string(wantRaw) != string(gotRaw) {
		t.Fatalf("raw tensor mismatch: want %v got %v", wantRaw, gotRaw)
	}
}
