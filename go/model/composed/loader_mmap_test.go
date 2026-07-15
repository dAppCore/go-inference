// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"bytes"
	"testing"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/model/quant/mlxaffine"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

// loader_mmap_test.go gates the composed-quant ZERO-COPY load: a quant checkpoint's packed weights must
// VIEW the mmap'd file (no heap copy) on the LoadComposedDir path, cutting resident memory by the packed
// weight (the dominant term of a quant checkpoint), while the copying LoadComposed contract the legacy
// metal loader relies on is preserved, and the model owns the mapping's lifetime (unmapped on Close).

// writeQuantCheckpoint writes a synthetic 4-layer hybrid with EVERY projection quantised (8-bit, gs 8, so
// no b1→b2 repack — every packed weight is eligible to alias) to a temp dir as config.json + a single
// model.safetensors, and returns the dir, the config bytes, and the dequantised reference for the embed
// table (a known tensor value to verify the aliased bytes decode correctly).
func writeQuantCheckpoint(t *testing.T) (dir string, cfg []byte, wantEmbed []float32) {
	t.Helper()
	ts, _ := mkHybridCheckpoint()
	for _, name := range allProjNames() {
		want := quantiseInPlace(t, ts, name, 8, 8)
		if name == "model.embed_tokens.weight" {
			wantEmbed = want
		}
	}
	// model_type routes LoadComposedDir to the composed hook; the quantization block marks it packed.
	cfg = []byte(`{"model_type":"qwen3_5","hidden_size":8,"num_hidden_layers":4,"intermediate_size":16,` +
		`"num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,"vocab_size":32,"rms_norm_eps":1e-5,` +
		`"rope_theta":1000000,"partial_rotary_factor":0.5,"full_attention_interval":2,` +
		`"quantization":{"group_size":8,"bits":8}}`)

	dir = t.TempDir()
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), string(cfg)); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	blob, err := safetensors.Encode(ts)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if err := coreio.Local.Write(core.PathJoin(dir, "model.safetensors"), string(blob)); err != nil {
		t.Fatalf("write model.safetensors: %v", err)
	}
	return dir, cfg, wantEmbed
}

// aliasesAnyShard reports whether b's backing array lies inside any of the mapping's shard regions — i.e.
// b is a VIEW into the mmap, not a heap copy. The house zero-copy check (see safetensors_mmap_test.go).
func aliasesAnyShard(dm *safetensors.DirMapping, b []byte) bool {
	if len(b) == 0 {
		return false
	}
	p := uintptr(unsafe.Pointer(&b[0]))
	for _, sh := range dm.Shards {
		if len(sh.Data) == 0 {
			continue
		}
		base := uintptr(unsafe.Pointer(&sh.Data[0]))
		if p >= base && p < base+uintptr(len(sh.Data)) {
			return true
		}
	}
	return false
}

// TestLoadComposed_ZeroCopyAliasesMmap is the RSS win, proven: on the zero-copy build every packed weight's
// bytes are a VIEW into the mmap (no heap copy), and those aliased bytes still dequantise to the reference —
// so the model reads the real weights straight out of the mapped file.
func TestLoadComposed_ZeroCopyAliasesMmap(t *testing.T) {
	dir, cfg, wantEmbed := writeQuantCheckpoint(t)

	dm, err := safetensors.LoadDirMmap(dir)
	if err != nil {
		t.Fatalf("LoadDirMmap: %v", err)
	}
	defer dm.Close() // this test never hands the mapping to the model, so it owns the unmap

	m, err := loadComposed(dm.Tensors, cfg, nil, true) // zero-copy build (the LoadComposedDir path)
	if err != nil {
		t.Fatalf("loadComposed(zeroCopy): %v", err)
	}
	if !m.mmapAliased {
		t.Fatal("mmapAliased not set — an all-8-bit quant checkpoint must alias its packed weights")
	}
	if m.EmbedQ == nil {
		t.Fatal("embed must be kept packed")
	}
	// The whole point: the packed weight is a view into the mmap, not a heap copy.
	if !aliasesAnyShard(dm, m.EmbedQ.Packed) {
		t.Fatal("embed packed bytes are a heap copy, not a view into the mmap — zero-copy broken")
	}
	// Spot-check every quantised projection aliases too, not just the embed.
	for li := range m.Layers {
		if mlp, ok := m.Layers[li].MLP.(*MLP); ok && mlp.GateQ != nil {
			if !aliasesAnyShard(dm, mlp.GateQ.Packed) {
				t.Fatalf("layer %d gate packed bytes are a heap copy, not an mmap view", li)
			}
		}
	}
	// The aliased bytes are the real weights: dequantising the embed reproduces the reference exactly.
	gotEmbed, err := mlxaffine.DequantizeTensor(m.EmbedQ.Packed, m.EmbedQ.Scales, m.EmbedQ.Biases, m.EmbedQ.OutDim, m.EmbedQ.InDim, m.EmbedQ.Bits, m.EmbedQ.GroupSize)
	if err != nil {
		t.Fatalf("dequantise aliased embed: %v", err)
	}
	assertF32Identical(t, "aliased-embed", gotEmbed, wantEmbed)
	t.Logf("zero-copy: %d-layer quant checkpoint's packed weights view the mmap (no heap copy), decode to the reference", len(m.Layers))
}

// TestLoadComposed_CopyPathOwnsPackedBytes pins the preserved legacy contract: the public LoadComposed
// COPIES the packed weights to owned heap buffers (mmapAliased stays false, the bytes are NOT an mmap view),
// so a caller that unmaps right after the build — the metal loadComposedTokenModel path — stays safe. The
// copied bytes are byte-identical to the checkpoint's, and the model declines the RetainMmap handshake.
func TestLoadComposed_CopyPathOwnsPackedBytes(t *testing.T) {
	dir, cfg, _ := writeQuantCheckpoint(t)

	dm, err := safetensors.LoadDirMmap(dir)
	if err != nil {
		t.Fatalf("LoadDirMmap: %v", err)
	}
	defer dm.Close()

	m, err := LoadComposed(dm.Tensors, cfg) // the copying contract
	if err != nil {
		t.Fatalf("LoadComposed: %v", err)
	}
	if m.mmapAliased {
		t.Fatal("copy path must not set mmapAliased")
	}
	if m.EmbedQ == nil {
		t.Fatal("embed must be kept packed")
	}
	if aliasesAnyShard(dm, m.EmbedQ.Packed) {
		t.Fatal("copy path packed bytes alias the mmap — the owned-copy contract is broken (metal unmaps after build)")
	}
	if !bytes.Equal(m.EmbedQ.Packed, dm.Tensors["model.embed_tokens.weight"].Data) {
		t.Fatal("copy path packed content diverged from the checkpoint bytes")
	}
	if NewTokenModel(m).RetainMmap(dm) {
		t.Fatal("a copied model must DECLINE RetainMmap so the loader unmaps now")
	}
}

// TestComposedTokenModel_CloseUnmapsMmap pins the lifetime guarantee: a zero-copy model takes ownership of
// the mapping via RetainMmap and unmaps it on Close — and Close is idempotent, so a double close (explicit
// then finalizer, or two callers) never double-unmaps.
func TestComposedTokenModel_CloseUnmapsMmap(t *testing.T) {
	dir, cfg, _ := writeQuantCheckpoint(t)

	dm, err := safetensors.LoadDirMmap(dir)
	if err != nil {
		t.Fatalf("LoadDirMmap: %v", err)
	}
	m, err := loadComposed(dm.Tensors, cfg, nil, true)
	if err != nil {
		t.Fatalf("loadComposed(zeroCopy): %v", err)
	}
	tm := NewTokenModel(m)
	if !tm.RetainMmap(dm) {
		t.Fatal("RetainMmap must take ownership of an aliasing model")
	}
	if err := tm.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if dm.Shards != nil {
		t.Fatal("Close did not unmap the shards")
	}
	if err := tm.Close(); err != nil {
		t.Fatalf("second Close (idempotent) errored: %v", err)
	}
}
