// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"bytes"
	"testing"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
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

// moeProjNames lists the dense-projection weights of the synthetic MoE checkpoint — embed, lm_head, and each
// layer's attention q/k/v/o. These are the weights buildFFN/buildAttn route through tensorAsQuant, so they
// are the ones the MoE zero-copy build aliases (the experts stay on buildMoE's dequant-to-f32 path).
func moeProjNames(nLayers int) []string {
	names := []string{"model.embed_tokens.weight", "lm_head.weight"}
	for i := range nLayers {
		ap := "model.layers." + itoa(i) + ".self_attn."
		names = append(names, ap+"q_proj.weight", ap+"k_proj.weight", ap+"v_proj.weight", ap+"o_proj.weight")
	}
	return names
}

// writeMoEQuantCheckpoint writes a small qwen2_moe-shaped checkpoint — every layer is full attention + a
// 4-expert top-2 MoE FFN (no shared expert, no gated-delta) — with its attention/embed/lm_head projections
// quantised (8-bit, gs 8, so no b1→b2 repack: every packed projection is eligible to alias) and its MoE
// experts left bf16 (buildMoE dequantises them; packed-expert MoE is a later slice). It returns the dir, the
// config bytes, the MoE Arch policy the registry hook would pass, and the dequantised embed reference.
func writeMoEQuantCheckpoint(t *testing.T) (dir string, cfg []byte, arch model.Arch, wantEmbed []float32) {
	t.Helper()
	const D, vocab, nLayers = 8, 32, 2
	const AH, AKVH, AHD = 4, 2, 8 // attention: q_proj rows = AH·AHD = 32, k/v = AKVH·AHD = 16
	const nE, moeFF = 4, 16
	ts := map[string]safetensors.Tensor{
		"model.embed_tokens.weight": bf16T(syn(vocab*D, 1), vocab, D),
		"model.norm.weight":         bf16T(syn(D, 2), D),
		"lm_head.weight":            bf16T(syn(vocab*D, 3), vocab, D),
	}
	for i := range nLayers {
		lp := "model.layers." + itoa(i) + "."
		ts[lp+"input_layernorm.weight"] = bf16T(syn(D, i*100+1), D)
		ts[lp+"post_attention_layernorm.weight"] = bf16T(syn(D, i*100+2), D)
		ap := lp + "self_attn."
		ts[ap+"q_proj.weight"] = bf16T(syn(AH*AHD*D, i*100+10), AH*AHD, D)
		ts[ap+"k_proj.weight"] = bf16T(syn(AKVH*AHD*D, i*100+11), AKVH*AHD, D)
		ts[ap+"v_proj.weight"] = bf16T(syn(AKVH*AHD*D, i*100+12), AKVH*AHD, D)
		ts[ap+"o_proj.weight"] = bf16T(syn(D*AH*AHD, i*100+13), D, AH*AHD)
		mp := lp + "mlp."
		ts[mp+"gate.weight"] = bf16T(syn(nE*D, i*100+20), nE, D) // router
		for e := range nE {
			ep := mp + "experts." + itoa(e) + "."
			ts[ep+"gate_proj.weight"] = bf16T(syn(moeFF*D, i*100+e*3+30), moeFF, D)
			ts[ep+"up_proj.weight"] = bf16T(syn(moeFF*D, i*100+e*3+31), moeFF, D)
			ts[ep+"down_proj.weight"] = bf16T(syn(D*moeFF, i*100+e*3+32), D, moeFF)
		}
	}
	for _, name := range moeProjNames(nLayers) {
		want := quantiseInPlace(t, ts, name, 8, 8)
		if name == "model.embed_tokens.weight" {
			wantEmbed = want
		}
	}
	cfg = []byte(`{"model_type":"qwen2_moe","hidden_size":8,"num_hidden_layers":2,"intermediate_size":16,` +
		`"num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,"vocab_size":32,"rms_norm_eps":1e-5,` +
		`"rope_theta":1000000,"num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":16,` +
		`"quantization":{"group_size":8,"bits":8}}`)
	arch = model.Arch{Experts: nE, TopK: 2, MoEGating: model.MoEGatingSoftmax, EmbedScale: 1}

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
	return dir, cfg, arch, wantEmbed
}

// TestLoadComposedWithArch_ZeroCopyAliasesMmap is the MoE-family RSS win, proven: LoadComposedWithArchMmap
// (the build the MoE arch registry hooks now use) keeps the packed quant projection weights as VIEWS into the
// mmap — the embed and every layer's attention q_proj alias the mapped shard, no heap copy — while the FFN is
// a real MoE (so this is the MoE arch path, not a dense fallback). The aliased embed still dequantises to the
// reference exactly, and the model owns the mapping's lifetime through the SAME RetainMmap/Close handshake as
// the base path (retain takes it, Close unmaps — no use-after-unmap because the finalizer only fires once
// every session/stepper holding the model is gone).
func TestLoadComposedWithArch_ZeroCopyAliasesMmap(t *testing.T) {
	dir, cfg, arch, wantEmbed := writeMoEQuantCheckpoint(t)

	dm, err := safetensors.LoadDirMmap(dir)
	if err != nil {
		t.Fatalf("LoadDirMmap: %v", err)
	}

	m, err := LoadComposedWithArchMmap(dm.Tensors, cfg, arch)
	if err != nil {
		t.Fatalf("LoadComposedWithArchMmap: %v", err)
	}
	if !m.mmapAliased {
		t.Fatal("mmapAliased not set — a quant MoE checkpoint's packed projection weights must alias the mmap")
	}
	if m.EmbedQ == nil {
		t.Fatal("embed must be kept packed")
	}
	// The whole point: the packed projection bytes are a view into the mmap, not a heap copy.
	if !aliasesAnyShard(dm, m.EmbedQ.Packed) {
		t.Fatal("embed packed bytes are a heap copy, not an mmap view — MoE zero-copy broken")
	}
	// This is genuinely the MoE arch path: every layer's FFN is a MoE, and its quant attention aliases too.
	for li := range m.Layers {
		if _, ok := m.Layers[li].MLP.(*MoEMLP); !ok {
			t.Fatalf("layer %d FFN %T, want *MoEMLP (the MoE arch path, not a dense fallback)", li, m.Layers[li].MLP)
		}
		am, ok := m.Layers[li].Mixer.(*attnMixer)
		if !ok {
			t.Fatalf("layer %d mixer %T, want *attnMixer", li, m.Layers[li].Mixer)
		}
		if am.w.QProjQ == nil || !aliasesAnyShard(dm, am.w.QProjQ.Packed) {
			t.Fatalf("layer %d q_proj packed bytes are a heap copy, not an mmap view", li)
		}
	}
	// The aliased bytes are the real weights: dequantising the embed reproduces the reference exactly.
	gotEmbed, err := mlxaffine.DequantizeTensor(m.EmbedQ.Packed, m.EmbedQ.Scales, m.EmbedQ.Biases, m.EmbedQ.OutDim, m.EmbedQ.InDim, m.EmbedQ.Bits, m.EmbedQ.GroupSize)
	if err != nil {
		t.Fatalf("dequantise aliased embed: %v", err)
	}
	assertF32Identical(t, "aliased-moe-embed", gotEmbed, wantEmbed)

	// Lifetime: the model takes ownership of the mapping and unmaps it on Close — exactly the base path.
	tm := NewTokenModel(m)
	if !tm.RetainMmap(dm) {
		t.Fatal("RetainMmap must take ownership of the aliasing MoE model")
	}
	if err := tm.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if dm.Shards != nil {
		t.Fatal("Close did not unmap the shards")
	}
	t.Logf("MoE zero-copy: %d-layer quant MoE checkpoint's packed projection weights view the mmap (no heap copy), decode to the reference", len(m.Layers))
}
