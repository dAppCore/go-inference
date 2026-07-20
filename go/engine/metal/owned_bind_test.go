// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	_ "dappco.re/go/inference/model/arch/Qwen/qwen2"   // register qwen2 (the b1 serve fixture's arch)
	_ "dappco.re/go/inference/model/arch/Qwen/qwenmoe" // register qwen2_moe (the packExperts fixture's arch)
	"dappco.re/go/inference/model/quant/mlxaffine"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

// owned_bind_test.go — receipts for the owned-weight device binding (#60, docs/
// design-owned-weight-binding.md): load-time synthesised heap tensors (the b1→b2 repack, the
// packExperts MoE packs) adopt as registered owned ranges and bind RESIDENT through the strict
// zero-copy resolver that used to refuse them ("weight is not a view into any mapped shard"),
// with their device buffers evicted when the owning session closes.

// b1FixtureLCG is a tiny deterministic byte stream for the synthetic 1-bit packed codes.
type b1FixtureLCG struct{ state uint32 }

func (s *b1FixtureLCG) bytes(n int) []byte {
	out := make([]byte, n)
	for i := range out {
		s.state = 1664525*s.state + 1013904223
		out[i] = byte(s.state >> 24)
	}
	return out
}

// b1Linear synthesises one 1-bit MLX affine weight: random packed codes [out, in/32] U32,
// uniform scales/biases [out, in/gs] BF16 (scale 0.05, bias -0.025 → dequantised entries
// ±0.025 — small, sane decode inputs). The b1 form is what Bonsai ships and what LoadLinear
// now widens to b2 at load.
func b1Linear(t map[string]safetensors.Tensor, s *b1FixtureLCG, prefix string, out, in, gs int) {
	packedWords := mlxaffine.PackedWords(in, 1)
	t[prefix+".weight"] = safetensors.Tensor{Dtype: "U32", Shape: []int{out, packedWords}, Data: s.bytes(out * packedWords * 4)}
	groups := in / gs
	scale, bias := safetensors.Float32ToBFloat16(0.05), safetensors.Float32ToBFloat16(-0.025)
	sb := make([]byte, out*groups*2)
	bb := make([]byte, out*groups*2)
	for i := 0; i < len(sb); i += 2 {
		sb[i], sb[i+1] = byte(scale), byte(scale>>8)
		bb[i], bb[i+1] = byte(bias), byte(bias>>8)
	}
	t[prefix+".scales"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{out, groups}, Data: sb}
	t[prefix+".biases"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{out, groups}, Data: bb}
}

// bf16Ones returns a BF16 all-ones vector tensor (an identity RMSNorm weight).
func bf16Ones(n int) safetensors.Tensor {
	one := safetensors.Float32ToBFloat16(1)
	data := make([]byte, n*2)
	for i := 0; i < len(data); i += 2 {
		data[i], data[i+1] = byte(one), byte(one>>8)
	}
	return safetensors.Tensor{Dtype: "BF16", Shape: []int{n}, Data: data}
}

// writeB1Qwen2Dir materialises a fully-1-bit-quantised dense qwen2 checkpoint directory —
// every linear (embed, q/k/v/o, gate/up/down, lm_head) ships b1 codes + scales/biases, the
// norms bf16 ones. The synthetic twin of Bonsai's format at toy geometry.
func writeB1Qwen2Dir(t *testing.T) string {
	t.Helper()
	const hidden, ff, vocab, layers, gs, headDim, kvDim = 64, 128, 32, 2, 32, 64, 64
	s := &b1FixtureLCG{state: 0xb17b0451}
	tensors := map[string]safetensors.Tensor{
		"model.norm.weight": bf16Ones(hidden),
	}
	b1Linear(tensors, s, "model.embed_tokens", vocab, hidden, gs)
	b1Linear(tensors, s, "lm_head", vocab, hidden, gs)
	for l := 0; l < layers; l++ {
		p := core.Sprintf("model.layers.%d.", l)
		tensors[p+"input_layernorm.weight"] = bf16Ones(hidden)
		tensors[p+"post_attention_layernorm.weight"] = bf16Ones(hidden)
		b1Linear(tensors, s, p+"self_attn.q_proj", 2*headDim, hidden, gs)
		b1Linear(tensors, s, p+"self_attn.k_proj", kvDim, hidden, gs)
		b1Linear(tensors, s, p+"self_attn.v_proj", kvDim, hidden, gs)
		b1Linear(tensors, s, p+"self_attn.o_proj", hidden, 2*headDim, gs)
		b1Linear(tensors, s, p+"mlp.gate_proj", ff, hidden, gs)
		b1Linear(tensors, s, p+"mlp.up_proj", ff, hidden, gs)
		b1Linear(tensors, s, p+"mlp.down_proj", hidden, ff, gs)
	}
	const config = `{"model_type":"qwen2","hidden_size":64,"intermediate_size":128,"num_hidden_layers":2,` +
		`"num_attention_heads":2,"num_key_value_heads":1,"head_dim":64,"vocab_size":32,"rms_norm_eps":1e-6,` +
		`"tie_word_embeddings":false,"max_position_embeddings":64,"quantization":{"bits":1,"group_size":32}}`
	dir := t.TempDir()
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	blob, err := safetensors.Encode(tensors)
	if err != nil {
		t.Fatalf("encode weights: %v", err)
	}
	if err := coreio.Local.Write(core.PathJoin(dir, "model.safetensors"), string(blob)); err != nil {
		t.Fatalf("write model.safetensors: %v", err)
	}
	return dir
}

// TestLoadDirB1RepackServe_Good is the synthetic end-to-end serve receipt for owned-weight
// binding on the STRICT quant bind path (mustBufFor/mustBufFor4 — the exact resolvers that
// refused owned buffers before #60): a fully-1-bit checkpoint loads through the LoadLinear
// b1→b2 repack (every projection, the embed and the head become OWNED heap buffers), the
// adoption sweep registers them, the session binds them resident, Generate decodes real
// tokens, and Close returns the owned device buffers (the resident cache shrinks back to its
// pre-load population). Before this change the same load died at session build.
func TestLoadDirB1RepackServe_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	dir := writeB1Qwen2Dir(t)

	// Adoption receipt (host-side): the repacked projection weights land in the tensor map as
	// owned registered ranges.
	lm, dm, err := model.Load(dir)
	if err != nil {
		t.Fatalf("model.Load: %v", err)
	}
	if lm.Embed == nil || lm.Embed.Bits != 2 {
		t.Fatalf("Embed.Bits = %v, want 2 (the b1→b2 repack)", lm.Embed)
	}
	q := dm.Tensors["model.layers.0.self_attn.q_proj.weight"]
	if !dm.IsOwned(q.Data) {
		t.Fatal("repacked q_proj not adopted as owned")
	}
	if !dm.IsOwned(lm.Layers[0].Q.Weight) {
		t.Fatal("assembled Q weight is not the adopted owned buffer")
	}
	if dm.IsOwned(dm.Tensors["model.layers.0.self_attn.q_proj.scales"].Data) {
		t.Fatal("scales must stay shard views (invariant under the widening), not owned")
	}
	if err := dm.Close(); err != nil {
		t.Fatalf("dm.Close: %v", err)
	}

	// Serve receipt (device): load → generate → close, TWICE. The owned weight buffers must
	// evict with each session (Close → evictResidentBufsForRanges over the dm's owned ranges);
	// the pooled ICB replay-scratch CONSTANTS (bf16ConstBuffer via newArchICBReplayScratch) are
	// process-lifetime by existing design and reach a steady state after the first cycle — so
	// the honest leak assertion is cycle-over-cycle equality, not emptiness: if owned weights
	// leaked, the second cycle would grow the cache by the full owned-weight count again.
	const vocab, n = 32, 4
	cycle := func() (int, []int32) {
		sess, err := LoadDir(dir, 16)
		if err != nil {
			t.Fatalf("LoadDir: %v (owned repacked weights must bind — this is the #60 wall)", err)
		}
		out, err := sess.Generate([]int32{1, 5, 3}, n, -1)
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		if len(out) != n {
			t.Fatalf("Generate returned %d ids, want %d", len(out), n)
		}
		for i, id := range out {
			if id < 0 || id >= vocab {
				t.Fatalf("generated id[%d] = %d outside vocab [0,%d)", i, id, vocab)
			}
		}
		if err := sess.Close(); err != nil {
			t.Fatalf("session Close: %v", err)
		}
		residentBufMu.Lock()
		defer residentBufMu.Unlock()
		return len(residentBufs), out
	}
	after1, out1 := cycle()
	after2, out2 := cycle()
	if after2 != after1 {
		t.Fatalf("resident cache grew across a load/serve/close cycle (%d → %d): owned weight buffers are not evicting with the session", after1, after2)
	}
	for i := range out1 {
		if out1[i] != out2[i] {
			t.Fatalf("cycle outputs diverged (%v vs %v) — the owned bind is not byte-stable", out1, out2)
		}
	}
	t.Logf("b1 checkpoint served through owned b2 repack: ids %v twice; resident cache steady at %d entries across cycles", out1, after1)
}

// writeTinyQwen2MoEBF16Dir materialises a bf16 qwen2_moe checkpoint directory whose routed
// per-expert tensors the qwenmoe NormalizeConfig hook packs (packExperts) into OWNED
// experts_packed synthesis at load — the exact latent class #60 names. Mirrors the qwenmoe
// load_test fixture's geometry with BF16 payloads (the engine's element type).
func writeTinyQwen2MoEBF16Dir(t *testing.T) string {
	t.Helper()
	const hidden, vocab, expertFF, sharedFF, heads, kvHeads, headDim, experts = 8, 32, 6, 10, 2, 1, 4, 4
	s := &b1FixtureLCG{state: 0x2a3e1234}
	bf16 := func(shape ...int) safetensors.Tensor {
		n := 1
		for _, d := range shape {
			n *= d
		}
		data := make([]byte, n*2)
		for i := 0; i < len(data); i += 2 {
			v := safetensors.Float32ToBFloat16(float32(int32(s.state>>24)-128) / 512)
			s.state = 1664525*s.state + 1013904223
			data[i], data[i+1] = byte(v), byte(v>>8)
		}
		return safetensors.Tensor{Dtype: "BF16", Shape: shape, Data: data}
	}
	tensors := map[string]safetensors.Tensor{
		"model.embed_tokens.weight":                         bf16(vocab, hidden),
		"model.norm.weight":                                 bf16Ones(hidden),
		"lm_head.weight":                                    bf16(vocab, hidden),
		"model.layers.0.input_layernorm.weight":             bf16Ones(hidden),
		"model.layers.0.post_attention_layernorm.weight":    bf16Ones(hidden),
		"model.layers.0.self_attn.q_proj.weight":            bf16(heads*headDim, hidden),
		"model.layers.0.self_attn.k_proj.weight":            bf16(kvHeads*headDim, hidden),
		"model.layers.0.self_attn.v_proj.weight":            bf16(kvHeads*headDim, hidden),
		"model.layers.0.self_attn.o_proj.weight":            bf16(hidden, heads*headDim),
		"model.layers.0.mlp.gate.weight":                    bf16(experts, hidden),
		"model.layers.0.mlp.shared_expert.gate_proj.weight": bf16(sharedFF, hidden),
		"model.layers.0.mlp.shared_expert.up_proj.weight":   bf16(sharedFF, hidden),
		"model.layers.0.mlp.shared_expert.down_proj.weight": bf16(hidden, sharedFF),
		"model.layers.0.mlp.shared_expert_gate.weight":      bf16(1, hidden),
	}
	for expert := range experts {
		prefix := core.Sprintf("model.layers.0.mlp.experts.%d.", expert)
		tensors[prefix+"gate_proj.weight"] = bf16(expertFF, hidden)
		tensors[prefix+"up_proj.weight"] = bf16(expertFF, hidden)
		tensors[prefix+"down_proj.weight"] = bf16(hidden, expertFF)
	}
	const config = `{"model_type":"qwen2_moe","hidden_size":8,"intermediate_size":10,"moe_intermediate_size":6,` +
		`"shared_expert_intermediate_size":10,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,` +
		`"num_experts":4,"num_experts_per_tok":2,"vocab_size":32,"norm_topk_prob":false,"tie_word_embeddings":false}`
	dir := t.TempDir()
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	blob, err := safetensors.Encode(tensors)
	if err != nil {
		t.Fatalf("encode weights: %v", err)
	}
	if err := coreio.Local.Write(core.PathJoin(dir, "model.safetensors"), string(blob)); err != nil {
		t.Fatalf("write model.safetensors: %v", err)
	}
	return dir
}

// TestFactoryLoadPackExperts_OwnedBind_Good is the latent class's bind receipt: a factory-loaded
// synthetic qwen2_moe checkpoint's packExperts tensors (owned heap synthesis at NormalizeConfig
// time) adopt as registered owned ranges and RESOLVE through the strict zero-copy binder over the
// real shard buffers — the exact call that failed "weight is not a view into any mapped shard"
// before #60. Full MoE Generate for this family is NOT asserted here: it is blocked past the
// binder by separate named gaps (the bf16 MoE step decode requires gemma's five sandwich norms +
// local MLP; packExperts packs .weight only, so a quantised pack has no packed scales/biases
// triple) — see docs/design-owned-weight-binding.md "What this does NOT do". The end-to-end serve
// receipt for owned binding is TestLoadDirB1RepackServe_Good on the same resolver path.
func TestFactoryLoadPackExperts_OwnedBind_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	dir := writeTinyQwen2MoEBF16Dir(t)
	lm, dm, err := model.Load(dir)
	if err != nil {
		t.Fatalf("model.Load: %v", err)
	}
	packed := dm.Tensors["model.layers.0.mlp.experts_packed.gate_proj.weight"]
	if len(packed.Data) == 0 {
		t.Fatal("packExperts synthesis absent — the qwen2_moe NormalizeConfig hook did not run")
	}
	if !dm.IsOwned(packed.Data) {
		t.Fatal("packExperts tensor not adopted as owned")
	}
	moe := lm.Layers[0].MoE
	if moe == nil || moe.ExpGate == nil {
		t.Fatal("factory load did not assemble the routed experts")
	}
	sb, err := buildShardBuffers(dm)
	if err != nil {
		t.Fatalf("buildShardBuffers: %v", err)
	}
	defer func() { _ = sb.Close() }()
	for role, w := range map[string][]byte{"ExpGate": moe.ExpGate.Weight, "ExpUp": moe.ExpUp.Weight, "ExpDown": moe.ExpDown.Weight} {
		v, err := sb.bufFor(w)
		if err != nil {
			t.Fatalf("bufFor(%s): %v — the owned packExperts synthesis must bind resident", role, err)
		}
		if v.buf == nil || v.off != 0 {
			t.Fatalf("bufFor(%s) = %+v, want a resident buffer at offset 0", role, v)
		}
	}
	// The checkpoint's own per-expert source tensors stay shard views — zero-copy untouched.
	src := dm.Tensors["model.layers.0.mlp.experts.0.gate_proj.weight"]
	if dm.IsOwned(src.Data) {
		t.Fatal("per-expert source tensor misregistered as owned (it is a shard view)")
	}
	if v, err := sb.bufFor(src.Data); err != nil || v.buf == nil {
		t.Fatalf("per-expert shard view stopped binding zero-copy: %+v %v", v, err)
	}
}

// TestBufForAligned_OwnedResident_Good — the resolver-level contract: a slice inside a
// REGISTERED owned range binds resident (offset 0, live buffer) while the shard-view path
// stays untouched. Uses a heap stand-in mapping (no mmap needed at this level).
func TestBufForAligned_OwnedResident_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("no Metal device: %v", err)
	}
	owned := make([]byte, 64)
	dm := &safetensors.DirMapping{Tensors: map[string]safetensors.Tensor{
		"synth": {Dtype: "BF16", Shape: []int{32}, Data: owned},
	}}
	if got := dm.AdoptOwnedTensors(); got != 1 {
		t.Fatalf("AdoptOwnedTensors = %d, want 1", got)
	}
	sb := &shardBuffers{dm: dm}
	v, err := sb.bufForAligned(owned, 4)
	if err != nil {
		t.Fatalf("bufForAligned(owned): %v", err)
	}
	if v.buf == nil || v.off != 0 {
		t.Fatalf("owned bind = %+v, want a resident buffer at offset 0", v)
	}
	// Same address → the cached entry (residentBytes keys by data pointer).
	v2, err := sb.bufForAligned(owned, 4)
	if err != nil || v2.buf != v.buf {
		t.Fatalf("second resolve = (%+v, %v), want the cached resident buffer", v2, err)
	}
	key := uintptr(unsafe.Pointer(&owned[0]))
	residentBufMu.Lock()
	_, cached := residentBufs[key]
	residentBufMu.Unlock()
	if !cached {
		t.Fatal("owned bind did not populate the resident cache")
	}
	// Session-scoped eviction: Close releases the owned entry with the session.
	if err := sb.Close(); err != nil {
		t.Fatalf("shardBuffers.Close: %v", err)
	}
	residentBufMu.Lock()
	_, cached = residentBufs[key]
	residentBufMu.Unlock()
	if cached {
		t.Fatal("owned resident entry survived shardBuffers.Close")
	}
}

// TestBufForAligned_OwnedResident_Bad — the wrong-mapping guard survives the fallback: an
// off-shard slice NO load pass registered still refuses to bind, and a nil-dm shardBuffers
// (the copy path) keeps its plain refusal.
func TestBufForAligned_OwnedResident_Bad(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	dm := &safetensors.DirMapping{Tensors: map[string]safetensors.Tensor{}}
	dm.AdoptOwnedTensors()
	sb := &shardBuffers{dm: dm}
	stranger := make([]byte, 32)
	if _, err := sb.bufForAligned(stranger, 4); err == nil {
		t.Fatal("unregistered off-shard slice bound — the wrong-mapping guard is gone")
	}
	if _, err := (&shardBuffers{}).bufForAligned(stranger, 4); err == nil {
		t.Fatal("nil-dm shardBuffers accepted an off-shard slice")
	}
}
