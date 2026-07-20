// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	_ "dappco.re/go/inference/model/arch/databricks/dbrx"   // register "dbrx" (this file's dbrx fixture's arch)
	_ "dappco.re/go/inference/model/arch/mistralai/mixtral" // register "mixtral" (this file's mixtral fixtures' arch)
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

// moe_zoo_generate_test.go — the end-to-end serve receipts for #59 item 3 (make the zoo MoE arches
// actually SERVE): a small mixtral-shaped and dbrx-shaped bf16 checkpoint, and a quantised
// mixtral-shaped checkpoint, load through the factory route and Generate real tokens — proving both
// named gaps from #60's "What this does NOT do" are closed: (a) the bf16/quant MoE step decode no
// longer assumes gemma4's five-sandwich-norm, always-on-local-MLP shape (moe_block.go, decode_forward_
// arch_quant.go), and (b) packExperts now packs the quantised scales/biases triple alongside the
// weight (model/arch/mistralai/mixtral/weights.go and its dbrx/olmoe/qwenmoe siblings).
//
// The quantised fixture calls LoadDir directly — the unmodified, literal factory route — because
// load_shared.go's loadedToQuant already wires a LoadedModel's MoE block onto the native quant
// struct (moeToQuant), so the quant route needed no upstream fix here.
//
// The bf16 fixtures do NOT call LoadDir/LoadTokenModelDir directly: load_shared.go's loadedToBF16
// maps every OTHER per-layer field from model.LoadedModel onto DecodeLayerWeights but never sets
// .MoE (grep confirms — the only production site that converts a LoadedMoE into a native MoE struct
// is moeToQuant, quant-only), so today's bf16 factory route drops every MoE layer's weights before
// decode is ever reached, for ANY architecture — a gap upstream of (and distinct from) the decode-
// shape fix, in load_shared.go, which #59's file fence names OUT OF BOUNDS here (a concurrent lane
// owns it). buildZooBF16Session below runs the REAL model.Load + the REAL loadedToBF16 + the REAL
// buildShardBuffers + the REAL newArchSessionShards — every production step LoadDir itself calls —
// and supplies ONLY the missing one-line wiring via moeLoadedToBF16 (a new, production, non-test
// function in moe_block.go: the bf16 sibling of moeToQuant, ready for load_shared.go's one-line
// call once that lane's fence lifts).

// bf16TensorVals bf16-encodes values as a safetensors.Tensor of the given logical shape.
func bf16TensorVals(values []float32, shape ...int) safetensors.Tensor {
	return safetensors.Tensor{Dtype: "BF16", Shape: shape, Data: toBF16Bytes(values)}
}

// writeCheckpointDir materialises tensors + configJSON as an on-disk checkpoint directory model.Load
// reads straight off disk — the shared body of writeTinyMixtralDir/writeTinyDBRXDir-style helpers
// across the arch packages' own load_test.go files, reused here as one generic version.
func writeCheckpointDir(t *testing.T, configJSON string, tensors map[string]safetensors.Tensor) string {
	t.Helper()
	dir := t.TempDir()
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), configJSON); err != nil {
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

// zooMixtralBF16Config is the SAME synthetic shape mixtral/load_test.go's TestTinyMixtralFactoryLoad_Good
// already proves loads through model.Load: 1 layer, 2 experts, top-1.
// hidden_size 64 / num_attention_heads 2 gives headDim 32 — the paged SDPA kernel's floor
// (encSDPAPagedDecodeStrided requires headDim a multiple of 32, at most 512).
const zooMixtralBF16Config = `{"model_type":"mixtral","hidden_size":64,"intermediate_size":12,"num_hidden_layers":1,` +
	`"num_attention_heads":2,"num_key_value_heads":1,"num_local_experts":2,"num_experts_per_tok":1,` +
	`"vocab_size":32,"rms_norm_eps":1e-5,"rope_theta":10000,"tie_word_embeddings":false}`

// writeZooMixtralBF16Dir writes a small bf16 Mixtral checkpoint: real per-expert w1/w2/w3 tensors
// (packExperts synthesises the packed triple at NormalizeConfig time), no shared expert, one
// pre-FFN norm — the llama-family zoo shape gap (a) makes legal.
func writeZooMixtralBF16Dir(t *testing.T) string {
	t.Helper()
	const hidden, ff, vocab, heads, kvHeads, headDim, experts = 64, 12, 32, 2, 1, 32, 2
	salt := 1
	next := func(n int) []float32 {
		v := syntheticFloat32(n, salt)
		salt++
		return v
	}
	tensors := map[string]safetensors.Tensor{
		"model.embed_tokens.weight":                      bf16TensorVals(next(vocab*hidden), vocab, hidden),
		"model.norm.weight":                              bf16Ones(hidden),
		"lm_head.weight":                                 bf16TensorVals(next(vocab*hidden), vocab, hidden),
		"model.layers.0.input_layernorm.weight":          bf16Ones(hidden),
		"model.layers.0.post_attention_layernorm.weight": bf16Ones(hidden),
		"model.layers.0.self_attn.q_proj.weight":         bf16TensorVals(next(heads*headDim*hidden), heads*headDim, hidden),
		"model.layers.0.self_attn.k_proj.weight":         bf16TensorVals(next(kvHeads*headDim*hidden), kvHeads*headDim, hidden),
		"model.layers.0.self_attn.v_proj.weight":         bf16TensorVals(next(kvHeads*headDim*hidden), kvHeads*headDim, hidden),
		"model.layers.0.self_attn.o_proj.weight":         bf16TensorVals(next(hidden*heads*headDim), hidden, heads*headDim),
		"model.layers.0.block_sparse_moe.gate.weight":    bf16TensorVals(next(experts*hidden), experts, hidden),
	}
	for e := range experts {
		prefix := core.Sprintf("model.layers.0.block_sparse_moe.experts.%d", e)
		tensors[prefix+".w1.weight"] = bf16TensorVals(next(ff*hidden), ff, hidden)
		tensors[prefix+".w3.weight"] = bf16TensorVals(next(ff*hidden), ff, hidden)
		tensors[prefix+".w2.weight"] = bf16TensorVals(next(hidden*ff), hidden, ff)
	}
	return writeCheckpointDir(t, zooMixtralBF16Config, tensors)
}

// zooDBRXBF16Config is the SAME synthetic shape dbrx/load_test.go's TestTinyDBRXFactoryLoad_Good
// already proves loads through model.Load: 1 layer, 4 experts, top-2, the RAW fused-tensor layout
// (NormalizeWeights + packExperts run inside model.Load's NormalizeConfig hook).
// d_model 64 / n_heads 2 gives headDim 32 — the paged SDPA kernel's floor (see zooMixtralBF16Config).
const zooDBRXBF16Config = `{"model_type":"dbrx","d_model":64,"n_heads":2,"n_layers":1,"vocab_size":32,` +
	`"attn_config":{"kv_n_heads":1,"rope_theta":10000},"ffn_config":{"ffn_hidden_size":12,"moe_num_experts":4,"moe_top_k":2}}`

// writeZooDBRXBF16Dir writes a small bf16 DBRX checkpoint in the real fused-tensor layout
// (transformer.blocks.0.ffn.experts.mlp.w1/v1/w2, one 3-D tensor per role covering every expert) —
// dbrx.NormalizeWeights slices+transposes it into per-expert tensors, then packExperts re-packs.
// NormalizeWeights reads tensor.Data by explicit byte width (BF16 ⇒ 2), never tensor.Shape, so the
// Shape metadata below only needs to carry the right total element count, exactly mirroring
// dbrx/load_test.go's own F32 fixture's shape choices.
func writeZooDBRXBF16Dir(t *testing.T) string {
	t.Helper()
	const hidden, vocab, expertFF, heads, kvHeads, headDim, experts = 64, 32, 12, 2, 1, 32, 4
	salt := 1
	next := func(n int) []float32 {
		v := syntheticFloat32(n, salt)
		salt++
		return v
	}
	tensors := map[string]safetensors.Tensor{
		"transformer.wte.weight":                                   bf16TensorVals(next(vocab*hidden), vocab, hidden),
		"transformer.norm_f.weight":                                bf16Ones(hidden),
		"lm_head.weight":                                           bf16TensorVals(next(vocab*hidden), vocab, hidden),
		"transformer.blocks.0.norm_attn_norm.norm_1.weight":        bf16Ones(hidden),
		"transformer.blocks.0.norm_attn_norm.norm_2.weight":        bf16Ones(hidden),
		"transformer.blocks.0.norm_attn_norm.attn.Wqkv.weight":     bf16TensorVals(next((heads+2*kvHeads)*headDim*hidden), (heads+2*kvHeads)*headDim, hidden),
		"transformer.blocks.0.norm_attn_norm.attn.out_proj.weight": bf16TensorVals(next(hidden*hidden), hidden, hidden),
		"transformer.blocks.0.ffn.router.layer.weight":             bf16TensorVals(next(experts*hidden), experts, hidden),
		"transformer.blocks.0.ffn.experts.mlp.w1":                  bf16TensorVals(next(experts*expertFF*hidden), experts, expertFF, hidden),
		"transformer.blocks.0.ffn.experts.mlp.v1":                  bf16TensorVals(next(experts*expertFF*hidden), experts, expertFF, hidden),
		"transformer.blocks.0.ffn.experts.mlp.w2":                  bf16TensorVals(next(experts*hidden*expertFF), experts, hidden, expertFF),
	}
	return writeCheckpointDir(t, zooDBRXBF16Config, tensors)
}

// zooMixtralQuantConfig is a small Mixtral shape sized for clean 4-bit affine geometry
// (hidden_size/intermediate_size both equal to the group size, 32 — the metallib's shipped affine_qmv
// kernels are group-size-specific, and 32 is the group size the existing quant test suite already
// uses, e.g. arch_quant_session_test.go) AND headDim 32 (hidden_size 64 / num_attention_heads 2) —
// the paged SDPA kernel's floor (see zooMixtralBF16Config).
const zooMixtralQuantConfig = `{"model_type":"mixtral","hidden_size":64,"intermediate_size":32,"num_hidden_layers":1,` +
	`"num_attention_heads":2,"num_key_value_heads":1,"num_local_experts":2,"num_experts_per_tok":1,` +
	`"vocab_size":32,"rms_norm_eps":1e-5,"rope_theta":10000,"tie_word_embeddings":false}`

// writeZooMixtralQuantDir writes a fully 4-bit-quantised Mixtral checkpoint — embed/attention/
// router/lm_head AND every per-expert w1/w2/w3 triple affine-packed via packAffineQuant (pure Go,
// self-consistent with this package's own dequantiser — see quantWeightFixture's doc). Quantising
// the embed routes the WHOLE model through LoadDir's quant path (quantised(lm) reads Embed.Quantised()).
// The per-expert quant triples prove gap (b): packExperts must pack .scales/.biases alongside
// .weight for a quantised checkpoint to serve at all — before that fix, LoadLinear would find no
// packed scales/biases tensor for the synthesised experts_packed.* role and the layer would load as
// an (incorrectly) unquantised or absent expert weight.
func writeZooMixtralQuantDir(t *testing.T) string {
	t.Helper()
	const hidden, ff, vocab, heads, kvHeads, headDim, experts, gs, bits = 64, 32, 32, 2, 1, 32, 2, 32, 4
	salt := 1
	mkNorm := func(n int) safetensors.Tensor {
		out := bf16TensorVals(syntheticFloat32(n, salt), n)
		salt++
		return out
	}
	set3 := func(tensors map[string]safetensors.Tensor, prefix string, outDim, inDim int) {
		packed, scales, biases := quantizeProj(t, outDim, inDim, gs, bits, salt)
		salt++
		tensors[prefix+".weight"] = safetensors.Tensor{Dtype: "U32", Shape: []int{outDim, inDim * bits / 32}, Data: packed}
		tensors[prefix+".scales"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{outDim, inDim / gs}, Data: scales}
		tensors[prefix+".biases"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{outDim, inDim / gs}, Data: biases}
	}
	tensors := map[string]safetensors.Tensor{
		"model.norm.weight":                              mkNorm(hidden),
		"model.layers.0.input_layernorm.weight":          mkNorm(hidden),
		"model.layers.0.post_attention_layernorm.weight": mkNorm(hidden),
	}
	set3(tensors, "model.embed_tokens", vocab, hidden)
	set3(tensors, "lm_head", vocab, hidden)
	set3(tensors, "model.layers.0.self_attn.q_proj", heads*headDim, hidden)
	set3(tensors, "model.layers.0.self_attn.k_proj", kvHeads*headDim, hidden)
	set3(tensors, "model.layers.0.self_attn.v_proj", kvHeads*headDim, hidden)
	set3(tensors, "model.layers.0.self_attn.o_proj", hidden, heads*headDim)
	set3(tensors, "model.layers.0.block_sparse_moe.gate", experts, hidden)
	for e := range experts {
		prefix := core.Sprintf("model.layers.0.block_sparse_moe.experts.%d", e)
		set3(tensors, prefix+".w1", ff, hidden)
		set3(tensors, prefix+".w3", ff, hidden)
		set3(tensors, prefix+".w2", hidden, ff)
	}
	return writeCheckpointDir(t, zooMixtralQuantConfig, tensors)
}

// buildZooBF16Session loads dir through the REAL factory parse/assemble (model.Load) and the REAL
// loadedToBF16 + buildShardBuffers + newArchSessionShards LoadDir itself calls, then patches in the
// ONE missing wire load_shared.go doesn't make yet — see this file's header doc and moeLoadedToBF16.
func buildZooBF16Session(t *testing.T, dir string, maxLen int) *ArchSession {
	t.Helper()
	lm, dm, err := model.Load(dir)
	if err != nil {
		t.Fatalf("model.Load: %v", err)
	}
	sb, err := buildShardBuffers(dm)
	if err != nil {
		_ = dm.Close()
		t.Fatalf("buildShardBuffers: %v", err)
	}
	g := loadedToBF16(lm)
	moeLayers := 0
	for i := range lm.Layers {
		if lm.Layers[i].MoE != nil {
			g.Layers[i].MoE = moeLoadedToBF16(lm.Layers[i].MoE, lm.Arch)
			moeLayers++
		}
	}
	if moeLayers == 0 {
		t.Fatal("no MoE layers assembled — fixture or arch registration is wrong")
	}
	sess, err := newArchSessionShards(g, lm.Arch, maxLen, sb)
	if err != nil {
		_ = sb.Close()
		t.Fatalf("newArchSessionShards: %v (owned/zoo-shaped MoE weights must bind and decode)", err)
	}
	sess.shards = sb
	return sess
}

// TestFactoryLoadMixtralBF16_Generate_Good is the bf16 end-to-end serve receipt for a llama-family
// zoo MoE arch with NO shared expert and NO local dense MLP (mixtral): factory-parsed, decoded on
// metal, deterministic, in-vocab, twice.
func TestFactoryLoadMixtralBF16_Generate_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	dir := writeZooMixtralBF16Dir(t)
	const vocab, n = 32, 4
	cycle := func() []int32 {
		sess := buildZooBF16Session(t, dir, 16)
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
		return out
	}
	out1, out2 := cycle(), cycle()
	for i := range out1 {
		if out1[i] != out2[i] {
			t.Fatalf("cycle outputs diverged (%v vs %v) — bf16 zoo MoE Generate is not deterministic", out1, out2)
		}
	}
	t.Logf("mixtral-shaped bf16 MoE checkpoint served through the factory route: ids %v twice", out1)
}

// TestFactoryLoadDBRXBF16_Generate_Good is the bf16 end-to-end serve receipt for DBRX's fused
// per-role expert tensors (NormalizeWeights' slice+transpose, then packExperts) reaching a real
// Generate, proving the SAME zoo shape fix a second, structurally distinct arch package.
func TestFactoryLoadDBRXBF16_Generate_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	dir := writeZooDBRXBF16Dir(t)
	const vocab, n = 32, 4
	cycle := func() []int32 {
		sess := buildZooBF16Session(t, dir, 16)
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
		return out
	}
	out1, out2 := cycle(), cycle()
	for i := range out1 {
		if out1[i] != out2[i] {
			t.Fatalf("cycle outputs diverged (%v vs %v) — bf16 zoo MoE Generate is not deterministic", out1, out2)
		}
	}
	t.Logf("dbrx-shaped bf16 MoE checkpoint served through the factory route: ids %v twice", out1)
}

// TestFactoryLoadMixtralQuant_Generate_Good is the quantised end-to-end serve receipt: gap (b)'s
// packExperts scales/biases packing proven by a REAL 4-bit Mixtral checkpoint reaching Generate
// through the literal, unmodified LoadDir — the quant route needed no upstream conversion fix
// (load_shared.go's loadedToQuant already wires LoadedMoE onto the native quant struct via
// moeToQuant), only the decode-shape + packExperts fixes this change makes.
func TestFactoryLoadMixtralQuant_Generate_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	dir := writeZooMixtralQuantDir(t)
	const vocab, n = 32, 4
	cycle := func() []int32 {
		sess, err := LoadDir(dir, 16)
		if err != nil {
			t.Fatalf("LoadDir: %v (quantised packExperts scales/biases must pack and bind — this is the #59 wall)", err)
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
		return out
	}
	out1, out2 := cycle(), cycle()
	for i := range out1 {
		if out1[i] != out2[i] {
			t.Fatalf("cycle outputs diverged (%v vs %v) — quantised zoo MoE Generate is not deterministic", out1, out2)
		}
	}
	t.Logf("mixtral-shaped 4-bit quantised MoE checkpoint served through LoadDir (the unmodified factory route): ids %v twice", out1)
}
