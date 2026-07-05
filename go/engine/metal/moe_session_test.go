// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"os"
	"testing"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	g4 "dappco.re/go/inference/model/gemma4"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

// moeQuantTensors builds a synthetic MIXED-PRECISION MoE gemma4 checkpoint (gemma4 26B-A4B
// shape): attention + embedding + experts 4-bit, local MLP + router 8-bit. The experts are the
// batched SwitchGLU layout. quant.For drives the per-tensor width.
func moeQuantTensors(t *testing.T, arch model.Arch, quant *model.QuantConfig) map[string]safetensors.Tensor {
	t.Helper()
	ts := map[string]safetensors.Tensor{}
	salt := 1
	mkBF16 := func(name string, elems int) {
		f := make([]float32, elems)
		for i := range f {
			f[i] = float32((i*salt+7)%83-41) * 0.02
		}
		ts[name] = safetensors.Tensor{Dtype: "BF16", Shape: []int{elems}, Data: toBF16Bytes(f)}
		salt++
	}
	mkQuant := func(prefix string, outDim, inDim int) {
		_, bits := quant.For(prefix)
		p, s, b := quantizeProj(t, outDim, inDim, 64, bits, salt)
		salt++
		ts[prefix+".weight"] = safetensors.Tensor{Dtype: "U32", Shape: []int{outDim, inDim * bits / 32}, Data: p}
		ts[prefix+".scales"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{outDim, inDim / 64}, Data: s}
		ts[prefix+".biases"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{outDim, inDim / 64}, Data: b}
	}
	dModel, headDim, dFF, vocab := arch.Hidden, arch.HeadDim, arch.FF, arch.Vocab
	qDim, kvDim := arch.Heads*headDim, arch.KVHeads*headDim
	nE, eFF := arch.Experts, arch.ExpertFF
	mkQuant("model.embed_tokens", vocab, dModel)
	mkBF16("model.norm.weight", dModel)
	for i := range arch.Layer {
		p := core.Sprintf("model.layers.%d", i)
		mkBF16(p+".input_layernorm.weight", dModel)
		mkBF16(p+".post_attention_layernorm.weight", dModel)
		mkBF16(p+".self_attn.q_norm.weight", headDim)
		mkBF16(p+".self_attn.k_norm.weight", headDim)
		mkQuant(p+".self_attn.q_proj", qDim, dModel)
		mkQuant(p+".self_attn.k_proj", kvDim, dModel)
		mkQuant(p+".self_attn.v_proj", kvDim, dModel)
		mkQuant(p+".self_attn.o_proj", dModel, qDim)
		// MoE dual-branch: 5 norms, local MLP (8-bit), router (8-bit), batched experts (4-bit).
		mkBF16(p+".pre_feedforward_layernorm.weight", dModel)
		mkBF16(p+".pre_feedforward_layernorm_2.weight", dModel)
		mkBF16(p+".post_feedforward_layernorm_1.weight", dModel)
		mkBF16(p+".post_feedforward_layernorm_2.weight", dModel)
		mkBF16(p+".post_feedforward_layernorm.weight", dModel)
		mkQuant(p+".mlp.gate_proj", dFF, dModel)
		mkQuant(p+".mlp.up_proj", dFF, dModel)
		mkQuant(p+".mlp.down_proj", dModel, dFF)
		mkBF16(p+".router.scale", dModel)
		mkBF16(p+".router.per_expert_scale", nE)
		mkQuant(p+".router.proj", nE, dModel)
		mkQuant(p+".experts.switch_glu.gate_proj", nE*eFF, dModel)
		mkQuant(p+".experts.switch_glu.up_proj", nE*eFF, dModel)
		mkQuant(p+".experts.switch_glu.down_proj", nE*dModel, eFF)
	}
	return ts
}

func moeQuantTensorsWithFusedGateUp(t *testing.T, arch model.Arch, quant *model.QuantConfig) map[string]safetensors.Tensor {
	t.Helper()
	ts := moeQuantTensors(t, arch, quant)
	for i := range arch.Layer {
		p := core.Sprintf("model.layers.%d.experts.switch_glu", i)
		fuseMoEGateUpTensorPair(t, ts, p+".gate_proj", p+".up_proj", p+".gate_up_proj", arch.Experts, arch.ExpertFF)
	}
	return ts
}

func fuseMoEGateUpTensorPair(t *testing.T, ts map[string]safetensors.Tensor, gatePrefix, upPrefix, fusedPrefix string, experts, expertFF int) {
	t.Helper()
	for _, suffix := range []string{".weight", ".scales", ".biases"} {
		gate, ok := ts[gatePrefix+suffix]
		if !ok {
			t.Fatalf("missing gate tensor %s", gatePrefix+suffix)
		}
		up, ok := ts[upPrefix+suffix]
		if !ok {
			t.Fatalf("missing up tensor %s", upPrefix+suffix)
		}
		if gate.Dtype != up.Dtype || len(gate.Shape) != 2 || len(up.Shape) != 2 || gate.Shape[0] != experts*expertFF || up.Shape[0] != experts*expertFF || gate.Shape[1] != up.Shape[1] {
			t.Fatalf("cannot fuse gate/up tensor %s: gate shape %v dtype %s, up shape %v dtype %s", suffix, gate.Shape, gate.Dtype, up.Shape, up.Dtype)
		}
		rowBytes := len(gate.Data) / gate.Shape[0]
		fused := make([]byte, 0, len(gate.Data)+len(up.Data))
		for e := 0; e < experts; e++ {
			start := e * expertFF * rowBytes
			end := start + expertFF*rowBytes
			fused = append(fused, gate.Data[start:end]...)
			fused = append(fused, up.Data[start:end]...)
		}
		ts[fusedPrefix+suffix] = safetensors.Tensor{Dtype: gate.Dtype, Shape: []int{experts, 2 * expertFF, gate.Shape[1]}, Data: fused}
		delete(ts, gatePrefix+suffix)
		delete(ts, upPrefix+suffix)
	}
}

// TestLoadGemma4QuantMoE gates the whole mixed-precision MoE path (gemma4 26B-A4B): a synthetic
// model (4-bit experts + attention, 8-bit local MLP + router) assembles into a session that
// generates; the first token equals the manual chain (embed → stepToken-with-MoEBlockQuant →
// lm_head → greedy); and a config.json carrying the per-tensor overrides dir-loads to the same.
func TestLoadGemma4QuantMoE(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, vocab = 64, 2, 1, 64, 32
	const dFF, expertDFF, numExperts, topK, numLayers = 128, 64, 4, 2, 2
	const maxLen, n = 16, 4
	// mixed precision: default 4-bit, local MLP + router 8-bit (the 26B-A4B QAT pattern).
	quant := &model.QuantConfig{GroupSize: 64, Bits: 4, Overrides: map[string]model.ModuleQuant{}}
	for i := 0; i < numLayers; i++ {
		for _, m := range []string{"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "router.proj"} {
			quant.Overrides[core.Sprintf("model.layers.%d.%s", i, m)] = model.ModuleQuant{GroupSize: 64, Bits: 8}
		}
	}
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: numLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim, VocabSize: vocab, RMSNormEps: 1e-6,
		EnableMoEBlock: true, NumExperts: numExperts, TopKExperts: topK, MoEIntermediateSize: expertDFF,
		Quantization: quant,
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if !arch.HasMoE() {
		t.Fatal("arch should be MoE")
	}
	ts := moeQuantTensors(t, arch, quant)
	prompt := []int32{1, 5, 3}

	lm, err := model.Assemble(ts, arch, model.StandardWeightNames())
	if err != nil {
		t.Fatalf("model.Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, quant.GroupSize, quant.Bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	if g.Layers[0].MoE == nil {
		t.Fatal("layer 0 should carry the quant MoE block")
	}
	if g.Layers[0].MoE.ExpertBits != 4 || g.Layers[0].MoE.LocalBits != 8 || g.Layers[0].MoE.RouterBits != 8 {
		t.Fatalf("per-component bits wrong: experts %d local %d router %d", g.Layers[0].MoE.ExpertBits, g.Layers[0].MoE.LocalBits, g.Layers[0].MoE.RouterBits)
	}
	sess, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession: %v", err)
	}
	gen, err := sess.Generate(prompt, n, -1)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	for i, id := range gen {
		if id < 0 || int(id) >= vocab {
			t.Fatalf("token %d = %d out of range", i, id)
		}
	}

	// manual chain: embed → stepToken (MoEBlockQuant via moeQuant) → lm_head → greedy.
	attnScale := arch.AttnScale // the model-declared scale (gemma4 1.0), matching the session
	embedScale := float32(math.Sqrt(float64(dModel)))
	var manualFirst int32
	withAutoreleasePool(func() {
		lb, moeQ, _ := buildQuantArchLayerBufs(g.Layers, arch.Layer, dModel, nHeads, nKV, headDim, dFF, maxLen, arch.SlidingWindow, nil)
		st := newArchDecodeState(arch.Layer, lb, make([]*MoELayerWeights, numLayers), dModel, nHeads, nKV, headDim, dFF, arch.SlidingWindow, arch.RotaryDim, arch.RotaryDimLocal, arch.RopeBase, arch.RopeLocalBase, attnScale, arch.Eps, false, 0)
		st.moeQuant = moeQ
		var hidden []byte
		for p, id := range prompt {
			embs, err := EmbedTokensQuant(g.Embed, g.EmbedScales, g.EmbedBiases, []int32{id}, vocab, dModel, 64, 4, embedScale)
			if err != nil {
				t.Fatalf("EmbedTokensQuant: %v", err)
			}
			if hidden, err = st.stepToken(embs[0], p); err != nil {
				t.Fatalf("stepToken: %v", err)
			}
		}
		logits, err := LMHeadQuant(hidden, g.FinalNorm, g.LMHead, g.LMHeadScales, g.LMHeadBiases, dModel, vocab, 64, 4, arch.Eps, arch.SoftCap)
		if err != nil {
			t.Fatalf("LMHeadQuant: %v", err)
		}
		if manualFirst, err = model.Greedy(logits, vocab); err != nil {
			t.Fatalf("Greedy: %v", err)
		}
	})
	if gen[0] != manualFirst {
		t.Fatalf("session first token %d != manual MoE chain %d", gen[0], manualFirst)
	}

	// dir-load: a config.json carrying the per-tensor overrides → LoadDir ≡ in-memory.
	configJSON := gemma4ConfigJSON(t, cfg)
	dir := t.TempDir()
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), string(configJSON)); err != nil {
		t.Fatalf("write config: %v", err)
	}
	blob, err := safetensors.Encode(ts)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if err := coreio.Local.Write(core.PathJoin(dir, "model.safetensors"), string(blob)); err != nil {
		t.Fatalf("write weights: %v", err)
	}
	dirSess, err := LoadDir(dir, maxLen)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	genDir, err := dirSess.Generate(prompt, n, -1)
	if err != nil {
		t.Fatalf("dir Generate: %v", err)
	}
	if !idsEqual(genDir, gen) {
		t.Fatalf("dir-loaded MoE %v != in-memory %v", genDir, gen)
	}
	t.Logf("mixed-precision MoE end to end: 4-bit experts + 8-bit local/router assemble → session generates %v; first token ≡ manual chain; config.json overrides dir-load ≡ in-memory", gen)
}

func TestLoadGemma4QuantMoEFusedGateUpMatchesSplitExperts(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, vocab = 64, 2, 1, 64, 32
	const dFF, expertDFF, numExperts, topK, numLayers = 128, 64, 4, 2, 1
	const maxLen = 8
	quant := &model.QuantConfig{GroupSize: 64, Bits: 4, Overrides: map[string]model.ModuleQuant{}}
	for i := 0; i < numLayers; i++ {
		for _, m := range []string{"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "router.proj"} {
			quant.Overrides[core.Sprintf("model.layers.%d.%s", i, m)] = model.ModuleQuant{GroupSize: 64, Bits: 8}
		}
	}
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: numLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim, VocabSize: vocab, RMSNormEps: 1e-6,
		EnableMoEBlock: true, NumExperts: numExperts, TopKExperts: topK, MoEIntermediateSize: expertDFF,
		Quantization: quant,
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	prompt := []int32{1, 5, 3}
	fusedGateUpPrefix := "model.layers.0.experts.switch_glu.gate_up_proj"
	dataPtr := func(b []byte) uintptr {
		if len(b) == 0 {
			return 0
		}
		return uintptr(unsafe.Pointer(&b[0]))
	}
	decode := func(name string, ts map[string]safetensors.Tensor, fused bool) []int32 {
		t.Helper()
		lm, err := model.Assemble(ts, arch, model.StandardWeightNames())
		if err != nil {
			t.Fatalf("%s model.Assemble: %v", name, err)
		}
		qm, err := loadedToQuant(lm, quant.GroupSize, quant.Bits)
		if err != nil {
			t.Fatalf("%s loadedToQuant: %v", name, err)
		}
		if qm.Layers[0].MoE == nil {
			t.Fatalf("%s did not populate MoE weights", name)
		}
		moe := qm.Layers[0].MoE
		if fused {
			fusedWeight := ts[fusedGateUpPrefix+".weight"].Data
			if len(moe.ExpGateUp.Packed) == 0 || dataPtr(moe.ExpGateUp.Packed) != dataPtr(fusedWeight) {
				t.Fatalf("%s did not keep fused gate_up packed backing: got ptr %x want %x", name, dataPtr(moe.ExpGateUp.Packed), dataPtr(fusedWeight))
			}
			if len(moe.ExpGate.Packed) != 0 || len(moe.ExpUp.Packed) != 0 {
				t.Fatalf("%s copied fused gate_up into split gate/up buffers (gate=%d up=%d)", name, len(moe.ExpGate.Packed), len(moe.ExpUp.Packed))
			}
		} else if len(moe.ExpGate.Packed) == 0 || len(moe.ExpUp.Packed) == 0 {
			t.Fatalf("%s did not populate split expert gate/up weights from MoE tensors", name)
		}
		sess, err := NewArchQuantSession(qm, arch, maxLen)
		if err != nil {
			t.Fatalf("%s NewArchQuantSession: %v", name, err)
		}
		out, err := sess.Generate(prompt, 3, -1)
		if err != nil {
			t.Fatalf("%s Generate: %v", name, err)
		}
		return out
	}

	want := decode("split", moeQuantTensors(t, arch, quant), false)
	got := decode("fused gate_up", moeQuantTensorsWithFusedGateUp(t, arch, quant), true)
	if !idsEqual(got, want) {
		t.Fatalf("fused gate_up generated %v, want split expert route %v", got, want)
	}
}

func TestLoadGemma4QuantMoEUsesShardViewsForMoEQuantTriples(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const dModel, nHeads, nKV, headDim, vocab = 64, 2, 1, 64, 32
	const dFF, expertDFF, numExperts, topK, numLayers = 128, 64, 4, 2, 1
	const maxLen = 8
	quant := &model.QuantConfig{GroupSize: 64, Bits: 4, Overrides: map[string]model.ModuleQuant{}}
	for _, m := range []string{"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "router.proj"} {
		quant.Overrides["model.layers.0."+m] = model.ModuleQuant{GroupSize: 64, Bits: 8}
	}
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: numLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim, VocabSize: vocab, RMSNormEps: 1e-6,
		EnableMoEBlock: true, NumExperts: numExperts, TopKExperts: topK, MoEIntermediateSize: expertDFF,
		Quantization: quant,
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	ts := moeQuantTensors(t, arch, quant)
	names := []string{
		"model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.gate_proj.scales", "model.layers.0.mlp.gate_proj.biases",
		"model.layers.0.mlp.up_proj.weight", "model.layers.0.mlp.up_proj.scales", "model.layers.0.mlp.up_proj.biases",
		"model.layers.0.mlp.down_proj.weight", "model.layers.0.mlp.down_proj.scales", "model.layers.0.mlp.down_proj.biases",
		"model.layers.0.router.proj.weight", "model.layers.0.router.proj.scales", "model.layers.0.router.proj.biases",
		"model.layers.0.experts.switch_glu.gate_proj.weight", "model.layers.0.experts.switch_glu.gate_proj.scales", "model.layers.0.experts.switch_glu.gate_proj.biases",
		"model.layers.0.experts.switch_glu.up_proj.weight", "model.layers.0.experts.switch_glu.up_proj.scales", "model.layers.0.experts.switch_glu.up_proj.biases",
		"model.layers.0.experts.switch_glu.down_proj.weight", "model.layers.0.experts.switch_glu.down_proj.scales", "model.layers.0.experts.switch_glu.down_proj.biases",
	}
	blob := alignedSafetensorsBlob(t, ts, names)

	dir := t.TempDir()
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), string(gemma4ConfigJSON(t, cfg))); err != nil {
		t.Fatalf("write config: %v", err)
	}
	if err := coreio.Local.Write(core.PathJoin(dir, "model.safetensors"), string(blob)); err != nil {
		t.Fatalf("write weights: %v", err)
	}
	sess, err := LoadDir(dir, maxLen)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	defer func() { _ = sess.Close() }()
	if _, err := sess.Generate([]int32{1, 5, 3}, 1, -1); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	moe := sess.state.moeQuant[0]
	if moe == nil {
		t.Fatal("loaded session missing MoE quant weights")
	}
	moeTriples := []struct {
		name string
		q    QuantWeight
	}{
		{"local gate", moe.LocalGate}, {"local up", moe.LocalUp}, {"local down", moe.LocalDown},
		{"router", moe.Router}, {"expert gate", moe.ExpGate}, {"expert up", moe.ExpUp}, {"expert down", moe.ExpDown},
	}
	key := func(b []byte) uintptr {
		if len(b) == 0 {
			return 0
		}
		return uintptr(unsafe.Pointer(&b[0]))
	}
	residentBufMu.Lock()
	copied := make([]string, 0)
	for _, triple := range moeTriples {
		parts := []struct {
			suffix string
			buf    []byte
		}{
			{"packed", triple.q.Packed},
			{"scales", triple.q.Scales},
			{"biases", triple.q.Biases},
		}
		for _, part := range parts {
			if _, ok := residentBufs[key(part.buf)]; ok {
				copied = append(copied, triple.name+" "+part.suffix)
			}
		}
	}
	residentBufMu.Unlock()
	if len(copied) != 0 {
		t.Fatalf("mmap MoE quant triples copied into resident buffers instead of shard views: %v", copied)
	}
}

func TestLoadGemma4QuantMoEUsesShardViewsForMoENormsAndScales(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, vocab = 64, 2, 1, 64, 32
	const dFF, expertDFF, numExperts, topK, numLayers = 128, 64, 4, 2, 1
	const maxLen = 8
	quant := &model.QuantConfig{GroupSize: 64, Bits: 4, Overrides: map[string]model.ModuleQuant{}}
	for _, m := range []string{"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "router.proj"} {
		quant.Overrides["model.layers.0."+m] = model.ModuleQuant{GroupSize: 64, Bits: 8}
	}
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: numLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim, VocabSize: vocab, RMSNormEps: 1e-6,
		EnableMoEBlock: true, NumExperts: numExperts, TopKExperts: topK, MoEIntermediateSize: expertDFF,
		Quantization: quant,
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	ts := moeQuantTensors(t, arch, quant)
	names := []string{
		"model.layers.0.pre_feedforward_layernorm.weight",
		"model.layers.0.pre_feedforward_layernorm_2.weight",
		"model.layers.0.post_feedforward_layernorm_1.weight",
		"model.layers.0.post_feedforward_layernorm_2.weight",
		"model.layers.0.post_feedforward_layernorm.weight",
		"model.layers.0.router.per_expert_scale",
	}
	blob := alignedSafetensorsBlob(t, ts, names)

	dir := t.TempDir()
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), string(gemma4ConfigJSON(t, cfg))); err != nil {
		t.Fatalf("write config: %v", err)
	}
	if err := coreio.Local.Write(core.PathJoin(dir, "model.safetensors"), string(blob)); err != nil {
		t.Fatalf("write weights: %v", err)
	}
	sess, err := LoadDir(dir, maxLen)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	defer func() { _ = sess.Close() }()
	if _, err := sess.Generate([]int32{1, 5, 3}, 1, -1); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	moe := sess.state.moeQuant[0]
	if moe == nil {
		t.Fatal("loaded session missing MoE quant weights")
	}
	views := []struct {
		name string
		view bufView
	}{
		{"pre ff norm", moe.preFFNormView},
		{"pre ff norm 2", moe.preFFNorm2View},
		{"post ff norm 1", moe.postFFNorm1View},
		{"post ff norm 2", moe.postFFNorm2View},
		{"post ff norm", moe.postFFNormView},
		{"per expert scale", moe.perExpertScaleView},
	}
	for _, item := range views {
		if item.view.buf == nil {
			t.Fatalf("%s did not keep a shard view", item.name)
		}
	}
	raw := []struct {
		name string
		buf  []byte
	}{
		{"pre ff norm", moe.PreFFNormW},
		{"pre ff norm 2", moe.PreFFNorm2W},
		{"post ff norm 1", moe.PostFFNorm1W},
		{"post ff norm 2", moe.PostFFNorm2W},
		{"post ff norm", moe.PostFFNormW},
		{"per expert scale", moe.PerExpertScale},
	}
	key := func(b []byte) uintptr { return uintptr(unsafe.Pointer(&b[0])) }
	residentBufMu.Lock()
	copied := make([]string, 0)
	for _, item := range raw {
		if _, ok := residentBufs[key(item.buf)]; ok {
			copied = append(copied, item.name)
		}
	}
	residentBufMu.Unlock()
	if len(copied) != 0 {
		t.Fatalf("mmap MoE norms/scales copied into resident buffers instead of shard views: %v", copied)
	}
}

func alignedSafetensorsBlob(t *testing.T, tensors map[string]safetensors.Tensor, alignedNames []string) []byte {
	t.Helper()
	offsetOf := func(blob []byte, tensor safetensors.Tensor) uintptr {
		if len(tensor.Data) == 0 {
			return 0
		}
		return uintptr(unsafe.Pointer(&tensor.Data[0])) - uintptr(unsafe.Pointer(&blob[0]))
	}
	for pad := 0; pad < 32; pad++ {
		candidate := make(map[string]safetensors.Tensor, len(tensors)+1)
		for name, tensor := range tensors {
			candidate[name] = tensor
		}
		if pad > 0 {
			candidate["000_alignment_pad"] = safetensors.Tensor{Dtype: "U8", Shape: []int{pad}, Data: make([]byte, pad)}
		}
		blob, err := safetensors.Encode(candidate)
		if err != nil {
			t.Fatalf("Encode: %v", err)
		}
		parsed, err := safetensors.Parse(blob)
		if err != nil {
			t.Fatalf("Parse: %v", err)
		}
		ok := true
		for _, name := range alignedNames {
			tns, exists := parsed[name]
			if !exists {
				t.Fatalf("aligned tensor %s missing from encoded checkpoint", name)
			}
			align := uintptr(2)
			if tns.Dtype == "U32" {
				align = 4
			}
			if offsetOf(blob, tns)%align != 0 {
				ok = false
				break
			}
		}
		if ok {
			return blob
		}
	}
	t.Fatal("could not build an aligned safetensors fixture")
	return nil
}
