// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

// TestComposedArchRegistered confirms init() registered every Qwen 3.6 hybrid model_type (the wrapper ids,
// their nested text_config aliases, qwen3_6/qwen3_6_moe, qwen3_next, and the generic composed/hybrid ids)
// with a Composed hook — the reactive loader can resolve each to the composed loader, and engine/metal's
// former hardcoded fallback switch (which named this exact id set) needs no equivalent any more.
func TestComposedArchRegistered(t *testing.T) {
	for _, mt := range []string{
		"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text",
		"qwen3_6", "qwen3_6_moe", "qwen3_next",
		"composed", "hybrid",
	} {
		spec, ok := model.LookupArch(mt)
		if !ok {
			t.Fatalf("model_type %q not registered — composed init() did not run", mt)
		}
		if spec.Composed == nil {
			t.Fatalf("model_type %q registered without a Composed hook", mt)
		}
	}
}

// TestComposedArchRoundTrip is the registration round-trip: model_type → LookupArch → ArchSpec → the
// Composed hook builds a serve-ready TokenModel from a synthetic hybrid checkpoint, and the two bookends +
// the decode seam drive it (Embed → DecodeForward → Head → vocab logits). No engine/metal switch.
func TestComposedArchRoundTrip(t *testing.T) {
	spec, ok := model.LookupArch("qwen3_5")
	if !ok || spec.Composed == nil {
		t.Fatal("qwen3_5 composed arch not registered")
	}
	ts, cfg := mkHybridCheckpoint()
	tm, err := spec.Composed(ts, cfg)
	if err != nil {
		t.Fatalf("Composed hook: %v", err)
	}
	if tm.Vocab() != 32 {
		t.Fatalf("Vocab() = %d, want 32", tm.Vocab())
	}
	emb, err := tm.Embed(1)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	hs, err := tm.DecodeForward([][]byte{emb})
	if err != nil {
		t.Fatalf("DecodeForward: %v", err)
	}
	logits, err := tm.Head(hs[0])
	if err != nil {
		t.Fatalf("Head: %v", err)
	}
	if len(logits) != 32*2 {
		t.Fatalf("Head logits = %d bytes, want %d (vocab bf16)", len(logits), 32*2)
	}
	t.Log("qwen3_5 → LookupArch → Composed hook → ComposedTokenModel: registration reaches the composed loader")
}

// TestLoadComposedDirRoundTrip is the end-to-end neutral path: a checkpoint dir (config.json carrying
// model_type qwen3_5 + a model.safetensors) is loaded by model.LoadComposedDir straight to a TokenModel via
// the registry — the registry-driven routing that replaces the backend's hardcoded switch. It also pins
// that model.Load REJECTS the same dir (a composed arch must not fall into the transformer Assemble path).
func TestLoadComposedDirRoundTrip(t *testing.T) {
	ts, _ := mkHybridCheckpoint()
	dir := t.TempDir()
	cfg := `{"model_type":"qwen3_5","hidden_size":8,"num_hidden_layers":4,"intermediate_size":16,` +
		`"num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,"vocab_size":32,"rms_norm_eps":1e-5,` +
		`"rope_theta":1000000,"partial_rotary_factor":0.5,"full_attention_interval":2}`
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), cfg); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	blob, err := safetensors.Encode(ts)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if err := coreio.Local.Write(core.PathJoin(dir, "model.safetensors"), string(blob)); err != nil {
		t.Fatalf("write model.safetensors: %v", err)
	}

	tm, ok, err := model.LoadComposedDir(dir)
	if err != nil {
		t.Fatalf("LoadComposedDir: %v", err)
	}
	if !ok {
		t.Fatal("LoadComposedDir ok=false — the qwen3_5 Composed hook was not reached")
	}
	if tm.Vocab() != 32 {
		t.Fatalf("Vocab() = %d, want 32", tm.Vocab())
	}
	if _, _, lerr := model.Load(dir); lerr == nil {
		t.Fatal("model.Load must reject a composed arch (route it to LoadComposedDir), not Assemble it")
	}
	t.Log("dir → LoadComposedDir → registry → Composed hook → TokenModel; model.Load rejects the composed arch")
}

// TestLoadComposedDir_RoutingContract pins the two branches engine/metal/load.go now delegates to the
// registry (the "registry over allow-list" contract): a registered composed model_type resolves through
// the ArchSpec.Composed hook (ok=true — the engine routes here, no hardcoded model_type switch), and a
// non-composed model_type declines (ok=false) so the caller falls through to the transformer loader. A
// future qwenX that registers a Composed hook therefore loads with zero engine edits.
func TestLoadComposedDir_RoutingContract(t *testing.T) {
	// Every registered composed arch is reachable through the registry — this is the coverage the engine
	// switch used to hardcode. qwen3_5 additionally has the full dir round-trip above.
	for _, mt := range []string{
		"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text",
		"qwen3_6", "qwen3_6_moe", "qwen3_next",
		"composed", "hybrid",
	} {
		if spec, ok := model.LookupArch(mt); !ok || spec.Composed == nil {
			t.Fatalf("composed model_type %q not reachable through the registry (LoadComposedDir would miss it)", mt)
		}
	}
	// A non-composed model_type: LoadComposedDir returns ok=false without touching tensors, so the engine
	// falls through to Load — the fall-through the old switch expressed as its default case.
	dir := t.TempDir()
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"),
		`{"model_type":"llama","hidden_size":8,"num_hidden_layers":2}`); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	tm, ok, err := model.LoadComposedDir(dir)
	if err != nil || ok || tm != nil {
		t.Fatalf("LoadComposedDir(llama) = (%v, %v, %v), want (nil, false, nil) so the caller falls through", tm, ok, err)
	}
}
