// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	g4 "dappco.re/go/inference/model/gemma4"
)

// cache_plan_test.go gates the #48 cache-plan instrument against three
// session shapes: a plain dense session (every layer native, the recorded
// ICB lane), a TQ-armed MoE session with a native sliding owner AND a shared
// follower beside the TQ owner (the state-lane carrier, tq_kv_state.go), and
// a paged token-model session (the stepToken lane's device-paged pool). Each
// asserts on the RETURNED CachePlanSummary's counts/bytes — never on the
// nativeTraceLog line's text.

// TestLayerCacheKindString_Good gates every LayerCacheKind's String() branch
// plus the out-of-range fallback.
func TestLayerCacheKindString_Good(t *testing.T) {
	cases := []struct {
		k    LayerCacheKind
		want string
	}{
		{CacheKindRecurrent, "recurrent"},
		{CacheKindShared, "shared"},
		{CacheKindNative, "native"},
		{CacheKindQ8, "q8"},
		{CacheKindTurboQuant, "turboquant"},
		{LayerCacheKind(99), "unknown"},
	}
	for _, c := range cases {
		if got := c.k.String(); got != c.want {
			t.Errorf("LayerCacheKind(%d).String() = %q, want %q", c.k, got, c.want)
		}
	}
}

// TestCachePlanDense_Good gates fixture (a): a plain dense (non-MoE)
// all-global session, no -kv-cache request. Every layer must classify
// native on the recorded-ICB lane, with byte totals matching the session's
// OWN resident buffers exactly (not a separately re-derived formula) — head
// dim 128 never qualifies for the ICB q8 default (kvQ8ICBOn requires 256/512),
// so this fixture is q8-default-proof.
func TestCachePlanDense_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := tqTestQuantModel(t)
	const maxLen = 32
	sess, err := newArchQuantSessionShardsWithHeadConfig(g, arch, maxLen, nil, nil, archSessionConfig{})
	if err != nil {
		t.Fatalf("native session: %v", err)
	}
	if sess.state.icb == nil {
		t.Fatal("fixture expects the recorded-ICB lane (dense, non-MoE, non-trace)")
	}

	plan := buildCachePlan(arch, &sess.state, "")
	if plan.Lane != "icb" {
		t.Fatalf("Lane = %q, want %q", plan.Lane, "icb")
	}
	if plan.NativeCount != len(arch.Layer) {
		t.Fatalf("NativeCount = %d, want %d (every layer native)", plan.NativeCount, len(arch.Layer))
	}
	if plan.Q8Count != 0 || plan.TQCount != 0 || plan.SharedCount != 0 || plan.RecurrentCount != 0 {
		t.Fatalf("expected only native layers, got %+v", plan)
	}
	if plan.PagedCount != 0 {
		t.Fatalf("PagedCount = %d, want 0 (the ICB lane never pages)", plan.PagedCount)
	}
	if plan.LinearCount != len(arch.Layer) {
		t.Fatalf("LinearCount = %d, want %d", plan.LinearCount, len(arch.Layer))
	}
	// Byte cross-check: the plan's per-layer total must equal the session's
	// OWN kCaches/vCaches buffer lengths — not a hand-rolled geometry formula.
	var wantBytes int64
	for li := range arch.Layer {
		lp := plan.Layers[li]
		if lp.Kind != CacheKindNative {
			t.Fatalf("layer %d: Kind = %s, want native", li, lp.Kind)
		}
		want := safeBufLen(sess.state.icb.kCaches[li]) + safeBufLen(sess.state.icb.vCaches[li])
		if lp.Bytes != want {
			t.Fatalf("layer %d: Bytes = %d, want %d (session's own kCaches/vCaches length)", li, lp.Bytes, want)
		}
		wantBytes += want
	}
	if plan.NativeBytes != wantBytes {
		t.Fatalf("NativeBytes = %d, want %d", plan.NativeBytes, wantBytes)
	}
	if wantBytes == 0 {
		t.Fatal("fixture is empty — the cross-check would pass vacuously")
	}
	t.Logf("dense plan: %s", plan.logLine())
}

// TestCachePlanTurboQuantMoEShared_Good gates fixture (b): a TQ-armed MoE
// session shaped exactly like arch_session_tq_moe_test.go's
// TestArchQuantSessionTurboQuantMoEShared_Good fixture (layer 0 full/global
// owner, layer 1 sliding owner, layer 2 shares layer 0's KV) — TQ owners AND
// native elsewhere AND a shared follower in the SAME session, on the
// state-lane carrier (no arch ICB exists for a MoE stack).
func TestCachePlanTurboQuantMoEShared_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const numLayers = 3
	quant := &model.QuantConfig{GroupSize: 64, Bits: 4, Overrides: map[string]model.ModuleQuant{}}
	for i := range numLayers {
		for _, m := range []string{"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "router.proj"} {
			quant.Overrides[core.Sprintf("model.layers.%d.%s", i, m)] = model.ModuleQuant{GroupSize: 64, Bits: 8}
		}
	}
	cfg := g4.Config{
		HiddenSize: 64, NumHiddenLayers: numLayers, IntermediateSize: 128,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 128, VocabSize: 32, RMSNormEps: 1e-6,
		SlidingWindow: 8, LayerTypes: []string{"full_attention", "sliding_attention", "full_attention"},
		NumKVSharedLayers: 1,
		EnableMoEBlock:    true, NumExperts: 4, TopKExperts: 2, MoEIntermediateSize: 64,
		Quantization: quant,
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if arch.Layer[2].OwnsCache() || arch.Layer[2].KVShareFrom != 0 {
		t.Fatalf("fixture wants layer 2 sharing layer 0's KV, got %+v", arch.Layer[2])
	}
	ts := moeQuantTensors(t, arch, quant)
	lm, err := model.Assemble(ts, arch, model.StandardWeightNames())
	if err != nil {
		t.Fatalf("model.Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, quant.GroupSize, quant.Bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	const maxLen = 32
	sess, err := newArchQuantSessionShardsWithHeadConfig(g, arch, maxLen, nil, nil, archSessionConfig{kvCacheMode: "turboquant:4"})
	if err != nil {
		t.Fatalf("turboquant session: %v", err)
	}
	if sess.state.icb != nil {
		t.Fatal("MoE session recorded an arch ICB — the state carrier gates assume stepToken decode")
	}
	if !sess.state.tqStateArmed() {
		t.Fatal("session did not arm the state-lane TQ carrier")
	}

	plan := buildCachePlan(arch, &sess.state, "turboquant:4")
	if plan.Lane != "state-tq" {
		t.Fatalf("Lane = %q, want %q", plan.Lane, "state-tq")
	}
	if len(plan.Layers) != numLayers {
		t.Fatalf("len(Layers) = %d, want %d", len(plan.Layers), numLayers)
	}
	if plan.Layers[0].Kind != CacheKindTurboQuant {
		t.Fatalf("layer 0 (global owner): Kind = %s, want turboquant", plan.Layers[0].Kind)
	}
	if plan.Layers[0].Bytes <= 0 {
		t.Fatal("layer 0: TurboQuant Bytes must be > 0")
	}
	if plan.Layers[1].Kind != CacheKindNative {
		t.Fatalf("layer 1 (sliding owner): Kind = %s, want native (sliding never arms TQ)", plan.Layers[1].Kind)
	}
	if !plan.Layers[1].Deferred {
		t.Fatal("layer 1: expected Deferred — the lb linear cache has not been touched yet at build time")
	}
	if plan.Layers[2].Kind != CacheKindShared {
		t.Fatalf("layer 2 (shares layer 0): Kind = %s, want shared", plan.Layers[2].Kind)
	}
	if plan.Layers[2].Owner != 0 {
		t.Fatalf("layer 2: Owner = %d, want 0", plan.Layers[2].Owner)
	}
	if plan.Layers[2].Bytes != 0 {
		t.Fatalf("layer 2: Bytes = %d, want 0 (a shared follower owns no rows)", plan.Layers[2].Bytes)
	}
	if plan.TQCount != 1 || plan.NativeCount != 1 || plan.SharedCount != 1 || plan.Q8Count != 0 || plan.RecurrentCount != 0 {
		t.Fatalf("counts = %+v, want TQ=1 native=1 shared=1 q8=0 recurrent=0", plan)
	}
	if plan.PagedCount != 0 {
		t.Fatalf("PagedCount = %d, want 0 (a state-armed session declines the paged pool wholesale)", plan.PagedCount)
	}
	if plan.LinearCount != 2 {
		t.Fatalf("LinearCount = %d, want 2 (both owning layers)", plan.LinearCount)
	}
	if plan.PagedPoolBytes != 0 {
		t.Fatalf("PagedPoolBytes = %d, want 0", plan.PagedPoolBytes)
	}
	// Residency cross-check against the carrier's own buffers (the tq_kv_state.go
	// receipt arch_session_tq_moe_test.go already gates decode correctness on).
	wantTQBytes := safeBufLen(sess.state.kvTQState.kCaches[0]) + safeBufLen(sess.state.kvTQState.vCaches[0]) +
		safeBufLen(sess.state.kvTQState.set.kGammas[0]) + safeBufLen(sess.state.kvTQState.set.vGammas[0])
	if plan.TQBytes != wantTQBytes {
		t.Fatalf("TQBytes = %d, want %d (kvTQState's own layer-0 buffers)", plan.TQBytes, wantTQBytes)
	}
	t.Logf("moe-shared plan: %s", plan.logLine())
}

// TestCachePlanPagedTokenModel_Good gates fixture (c): a paged token-model
// session — the LoadTokenModelDir path (NewQuantTokenModel + OpenSession,
// not a direct newArchQuantSessionShardsWithHeadConfig call), no -kv-cache
// request, over a MoE arch (the stepToken/paged decode lane — 26B's shape).
// The kind (native vs q8) is read from the ambient kvQ8Enabled default
// rather than hardcoded, since sdpa_paged.go's LTHN_KV_Q8 switch is
// environment-controlled and this fixture's geometry qualifies for q8
// either way — what this test actually gates is the PAGED/LINEAR axis.
func TestCachePlanPagedTokenModel_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := tqMoETestQuantModel(t)
	const maxLen = 32
	tm, err := NewQuantTokenModel(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewQuantTokenModel: %v", err)
	}
	stepper, err := tm.OpenSession()
	if err != nil {
		t.Fatalf("OpenSession: %v", err)
	}
	sess, ok := stepper.(*ArchSession)
	if !ok {
		t.Fatalf("OpenSession returned %T, want *ArchSession", stepper)
	}
	if sess.state.icb != nil {
		t.Fatal("fixture expects the stepToken lane (MoE, no -kv-cache) — an ICB should not have recorded")
	}
	if sess.state.tqStateArmed() {
		t.Fatal("fixture requested no -kv-cache — the TQ state carrier must stay unarmed")
	}
	if !sess.state.hasDevicePagedKV() {
		t.Fatal("fixture expects the device-paged pool alive (MoE stepToken decode reads it)")
	}

	plan := buildCachePlan(arch, &sess.state, "")
	if plan.Lane != "stepToken" {
		t.Fatalf("Lane = %q, want %q", plan.Lane, "stepToken")
	}
	if plan.PagedCount != len(arch.Layer) {
		t.Fatalf("PagedCount = %d, want %d (every owning layer reads the paged pool)", plan.PagedCount, len(arch.Layer))
	}
	if plan.LinearCount != 0 {
		t.Fatalf("LinearCount = %d, want 0", plan.LinearCount)
	}
	if plan.TQCount != 0 || plan.SharedCount != 0 || plan.RecurrentCount != 0 {
		t.Fatalf("expected no TQ/shared/recurrent layers, got %+v", plan)
	}
	wantQ8 := kvQ8Enabled // MoE => pagedDecodeLane=true; this fixture's geometry (nHeads=2*kvHeads, hd=128<=256, kvd%64==0) always qualifies
	if wantQ8 {
		if plan.Q8Count != len(arch.Layer) || plan.NativeCount != 0 {
			t.Fatalf("LTHN_KV_Q8 default enabled: counts = %+v, want q8=%d native=0", plan, len(arch.Layer))
		}
	} else {
		if plan.NativeCount != len(arch.Layer) || plan.Q8Count != 0 {
			t.Fatalf("LTHN_KV_Q8 disabled: counts = %+v, want native=%d q8=0", plan, len(arch.Layer))
		}
	}
	for li, lp := range plan.Layers {
		if !lp.Paged {
			t.Fatalf("layer %d: Paged = false, want true", li)
		}
	}
	t.Logf("paged token-model plan (LTHN_KV_Q8 default enabled=%v): %s", wantQ8, plan.logLine())
}
