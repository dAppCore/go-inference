// SPDX-Licence-Identifier: EUPL-1.2

package memory

import (
	"testing"

	mp "dappco.re/go/inference/modelpack"
	"dappco.re/go/inference/profile"
)

// TestMemory_KVWidthPerLayer_PackFallback covers the Pack branch of
// kvWidthPerLayer: when ModelInfo carries no KV dims (nil, or zero heads/dim)
// but the Pack declares num_kv_heads * head_dim, the per-layer width comes from
// the Pack. Returns 0 only when neither source declares the GQA dims.
func TestMemory_KVWidthPerLayer_PackFallback(t *testing.T) {
	// Pack-only KV dims (no ModelInfo) → Pack width.
	packOnly := Input{Pack: &mp.ModelPack{NumKVHeads: 8, HeadDim: 256}}
	if got := kvWidthPerLayer(packOnly); got != 8*256 {
		t.Fatalf("kvWidthPerLayer(pack-only) = %d, want %d", got, 8*256)
	}
	// ModelInfo present but without KV dims → still falls through to the Pack.
	modelNoKV := Input{
		ModelInfo: &ModelInfo{Architecture: "qwen2", NumLayers: 28},
		Pack:      &mp.ModelPack{NumKVHeads: 4, HeadDim: 128},
	}
	if got := kvWidthPerLayer(modelNoKV); got != 4*128 {
		t.Fatalf("kvWidthPerLayer(model-no-kv, pack-kv) = %d, want %d", got, 4*128)
	}
	// ModelInfo KV dims take precedence over the Pack when both are present.
	both := Input{
		ModelInfo: &ModelInfo{NumKVHeads: 2, HeadDim: 64},
		Pack:      &mp.ModelPack{NumKVHeads: 8, HeadDim: 256},
	}
	if got := kvWidthPerLayer(both); got != 2*64 {
		t.Fatalf("kvWidthPerLayer(both) = %d, want ModelInfo %d", got, 2*64)
	}
	// Neither declares KV dims → unknown width.
	if got := kvWidthPerLayer(Input{Pack: &mp.ModelPack{}}); got != 0 {
		t.Fatalf("kvWidthPerLayer(no-kv) = %d, want 0", got)
	}
}

// TestMemory_PerTokenKVBytes_Shapes covers perTokenKVBytes's width derivation:
// the GQA-declared width path and the hidden_size fallback. perTokenKVBytes
// carries no zero-shape guards (it can't return 0 — see
// TestMemory_PerTokenKVBytes_NeverZero and TestMemory_KVEstimateShape_AlwaysPositive
// for that invariant); these cases pin that the GQA width is narrower than the
// hidden-size fallback when the model declares its KV dims.
func TestMemory_PerTokenKVBytes_Shapes(t *testing.T) {
	plan := Plan{MachineClass: ClassApple96GB, CacheMode: KVCacheModeFP16}
	input := Input{ModelInfo: &ModelInfo{NumLayers: 32, HiddenSize: 3072}}
	if got := perTokenKVBytes(plan, input); got == 0 {
		t.Fatal("perTokenKVBytes(known shape) = 0, want > 0")
	}
	// Pack-declared GQA width is used in place of hidden_size.
	gqaInput := Input{Pack: &mp.ModelPack{NumLayers: 28, HiddenSize: 2048, NumKVHeads: 4, HeadDim: 256}}
	wide := perTokenKVBytes(plan, Input{Pack: &mp.ModelPack{NumLayers: 28, HiddenSize: 2048}})
	narrow := perTokenKVBytes(plan, gqaInput)
	if narrow == 0 || wide == 0 {
		t.Fatalf("perTokenKVBytes = wide:%d narrow:%d, want both > 0", wide, narrow)
	}
	if narrow >= wide {
		t.Fatalf("GQA per-token KV = %d, want < hidden-width %d (4*256=1024 < 2048)", narrow, wide)
	}
}

// TestMemory_PerTokenKVBytes_UnknownClassStillResolves pins the documented
// contract on the ClassUnknown path: with neither a ModelInfo/Pack shape nor a
// named class, kvEstimateShape resolves the default (48/5120) shape, so the
// per-token cost is total and non-zero. (The full totality matrix lives in
// TestMemory_PerTokenKVBytes_NeverZero.)
func TestMemory_PerTokenKVBytes_UnknownClassStillResolves(t *testing.T) {
	plan := Plan{MachineClass: ClassUnknown, CacheMode: KVCacheModeFP16}
	if got := perTokenKVBytes(plan, Input{}); got == 0 {
		t.Fatal("perTokenKVBytes(unknown class, no model) = 0, want > 0 (class default shape)")
	}
}

// TestMemory_FitContextLength_Guards walks every early-return and fallback branch
// of fitContextLength that the public context-fit tests do not isolate:
//   - missing weight bytes / over-budget weights → 0
//   - a valid shape → a positive fit (perToken is always > 0, so there is no
//     zero-per-token exit; the divisor is never zero)
//   - a fit below the 4096 alignment floor → 0
//   - modelContext<=0 → ceiling falls back to plan.ContextLength
func TestMemory_FitContextLength_Guards(t *testing.T) {
	base := Plan{
		MachineClass:     ClassApple96GB,
		MemoryLimitBytes: 80 * GiB,
		ContextLength:    131072,
		ParallelSlots:    1,
		CacheMode:        KVCacheModeFP16,
	}
	shapedInput := Input{ModelInfo: &ModelInfo{NumLayers: 32, HiddenSize: 3072}}

	// No weight bytes → cannot compute a real fit → 0.
	if got := fitContextLength(base, 262144, 0, shapedInput); got != 0 {
		t.Fatalf("fitContextLength(no weight bytes) = %d, want 0", got)
	}
	// Weights exceed the memory limit → no post-weights budget → 0.
	if got := fitContextLength(base, 262144, base.MemoryLimitBytes+GiB, shapedInput); got != 0 {
		t.Fatalf("fitContextLength(weights over budget) = %d, want 0", got)
	}
	// A resolvable shape always yields a positive per-token KV (kvEstimateShape
	// resolves a class default when the model declares none), so a valid plan
	// produces a positive fit — there is no zero-per-token exit to hit.
	if got := fitContextLength(base, 262144, 8*GiB, shapedInput); got <= 0 {
		t.Fatalf("fitContextLength(valid) = %d, want > 0", got)
	}

	// fit below the alignment floor → 0. A tiny post-weights budget with a heavy
	// per-token KV makes kvBudget/(perToken*slots) < 4096.
	tight := base
	tight.MemoryLimitBytes = 8*GiB + 16*1024*1024 // 16 MiB of headroom over 8 GiB weights
	heavyKV := Input{ModelInfo: &ModelInfo{NumLayers: 80, HiddenSize: 8192}}
	if got := fitContextLength(tight, 262144, 8*GiB, heavyKV); got != 0 {
		t.Fatalf("fitContextLength(sub-alignment fit) = %d, want 0 (below the 4096 floor)", got)
	}

	// modelContext<=0 → ceiling = plan.ContextLength. With a generous budget the
	// raw fit exceeds the plan's 131072 baseline, so the ceiling (plan.ContextLength)
	// is what caps the result rather than the model's (absent) declared maximum.
	big := base
	big.MemoryLimitBytes = 512 * GiB
	big.ContextLength = 8192 // a low ceiling so the cap is observable
	got := fitContextLength(big, 0, 8*GiB, shapedInput)
	if got != 8192 {
		t.Fatalf("fitContextLength(modelContext=0) = %d, want plan.ContextLength ceiling 8192", got)
	}
}

// TestMemory_FitContextLength_SlotsZeroDefaultsToOne pins the slots==0 → 1
// normalisation: a plan with ParallelSlots 0 must divide the KV budget by one
// slot, not by zero. The derived fit equals the same plan with ParallelSlots 1.
func TestMemory_FitContextLength_SlotsZeroDefaultsToOne(t *testing.T) {
	base := Plan{
		MachineClass:     ClassApple96GB,
		MemoryLimitBytes: 80 * GiB,
		ContextLength:    131072,
		CacheMode:        KVCacheModeFP16,
	}
	input := Input{ModelInfo: &ModelInfo{NumLayers: 32, HiddenSize: 3072}}

	zeroSlots := base
	zeroSlots.ParallelSlots = 0
	oneSlot := base
	oneSlot.ParallelSlots = 1

	gotZero := fitContextLength(zeroSlots, 262144, 8*GiB, input)
	gotOne := fitContextLength(oneSlot, 262144, 8*GiB, input)
	if gotZero <= 0 {
		t.Fatalf("fitContextLength(slots=0) = %d, want > 0 (normalised to one slot)", gotZero)
	}
	if gotZero != gotOne {
		t.Fatalf("fitContextLength(slots=0) = %d, want == slots=1 result %d", gotZero, gotOne)
	}
}

// TestMemory_ConcurrentContextsThatFit_Guards covers the early returns of
// concurrentContextsThatFit that the public no-inversion tests do not isolate:
// a non-positive modelContext, missing weight bytes, and over-budget weights
// each return 0 — telling NewPlan to keep the honest one-slot default. (There
// is no zero-per-token exit: perToken is always > 0 and modelContext > 0 here,
// so windowBytes and the divisor are never zero.) A single window that fits
// returns at least 1.
func TestMemory_ConcurrentContextsThatFit_Guards(t *testing.T) {
	base := Plan{
		MachineClass:     ClassApple96GB,
		MemoryLimitBytes: 80 * GiB,
		CacheMode:        KVCacheModeFP16,
	}
	input := Input{ModelInfo: &ModelInfo{NumLayers: 32, HiddenSize: 3072}}

	if got := concurrentContextsThatFit(base, 0, 8*GiB, input); got != 0 {
		t.Fatalf("concurrentContextsThatFit(modelContext=0) = %d, want 0", got)
	}
	if got := concurrentContextsThatFit(base, 32768, 0, input); got != 0 {
		t.Fatalf("concurrentContextsThatFit(no weight bytes) = %d, want 0", got)
	}
	if got := concurrentContextsThatFit(base, 32768, base.MemoryLimitBytes+GiB, input); got != 0 {
		t.Fatalf("concurrentContextsThatFit(weights over budget) = %d, want 0", got)
	}
	// A model whose single context window exceeds the post-weights budget still
	// returns at least one slot (the floor), not zero.
	tight := base
	tight.MemoryLimitBytes = 8*GiB + 64*1024*1024
	huge := Input{ModelInfo: &ModelInfo{NumLayers: 80, HiddenSize: 8192}}
	if got := concurrentContextsThatFit(tight, 262144, 8*GiB, huge); got != 1 {
		t.Fatalf("concurrentContextsThatFit(one window over budget) = %d, want 1 (floor)", got)
	}
	// A roomy machine fits more than one window.
	roomy := base
	roomy.MemoryLimitBytes = 512 * GiB
	if got := concurrentContextsThatFit(roomy, 8192, 8*GiB, input); got < 2 {
		t.Fatalf("concurrentContextsThatFit(roomy, small ctx) = %d, want >= 2", got)
	}
}

// TestMemory_EstimateKVCacheBytesWithProfile_GenerationProfileHint pins the two
// live early-return gates of estimateKVCacheBytesWithProfile via the profile
// hint: a generation profile on a resolvable shape returns a positive estimate,
// and an embedding profile disables the cache (→ 0) even with a positive context
// and a resolvable shape. (There is no zero-shape gate — kvEstimateShape always
// resolves a positive shape; see TestMemory_KVEstimateShape_AlwaysPositive.)
func TestMemory_EstimateKVCacheBytesWithProfile_GenerationProfileHint(t *testing.T) {
	plan := Plan{MachineClass: ClassApple96GB, ContextLength: 8192}
	genHint := &profile.ModelArchitectureProfile{ID: "qwen2"}
	input := Input{ModelInfo: &ModelInfo{NumLayers: 32, HiddenSize: 3072}}
	if got := estimateKVCacheBytesWithProfile(plan, input, KVCacheModeFP16, genHint); got == 0 {
		t.Fatal("estimateKVCacheBytesWithProfile(generation hint) = 0, want > 0")
	}
	// An embedding profile hint disables the cache → 0, even with a positive
	// context and a resolvable shape.
	embedHint := &profile.ModelArchitectureProfile{ID: "bert", Embeddings: true}
	if got := estimateKVCacheBytesWithProfile(plan, input, KVCacheModeFP16, embedHint); got != 0 {
		t.Fatalf("estimateKVCacheBytesWithProfile(embedding hint) = %d, want 0", got)
	}
}

// TestMemory_EstimateKVCacheBytesWithProfile_ResolvableShape pins that a model
// declaring no KV dims still produces a positive estimate: kvEstimateShape
// resolves the class-default shape (the defaults are unconditional), so there is
// no zero-shape path. (The totality invariant itself is locked in
// TestMemory_KVEstimateShape_AlwaysPositive.)
func TestMemory_EstimateKVCacheBytesWithProfile_ResolvableShape(t *testing.T) {
	plan := Plan{MachineClass: ClassApple16GB, ContextLength: 4096}
	input := Input{ModelInfo: &ModelInfo{Architecture: "qwen2"}} // no dims → class default shape
	if got := estimateKVCacheBytesWithProfile(plan, input, KVCacheModeFP16, nil); got == 0 {
		t.Fatal("estimateKVCacheBytesWithProfile(class-default shape) = 0, want > 0")
	}
}

// TestMemory_ApplyArchitectureHints_NormalizeFallback covers the
// `else if architecture != ""` branch where the profile registry has no entry
// (profileHint nil) so the architecture string is normalised directly. An
// architecture the registry does not know normalises to a value the hint switch
// does not match, so no notes are emitted and the plan is unchanged — proving
// the normalise-fallback path runs without a registry hit.
func TestMemory_ApplyArchitectureHints_NormalizeFallback(t *testing.T) {
	plan := Plan{MachineClass: ClassApple96GB, ContextLength: 131072}
	before := len(plan.Notes)
	// Unknown architecture, nil profile hint → normalise fallback, no switch hit.
	applyArchitectureHints(&plan, "totally-unknown-arch-xyz", nil)
	if len(plan.Notes) != before {
		t.Fatalf("Notes grew to %d on an unknown architecture, want unchanged %d", len(plan.Notes), before)
	}
	// Empty architecture, nil hint → the normalise branch is skipped entirely.
	applyArchitectureHints(&plan, "", nil)
	if len(plan.Notes) != before {
		t.Fatalf("Notes grew to %d on empty architecture, want unchanged %d", len(plan.Notes), before)
	}
	// A normalise-only known architecture (nil hint, registry would normally
	// supply it) still emits its note via the NormalizeArchitecture fallback.
	q2 := Plan{MachineClass: ClassApple96GB, ContextLength: 131072}
	applyArchitectureHints(&q2, "qwen2", nil)
	if !hasNote(q2, "native Qwen decoder") {
		t.Fatalf("Notes = %+v, want qwen2 note via normalise fallback", q2.Notes)
	}
}

// TestMemory_UsesGenerationKVCacheWithProfile_LegacyLookup covers the legacy
// single-call registry path (profileHint nil, no Pack-resident profile): an
// embedding architecture named only by string is resolved through the registry
// and disables the generation cache. A generation architecture string keeps it
// enabled.
func TestMemory_UsesGenerationKVCacheWithProfile_LegacyLookup(t *testing.T) {
	// Pack with an embedding architecture string but NO cached ArchitectureProfile
	// and NO Embedding/Rerank struct → falls through to the registry lookup at the
	// bottom of the function, which finds bert (Embeddings=true) → false.
	packBert := Input{Pack: &mp.ModelPack{Architecture: "bert"}}
	if usesGenerationKVCacheWithProfile(packBert, nil) {
		t.Fatal("usesGenerationKVCacheWithProfile(bert string, legacy lookup) = true, want false")
	}
	// ModelInfo-only embedding architecture → same legacy lookup path → false.
	modelBert := Input{ModelInfo: &ModelInfo{Architecture: "bert"}}
	if usesGenerationKVCacheWithProfile(modelBert, nil) {
		t.Fatal("usesGenerationKVCacheWithProfile(bert ModelInfo, legacy lookup) = true, want false")
	}
	// A rerank architecture string → registry rerank=true → false.
	modelRerank := Input{ModelInfo: &ModelInfo{Architecture: "bert_rerank"}}
	if usesGenerationKVCacheWithProfile(modelRerank, nil) {
		t.Fatal("usesGenerationKVCacheWithProfile(bert_rerank ModelInfo) = true, want false")
	}
	// A generation architecture string → registry generation → true.
	if !usesGenerationKVCacheWithProfile(Input{ModelInfo: &ModelInfo{Architecture: "qwen2"}}, nil) {
		t.Fatal("usesGenerationKVCacheWithProfile(qwen2 ModelInfo) = false, want true")
	}
	// An architecture the registry does not know → defaults to generation (true).
	if !usesGenerationKVCacheWithProfile(Input{ModelInfo: &ModelInfo{Architecture: "unknown-xyz"}}, nil) {
		t.Fatal("usesGenerationKVCacheWithProfile(unknown arch) = false, want true (default)")
	}
}

// TestMemory_ApplyGenericMoEResidency_NilPlanAndGuards covers the guard returns
// of applyGenericMoEResidency: a nil plan is a no-op (must not panic), a nil
// profile hint is a no-op, and a non-MoE profile hint is a no-op. Only an MoE
// profile hint installs the residency plan.
func TestMemory_ApplyGenericMoEResidency_NilPlanAndGuards(t *testing.T) {
	// Nil plan → early return, no panic.
	applyGenericMoEResidency(nil, &mp.ModelPack{Architecture: "qwen3_moe"}, &profile.ModelArchitectureProfile{ID: "qwen3_moe", MoE: true})

	// Nil profile hint → no residency installed.
	plan := Plan{MachineClass: ClassApple96GB}
	applyGenericMoEResidency(&plan, &mp.ModelPack{Architecture: "qwen2"}, nil)
	if plan.ExpertResidency.Enabled {
		t.Fatal("applyGenericMoEResidency(nil hint) enabled residency, want untouched")
	}

	// Non-MoE profile hint → no residency installed.
	denseHint := &profile.ModelArchitectureProfile{ID: "qwen2", MoE: false}
	applyGenericMoEResidency(&plan, &mp.ModelPack{Architecture: "qwen2"}, denseHint)
	if plan.ExpertResidency.Enabled {
		t.Fatal("applyGenericMoEResidency(non-MoE hint) enabled residency, want untouched")
	}

	// MoE profile hint → residency installed with the lazy policy.
	moeHint := &profile.ModelArchitectureProfile{ID: "qwen3_moe", MoE: true}
	applyGenericMoEResidency(&plan, &mp.ModelPack{Architecture: "qwen3_moe"}, moeHint)
	if !plan.ExpertResidency.Enabled || plan.ExpertResidency.Mode != ExpertResidencyModeLazy {
		t.Fatalf("applyGenericMoEResidency(MoE hint) = %+v, want enabled lazy residency", plan.ExpertResidency)
	}
	if plan.ExpertResidency.Architecture != "qwen3_moe" {
		t.Fatalf("residency Architecture = %q, want qwen3_moe", plan.ExpertResidency.Architecture)
	}
}

// TestMemory_NewPlan_PackCachedArchitectureProfile covers the NewPlan branch that
// reuses a Pack's pre-resolved ArchitectureProfile instead of hitting the
// registry: a Pack carrying its own ArchitectureProfile (the native-load path)
// drives the architecture hints and MoE residency from the cached pointer. With
// no ModelInfo override, packArch == hintsArch, so the cached profile serves both
// the hints and the generation-cache call sites.
func TestMemory_NewPlan_PackCachedArchitectureProfile(t *testing.T) {
	cached := &profile.ModelArchitectureProfile{ID: "qwen3_moe", MoE: true}
	pack := mp.ModelPack{
		Architecture:        "qwen3_moe",
		ArchitectureProfile: cached,
		ContextLength:       32768,
		NumLayers:           48, HiddenSize: 4096, QuantBits: 4,
	}
	plan := NewPlan(Input{
		Device: DeviceInfo{MemorySize: 96 * GiB, MaxRecommendedWorkingSetSize: 90 * GiB},
		Pack:   &pack,
	})
	// MoE residency must be installed from the cached profile (MoE=true).
	if !plan.ExpertResidency.Enabled {
		t.Fatalf("ExpertResidency not enabled from cached MoE profile: %+v", plan.ExpertResidency)
	}
	if !hasNote(plan, "Qwen3-MoE") {
		t.Fatalf("Notes = %+v, want Qwen3-MoE architecture hint from the cached profile", plan.Notes)
	}
}

// TestMemory_NewPlan_PackCachedProfile_ModelInfoArchOverride covers the NewPlan
// branch where the Pack carries a cached ArchitectureProfile but ModelInfo
// overrides the architecture string to a DIFFERENT value. Because hintsArch
// (ModelInfo) differs from packArch (Pack), the cached pointer cannot serve the
// hints call site, so NewPlan resolves hintsPtr via a registry lookup while
// reusing the Pack pointer for the pack-precedence call sites — the
// `if packArch == hintsArch` false branch and the divergent-arch hints lookup.
func TestMemory_NewPlan_PackCachedProfile_ModelInfoArchOverride(t *testing.T) {
	cached := &profile.ModelArchitectureProfile{ID: "qwen3_moe", MoE: true}
	pack := mp.ModelPack{
		Architecture:        "qwen3_moe",
		ArchitectureProfile: cached,
		ContextLength:       32768,
		NumLayers:           48, HiddenSize: 4096, QuantBits: 4,
	}
	// ModelInfo declares a different (dense) architecture → hintsArch=qwen2,
	// packArch=qwen3_moe. The hints come from qwen2 (registry lookup), the MoE
	// residency from the Pack's cached qwen3_moe profile.
	plan := NewPlan(Input{
		Device:    DeviceInfo{MemorySize: 96 * GiB, MaxRecommendedWorkingSetSize: 90 * GiB},
		ModelInfo: &ModelInfo{Architecture: "qwen2"},
		Pack:      &pack,
	})
	// Hints reflect qwen2 (the ModelInfo override), resolved by a fresh lookup.
	if !hasNote(plan, "native Qwen decoder") {
		t.Fatalf("Notes = %+v, want qwen2 hint from the ModelInfo-override lookup", plan.Notes)
	}
	// Residency still reflects the Pack's cached MoE profile.
	if !plan.ExpertResidency.Enabled {
		t.Fatalf("ExpertResidency not enabled from cached Pack MoE profile: %+v", plan.ExpertResidency)
	}
}

// TestMemory_KVEstimateShape_AlwaysPositive is the regression lock on the
// totality invariant that the KV-budget derivations depend on:
// kvEstimateShape ALWAYS returns positive (layers, hidden), for every Class
// (named, unmapped, or empty) and every partial / zero / negative / mixed shape
// the model metadata can carry. This invariant is what makes the division in
// fitContextLength (kvBudget / (perToken*slots)) and concurrentContextsThatFit
// (kvBudget / windowBytes) safe without a per-token==0 guard, and what lets
// perTokenKVBytes and estimateKVCacheBytesWithProfile drop their zero-shape
// returns. If a future change to kvEstimateShape can yield a non-positive dim,
// this test fails — restore the guards (or the totality) before shipping it.
//
// The static proof: kvEstimateShape has exactly two return sites — the
// `if layers>0 && hidden>0` return (both strictly positive by the guard) and
// the trailing `switch class`, whose every branch (including default, which
// catches ClassUnknown and any unmapped Class) returns hardcoded positive
// constants. There is no third return, so the result is unconditionally
// (positive, positive). The table below pins representative + edge inputs; the
// proof carries the rest of the infinite int space.
func TestMemory_KVEstimateShape_AlwaysPositive(t *testing.T) {
	classes := []Class{
		ClassUnknown, ClassApple16GB, ClassApple24GB, ClassApple32GB,
		ClassApple64GB, ClassApple96GB, ClassApple128GB,
		Class("unmapped-class"), Class(""),
	}
	dims := []struct{ l, h int }{
		{0, 0}, {0, 4096}, {4096, 0},
		{-1, -1}, {-1, 4096}, {4096, -1},
		{-5, 0}, {0, -5}, {-5, -5},
		{32, 3072},
	}
	check := func(name string, in Input, c Class) {
		t.Helper()
		l, h := kvEstimateShape(in, c)
		if l <= 0 || h <= 0 {
			t.Errorf("kvEstimateShape(%s, class=%q) = (%d, %d), want both > 0 (totality invariant)", name, c, l, h)
		}
	}
	for _, c := range classes {
		check("empty", Input{}, c)
		for _, d := range dims {
			check("modelinfo", Input{ModelInfo: &ModelInfo{NumLayers: d.l, HiddenSize: d.h}}, c)
			check("pack", Input{Pack: &mp.ModelPack{NumLayers: d.l, HiddenSize: d.h}}, c)
			// Mixed: ModelInfo supplies one dim, the Pack the other (the Pack only
			// fills a dim the ModelInfo left at exactly 0).
			check("mixed", Input{
				ModelInfo: &ModelInfo{NumLayers: d.l},
				Pack:      &mp.ModelPack{HiddenSize: d.h},
			}, c)
			for _, d2 := range dims {
				check("both", Input{
					ModelInfo: &ModelInfo{NumLayers: d.l, HiddenSize: d.h},
					Pack:      &mp.ModelPack{NumLayers: d2.l, HiddenSize: d2.h},
				}, c)
			}
		}
	}
}

// TestMemory_PerTokenKVBytes_NeverZero locks the corollary the div-by-zero-free
// derivations rely on: perTokenKVBytes is always > 0. width is either the
// GQA-declared product (kvWidthPerLayer is >0-guarded on both factors) or the
// always-positive hidden from kvEstimateShape, and layers is always positive —
// so a partial, zero, negative or mixed KV shape on any Class still costs a
// positive number of bytes per token.
func TestMemory_PerTokenKVBytes_NeverZero(t *testing.T) {
	classes := []Class{
		ClassUnknown, ClassApple16GB, ClassApple128GB, Class("unmapped"), Class(""),
	}
	modes := []KVCacheMode{
		KVCacheModeFP16, KVCacheModeQ8, KVCacheModeKQ8VQ4,
		KVCacheModeTurboQuant, KVCacheMode("unknown-mode"),
	}
	shapes := []struct{ l, h int }{{0, 0}, {-1, 4096}, {4096, -1}, {-5, -5}, {32, 3072}}
	kvDims := []struct{ kvh, hd int }{{0, 0}, {-1, 4}, {4, -1}, {-4, -4}, {4, 256}, {4, 0}}
	for _, c := range classes {
		for _, m := range modes {
			plan := Plan{MachineClass: c, CacheMode: m}
			for _, s := range shapes {
				for _, kv := range kvDims {
					in := Input{ModelInfo: &ModelInfo{
						NumLayers: s.l, HiddenSize: s.h,
						NumKVHeads: kv.kvh, HeadDim: kv.hd,
					}}
					if got := perTokenKVBytes(plan, in); got == 0 {
						t.Errorf("perTokenKVBytes(class=%q mode=%q layers=%d hidden=%d kvh=%d hd=%d) = 0, want > 0", c, m, s.l, s.h, kv.kvh, kv.hd)
					}
				}
			}
		}
	}
}

// TestMemory_NewPlan_ModelInfoArchOverride_NoPackProfile covers the second
// registry lookup at the divergent-architecture path: ModelInfo declares one
// architecture, the Pack declares another, and the Pack has NO cached profile —
// so packArch != hintsArch && packArch != "" forces NewPlan to look the Pack
// architecture up in the registry (the `if packPtr == nil && packArch != hintsArch`
// branch). The MoE Pack architecture drives the residency via that lookup.
func TestMemory_NewPlan_ModelInfoArchOverride_NoPackProfile(t *testing.T) {
	pack := mp.ModelPack{
		Architecture:  "qwen3_moe", // MoE, but no cached ArchitectureProfile
		ContextLength: 32768,
		NumLayers:     48, HiddenSize: 4096, QuantBits: 4,
	}
	plan := NewPlan(Input{
		Device:    DeviceInfo{MemorySize: 96 * GiB, MaxRecommendedWorkingSetSize: 90 * GiB},
		ModelInfo: &ModelInfo{Architecture: "qwen2"}, // override → hintsArch != packArch
		Pack:      &pack,
	})
	// Hints reflect the ModelInfo qwen2 override.
	if !hasNote(plan, "native Qwen decoder") {
		t.Fatalf("Notes = %+v, want qwen2 hint from the ModelInfo override", plan.Notes)
	}
	// Residency comes from the Pack qwen3_moe architecture, resolved via the
	// divergent-arch registry lookup.
	if !plan.ExpertResidency.Enabled {
		t.Fatalf("ExpertResidency not enabled from the Pack-arch registry lookup: %+v", plan.ExpertResidency)
	}
}
