// SPDX-Licence-Identifier: EUPL-1.2

package memory

import (
	"strings"
	"testing"

	mp "dappco.re/go/inference/modelpack"
	"dappco.re/go/inference/profile"
)

func hasNote(plan Plan, fragment string) bool {
	for _, note := range plan.Notes {
		if strings.Contains(note, fragment) {
			return true
		}
	}
	return false
}

// TestMemory_NewPlan_Good is the canonical happy-path triplet member for the
// public NewPlan entry point: a measured 16GB device yields the M1-class plan
// with its rotating compact cache, single foreground slot, and allocator limits.
func TestMemory_NewPlan_Good(t *testing.T) {
	plan := NewPlan(Input{
		Device: DeviceInfo{
			Architecture:                 "apple7",
			MemorySize:                   16 * GiB,
			MaxRecommendedWorkingSetSize: 14 * GiB,
		},
	})
	if plan.MachineClass != ClassApple16GB {
		t.Fatalf("MachineClass = %q, want %q", plan.MachineClass, ClassApple16GB)
	}
	if plan.ContextLength != 8192 || plan.CachePolicy != KVCacheRotating || plan.CacheMode != KVCacheModeKQ8VQ4 {
		t.Fatalf("plan shape = %+v", plan)
	}
	if plan.BatchSize != 1 || plan.PrefillChunkSize != 512 {
		t.Fatalf("batch/prefill = %d/%d, want 1/512", plan.BatchSize, plan.PrefillChunkSize)
	}
	if plan.PromptCache {
		t.Fatal("PromptCache = true, want false on 16GB class")
	}
	if plan.MemoryLimitBytes == 0 || plan.CacheLimitBytes == 0 || plan.WiredLimitBytes == 0 {
		t.Fatalf("allocator limits unset: %+v", plan)
	}
}

func TestMemory_NewPlan_M3Ultra96GB(t *testing.T) {
	plan := NewPlan(Input{
		Device: DeviceInfo{
			Architecture:                 "apple9",
			MemorySize:                   96 * GiB,
			MaxRecommendedWorkingSetSize: 90 * GiB,
		},
	})
	if plan.MachineClass != ClassApple96GB {
		t.Fatalf("MachineClass = %q, want %q", plan.MachineClass, ClassApple96GB)
	}
	if plan.ContextLength != 131072 || plan.CacheMode != KVCacheModeDefault {
		t.Fatalf("shape = ctx:%d mode:%q, want default (bounded) cache", plan.ContextLength, plan.CacheMode)
	}
	if plan.BatchSize != 1 || plan.PrefillChunkSize != 4096 || plan.ParallelSlots != 1 {
		t.Fatalf("cold-start shape = batch %d prefill %d slots %d, want 1/4096/1 (no model → honest local default; concurrency capacity is derived once a model is known)", plan.BatchSize, plan.PrefillChunkSize, plan.ParallelSlots)
	}
	if !plan.PromptCache {
		t.Fatal("PromptCache = false, want true on 96GB class")
	}
}

func TestMemory_NewPlan_Apple64GBUsesWidePrefill(t *testing.T) {
	plan := NewPlan(Input{
		Device: DeviceInfo{
			Architecture:                 "apple9",
			MemorySize:                   64 * GiB,
			MaxRecommendedWorkingSetSize: 60 * GiB,
		},
	})
	if plan.MachineClass != ClassApple64GB {
		t.Fatalf("MachineClass = %q, want %q", plan.MachineClass, ClassApple64GB)
	}
	if plan.BatchSize != 1 || plan.PrefillChunkSize != 4096 || plan.ParallelSlots != 1 {
		t.Fatalf("cold-start shape = batch %d prefill %d slots %d, want 1/4096/1 (no model → honest local default)", plan.BatchSize, plan.PrefillChunkSize, plan.ParallelSlots)
	}
	if plan.CacheMode != KVCacheModeDefault || !plan.PromptCache {
		t.Fatalf("cache = mode %q prompt %t, want default (bounded) cache + prompt cache", plan.CacheMode, plan.PromptCache)
	}
}

func TestMemory_NewPlan_CapsContextToModelPack(t *testing.T) {
	pack := mp.ModelPack{ContextLength: 40960, QuantBits: 4}
	plan := NewPlan(Input{
		Device: DeviceInfo{MemorySize: 96 * GiB},
		Pack:   &pack,
	})
	if plan.ContextLength != 40960 {
		t.Fatalf("ContextLength = %d, want model cap 40960", plan.ContextLength)
	}
	if plan.ModelQuantization != 4 {
		t.Fatalf("quantization = model %d, want 4", plan.ModelQuantization)
	}
}

func TestMemory_NewPlan_QwenMoEHints(t *testing.T) {
	pack := mp.ModelPack{
		Architecture: "qwen3_moe", ContextLength: 32768,
		NumLayers: 48, HiddenSize: 4096, QuantBits: 4,
	}
	plan := NewPlan(Input{
		Device: DeviceInfo{MemorySize: 16 * GiB, MaxRecommendedWorkingSetSize: 13 * GiB},
		Pack:   &pack,
	})
	if plan.CacheMode != KVCacheModeKQ8VQ4 {
		t.Fatalf("CacheMode = %q, want %q for Qwen3-MoE on 16GB", plan.CacheMode, KVCacheModeKQ8VQ4)
	}
	if !hasNote(plan, "Qwen3-MoE") || !hasNote(plan, "expert") {
		t.Fatalf("Notes = %+v", plan.Notes)
	}
}

func TestMemory_NewPlan_MiniMaxArchitectureHintsAndCaps(t *testing.T) {
	pack := mp.ModelPack{
		Architecture:  "minimax_m2",
		ContextLength: 196608,
		NumLayers:     62, HiddenSize: 3072,
	}
	plan := NewPlan(Input{
		Device: DeviceInfo{MemorySize: 96 * GiB, MaxRecommendedWorkingSetSize: 90 * GiB},
		Pack:   &pack,
	})
	if plan.ContextLength != 32768 || plan.BatchSize != 1 {
		t.Fatalf("MiniMax shape = ctx:%d batch:%d, want 32768/1", plan.ContextLength, plan.BatchSize)
	}
	if !hasNote(plan, "MiniMax M2") {
		t.Fatalf("Notes = %+v, want MiniMax hint", plan.Notes)
	}
}

func TestMemory_NewPlan_BertEmbeddingDisablesGenerationCache(t *testing.T) {
	pack := mp.ModelPack{
		Architecture: "bert", ContextLength: 512,
		NumLayers: 12, HiddenSize: 768,
		Embedding:   &mp.ModelEmbeddingProfile{Dimension: 768, Pooling: "mean", MaxSequenceLength: 512},
		WeightBytes: 420 * 1024 * 1024,
		QuantBits:   16, QuantType: "fp16", QuantFamily: "dense",
	}
	plan := NewPlan(Input{
		Device: DeviceInfo{MemorySize: 16 * GiB, MaxRecommendedWorkingSetSize: 13 * GiB},
		Pack:   &pack,
	})
	if plan.ContextLength != 512 {
		t.Fatalf("ContextLength = %d, want BERT max 512", plan.ContextLength)
	}
	if plan.CachePolicy != KVCacheDefault || plan.CacheMode != KVCacheModeDefault || plan.PromptCache {
		t.Fatalf("cache policy = %+v, want disabled generation cache", plan)
	}
	if plan.EstimatedKVCacheBytes != 0 || plan.EstimatedKVCacheModeBytes != 0 {
		t.Fatalf("KV estimates = fp:%d mode:%d, want zero for encoder", plan.EstimatedKVCacheBytes, plan.EstimatedKVCacheModeBytes)
	}
	if plan.BatchSize < 4 || !hasNote(plan, "embedding encoder") {
		t.Fatalf("plan = %+v, want embedding throughput hint", plan)
	}
}

// TestMemory_NewPlan_Bad is the canonical bad-input triplet member for NewPlan:
// an empty Input (zero device memory, no model) must not panic or over-allocate
// — it falls back to the honest unknown-class local default.
func TestMemory_NewPlan_Bad(t *testing.T) {
	plan := NewPlan(Input{})
	if plan.MachineClass != ClassUnknown {
		t.Fatalf("MachineClass = %q, want unknown", plan.MachineClass)
	}
	if plan.ContextLength != defaultLocalContextLength || plan.BatchSize != 1 {
		t.Fatalf("fallback = %+v", plan)
	}
}

// TestMemory_NewPlan_Ugly is the canonical edge-case triplet member for NewPlan:
// a constrained model whose declared context (4096) is far below the device's
// 24GB-class baseline must cap DOWN to the metadata limit and emit a note — the
// awkward case where model metadata overrides the hardware tier.
func TestMemory_NewPlan_Ugly(t *testing.T) {
	plan := NewPlan(Input{
		Device:    DeviceInfo{MemorySize: 24 * GiB},
		ModelInfo: &ModelInfo{ContextLength: 4096, QuantBits: 2},
	})
	if plan.ContextLength != 4096 {
		t.Fatalf("ContextLength = %d, want metadata cap 4096", plan.ContextLength)
	}
	if len(plan.Notes) == 0 {
		t.Fatal("expected notes for constrained model metadata")
	}
}

func TestMemory_NewPlan_KVCacheQ8ForMiddleClass(t *testing.T) {
	plan := NewPlan(Input{
		Device: DeviceInfo{MemorySize: 32 * GiB, MaxRecommendedWorkingSetSize: 28 * GiB},
	})
	if plan.CacheMode != KVCacheModeQ8 {
		t.Fatalf("CacheMode = %q, want %q", plan.CacheMode, KVCacheModeQ8)
	}
	if plan.EstimatedKVCacheBytes == 0 || plan.EstimatedKVCacheModeBytes == 0 {
		t.Fatalf("KV estimates unset: %+v", plan)
	}
	if plan.EstimatedKVCacheModeBytes >= plan.EstimatedKVCacheBytes {
		t.Fatalf("mode bytes %d >= fp bytes %d", plan.EstimatedKVCacheModeBytes, plan.EstimatedKVCacheBytes)
	}
}

func TestMemory_NewPlan_TurboQuantKVCacheEstimateResearchMode(t *testing.T) {
	const elements uint64 = 32

	got := scaleKVElements(elements, KVCacheModeTurboQuant)

	if got != 14 {
		t.Fatalf("TurboQuant bytes = %d, want 14 for 32 KV elements at 3.5 bits/element", got)
	}
}

func TestMemory_NewPlan_TurboQuantIsNeverDefault(t *testing.T) {
	plan := NewPlan(Input{
		Device: DeviceInfo{MemorySize: 96 * GiB, MaxRecommendedWorkingSetSize: 90 * GiB},
	})

	if plan.CacheMode == KVCacheModeTurboQuant {
		t.Fatal("CacheMode = turboquant, want opt-in research mode only")
	}
}

func TestMemory_NewPlan_GenericMoEResidencyEnabled(t *testing.T) {
	// MoE architecture without MiniMax-specific tensor plan should still get
	// generic lazy residency from the architecture profile.
	pack := mp.ModelPack{Architecture: "qwen3_moe", NumLayers: 48, HiddenSize: 4096}
	plan := NewPlan(Input{
		Device: DeviceInfo{MemorySize: 96 * GiB, MaxRecommendedWorkingSetSize: 90 * GiB},
		Pack:   &pack,
	})
	if !plan.ExpertResidency.Enabled || plan.ExpertResidency.Mode != ExpertResidencyModeLazy {
		t.Fatalf("ExpertResidency = %+v, want lazy residency for MoE", plan.ExpertResidency)
	}
	if plan.ExpertResidency.EvictionPolicy != ExpertEvictionLRU {
		t.Fatalf("EvictionPolicy = %q, want LRU", plan.ExpertResidency.EvictionPolicy)
	}
}

// TestMemory_ClassForBytes_Good is the canonical happy-path triplet member for
// the public ClassForBytes classifier: each nominal device-memory size maps to
// the Apple tier it advertises.
func TestMemory_ClassForBytes_Good(t *testing.T) {
	cases := []struct {
		bytes uint64
		want  Class
	}{
		{16 * GiB, ClassApple16GB},
		{24 * GiB, ClassApple24GB},
		{32 * GiB, ClassApple32GB},
		{64 * GiB, ClassApple64GB},
		{96 * GiB, ClassApple96GB},
		{128 * GiB, ClassApple128GB},
	}
	for _, c := range cases {
		if got := ClassForBytes(c.bytes); got != c.want {
			t.Fatalf("ClassForBytes(%d) = %q, want %q", c.bytes, got, c.want)
		}
	}
}

// TestMemory_ClassForBytes_Bad is the canonical bad-input triplet member: a zero
// byte count is not a measured device — ClassForBytes must report ClassUnknown,
// never the smallest real tier, so a missing measurement can never be mistaken
// for a 16GB machine.
func TestMemory_ClassForBytes_Bad(t *testing.T) {
	if got := ClassForBytes(0); got != ClassUnknown {
		t.Fatalf("ClassForBytes(0) = %q, want %q for an unmeasured device", got, ClassUnknown)
	}
	// A single byte is still not a real tier floor — anything under the 16GB
	// band rounds up to the smallest Apple class, never to unknown.
	if got := ClassForBytes(1); got != ClassApple16GB {
		t.Fatalf("ClassForBytes(1) = %q, want %q (a non-zero sub-tier size rounds up, not to unknown)", got, ClassApple16GB)
	}
}

// TestMemory_ClassForBytes_Ugly is the canonical edge-case triplet member: it
// pins the ceiling-rounding boundaries between bands. The classifier rounds the
// raw byte count UP to whole GiB then bands it, so an off-nominal size just below
// a boundary stays in the lower tier and the exact boundary tips into the next —
// the awkward arithmetic the nominal Good cases never exercise.
func TestMemory_ClassForBytes_Ugly(t *testing.T) {
	cases := []struct {
		name  string
		bytes uint64
		want  Class
	}{
		{"18GiB still 16-class", 18 * GiB, ClassApple16GB},
		{"19GiB tips to 24-class", 19 * GiB, ClassApple24GB},
		{"26GiB still 24-class", 26 * GiB, ClassApple24GB},
		{"40GiB still 32-class", 40 * GiB, ClassApple32GB},
		{"41GiB tips to 64-class", 41 * GiB, ClassApple64GB},
		{"112GiB still 96-class", 112 * GiB, ClassApple96GB},
		{"113GiB tips to 128-plus", 113 * GiB, ClassApple128GB},
		// One byte over a whole-GiB count still rounds the same band up: 80GiB
		// is the top of 64-class, 80GiB+1 byte rounds to 81 GiB → 96-class.
		{"64-class ceiling + 1 byte", 80*GiB + 1, ClassApple96GB},
	}
	for _, c := range cases {
		if got := ClassForBytes(c.bytes); got != c.want {
			t.Fatalf("%s: ClassForBytes(%d) = %q, want %q", c.name, c.bytes, got, c.want)
		}
	}
}

func TestMemory_MinPositive_FavoursPositive(t *testing.T) {
	if minPositive(0, 5) != 5 {
		t.Fatal("minPositive(0,5) != 5")
	}
	if minPositive(5, 0) != 5 {
		t.Fatal("minPositive(5,0) != 5")
	}
	if minPositive(3, 7) != 3 {
		t.Fatal("minPositive(3,7) != 3")
	}
	if minPositive(0, 0) != 0 {
		t.Fatal("minPositive(0,0) != 0")
	}
	// b < a, both positive → returns b (the second-arg branch the existing
	// a<b case does not reach).
	if minPositive(7, 3) != 3 {
		t.Fatal("minPositive(7,3) != 3")
	}
}

// TestUsesGenerationKVCacheWithProfile_ShortCircuits covers the cached-profile
// short-circuits of usesGenerationKVCacheWithProfile: a Pack carrying its own
// embedding/rerank ArchitectureProfile, and a separately-supplied profileHint,
// both disable the generation cache without a registry lookup; a generation
// profileHint enables it.
func TestUsesGenerationKVCacheWithProfile_ShortCircuits(t *testing.T) {
	embedProfile := &profile.ModelArchitectureProfile{ID: "bert", Embeddings: true}
	rerankProfile := &profile.ModelArchitectureProfile{ID: "bert_rerank", Rerank: true}
	genProfile := &profile.ModelArchitectureProfile{ID: "qwen2"}

	// Pack-resident ArchitectureProfile (embeddings) → false, via the Pack
	// short-circuit before any profileHint is consulted.
	packEmbed := Input{Pack: &mp.ModelPack{Architecture: "bert", ArchitectureProfile: embedProfile}}
	if usesGenerationKVCacheWithProfile(packEmbed, nil) {
		t.Fatal("usesGenerationKVCacheWithProfile(pack embedding profile) = true, want false")
	}
	// Pack-resident ArchitectureProfile (rerank) → false.
	packRerank := Input{Pack: &mp.ModelPack{Architecture: "bert_rerank", ArchitectureProfile: rerankProfile}}
	if usesGenerationKVCacheWithProfile(packRerank, nil) {
		t.Fatal("usesGenerationKVCacheWithProfile(pack rerank profile) = true, want false")
	}
	// Supplied profileHint (embeddings) → false.
	if usesGenerationKVCacheWithProfile(Input{}, embedProfile) {
		t.Fatal("usesGenerationKVCacheWithProfile(embedding hint) = true, want false")
	}
	// Supplied profileHint (generation) → true.
	if !usesGenerationKVCacheWithProfile(Input{}, genProfile) {
		t.Fatal("usesGenerationKVCacheWithProfile(generation hint) = false, want true")
	}
}

func TestMemory_PercentBytes_GuardsAgainstZero(t *testing.T) {
	if percentBytes(0, 50) != 0 {
		t.Fatal("percentBytes(0,50) != 0")
	}
	if percentBytes(100, 25) != 25 {
		t.Fatal("percentBytes(100,25) != 25")
	}
}

// TestMemory_IsKnownKVCacheMode_Good walks every mode named in the public
// KV-cache contract — including the empty default — and asserts each is reported
// known. The empty string IS a contract member (KVCacheModeDefault), so "unset"
// must read as known, not unknown.
func TestMemory_IsKnownKVCacheMode_Good(t *testing.T) {
	for _, mode := range []KVCacheMode{
		KVCacheModeDefault, // == "" — the unset/default case is a contract member
		KVCacheModeFP16,
		KVCacheModeQ8,
		KVCacheModeKQ8VQ4,
		KVCacheModePaged,
		KVCacheModeTurboQuant,
	} {
		if !IsKnownKVCacheMode(mode) {
			t.Fatalf("IsKnownKVCacheMode(%q) = false, want true (contract member)", mode)
		}
	}
}

// TestMemory_IsKnownKVCacheMode_Bad feeds a non-empty string that is not in the
// contract and asserts it is rejected — the discrimination the function exists
// for.
func TestMemory_IsKnownKVCacheMode_Bad(t *testing.T) {
	if IsKnownKVCacheMode(KVCacheMode("q3")) {
		t.Fatal(`IsKnownKVCacheMode("q3") = true, want false for an out-of-contract mode`)
	}
	if IsKnownKVCacheMode(KVCacheMode("not-a-mode")) {
		t.Fatal(`IsKnownKVCacheMode("not-a-mode") = true, want false for garbage input`)
	}
}

// TestMemory_IsKnownKVCacheMode_Ugly pins the subtle case: TurboQuant is a
// research mode a backend may fail closed on, yet it is still part of the
// published contract, so IsKnownKVCacheMode reports it known. "known" means
// "named in the contract", not "every backend implements it".
func TestMemory_IsKnownKVCacheMode_Ugly(t *testing.T) {
	if !IsKnownKVCacheMode(KVCacheModeTurboQuant) {
		t.Fatal("IsKnownKVCacheMode(turboquant) = false, want true — research mode is still a contract member")
	}
	// Contrast: a near-miss that is NOT in the contract must be rejected, so the
	// known set is a closed enumeration and not an accept-anything check.
	if IsKnownKVCacheMode(KVCacheMode("turbo")) {
		t.Fatal(`IsKnownKVCacheMode("turbo") = true, want false — only the exact contract spelling is known`)
	}
}

// TestMemory_IsKnownKVCacheMode_RegistryAuthoritative pins the registry-backed
// edges the old hard-coded list could not express. "fixed" is a registered
// StateKVCache scheme the pre-registry list omitted; with the scheme registry as
// the single authority it now reads as a known KV-cache mode. "recurrent" is a
// registered cache scheme too, but it serves StateRecurrent — a recurrent-state
// holder, not a KV cache — so it is correctly NOT a known KV-cache mode.
func TestMemory_IsKnownKVCacheMode_RegistryAuthoritative(t *testing.T) {
	if !IsKnownKVCacheMode(KVCacheMode("fixed")) {
		t.Fatal(`IsKnownKVCacheMode("fixed") = false, want true — a registered KV-cache scheme is a known mode`)
	}
	if IsKnownKVCacheMode(KVCacheMode("recurrent")) {
		t.Fatal(`IsKnownKVCacheMode("recurrent") = true, want false — a recurrent-state holder is not a KV-cache mode`)
	}
}

// TestMemory_ScaleKVElements_Good is the golden parity table for scaleKVElements
// after the rewire through the scheme registry: each KV mode scales the raw
// element count by the exact per-element byte arithmetic the pre-registry switch
// produced. elements is divisible by 16 so every ratio lands on an integer — the
// Good case pins the ratios, the Ugly case pins their rounding. "fixed" carries
// no memory-package const but is a registered full-precision KV scheme, so it
// sizes at 2:1 exactly like the default lane.
func TestMemory_ScaleKVElements_Good(t *testing.T) {
	const elements uint64 = 4096 // 16 | 4096, so every ratio is exact
	cases := []struct {
		name string
		mode KVCacheMode
		want uint64
	}{
		{"q8 1:1", KVCacheModeQ8, 4096},                  // elements
		{"k-q8-v-q4 3:4", KVCacheModeKQ8VQ4, 3072},       // elements * 3 / 4
		{"turboquant 7:16", KVCacheModeTurboQuant, 1792}, // ceil(elements*7/16); 4096*7/16 = 1792 exact
		{"fp16 2:1", KVCacheModeFP16, 8192},              // elements * 2
		{"default 2:1", KVCacheModeDefault, 8192},        // "" → "default" → elements * 2
		{"paged 2:1", KVCacheModePaged, 8192},            // elements * 2 (was the switch default lane)
		{"fixed 2:1", KVCacheMode("fixed"), 8192},        // registered full-precision KV scheme
	}
	for _, c := range cases {
		if got := scaleKVElements(elements, c.mode); got != c.want {
			t.Fatalf("%s: scaleKVElements(%d, %q) = %d, want %d", c.name, elements, c.mode, got, c.want)
		}
	}
}

// TestMemory_ScaleKVElements_Bad covers the unknown-mode lane: a mode the scheme
// registry never registered has no width to probe, so the planner falls back to
// the conservative fp16-equivalent 2 bytes/element — the same safe over-estimate
// the pre-registry switch default produced, never a zero that would under-size a
// plan.
func TestMemory_ScaleKVElements_Bad(t *testing.T) {
	const elements uint64 = 4096
	if got := scaleKVElements(elements, KVCacheMode("not-a-real-mode")); got != elements*2 {
		t.Fatalf("scaleKVElements(unknown mode) = %d, want %d (conservative 2:1 default)", got, elements*2)
	}
}

// TestMemory_ScaleKVElements_Ugly pins the subtle parity detail the two
// fractional modes carry: k-q8-v-q4 TRUNCATES its 3/4 ratio while TurboQuant
// ROUNDS UP its 7/16 ratio — opposite rounding the width capability's roundUp
// flag preserves exactly. It also pins the recurrent path: "recurrent" resolves
// in the registry but carries no KV width, so the probe misses and it falls to
// the 2:1 default lane rather than being sized as a KV storage format.
func TestMemory_ScaleKVElements_Ugly(t *testing.T) {
	// k-q8-v-q4 truncates: 6*3/4 = 4.5 → 4, never 5. Proves floor, not ceil.
	if got := scaleKVElements(6, KVCacheModeKQ8VQ4); got != 4 {
		t.Fatalf("scaleKVElements(6, k-q8-v-q4) = %d, want 4 (truncated 3/4, not rounded up)", got)
	}
	// turboquant rounds up: 33*7/16 = 14.4375 → 15, never 14. Proves ceil.
	if got := scaleKVElements(33, KVCacheModeTurboQuant); got != 15 {
		t.Fatalf("scaleKVElements(33, turboquant) = %d, want 15 (7/16 rounded up)", got)
	}
	// recurrent: registered, but StateRecurrent carries no KV width → 2:1 default.
	if got := scaleKVElements(4096, KVCacheMode("recurrent")); got != 8192 {
		t.Fatalf("scaleKVElements(recurrent) = %d, want 8192 (no KV width → 2:1 default)", got)
	}
}

// TestMemory_NewPlan_Qwen2HintNote exercises the qwen2 branch of
// applyArchitectureHints via the public NewPlan: a Qwen2 pack emits the native
// decoder note and leaves the cache policy on its class baseline.
func TestMemory_NewPlan_Qwen2HintNote(t *testing.T) {
	pack := mp.ModelPack{Architecture: "qwen2", ContextLength: 32768, NumLayers: 28, HiddenSize: 3584, QuantBits: 4}
	plan := NewPlan(Input{
		Device: DeviceInfo{MemorySize: 96 * GiB, MaxRecommendedWorkingSetSize: 90 * GiB},
		Pack:   &pack,
	})
	if !hasNote(plan, "Qwen2.x uses the native Qwen decoder") {
		t.Fatalf("Notes = %+v, want qwen2 native-decoder hint", plan.Notes)
	}
}

// TestMemory_NewPlan_Qwen36ClampsParallelAndPrefill exercises the qwen3_6 hybrid
// linear-attention branch: it forces ParallelSlots to 1 and clamps a wide class
// baseline PrefillChunkSize (4096 on 64GB) down to 2048.
func TestMemory_NewPlan_Qwen36ClampsParallelAndPrefill(t *testing.T) {
	pack := mp.ModelPack{Architecture: "qwen3_6", ContextLength: 40960, NumLayers: 28, HiddenSize: 2048, QuantBits: 4}
	plan := NewPlan(Input{
		Device: DeviceInfo{MemorySize: 64 * GiB, MaxRecommendedWorkingSetSize: 60 * GiB},
		Pack:   &pack,
	})
	if plan.ParallelSlots != 1 {
		t.Fatalf("ParallelSlots = %d, want 1 (qwen3.6 hybrid attention pins to one slot)", plan.ParallelSlots)
	}
	if plan.PrefillChunkSize != 2048 {
		t.Fatalf("PrefillChunkSize = %d, want 2048 (clamped from the 64GB baseline 4096)", plan.PrefillChunkSize)
	}
	if !hasNote(plan, "hybrid linear attention") {
		t.Fatalf("Notes = %+v, want hybrid-attention hint", plan.Notes)
	}
}

// TestMemory_NewPlan_Qwen36MoESmallClassCompactCache exercises the qwen3_6_moe
// branch on a constrained class: it pins one slot, clamps prefill, and forces
// the asymmetric K@q8,V@q4 compact cache below 64GB.
func TestMemory_NewPlan_Qwen36MoESmallClassCompactCache(t *testing.T) {
	pack := mp.ModelPack{Architecture: "qwen3_6_moe", ContextLength: 40960, NumLayers: 48, HiddenSize: 4096, QuantBits: 4}
	plan := NewPlan(Input{
		Device: DeviceInfo{MemorySize: 32 * GiB, MaxRecommendedWorkingSetSize: 28 * GiB},
		Pack:   &pack,
	})
	if plan.CacheMode != KVCacheModeKQ8VQ4 {
		t.Fatalf("CacheMode = %q, want %q (qwen3.6-MoE compact cache below 64GB)", plan.CacheMode, KVCacheModeKQ8VQ4)
	}
	if plan.ParallelSlots != 1 {
		t.Fatalf("ParallelSlots = %d, want 1", plan.ParallelSlots)
	}
	if !hasNote(plan, "routed experts") || !hasNote(plan, "asymmetric K@q8,V@q4") {
		t.Fatalf("Notes = %+v, want MoE + asymmetric-cache hints", plan.Notes)
	}
}

// TestMemory_NewPlan_Qwen36MoEWideClassClampsPrefill exercises the qwen3_6_moe
// prefill clamp on a WIDE class: the 96GB baseline PrefillChunkSize (4096) is
// clamped down to 2048 by the hybrid-attention branch, and the compact-cache
// override does NOT fire above 64GB.
func TestMemory_NewPlan_Qwen36MoEWideClassClampsPrefill(t *testing.T) {
	pack := mp.ModelPack{Architecture: "qwen3_6_moe", ContextLength: 40960, NumLayers: 48, HiddenSize: 4096, QuantBits: 4}
	plan := NewPlan(Input{
		Device: DeviceInfo{MemorySize: 96 * GiB, MaxRecommendedWorkingSetSize: 90 * GiB},
		Pack:   &pack,
	})
	if plan.PrefillChunkSize != 2048 {
		t.Fatalf("PrefillChunkSize = %d, want 2048 (clamped from the 96GB baseline 4096)", plan.PrefillChunkSize)
	}
	if plan.CacheMode == KVCacheModeKQ8VQ4 {
		t.Fatalf("CacheMode = %q, want the wide-class default (no sub-64GB compact override)", plan.CacheMode)
	}
}

// TestMemory_NewPlan_MiniMaxSmallClassFloorsContext exercises the MiniMax M2
// sub-64GB branch: context is floored to 8192 (via minPositive) and the cache
// forced to asymmetric K@q8,V@q4 — the path the 96GB MiniMax test cannot reach.
func TestMemory_NewPlan_MiniMaxSmallClassFloorsContext(t *testing.T) {
	pack := mp.ModelPack{
		Architecture:  "minimax_m2",
		ContextLength: 196608,
		NumLayers:     62, HiddenSize: 3072,
	}
	plan := NewPlan(Input{
		Device: DeviceInfo{MemorySize: 24 * GiB, MaxRecommendedWorkingSetSize: 21 * GiB},
		Pack:   &pack,
	})
	if plan.ContextLength != 8192 {
		t.Fatalf("ContextLength = %d, want 8192 (MiniMax floored below 64GB)", plan.ContextLength)
	}
	if plan.CacheMode != KVCacheModeKQ8VQ4 {
		t.Fatalf("CacheMode = %q, want %q below 64GB", plan.CacheMode, KVCacheModeKQ8VQ4)
	}
	if !hasNote(plan, "asymmetric compact KV cache below 64GB") {
		t.Fatalf("Notes = %+v, want sub-64GB MiniMax compact-cache note", plan.Notes)
	}
}

// TestMemory_NewPlan_EncoderUnknownClassBatchFloor exercises the default branch of
// applyEncoderHints: an encoder pack on an unknown-memory device gets the
// conservative batch-4 floor.
func TestMemory_NewPlan_EncoderUnknownClassBatchFloor(t *testing.T) {
	pack := mp.ModelPack{
		Architecture: "bert", ContextLength: 512,
		NumLayers: 12, HiddenSize: 768,
		Embedding: &mp.ModelEmbeddingProfile{Dimension: 768, Pooling: "mean", MaxSequenceLength: 512},
	}
	plan := NewPlan(Input{Device: DeviceInfo{MemorySize: 0}, Pack: &pack})
	if plan.MachineClass != ClassUnknown {
		t.Fatalf("MachineClass = %q, want unknown for zero memory", plan.MachineClass)
	}
	if plan.BatchSize != 4 {
		t.Fatalf("BatchSize = %d, want 4 (unknown-class encoder floor)", plan.BatchSize)
	}
}

// TestMemory_NewPlan_MoEUnknownClassResidentFloor exercises the default branch of
// genericMoEResidentExpertLimit: a generic MoE pack on an unknown-memory device
// gets the conservative resident-expert floor of 2.
func TestMemory_NewPlan_MoEUnknownClassResidentFloor(t *testing.T) {
	pack := mp.ModelPack{Architecture: "qwen3_moe", NumLayers: 48, HiddenSize: 4096, QuantBits: 4}
	plan := NewPlan(Input{Device: DeviceInfo{MemorySize: 0}, Pack: &pack})
	if plan.MachineClass != ClassUnknown {
		t.Fatalf("MachineClass = %q, want unknown for zero memory", plan.MachineClass)
	}
	if !plan.ExpertResidency.Enabled || plan.ExpertResidency.MaxResidentExperts != 2 {
		t.Fatalf("MaxResidentExperts = %d (enabled=%t), want 2 (unknown-class floor)", plan.ExpertResidency.MaxResidentExperts, plan.ExpertResidency.Enabled)
	}
}

// TestMemory_NewPlan_Qwen3NextHintNote exercises the qwen3_next branch: it emits
// the nested-text_config note and otherwise keeps the class baseline.
func TestMemory_NewPlan_Qwen3NextHintNote(t *testing.T) {
	pack := mp.ModelPack{Architecture: "qwen3_next", ContextLength: 32768, NumLayers: 48, HiddenSize: 2048, QuantBits: 4}
	plan := NewPlan(Input{
		Device: DeviceInfo{MemorySize: 96 * GiB, MaxRecommendedWorkingSetSize: 90 * GiB},
		Pack:   &pack,
	})
	if !hasNote(plan, "nested text_config") {
		t.Fatalf("Notes = %+v, want qwen3-next nested-config hint", plan.Notes)
	}
}

// TestMemory_NewPlan_BertRerankDisablesGenerationCache exercises the bert_rerank
// branch and the rerank early-return in usesGenerationKVCache: a cross-encoder
// rerank pack disables the generation cache and emits no KV estimate.
func TestMemory_NewPlan_BertRerankDisablesGenerationCache(t *testing.T) {
	pack := mp.ModelPack{
		Architecture: "bert_rerank", ContextLength: 512,
		NumLayers: 12, HiddenSize: 768,
		Rerank:      &mp.ModelRerankProfile{Method: "cross-encoder", MaxSequenceLength: 512},
		WeightBytes: 420 * 1024 * 1024,
		QuantBits:   16, QuantType: "fp16", QuantFamily: "dense",
	}
	plan := NewPlan(Input{
		Device: DeviceInfo{MemorySize: 32 * GiB, MaxRecommendedWorkingSetSize: 28 * GiB},
		Pack:   &pack,
	})
	if plan.CachePolicy != KVCacheDefault || plan.PromptCache {
		t.Fatalf("plan = %+v, want disabled generation cache for rerank encoder", plan)
	}
	if plan.EstimatedKVCacheBytes != 0 || plan.EstimatedKVCacheModeBytes != 0 {
		t.Fatalf("KV estimates = fp:%d mode:%d, want zero for a rerank encoder", plan.EstimatedKVCacheBytes, plan.EstimatedKVCacheModeBytes)
	}
	if !hasNote(plan, "cross-encoder rerank") {
		t.Fatalf("Notes = %+v, want rerank encoder hint", plan.Notes)
	}
}

// TestMemory_NewPlan_EncoderBatchScalesWithClass walks the applyEncoderHints batch
// tiers across machine classes — the throughput floor an embedding encoder gets
// rises with available memory (16/24→8, 32→16, 64/96→32, 128→48).
func TestMemory_NewPlan_EncoderBatchScalesWithClass(t *testing.T) {
	cases := []struct {
		mem       uint64
		wantBatch int
	}{
		{16 * GiB, 8},
		{24 * GiB, 8},
		{32 * GiB, 16},
		{64 * GiB, 32},
		{96 * GiB, 32},
		{128 * GiB, 48},
	}
	for _, c := range cases {
		pack := mp.ModelPack{
			Architecture: "bert", ContextLength: 512,
			NumLayers: 12, HiddenSize: 768,
			Embedding: &mp.ModelEmbeddingProfile{Dimension: 768, Pooling: "mean", MaxSequenceLength: 512},
		}
		plan := NewPlan(Input{
			Device: DeviceInfo{MemorySize: c.mem, MaxRecommendedWorkingSetSize: c.mem - 2*GiB},
			Pack:   &pack,
		})
		if plan.BatchSize != c.wantBatch {
			t.Fatalf("%dGiB encoder BatchSize = %d, want %d", c.mem/GiB, plan.BatchSize, c.wantBatch)
		}
		if plan.PrefillChunkSize != 512 {
			t.Fatalf("%dGiB encoder PrefillChunkSize = %d, want 512", c.mem/GiB, plan.PrefillChunkSize)
		}
	}
}

// TestMemory_NewPlan_GenericMoEResidentLimitScalesWithClass walks the
// genericMoEResidentExpertLimit tiers via the public plan: a generic MoE pack's
// MaxResidentExperts rises with the machine class (16/24→2, 32→4, 64→8, 96→16,
// 128→24).
func TestMemory_NewPlan_GenericMoEResidentLimitScalesWithClass(t *testing.T) {
	cases := []struct {
		mem       uint64
		wantLimit int
	}{
		{16 * GiB, 2},
		{24 * GiB, 2},
		{32 * GiB, 4},
		{64 * GiB, 8},
		{96 * GiB, 16},
		{128 * GiB, 24},
	}
	for _, c := range cases {
		pack := mp.ModelPack{Architecture: "qwen3_moe", NumLayers: 48, HiddenSize: 4096, QuantBits: 4}
		plan := NewPlan(Input{
			Device: DeviceInfo{MemorySize: c.mem, MaxRecommendedWorkingSetSize: c.mem - 4*GiB},
			Pack:   &pack,
		})
		if !plan.ExpertResidency.Enabled {
			t.Fatalf("%dGiB MoE residency disabled, want enabled", c.mem/GiB)
		}
		if plan.ExpertResidency.MaxResidentExperts != c.wantLimit {
			t.Fatalf("%dGiB MaxResidentExperts = %d, want %d", c.mem/GiB, plan.ExpertResidency.MaxResidentExperts, c.wantLimit)
		}
	}
}

// TestMemory_NewPlan_JangtqQuantizationNote exercises applyQuantizationHints: a
// JANGTQ/JANG mixed-precision pack emits the measured-weight-bytes guidance note.
func TestMemory_NewPlan_JangtqQuantizationNote(t *testing.T) {
	pack := mp.ModelPack{
		Architecture: "qwen2", ContextLength: 32768,
		NumLayers: 28, HiddenSize: 3584,
		QuantBits: 4, QuantType: "jangtq", QuantFamily: "jang",
	}
	plan := NewPlan(Input{
		Device: DeviceInfo{MemorySize: 96 * GiB, MaxRecommendedWorkingSetSize: 90 * GiB},
		Pack:   &pack,
	})
	if plan.ModelQuantizationFamily != "jang" {
		t.Fatalf("ModelQuantizationFamily = %q, want jang", plan.ModelQuantizationFamily)
	}
	if !hasNote(plan, "JANGTQ/JANG mixed precision") {
		t.Fatalf("Notes = %+v, want JANGTQ guidance note", plan.Notes)
	}
}

// TestMemory_NewPlan_DerivedKVCacheSavingsRatio proves the savings ratio is
// populated when a compact cache mode is selected: a Q8 plan with a real KV
// estimate reports a positive, sub-1.0 savings ratio versus the FP16 baseline.
func TestMemory_NewPlan_DerivedKVCacheSavingsRatio(t *testing.T) {
	plan := NewPlan(Input{
		Device:    DeviceInfo{MemorySize: 32 * GiB, MaxRecommendedWorkingSetSize: 28 * GiB},
		ModelInfo: &ModelInfo{Architecture: "qwen2", NumLayers: 32, HiddenSize: 3072, ContextLength: 16384},
	})
	if plan.CacheMode != KVCacheModeQ8 {
		t.Fatalf("CacheMode = %q, want q8 baseline for 32GB", plan.CacheMode)
	}
	if plan.KVCacheSavingsRatio <= 0 || plan.KVCacheSavingsRatio >= 1 {
		t.Fatalf("KVCacheSavingsRatio = %v, want (0,1) for a compact cache vs fp16", plan.KVCacheSavingsRatio)
	}
}

// TestMemory_ScaleElementsByByteRatioCeil_ZeroGuards covers the zero-input guards
// (the uncovered branch) and the ceiling rounding of the byte-ratio scaler.
func TestMemory_ScaleElementsByByteRatioCeil_ZeroGuards(t *testing.T) {
	if got := scaleElementsByByteRatioCeil(0, 7, 16); got != 0 {
		t.Fatalf("scaleElementsByByteRatioCeil(0,…) = %d, want 0", got)
	}
	if got := scaleElementsByByteRatioCeil(32, 0, 16); got != 0 {
		t.Fatalf("scaleElementsByByteRatioCeil(…,0,…) = %d, want 0", got)
	}
	if got := scaleElementsByByteRatioCeil(32, 7, 0); got != 0 {
		t.Fatalf("scaleElementsByByteRatioCeil(…,0 denom) = %d, want 0", got)
	}
	// 33*7 = 231; ceil(231/16) = 15 (231/16 = 14.4375) — proves ceiling, not floor.
	if got := scaleElementsByByteRatioCeil(33, 7, 16); got != 15 {
		t.Fatalf("scaleElementsByByteRatioCeil(33,7,16) = %d, want 15 (ceil)", got)
	}
}

// TestEstimateKVCacheBytes_GenerationVsEncoder covers the unexported
// estimateKVCacheBytes wrapper (no live callers today; retained from the
// profile-caching refactor). It must return a positive FP16 estimate for a
// generation model and zero for an encoder / zero-context plan.
func TestEstimateKVCacheBytes_GenerationVsEncoder(t *testing.T) {
	genPlan := Plan{MachineClass: ClassApple96GB, ContextLength: 8192, CacheMode: KVCacheModeFP16}
	genInput := Input{ModelInfo: &ModelInfo{Architecture: "qwen2", NumLayers: 32, HiddenSize: 3072}}
	if got := estimateKVCacheBytes(genPlan, genInput, KVCacheModeFP16); got == 0 {
		t.Fatal("estimateKVCacheBytes(generation) = 0, want > 0")
	}
	// Zero context → zero estimate regardless of architecture.
	zeroCtx := genPlan
	zeroCtx.ContextLength = 0
	if got := estimateKVCacheBytes(zeroCtx, genInput, KVCacheModeFP16); got != 0 {
		t.Fatalf("estimateKVCacheBytes(zero-context) = %d, want 0", got)
	}
	// Encoder pack → generation cache disabled → zero estimate.
	encInput := Input{Pack: &mp.ModelPack{Architecture: "bert", NumLayers: 12, HiddenSize: 768, Embedding: &mp.ModelEmbeddingProfile{Dimension: 768}}}
	if got := estimateKVCacheBytes(genPlan, encInput, KVCacheModeFP16); got != 0 {
		t.Fatalf("estimateKVCacheBytes(encoder) = %d, want 0", got)
	}
}

// TestUsesGenerationKVCache_GenerationVsEncoder covers the unexported
// usesGenerationKVCache wrapper (no live callers today; retained from the
// profile-caching refactor). A generation architecture uses the cache; an
// embedding pack does not.
func TestUsesGenerationKVCache_GenerationVsEncoder(t *testing.T) {
	if !usesGenerationKVCache(Input{ModelInfo: &ModelInfo{Architecture: "qwen2"}}) {
		t.Fatal("usesGenerationKVCache(qwen2) = false, want true for a generation model")
	}
	if usesGenerationKVCache(Input{Pack: &mp.ModelPack{Architecture: "bert", Embedding: &mp.ModelEmbeddingProfile{Dimension: 768}}}) {
		t.Fatal("usesGenerationKVCache(bert embedding) = true, want false for an encoder")
	}
}
