// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/engine"
	rocmprofile "dappco.re/go/inference/engine/hip/profile"
	"dappco.re/go/inference/kv"
	"dappco.re/go/inference/model/arch/mamba2"
	coreio "dappco.re/go/io"
)

// hipComposedTestValues is a deterministic synthetic vector (seeded), values in [-1, 1) — the same
// generator shape model/arch/mamba2's own tests use (see that package's scan_test.go: syn).
func hipComposedTestValues(n, seed int) []float32 {
	out := make([]float32, n)
	for index := range out {
		out[index] = float32((index*seed+7)%101-50) * 0.02
	}
	return out
}

// hipComposedTestBlockConfig mirrors model/arch/mamba2's own small test geometry (see that package's
// block_test.go / model_test.go): NumHeads=2, HeadDim=8, StateDim=8, NumGroups=1, ConvKernel=4.
func hipComposedTestBlockConfig() mamba2.BlockConfig {
	return mamba2.BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
}

// hipComposedTestBlockWeights builds one layer's synthetic block weights. BlockConfig's dInner/convDim/
// projDim helpers (block.go) are mamba2-package-private, so the dims are derived here from the same
// formulas BlockWeights' own field comments document (InProj is [projDim,D], ConvWeight is
// [convDim,K], the rest are per-head/per-channel vectors).
func hipComposedTestBlockWeights(cfg mamba2.BlockConfig, d, seed int) *mamba2.BlockWeights {
	dInner := cfg.NumHeads * cfg.HeadDim
	convDim := dInner + 2*cfg.NumGroups*cfg.StateDim
	projDim := 2*dInner + 2*cfg.NumGroups*cfg.StateDim + cfg.NumHeads
	return &mamba2.BlockWeights{
		InProj:     hipComposedTestValues(projDim*d, seed+1),
		ConvWeight: hipComposedTestValues(convDim*cfg.ConvKernel, seed+2),
		ConvBias:   hipComposedTestValues(convDim, seed+3),
		ALog:       hipComposedTestValues(cfg.NumHeads, seed+4),
		D:          hipComposedTestValues(cfg.NumHeads, seed+5),
		DtBias:     hipComposedTestValues(cfg.NumHeads, seed+6),
		Norm:       hipComposedTestValues(dInner, seed+7),
		OutProj:    hipComposedTestValues(d*dInner, seed+8),
	}
}

// hipComposedTestTokenModel builds a small, deterministic sharedmodel.SessionModel fixture for
// exercising hipComposedTextModel/hipComposedEngineSession — the generic token-prefix bridge these
// tests target. It is backed by model/arch/mamba2 (the SAME package mamba2_runtime.go's
// loadHIPMamba2TextModel uses in production), not the composed engine: #50 retired composed_runtime.go's use
// of the composed engine, and this fixture follows suit rather than keeping a needless test-only import
// alive. mamba2.MambaTokenModel's HiddenSize/NumLayers methods carry the layer/hidden
// geometry shape the serve bridge probes, so it is a faithful drop-in for what these tests need.
func hipComposedTestTokenModel(layers int) *mamba2.MambaTokenModel {
	const d, vocab = 8, 32
	cfg := hipComposedTestBlockConfig()
	blocks := make([]mamba2.MambaLayer, layers)
	for layer := range blocks {
		blocks[layer] = mamba2.MambaLayer{
			Norm: hipComposedTestValues(d, layer*13+1),
			W:    hipComposedTestBlockWeights(cfg, d, layer*13+20),
		}
	}
	return mamba2.NewTokenModel(&mamba2.MambaModel{
		Embed:  hipComposedTestValues(vocab*d, 100),
		NormF:  hipComposedTestValues(d, 101),
		Layers: blocks,
		Cfg:    cfg,
		D:      d,
		Vocab:  vocab,
	})
}

func TestHIPComposedEngineSession_StateIncludesGeneratedTokens_Good(t *testing.T) {
	const layers = 2
	session := &hipComposedEngineSession{model: hipComposedTestTokenModel(layers), architecture: "mamba2", numLayers: layers}
	core.RequireNoError(t, session.PrefillTokens([]int32{1, 5, 9, 2}))
	generated, err := session.GenerateFromCacheEach(4, -1, func(int32) bool { return true })
	core.RequireNoError(t, err)
	core.AssertEqual(t, 8, session.Pos())

	snapshot, err := session.CaptureKVWithOptions(kv.CaptureOptions{})
	core.RequireNoError(t, err)
	core.AssertEqual(t, append([]int32{1, 5, 9, 2}, generated...), snapshot.Tokens)

	resumed := &hipComposedEngineSession{model: hipComposedTestTokenModel(layers), architecture: "mamba2", numLayers: layers}
	core.RequireNoError(t, resumed.RestoreFromKV(context.Background(), snapshot))
	want, err := session.GenerateFromCacheEach(3, -1, func(int32) bool { return true })
	core.RequireNoError(t, err)
	got, err := resumed.GenerateFromCacheEach(3, -1, func(int32) bool { return true })
	core.RequireNoError(t, err)
	core.AssertEqual(t, want, got)
}

func TestHIPComposedEngineSession_CaptureAndRangeRejectInvalidState_Bad(t *testing.T) {
	session := &hipComposedEngineSession{model: hipComposedTestTokenModel(1), architecture: "mamba2", numLayers: 1}
	_, err := session.CaptureKVWithOptions(kv.CaptureOptions{})
	core.AssertError(t, err)
	core.RequireNoError(t, session.PrefillTokens([]int32{1, 2, 3}))
	err = session.RangeKVBlocks(2, kv.CaptureOptions{BlockStartToken: -1}, func(kv.Block) (bool, error) { return true, nil })
	core.AssertError(t, err)
}

func TestHIPComposedEngineSession_RangeHonorsTrustedPrefix_Ugly(t *testing.T) {
	session := &hipComposedEngineSession{model: hipComposedTestTokenModel(1), architecture: "mamba2", numLayers: 1}
	core.RequireNoError(t, session.PrefillTokens([]int32{1, 2, 3, 4, 5, 6, 7}))
	var blocks []kv.Block
	err := session.RangeKVBlocks(3, kv.CaptureOptions{BlockStartToken: 3}, func(block kv.Block) (bool, error) {
		blocks = append(blocks, block)
		return true, nil
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, []int{1, 2}, []int{blocks[0].Index, blocks[1].Index})
	core.AssertEqual(t, []int32{4, 5, 6}, blocks[0].Snapshot.Tokens)
	core.AssertEqual(t, []int32{7}, blocks[1].Snapshot.Tokens)
}

func TestHIPComposedTextModel_DeclaresQwenChatML_Good(t *testing.T) {
	source := &hipComposedTextModel{model: hipComposedTestTokenModel(1), modelType: "qwen3_6", numLayers: 1}
	var memory engine.MemoryReporter = source
	template, ok := source.DeclaredChatTemplate()
	core.RequireTrue(t, ok)
	core.AssertEqual(t, "<|im_start|>", template.Open)
	core.AssertEqual(t, "assistant", template.AssistantRole)
	core.AssertEqual(t, []string{"<|im_end|>"}, template.Stops)
	core.AssertEqual(t, uint64(0), memory.ActiveMemoryBytes())
	core.AssertEqual(t, uint64(0), memory.PeakMemoryBytes())
}

// TestHIPComposedTextModel_DeclaresGemmaChatForNonQwenArchitecture_Bad proves DeclaredChatTemplate's
// replacement lookup (hipArchitectureChatTemplate, via HIP's own architecture-profile registry — #50
// dropped the composed engine's ChatMLDialect call this used to make) falls back to the Gemma template for
// an architecture that is not qwen-family, exactly as the retired composed.ChatMLDialect check did.
func TestHIPComposedTextModel_DeclaresGemmaChatForNonQwenArchitecture_Bad(t *testing.T) {
	source := &hipComposedTextModel{model: hipComposedTestTokenModel(1), modelType: "mamba2", numLayers: 1}
	template, ok := source.DeclaredChatTemplate()
	core.RequireTrue(t, ok)
	core.AssertTrue(t, template.Open != "<|im_start|>")
}

// TestLoadHIPComposedTextModel_DeclinesRegisteredComposedArchitecture_Good proves a checkpoint whose
// config.json declares a retired-composed family model_type (rocmRetiredComposedArchDecline) gets a
// clean, named decline — #50 retired HIP's composed-engine detour, so nothing in this package can
// serve these architectures any more.
func TestLoadHIPComposedTextModel_DeclinesRegisteredComposedArchitecture_Good(t *testing.T) {
	const modelType = "qwen3_6"
	dir := t.TempDir()
	core.RequireNoError(t, coreio.Local.Write(core.PathJoin(dir, "config.json"), `{"model_type":"`+modelType+`"}`))

	loaded, matched, err := loadHIPComposedTextModel(dir, inference.LoadConfig{})
	core.AssertTrue(t, loaded == nil)
	core.RequireTrue(t, matched)
	core.AssertError(t, err)
	core.AssertTrue(t, core.Contains(err.Error(), modelType))
}

// TestLoadHIPComposedTextModel_UnreadableConfigDefersToNextLoader_Bad proves a directory whose
// config.json cannot be read is NOT this function's concern to report: matched=false, err=nil, deferring
// to whichever loader native.go's pipeline tries next — the same shape loadHIPMamba2TextModel's sibling
// check (mamba2_runtime.go) already uses for the identical situation.
func TestLoadHIPComposedTextModel_UnreadableConfigDefersToNextLoader_Bad(t *testing.T) {
	dir := t.TempDir() // no config.json written

	loaded, matched, err := loadHIPComposedTextModel(dir, inference.LoadConfig{})
	core.AssertTrue(t, loaded == nil)
	core.AssertFalse(t, matched)
	core.RequireNoError(t, err)
}

// TestLoadHIPComposedTextModel_DeclinesNestedTextConfigComposedArchitecture_Ugly proves the
// multimodal-wrapper fallback (top-level model_type unknown, nested text_config.model_type carries a
// retired-composed family) still declines correctly, naming the NESTED architecture rather than
// silently passing the wrapper through to HIP's native pipeline.
func TestLoadHIPComposedTextModel_DeclinesNestedTextConfigComposedArchitecture_Ugly(t *testing.T) {
	const textModelType = "qwen3_5_moe"
	dir := t.TempDir()
	core.RequireNoError(t, coreio.Local.Write(core.PathJoin(dir, "config.json"),
		`{"model_type":"hip_composed_retired_wrapper_test","text_config":{"model_type":"`+textModelType+`"}}`))

	loaded, matched, err := loadHIPComposedTextModel(dir, inference.LoadConfig{})
	core.AssertTrue(t, loaded == nil)
	core.RequireTrue(t, matched)
	core.AssertError(t, err)
	core.AssertTrue(t, core.Contains(err.Error(), textModelType))
}

type hipComposedRouteRuntime struct{ calls int }

func (runtime *hipComposedRouteRuntime) Available() bool {
	runtime.calls++
	return false
}

func (*hipComposedRouteRuntime) DeviceInfo() nativeDeviceInfo { return nativeDeviceInfo{} }

func (runtime *hipComposedRouteRuntime) LoadModel(string, nativeLoadConfig) (nativeModel, error) {
	runtime.calls++
	return nil, core.NewError("native runtime must not receive a checkpoint the composed decline should have caught first")
}

// TestROCmBackend_DeclinesRegisteredComposedArchitectureBeforeNativeRuntime_Good proves the full
// loadModelWithROCmConfigMode pipeline still short-circuits BEFORE ever reaching the native ROCm
// runtime for a composed-registered architecture — only now the short circuit is a clean decline
// (#50) rather than a successful load through the retired composed engine.
func TestROCmBackend_DeclinesRegisteredComposedArchitectureBeforeNativeRuntime_Good(t *testing.T) {
	const modelType = "mixtral"
	dir := t.TempDir()
	core.RequireNoError(t, coreio.Local.Write(core.PathJoin(dir, "config.json"), `{"model_type":"`+modelType+`","max_position_embeddings":8192}`))

	runtime := &hipComposedRouteRuntime{}
	loaded, err := newROCmBackendWithRuntime(runtime).loadModelWithROCmConfigMode(
		dir,
		inference.LoadConfig{},
		ROCmLoadConfig{},
		false,
	)
	core.AssertTrue(t, loaded == nil)
	core.AssertError(t, err)
	core.AssertEqual(t, 0, runtime.calls)
	core.AssertTrue(t, core.Contains(err.Error(), modelType))
}

// TestReactiveSequenceMixerReport_ComposedRunnerRetired_Good proves reactive_sequence_mixer.go's
// baseReactiveSequenceMixerReport and engine/hip/profile's ArchitectureProfileNotes/LookupArchitectureProfile
// now tell the truth about #50's sever: composed_runtime.go's loadHIPComposedTextModel declines every
// config-composed/hybrid architecture outright (no native ROCm execution path, and nothing replaced the
// retired model/composed detour), so RunnerReady, the runner-status label, the architecture notes, and
// NativeRuntime must all say so rather than describe a runner that no longer exists. This test used to be
// named TestReactiveSequenceMixerReport_ComposedRunnerLinked_Good and asserted the opposite
// (RunnerReady=true, NativeRuntime=true, "portable composed incremental runner is linked") — that was
// accurate before #50, when loadHIPComposedTextModel actually routed these checkpoints through
// model/composed; it went stale the moment the severing lane retired that route without updating the report
// that described it (flagged, not fixed, in that lane's own doc comment — this lane closes the gap).
func TestReactiveSequenceMixerReport_ComposedRunnerRetired_Good(t *testing.T) {
	report := baseReactiveSequenceMixerReport("/models/qwen", nil)
	core.AssertFalse(t, report.RunnerReady)
	core.AssertEqual(t, "composed_route_retired", report.Labels["sequence_mixer_runner_status"])
	core.AssertEqual(t,
		[]string{"composed route retired (#50): the gated-delta hybrid has no native ROCm execution path; loadHIPComposedTextModel declines it outright, factory-native port pending"},
		rocmprofile.ArchitectureProfileNotes("qwen3_6"),
	)
	core.AssertEqual(t,
		[]string{"composed route retired (#50): the gated-delta MoE hybrid has no native ROCm execution path; loadHIPComposedTextModel declines it outright, factory-native port pending"},
		rocmprofile.ArchitectureProfileNotes("qwen3_6_moe"),
	)
	core.AssertEqual(t,
		[]string{"composed route retired (#50): no native ROCm execution path; loadHIPComposedTextModel declines this architecture outright, factory-native port pending"},
		rocmprofile.ArchitectureProfileNotes("composed"),
	)

	// The Qwen 3.5/3.6 gated-delta hybrid family and the generic composed/hybrid ids: #50 declined them
	// outright, so NativeRuntime must flip false — but Generation/Chat stay true (unchanged; out of this
	// lane's flagged scope). ROCmChatTemplateID's family fallback (architecture_registry.go) still resolves
	// "qwen" from profile.Family alone even with ChatTemplate cleared, so engine/hip's own chat-template
	// tests for these ids (TestHIPInferenceModel_DeclaredQwenChatTemplate_Good,
	// TestHIPRuntime_ApplyChatTemplateUsesQwenChatML_Good) are unaffected by this change.
	for _, architecture := range []string{"composed", "hybrid", "qwen3_6", "qwen3_6_moe", "qwen3_next"} {
		profile, ok := rocmprofile.LookupArchitectureProfile(architecture)
		core.RequireTrue(t, ok)
		core.AssertFalse(t, profile.NativeRuntime)
		core.AssertEqual(t, inference.FeatureRuntimeMetadataOnly, profile.RuntimeStatus)
		core.AssertTrue(t, profile.Generation)
		core.AssertTrue(t, profile.Chat)
	}

	// mixtral/qwen3_moe/deepseek/deepseek_r1 already reported Generation=false before #50 (their MoE expert
	// decode was never wired even while composed served them) — #50 only moves NativeRuntime here, from a
	// stale true to a truthful false; Generation/Chat were already honest and stay untouched.
	for _, architecture := range []string{"qwen3_moe", "mixtral", "deepseek", "deepseek_r1"} {
		profile, ok := rocmprofile.LookupArchitectureProfile(architecture)
		core.RequireTrue(t, ok)
		core.AssertFalse(t, profile.NativeRuntime)
		core.AssertEqual(t, inference.FeatureRuntimeMetadataOnly, profile.RuntimeStatus)
		core.AssertFalse(t, profile.Generation)
	}

	// mamba2 is untouched by #50: HIP serves it through its own native loader (mamba2_runtime.go), never
	// through model/composed, so it must keep reporting a real runner.
	mamba2Profile, ok := rocmprofile.LookupArchitectureProfile("mamba2")
	core.RequireTrue(t, ok)
	core.AssertTrue(t, mamba2Profile.NativeRuntime)
}
