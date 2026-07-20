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
	sharedmodel "dappco.re/go/inference/model"
	"dappco.re/go/inference/model/arch/mamba2"
	"dappco.re/go/inference/model/safetensors"
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
// loadHIPMamba2TextModel uses in production), not model/composed: #50 retired composed_runtime.go's use
// of model/composed, and this fixture follows suit rather than keeping a needless test-only import
// alive. mamba2.MambaTokenModel's HiddenSize/NumLayers methods are documented (token_model.go) as
// mirroring model/composed.ComposedTokenModel's, so it is a faithful drop-in for what these tests need.
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
// dropped the model/composed.ChatMLDialect call this used to make) falls back to the Gemma template for
// an architecture that is not qwen-family, exactly as the retired composed.ChatMLDialect check did.
func TestHIPComposedTextModel_DeclaresGemmaChatForNonQwenArchitecture_Bad(t *testing.T) {
	source := &hipComposedTextModel{model: hipComposedTestTokenModel(1), modelType: "mamba2", numLayers: 1}
	template, ok := source.DeclaredChatTemplate()
	core.RequireTrue(t, ok)
	core.AssertTrue(t, template.Open != "<|im_start|>")
}

func hipComposedTestRegisterComposedArch(modelType string) {
	sharedmodel.RegisterArch(sharedmodel.ArchSpec{
		ModelTypes: []string{modelType},
		Composed: func(map[string]safetensors.Tensor, []byte) (sharedmodel.TokenModel, error) {
			return nil, core.NewError("hip test double: composed hook must never run — #50 retired HIP's model/composed route")
		},
	})
}

// TestLoadHIPComposedTextModel_DeclinesRegisteredComposedArchitecture_Good proves a checkpoint whose
// config.json declares a model_type registered with an ArchSpec.Composed hook gets a clean, named
// decline — #50 retired HIP's model/composed detour, so nothing in this package calls into
// LoadComposedDir (or the arch's own Composed closure) any more.
func TestLoadHIPComposedTextModel_DeclinesRegisteredComposedArchitecture_Good(t *testing.T) {
	const modelType = "hip_composed_retired_direct_test"
	hipComposedTestRegisterComposedArch(modelType)
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
// multimodal-wrapper fallback LoadComposedDir used to apply (top-level model_type unregistered, nested
// text_config.model_type carries the Composed hook) still declines correctly, naming the NESTED
// architecture rather than silently passing the wrapper through to HIP's native pipeline.
func TestLoadHIPComposedTextModel_DeclinesNestedTextConfigComposedArchitecture_Ugly(t *testing.T) {
	const textModelType = "hip_composed_retired_text_wrapper_test"
	hipComposedTestRegisterComposedArch(textModelType)
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
// (#50) rather than a successful load through model/composed.
func TestROCmBackend_DeclinesRegisteredComposedArchitectureBeforeNativeRuntime_Good(t *testing.T) {
	const modelType = "hip_composed_retired_backend_test"
	hipComposedTestRegisterComposedArch(modelType)
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

// TestReactiveSequenceMixerReport_ComposedRunnerLinked_Good is UNCHANGED by #50's sever — it tests
// reactive_sequence_mixer.go's baseReactiveSequenceMixerReport and engine/hip/profile's
// ArchitectureProfileNotes/LookupArchitectureProfile, none of which this change touches (out of this
// lane's file fence). Flagging it honestly: those two files now make a STALE claim.
// baseReactiveSequenceMixerReport hardcodes RunnerReady=true and
// labels["sequence_mixer_runner_status"]="portable_composed_session_model" unconditionally, and
// profile/architecture.go's ArchitectureProfileNotes/LookupArchitectureProfile report
// NativeRuntime=true plus "portable composed incremental runner is linked" for qwen3_6/qwen3_6_moe/
// composed/hybrid — all true only because loadHIPComposedTextModel used to actually serve those
// checkpoints via model/composed. It no longer does (see loadHIPComposedTextModel's doc comment in
// composed_runtime.go); these two files were not in scope here and need their own follow-up so HIP's
// capability reporting stops claiming a runner that composed_runtime.go's retirement removed.
func TestReactiveSequenceMixerReport_ComposedRunnerLinked_Good(t *testing.T) {
	report := baseReactiveSequenceMixerReport("/models/qwen", nil)
	core.AssertTrue(t, report.RunnerReady)
	core.AssertEqual(t, "portable_composed_session_model", report.Labels["sequence_mixer_runner_status"])
	core.AssertEqual(t,
		[]string{"portable composed incremental runner is linked; projection hooks remain available for HIP++ acceleration"},
		rocmprofile.ArchitectureProfileNotes("qwen3_6"),
	)
	for _, architecture := range []string{"composed", "hybrid", "qwen3_6", "qwen3_6_moe"} {
		profile, ok := rocmprofile.LookupArchitectureProfile(architecture)
		core.RequireTrue(t, ok)
		core.RequireTrue(t, profile.NativeRuntime)
		core.RequireTrue(t, profile.Generation)
		core.RequireTrue(t, profile.Chat)
	}
}
