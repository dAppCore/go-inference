// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"

	core "dappco.re/go"
)

type hipNativeProjectionKernelSet struct {
	hipKernelStub
	moduleSource string
}

func newHIPRuntimeKernelSet(driver nativeHIPDriver) hipKernelSet {
	if driver == nil {
		return newDefaultHIPKernelSet()
	}
	if _, ok := driver.(nativeHIPKernelLauncher); !ok {
		return newDefaultHIPKernelSet()
	}
	resolution := resolveHIPKernelModule()
	if core.Trim(resolution.Path) == "" {
		return newDefaultHIPKernelSet()
	}
	return hipNativeProjectionKernelSet{moduleSource: resolution.Source}
}

func (kernels hipNativeProjectionKernelSet) Status() hipKernelStatus {
	return hipKernelStatus{
		CrossEntropy: hipKernelStatusLinked,
		Decode:       hipKernelStatusNotLinked,
		Distillation: hipKernelStatusLinked,
		Embedding:    hipKernelStatusLinked,
		GRPO:         hipKernelStatusLinked,
		Prefill:      hipKernelStatusNotLinked,
		Projection:   hipKernelStatusLinked,
		Rerank:       hipKernelStatusLinked,
		KVCache:      hipKernelStatusPlanned,
		Reason:       "native projection, embedding, rerank, and toy loss kernels configured by " + hipKernelModuleSourceLabel(kernels.moduleSource) + "; prefill/decode kernels are not linked yet",
	}
}

func (kernels hipNativeProjectionKernelSet) Project(ctx context.Context, model *hipLoadedModel, req hipProjectionRequest) ([]float32, error) {
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
	}
	if model == nil || model.driver == nil {
		return nil, core.E("rocm.hip.Project", "HIP driver is nil", nil)
	}
	return hipRunProjectionKernel(ctx, model.driver, req)
}

func (kernels hipNativeProjectionKernelSet) OpenROCmDiffusionSession(ctx context.Context, model *hipLoadedModel) (ROCmDiffusionSession, error) {
	const op = "rocm.hip.OpenROCmDiffusionSession"
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	session, err := newHIPROCmNativeDiffusionSession(model)
	if err != nil {
		return nil, core.E(op, "open native DiffusionGemma session", err)
	}
	return session, nil
}
