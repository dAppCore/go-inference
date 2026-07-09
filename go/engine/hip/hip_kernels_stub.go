// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"iter"

	"dappco.re/go/inference"
)

type hipKernelStub struct{}

func newDefaultHIPKernelSet() hipKernelSet {
	return hipKernelStub{}
}

func (hipKernelStub) Status() hipKernelStatus {
	return defaultHIPKernelStatus()
}

func (stub hipKernelStub) Generate(ctx context.Context, _ *hipLoadedModel, _ string, _ inference.GenerateConfig) (iter.Seq[inference.Token], func() error) {
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return emptyTokenSeq, func() error { return err }
		}
	}
	return emptyTokenSeq, func() error {
		return hipKernelNotLinkedError("rocm.hip.Generate", hipKernelDecode, stub.Status())
	}
}

func (stub hipKernelStub) Chat(ctx context.Context, _ *hipLoadedModel, messages []inference.Message, _ inference.GenerateConfig) (iter.Seq[inference.Token], func() error) {
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return emptyTokenSeq, func() error { return err }
		}
	}
	if err := validateROCmChatMessages("rocm.hip.Chat", messages); err != nil {
		return emptyTokenSeq, func() error { return err }
	}
	return emptyTokenSeq, func() error {
		return hipKernelNotLinkedError("rocm.hip.Chat", hipKernelDecode, stub.Status())
	}
}

func (stub hipKernelStub) Classify(ctx context.Context, _ *hipLoadedModel, prompts []string, _ inference.GenerateConfig) ([]inference.ClassifyResult, error) {
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
	}
	if err := validateROCmPromptBatch("rocm.hip.Classify", prompts); err != nil {
		return nil, err
	}
	return nil, hipKernelNotLinkedError("rocm.hip.Classify", hipKernelPrefill, stub.Status())
}

func (stub hipKernelStub) BatchGenerate(ctx context.Context, _ *hipLoadedModel, prompts []string, _ inference.GenerateConfig) ([]inference.BatchResult, error) {
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
	}
	if err := validateROCmPromptBatch("rocm.hip.BatchGenerate", prompts); err != nil {
		return nil, err
	}
	return nil, hipKernelNotLinkedError("rocm.hip.BatchGenerate", hipKernelDecode, stub.Status())
}

func (stub hipKernelStub) Project(ctx context.Context, _ *hipLoadedModel, req hipProjectionRequest) ([]float32, error) {
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
	}
	if err := req.validate(); err != nil {
		return nil, err
	}
	return nil, hipKernelNotLinkedError("rocm.hip.Project", hipKernelProjection, stub.Status())
}

func (stub hipKernelStub) Prefill(ctx context.Context, _ *hipLoadedModel, req hipPrefillRequest) (hipPrefillResult, error) {
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return hipPrefillResult{}, err
		}
	}
	if err := req.validate(); err != nil {
		return hipPrefillResult{}, err
	}
	return hipPrefillResult{}, hipKernelNotLinkedError("rocm.hip.Prefill", hipKernelPrefill, stub.Status())
}

func (stub hipKernelStub) Decode(ctx context.Context, _ *hipLoadedModel, req hipDecodeRequest) (hipDecodeResult, error) {
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return hipDecodeResult{}, err
		}
	}
	if req.DeviceKV != nil || req.DescriptorTable != nil {
		if _, err := req.decodeLaunchArgsBytes(); err != nil {
			return hipDecodeResult{}, err
		}
	}
	return hipDecodeResult{}, hipKernelNotLinkedError("rocm.hip.Decode", hipKernelDecode, stub.Status())
}
