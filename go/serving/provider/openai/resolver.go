// SPDX-Licence-Identifier: EUPL-1.2

// Resolvers map request model names to loaded inference.TextModel values.
package openai

import (
	"context"
	"sync"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// Resolver maps request model names to loaded inference models.
type Resolver interface {
	ResolveModel(ctx context.Context, name string) (inference.TextModel, error)
}

type ResolverFunc func(context.Context, string) (inference.TextModel, error)

func (fn ResolverFunc) ResolveModel(ctx context.Context, name string) (inference.TextModel, error) {
	if fn == nil {
		return nil, core.E("openai.ResolverFunc", "resolver is nil", nil)
	}
	return fn(ctx, name)
}

type StaticResolver struct {
	models map[string]inference.TextModel
}

func NewStaticResolver(models map[string]inference.TextModel) *StaticResolver {
	resolver := &StaticResolver{models: make(map[string]inference.TextModel, len(models))}
	for name, model := range models {
		resolver.models[core.Lower(core.Trim(name))] = model
	}
	return resolver
}

func (r *StaticResolver) ResolveModel(ctx context.Context, name string) (inference.TextModel, error) {
	if ctx != nil {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
	}
	if r == nil {
		return nil, core.E("openai.StaticResolver", "resolver is nil", nil)
	}
	model, ok := r.models[core.Lower(core.Trim(name))]
	if !ok || model == nil {
		return nil, core.E("openai.StaticResolver", core.Sprintf("model %q not found", name), nil)
	}
	return model, nil
}

// BackendResolver lazily loads one model through the inference backend registry.
type BackendResolver struct {
	BackendName string
	ModelPath   string
	LoadOptions []inference.LoadOption

	mu    sync.Mutex
	model inference.TextModel
}

func NewBackendResolver(backendName, modelPath string, opts ...inference.LoadOption) *BackendResolver {
	return &BackendResolver{
		BackendName: core.Trim(backendName),
		ModelPath:   core.Trim(modelPath),
		LoadOptions: append([]inference.LoadOption(nil), opts...),
	}
}

func (r *BackendResolver) ResolveModel(ctx context.Context, _ string) (inference.TextModel, error) {
	if ctx != nil {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
	}
	if r == nil {
		return nil, core.E("openai.BackendResolver", "resolver is nil", nil)
	}
	if r.ModelPath == "" {
		return nil, core.E("openai.BackendResolver", "model path is required", nil)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.model != nil {
		return r.model, nil
	}
	opts := append([]inference.LoadOption(nil), r.LoadOptions...)
	if r.BackendName != "" {
		opts = append(opts, inference.WithBackend(r.BackendName))
	}
	result := inference.LoadModel(r.ModelPath, opts...)
	if !result.OK {
		return nil, result.Err()
	}
	model, ok := result.Value.(inference.TextModel)
	if !ok || model == nil {
		return nil, core.E("openai.BackendResolver", "loaded value is not an inference.TextModel", nil)
	}
	r.model = model
	return model, nil
}
