// SPDX-Licence-Identifier: EUPL-1.2

// serve_embed.go wires an optional embeddings/rerank model into a serve
// alongside (or instead of) the chat model — the -embed-model flag's library
// half (see cmd/lem/serve.go). It never introduces a new route: the compat
// mux's existing /v1/embeddings and /v1/rerank handlers already resolve
// through the SAME resolver every chat route uses and unwrap to whatever
// capability the resolved model carries (inference.BaseTextModel — see
// serving/provider/openai/services.go); this file only teaches the resolver
// one more name.
package serving

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/model/arch/bert"
	openai "dappco.re/go/inference/serving/provider/openai"
)

// loadEmbedModel reads a bert/BGE-class encoder snapshot from path (see
// model/bert — config.json + vocab.txt + model.safetensors, plus the optional
// sentence-transformers pooling/normalise files) and wraps it as a servable
// inference.TextModel. id is the caller's requested request-facing name
// (ServeConfig.EmbedModelID); when empty the pack's own basename is returned
// instead, mirroring how the chat model's default id is derived
// (core.PathBase — see serveHost.listModels in RunServe).
//
//	model, id, err := loadEmbedModel("/path/to/bge-small-en-v1.5", "")
func loadEmbedModel(path, id string) (inference.TextModel, string, error) {
	loaded, err := bert.Load(path)
	if err != nil {
		return nil, "", core.E("serving.loadEmbedModel", core.Sprintf("load embeddings model %q", path), err)
	}
	if core.Trim(id) == "" {
		id = core.PathBase(path)
	}
	return bert.NewServeModel(loaded), id, nil
}

// wrapEmbedResolver decorates inner so a request naming id (case-insensitive,
// trimmed — matching openai.StaticResolver's own comparison) resolves to model
// without ever reaching inner; every other name falls through unchanged. A
// serve started with -model "" (model-less) and -embed-model set routes
// EVERY other name to inner's "no model loaded" error exactly as it does
// today — this wrap adds a name, it never removes one.
//
//	resolver = wrapEmbedResolver(resolver, "bge-small-en-v1.5", embedModel)
func wrapEmbedResolver(inner Resolver, id string, model inference.TextModel) Resolver {
	canonical := core.Lower(core.Trim(id))
	return openai.ResolverFunc(func(ctx context.Context, name string) (inference.TextModel, error) {
		if core.Lower(core.Trim(name)) == canonical {
			return model, nil
		}
		return inner.ResolveModel(ctx, name)
	})
}
