// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

// Features is the Gemma 4 architecture's feature surface: what the engine reads
// off a loaded config to configure itself. It is deliberately NOT a list of
// models — there are hundreds of Gemma 4 builds across orgs, quants, and
// fine-tunes, and the engine reacts to what a config declares, never to a model
// name or quant. Adding a new member of the family is "load its config"; the
// engine asks FeaturesOf and reacts, with no code change.
//
//	f := gemma4.FeaturesOf(model.Cfg)
//	if f.Mixture { /* route through the MoE experts path */ }
//	if f.Vision { /* load the vision tower */ }
type Features struct {
	Mixture     bool           // mixture-of-experts block active (vs a dense MLP)
	NumExperts  int            // total experts when Mixture, 0 when dense
	TopKExperts int            // experts routed per token when Mixture, 0 when dense
	Vision      bool           // vision encoder present
	Audio       bool           // audio encoder present
	Attention   AttentionClass // the attention topology the engine must provide
}

// AttentionClass is the attention topology a Gemma-4 build declares from its
// config, so the engine selects kernels (sliding-window local vs full global,
// shared-KV reuse) by what the model IS — never by its name. A future family
// that needs flash or sparse attention declares it the same way and the engine
// reacts; the engine never name-branches on "gemma4".
type AttentionClass struct {
	// SlidingWindow is the local-attention span. 0 = full attention on every
	// layer. >0 = the build alternates sliding-window local layers with
	// periodic full-attention (global) layers — Gemma-4's hybrid attention.
	SlidingWindow int
	// SlidingPattern is the cadence of full-attention layers among sliding ones
	// (e.g. 6 → every 6th layer is full attention). 0 when not hybrid.
	SlidingPattern int
	// SharedKVLayers is the count of trailing layers that reuse an earlier
	// layer's KV cache (Gemma-4 shared-KV). 0 when none.
	SharedKVLayers int
}

// Hybrid reports whether the build alternates sliding-window and full attention
// (vs a single dense attention on every layer). Drives the fixed-sliding KV
// cache selection.
func (a AttentionClass) Hybrid() bool { return a.SlidingWindow > 0 }

// FeaturesOf reads the feature surface from a loaded Gemma 4 config. A nil config
// reports the zero surface (dense, text-only). This is the single place that
// answers "what is this model" from its settings, so callers react to the
// returned Features rather than poking config fields — a new family member then
// needs no engine change, only a config.
func FeaturesOf(cfg *Gemma4TextConfig) Features {
	if cfg == nil {
		return Features{}
	}
	f := Features{
		Mixture: cfg.EnableMoEBlock,
		Vision:  cfg.VisionConfig != nil,
		Audio:   cfg.AudioConfig != nil,
		Attention: AttentionClass{
			SlidingWindow:  int(cfg.SlidingWindow),
			SlidingPattern: int(cfg.SlidingWindowPattern),
			SharedKVLayers: int(cfg.NumKVSharedLayers),
		},
	}
	if f.Mixture {
		if cfg.NumExperts != nil {
			f.NumExperts = int(*cfg.NumExperts)
		}
		if cfg.TopKExperts != nil {
			f.TopKExperts = int(*cfg.TopKExperts)
		}
	}
	return f
}
