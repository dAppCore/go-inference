// SPDX-Licence-Identifier: EUPL-1.2

// Package registry is the model catalogue — the handoff seam between the training side
// and the serving side. The training side writes entries; the serving side reads them to
// route requests and decide residency. It holds no model weights, only the
// metadata placement and serving need.
//
//	r := registry.New()
//	r.Put(registry.Entry{ID: "gemma-4-4b-it", Aliases: []string{"lemma"}, MemoryBytes: 4_500_000_000})
//	e := r.Resolve("lemma")           // id-or-alias → Entry
//	fits := r.FitsDevice(8 << 30)     // entries that fit an 8 GiB budget
package registry

// Format is the on-disk weight format of a catalogue entry.
//
//	if e.Format == registry.FormatGGUF { ... }
type Format string

const (
	// FormatSafetensors is the HuggingFace safetensors layout.
	FormatSafetensors Format = "safetensors"
	// FormatGGUF is the llama.cpp GGUF single-file layout.
	FormatGGUF Format = "gguf"
	// FormatPEFT is a LoRA / PEFT adapter applied over a base model.
	FormatPEFT Format = "peft"
)

// Status is the lifecycle state of a catalogue entry.
//
//	if e.Status == registry.StatusReady { serve(e) }
type Status string

const (
	// StatusDraft is registered but not yet servable (training / converting).
	StatusDraft Status = "draft"
	// StatusReady is published and available to serve.
	StatusReady Status = "ready"
	// StatusArchived is retained for provenance but withdrawn from serving.
	StatusArchived Status = "archived"
)

// Capabilities advertises what a loaded model can do — read by the serving router
// to match a request to a capable model.
//
//	if req.NeedsTools && !e.Capabilities.Tools { skip }
type Capabilities struct {
	Tools     bool `json:"tools"`     // native function / tool calling
	Vision    bool `json:"vision"`    // image input
	Grammar   bool `json:"grammar"`   // grammar-constrained / structured output
	Streaming bool `json:"streaming"` // token streaming
}

// Lineage records adapter / base relationships and the run that produced the
// model — for management, reproducibility, and model cards.
//
//	e.Lineage = registry.Lineage{Base: "gemma-4-4b-it", RunID: "lem-2026-06-14", Checkpoint: "step-3000"}
type Lineage struct {
	Base       string   `json:"base,omitempty"`       // base model id this entry derives from
	Adapters   []string `json:"adapters,omitempty"`   // applied adapter ids (LoRA / PEFT)
	RunID      string   `json:"run_id,omitempty"`     // training run that produced it
	Checkpoint string   `json:"checkpoint,omitempty"` // checkpoint / step within the run
}

// Source is where the weights live — a local path or a remote provider.
//
//	registry.Source{LocalPath: "/models/lemma"}
//	registry.Source{Provider: "openrouter", Remote: "google/gemma-4-4b-it"}
type Source struct {
	LocalPath string `json:"local_path,omitempty"` // on-disk model / adapter directory
	Provider  string `json:"provider,omitempty"`   // remote provider id (e.g. "openrouter")
	Remote    string `json:"remote,omitempty"`     // provider-side model identifier
}

// Entry is a model catalogue entry. It carries the identity, capability,
// placement, lineage, and source metadata that the registry indexes. It mirrors
// the fields of inference.ModelInfo (Architecture, params, quant) without
// importing it, so this subpackage stays pure-Go and CGO-free.
//
//	e := registry.Entry{
//	    ID:           "gemma-4-4b-it",
//	    Aliases:      []string{"lemma"},
//	    Architecture: "gemma4",
//	    Params:       4_500_000_000,
//	    MemoryBytes:  4_500_000_000,
//	    Format:       registry.FormatGGUF,
//	    Status:       registry.StatusReady,
//	}
type Entry struct {
	ID            string       `json:"id"`                       // canonical identifier
	Aliases       []string     `json:"aliases,omitempty"`        // alternative names that resolve to this entry
	Architecture  string       `json:"architecture,omitempty"`   // e.g. "gemma4", "qwen3"
	Params        int64        `json:"params,omitempty"`         // parameter count
	ContextLength int          `json:"context_length,omitempty"` // max context window in tokens
	Quantisation  string       `json:"quantisation,omitempty"`   // e.g. "Q4_K_M", "bf16"
	Format        Format       `json:"format,omitempty"`         // safetensors / gguf / peft
	MemoryBytes   uint64       `json:"memory_bytes,omitempty"`   // resident memory footprint
	DeviceFit     []string     `json:"device_fit,omitempty"`     // device / runtime ids that can hold it
	Lineage       Lineage      `json:"lineage,omitempty"`        // adapter / base / provenance
	Capabilities  Capabilities `json:"capabilities,omitempty"`   // tools / vision / grammar / streaming
	Source        Source       `json:"source,omitempty"`         // local path or remote provider
	Status        Status       `json:"status,omitempty"`         // draft / ready / archived
	Card          *ModelCard   `json:"card,omitempty"`           // EU AI Act model card (intended use, provenance, eval, limitations)
}

// Filter selects entries by capability and status. The zero Filter matches
// everything; each true field narrows the set. Used by Filter, FitsDeviceWith.
//
//	r.Filter(registry.Filter{Tools: true, ReadyOnly: true})
type Filter struct {
	Tools     bool // require tool calling
	Vision    bool // require image input
	Grammar   bool // require grammar / structured output
	Streaming bool // require token streaming
	ReadyOnly bool // require Status == StatusReady
}

// matches reports whether e satisfies every set field of f.
//
//	if (registry.Filter{Tools: true}).matches(e) { ... }
func (f Filter) matches(e Entry) bool {
	if f.Tools && !e.Capabilities.Tools {
		return false
	}
	if f.Vision && !e.Capabilities.Vision {
		return false
	}
	if f.Grammar && !e.Capabilities.Grammar {
		return false
	}
	if f.Streaming && !e.Capabilities.Streaming {
		return false
	}
	if f.ReadyOnly && e.Status != StatusReady {
		return false
	}
	return true
}
