// SPDX-Licence-Identifier: EUPL-1.2

package state

// ModelIdentity carries backend-neutral model metadata for state bundles,
// benchmark reports, fit planning, and adapter compatibility checks.
type ModelIdentity struct {
	ID            string            `json:"id,omitempty"`
	Path          string            `json:"path,omitempty"`
	Architecture  string            `json:"architecture,omitempty"`
	Revision      string            `json:"revision,omitempty"`
	Hash          string            `json:"hash,omitempty"`
	QuantBits     int               `json:"quant_bits,omitempty"`
	QuantGroup    int               `json:"quant_group,omitempty"`
	QuantType     string            `json:"quant_type,omitempty"`
	ContextLength int               `json:"context_length,omitempty"`
	NumLayers     int               `json:"num_layers,omitempty"`
	HiddenSize    int               `json:"hidden_size,omitempty"`
	VocabSize     int               `json:"vocab_size,omitempty"`
	Labels        map[string]string `json:"labels,omitempty"`
}

// TokenizerIdentity carries tokenizer and chat-template metadata without
// exposing backend-specific tokenizer implementations.
type TokenizerIdentity struct {
	Kind         string            `json:"kind,omitempty"`
	Path         string            `json:"path,omitempty"`
	Hash         string            `json:"hash,omitempty"`
	ChatTemplate string            `json:"chat_template,omitempty"`
	BOSID        int32             `json:"bos_id,omitempty"`
	EOSID        int32             `json:"eos_id,omitempty"`
	PADID        int32             `json:"pad_id,omitempty"`
	Labels       map[string]string `json:"labels,omitempty"`
}

// AdapterIdentity is the portable identity for an active or saved adapter.
type AdapterIdentity struct {
	Path          string            `json:"path,omitempty"`
	Hash          string            `json:"hash,omitempty"`
	Format        string            `json:"format,omitempty"`
	Rank          int               `json:"rank,omitempty"`
	Alpha         float32           `json:"alpha,omitempty"`
	TargetKeys    []string          `json:"target_keys,omitempty"`
	BaseModelHash string            `json:"base_model_hash,omitempty"`
	Labels        map[string]string `json:"labels,omitempty"`
}

// RuntimeIdentity records runtime and device metadata for reproducibility.
type RuntimeIdentity struct {
	Backend       string            `json:"backend,omitempty"`
	Device        string            `json:"device,omitempty"`
	Version       string            `json:"version,omitempty"`
	CacheMode     string            `json:"cache_mode,omitempty"`
	NativeRuntime bool              `json:"native_runtime,omitempty"`
	Labels        map[string]string `json:"labels,omitempty"`
}

// SamplerConfig is the serializable form of generation sampler settings.
type SamplerConfig struct {
	MaxTokens     int      `json:"max_tokens,omitempty"`
	Temperature   float32  `json:"temperature,omitempty"`
	TopK          int      `json:"top_k,omitempty"`
	TopP          float32  `json:"top_p,omitempty"`
	RepeatPenalty float32  `json:"repeat_penalty,omitempty"`
	StopTokens    []int32  `json:"stop_tokens,omitempty"`
	StopSequences []string `json:"stop_sequences,omitempty"`
	ReturnLogits  bool     `json:"return_logits,omitempty"`
}

// StateRef points to backend-owned binary state, probe, or knowledge-pack data.
type StateRef struct {
	Kind      string            `json:"kind,omitempty"`
	URI       string            `json:"uri,omitempty"`
	Hash      string            `json:"hash,omitempty"`
	SizeBytes uint64            `json:"size_bytes,omitempty"`
	Encoding  string            `json:"encoding,omitempty"`
	Labels    map[string]string `json:"labels,omitempty"`
}

// Bundle is a portable state envelope. It contains metadata and references,
// not backend tensor objects.
type Bundle struct {
	Version         string            `json:"version,omitempty"`
	CreatedAtUnix   int64             `json:"created_at_unix,omitempty"`
	Model           ModelIdentity     `json:"model,omitempty"`
	Tokenizer       TokenizerIdentity `json:"tokenizer,omitempty"`
	Adapter         AdapterIdentity   `json:"adapter,omitempty"`
	Sampler         SamplerConfig     `json:"sampler,omitempty"`
	Runtime         RuntimeIdentity   `json:"runtime,omitempty"`
	PromptHash      string            `json:"prompt_hash,omitempty"`
	PromptTokens    int               `json:"prompt_tokens,omitempty"`
	GeneratedTokens int               `json:"generated_tokens,omitempty"`
	KVRefs          []StateRef        `json:"kv_refs,omitempty"`
	ProbeRefs       []StateRef        `json:"probe_refs,omitempty"`
	MemvidRefs      []StateRef        `json:"memvid_refs,omitempty"`
	Labels          map[string]string `json:"labels,omitempty"`
}

// StateBundle keeps the previous package-level name available for callers
// that want the longer explicit spelling.
type StateBundle = Bundle
