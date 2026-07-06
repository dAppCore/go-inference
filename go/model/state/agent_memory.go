// SPDX-Licence-Identifier: EUPL-1.2

package state

import "context"

// Ref identifies a durable model-state span. It is URI-first so runtimes can
// back it with memvid, a local file log, object storage, or another store
// without depending on a concrete adapter.
type Ref struct {
	URI        string            `json:"uri,omitempty"`
	BundleURI  string            `json:"bundle_uri,omitempty"`
	IndexURI   string            `json:"index_uri,omitempty"`
	Title      string            `json:"title,omitempty"`
	Kind       string            `json:"kind,omitempty"`
	Hash       string            `json:"hash,omitempty"`
	TokenStart int               `json:"token_start,omitempty"`
	TokenCount int               `json:"token_count,omitempty"`
	ByteStart  int64             `json:"byte_start,omitempty"`
	ByteCount  int64             `json:"byte_count,omitempty"`
	StateRefs  []StateRef        `json:"state_refs,omitempty"`
	Labels     map[string]string `json:"labels,omitempty"`
}

// WakeRequest selects a durable state prefix to restore. Store is an opaque
// runtime-owned handle and is deliberately omitted from JSON.
type WakeRequest struct {
	Store                  any               `json:"-"`
	IndexURI               string            `json:"index_uri,omitempty"`
	EntryURI               string            `json:"entry_uri,omitempty"`
	Model                  ModelIdentity     `json:"model"`
	Tokenizer              TokenizerIdentity `json:"tokenizer"`
	Adapter                AdapterIdentity   `json:"adapter"`
	Runtime                RuntimeIdentity   `json:"runtime"`
	SkipCompatibilityCheck bool              `json:"skip_compatibility_check,omitempty"`
	Labels                 map[string]string `json:"labels,omitempty"`
}

// WakeResult reports the durable prefix restored into a session.
type WakeResult struct {
	Entry        Ref               `json:"entry"`
	Bundle       StateRef          `json:"bundle"`
	Index        StateRef          `json:"index"`
	PrefixTokens int               `json:"prefix_tokens,omitempty"`
	BundleTokens int               `json:"bundle_tokens,omitempty"`
	BlockSize    int               `json:"block_size,omitempty"`
	BlocksRead   int               `json:"blocks_read,omitempty"`
	Labels       map[string]string `json:"labels,omitempty"`
}

// SleepRequest asks a live session to persist its current state. Store is an
// opaque runtime-owned handle and is deliberately omitted from JSON.
type SleepRequest struct {
	Store             any               `json:"-"`
	EntryURI          string            `json:"entry_uri,omitempty"`
	BundleURI         string            `json:"bundle_uri,omitempty"`
	IndexURI          string            `json:"index_uri,omitempty"`
	ParentEntryURI    string            `json:"parent_entry_uri,omitempty"`
	ParentBundleURI   string            `json:"parent_bundle_uri,omitempty"`
	ParentIndexURI    string            `json:"parent_index_uri,omitempty"`
	Title             string            `json:"title,omitempty"`
	Model             ModelIdentity     `json:"model"`
	Tokenizer         TokenizerIdentity `json:"tokenizer"`
	Adapter           AdapterIdentity   `json:"adapter"`
	Runtime           RuntimeIdentity   `json:"runtime"`
	ReuseParentPrefix bool              `json:"reuse_parent_prefix,omitempty"`
	BlockSize         int               `json:"block_size,omitempty"`
	Encoding          string            `json:"encoding,omitempty"`
	Labels            map[string]string `json:"labels,omitempty"`
	Metadata          map[string]string `json:"metadata,omitempty"`
}

// SleepResult reports the durable state written by a session.
type SleepResult struct {
	Entry         Ref               `json:"entry"`
	Parent        Ref               `json:"parent"`
	Bundle        StateRef          `json:"bundle"`
	Index         StateRef          `json:"index"`
	TokenCount    int               `json:"token_count,omitempty"`
	BlockSize     int               `json:"block_size,omitempty"`
	BlocksWritten int               `json:"blocks_written,omitempty"`
	BlocksReused  int               `json:"blocks_reused,omitempty"`
	Encoding      string            `json:"encoding,omitempty"`
	Labels        map[string]string `json:"labels,omitempty"`
}

// Session is implemented by live sessions that can wake from and sleep to
// durable model-state storage.
type Session interface {
	WakeState(ctx context.Context, req WakeRequest) (*WakeResult, error)
	SleepState(ctx context.Context, req SleepRequest) (*SleepResult, error)
}

// Forker creates an independent live session from durable state.
type Forker interface {
	ForkState(ctx context.Context, req WakeRequest) (Session, *WakeResult, error)
}

type AgentMemoryRef = Ref
type AgentMemoryWakeRequest = WakeRequest
type AgentMemoryWakeResult = WakeResult
type AgentMemorySleepRequest = SleepRequest
type AgentMemorySleepResult = SleepResult
type AgentMemorySession = Session
type AgentMemoryForker = Forker
