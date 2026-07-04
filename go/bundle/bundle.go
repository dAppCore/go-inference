// SPDX-Licence-Identifier: EUPL-1.2

// Package bundle is the portable model-state artifact for go-mlx
// sessions: a kv.Snapshot plus the tokenizer, runtime, adapter, and
// sampler identity needed to safely replay it on a different host.
//
//	b, err := bundle.New(snapshot, bundle.Options{
//	    Model: "gemma4-e4b", ModelPath: "/models/gemma4",
//	    Source: bundle.ModelInfo{Architecture: "gemma4_text", NumLayers: 32},
//	})
package bundle

import (
	"context"
	"crypto/sha256"
	"io"
	"strconv"

	core "dappco.re/go"
	"dappco.re/go/inference/kv"
	"dappco.re/go/inference/lora"
	state "dappco.re/go/inference/state"
)

const (
	// Version is the portable bundle schema version.
	Version = 1
	// Kind identifies go-mlx state-bundle JSON payloads.
	Kind = "go-mlx/state-bundle"
	// RefState identifies a State cold-storage reference.
	RefState = "state"
	// RefMemvid identifies an old memvid cold-storage reference.
	//
	// Deprecated: use RefState.
	RefMemvid = "memvid"
)

// Constant validation errors hoisted to package vars — each previously
// allocated a fresh core.NewError on the (rare but hot under churn)
// failure path. errBundleNil fires 4×, errBundleKVHash 3×,
// errBundleNoSnapshot 2× from validation/load/restore guards.
var (
	errBundleNil                = core.NewError("bundle: state bundle is nil")
	errBundleKVHash             = core.NewError("bundle: state bundle KV hash mismatch")
	errBundleNoSnapshot         = core.NewError("bundle: state bundle has no KV snapshot")
	errCoreResultFailed         = core.NewError("core result failed")
	errBundleUnsupportedVersion = core.NewError("bundle: unsupported state bundle version")
	errBundleNeedsLoRA          = core.NewError("bundle: state bundle requires a LoRA adapter but model has none")
	errBundleLayerMismatch      = core.NewError("bundle: state bundle model layer mismatch")
	errBundleArchMismatch       = core.NewError("bundle: state bundle model architecture mismatch")
	errBundleLoRARank           = core.NewError("bundle: state bundle LoRA adapter rank mismatch")
	errBundleLoRAPath           = core.NewError("bundle: state bundle LoRA adapter path mismatch")
	errBundleLoRAHash           = core.NewError("bundle: state bundle LoRA adapter hash mismatch")
	errBundleLoRAAlpha          = core.NewError("bundle: state bundle LoRA adapter alpha mismatch")
	errBundleNoStateKVSnapshot  = core.NewError("bundle: state bundle has no State KV snapshot")
	errBundleKVSnapshotNil      = core.NewError("bundle: KV snapshot is nil")
	errBundleInvalidKind        = core.NewError("bundle: invalid state bundle kind")
)

// Options labels a bundle with caller-owned provenance.
type Options struct {
	Model       string
	ModelPath   string
	Source      ModelInfo
	Prompt      string
	Tokenizer   Tokenizer
	Runtime     Runtime
	Adapter     Adapter
	AdapterPath string
	KVPath      string
	Sampler     Sampler
	Analysis    *kv.Analysis
	SAMI        *SAMIResult
	Refs        []Ref
	StateRefs   []state.ChunkRef
	// Deprecated: use StateRefs.
	MemvidRefs []state.ChunkRef
	Meta       map[string]string
}

// ModelInfo describes the model expected by a bundle. Mirrors the
// mlx-root ModelInfo struct; converters at the boundary keep the two in
// sync.
type ModelInfo struct {
	Architecture  string
	VocabSize     int
	NumLayers     int
	HiddenSize    int
	QuantBits     int
	QuantGroup    int
	ContextLength int
	Adapter       lora.AdapterInfo
}

// Bundle is a portable, strict model-state artifact.
type Bundle struct {
	Version   int               `json:"version"`
	Kind      string            `json:"kind"`
	Model     Model             `json:"model"`
	Prompt    Prompt            `json:"prompt"`
	Tokenizer Tokenizer         `json:"tokenizer"`
	Runtime   Runtime           `json:"runtime"`
	Adapter   Adapter           `json:"adapter"`
	Sampler   Sampler           `json:"sampler"`
	KV        *kv.Snapshot      `json:"kv,omitempty"`
	KVPath    string            `json:"kv_path,omitempty"`
	KVHash    string            `json:"kv_hash"`
	Analysis  *kv.Analysis      `json:"analysis,omitempty"`
	SAMI      *SAMIResult       `json:"sami,omitempty"`
	Refs      []Ref             `json:"refs,omitempty"`
	Meta      map[string]string `json:"meta,omitempty"`
}

// Model identifies the model captured by the bundle.
type Model struct {
	Name          string `json:"name,omitempty"`
	Path          string `json:"path,omitempty"`
	Architecture  string `json:"architecture"`
	VocabSize     int    `json:"vocab_size,omitempty"`
	NumLayers     int    `json:"num_layers,omitempty"`
	HiddenSize    int    `json:"hidden_size,omitempty"`
	QuantBits     int    `json:"quant_bits,omitempty"`
	QuantGroup    int    `json:"quant_group,omitempty"`
	ContextLength int    `json:"context_length,omitempty"`
	Hash          string `json:"hash,omitempty"`
}

// Prompt identifies the prompt/token state captured by the bundle.
type Prompt struct {
	Text        string `json:"text,omitempty"`
	Hash        string `json:"hash,omitempty"`
	TokenCount  int    `json:"token_count"`
	TokenOffset int    `json:"token_offset"`
}

// Tokenizer identifies tokenizer and chat-template compatibility.
type Tokenizer struct {
	Kind             string `json:"kind,omitempty"`
	Path             string `json:"path,omitempty"`
	Version          string `json:"version,omitempty"`
	Hash             string `json:"hash,omitempty"`
	VocabSize        int    `json:"vocab_size,omitempty"`
	BOS              int32  `json:"bos,omitempty"`
	EOS              int32  `json:"eos,omitempty"`
	ChatTemplate     string `json:"chat_template,omitempty"`
	ChatTemplateHash string `json:"chat_template_hash,omitempty"`
}

// Runtime identifies the go-mlx runtime that created the bundle.
type Runtime struct {
	Name     string `json:"name,omitempty"`
	Version  string `json:"version,omitempty"`
	Build    string `json:"build,omitempty"`
	Platform string `json:"platform,omitempty"`
}

// Adapter identifies an optional LoRA adapter applied to the model.
type Adapter struct {
	Name       string   `json:"name,omitempty"`
	Path       string   `json:"path,omitempty"`
	Hash       string   `json:"hash,omitempty"`
	Rank       int      `json:"rank,omitempty"`
	Alpha      float32  `json:"alpha,omitempty"`
	Scale      float32  `json:"scale,omitempty"`
	TargetKeys []string `json:"target_keys,omitempty"`
}

// Sampler stores generation settings needed for reproducible replay.
type Sampler struct {
	MaxTokens     int     `json:"max_tokens"`
	Temperature   float32 `json:"temperature"`
	TopK          int     `json:"top_k"`
	TopP          float32 `json:"top_p"`
	MinP          float32 `json:"min_p"`
	StopTokens    []int32 `json:"stop_tokens,omitempty"`
	RepeatPenalty float32 `json:"repeat_penalty"`
}

// Ref links external cold-storage artifacts such as State chunks.
type Ref struct {
	Kind   string         `json:"kind"`
	URI    string         `json:"uri"`
	Hash   string         `json:"hash,omitempty"`
	Title  string         `json:"title,omitempty"`
	Track  string         `json:"track,omitempty"`
	State  state.ChunkRef `json:"state"`
	Memvid state.ChunkRef `json:"memvid"`
}

// New builds a portable bundle around a restorable kv.Snapshot.
//
//	b, err := bundle.New(snapshot, bundle.Options{Model: "gemma4-e4b"})
func New(snapshot *kv.Snapshot, opts Options) (*Bundle, error) {
	if snapshot == nil {
		return nil, errBundleKVSnapshotNil
	}
	snap := snapshot.Clone()
	if snap.Version == 0 {
		snap.Version = kv.SnapshotVersion
	}
	tokenCount := len(snap.Tokens)
	if snap.TokenOffset == 0 {
		snap.TokenOffset = tokenCount
	}
	kvHash, err := kv.HashSnapshot(snap)
	if err != nil {
		return nil, err
	}
	analysis := opts.Analysis
	if analysis == nil {
		analysis = kv.Analyze(snap)
	}
	sami := opts.SAMI
	if sami == nil {
		result := SAMIFromKV(snap, analysis, SAMIOptions{Model: opts.Model, Prompt: opts.Prompt})
		sami = &result
	}
	model := buildModel(snap, opts)
	tokenizer := NormaliseTokenizer(opts.Tokenizer)
	runtime := normaliseRuntime(opts.Runtime)
	adapter := buildAdapter(opts.Adapter, opts.AdapterPath, opts.Source.Adapter)
	b := &Bundle{
		Version: Version,
		Kind:    Kind,
		Model:   model,
		Prompt: Prompt{
			Text:        opts.Prompt,
			Hash:        HashString(opts.Prompt),
			TokenCount:  tokenCount,
			TokenOffset: snap.TokenOffset,
		},
		Tokenizer: tokenizer,
		Runtime:   runtime,
		Adapter:   adapter,
		Sampler:   opts.Sampler,
		KV:        snap,
		KVPath:    opts.KVPath,
		KVHash:    kvHash,
		Analysis:  analysis,
		SAMI:      sami,
		Refs:      buildRefs(opts.Refs, joinChunkRefs(opts.StateRefs, opts.MemvidRefs)),
		Meta:      cloneMeta(opts.Meta),
	}
	if AdapterEmpty(b.Adapter) {
		b.Adapter = Adapter{}
	}
	return b, nil
}

// Save writes the bundle as stable indented JSON.
//
//	if err := b.Save(path); err != nil { … }
//
// The two-space indent is the human-debug contract: `Save` output is the
// canonical artifact developers `cat` / diff during a session crash or a
// bundle-shape audit. Switching this to compact JSON would break that
// contract — use SaveCompact when disk footprint matters more than
// readability (cold-storage, State-container packaging, archive tiers).
func (b *Bundle) Save(path string) error {
	if err := b.Validate(); err != nil {
		return err
	}
	data := core.JSONMarshalIndent(b, "", "  ")
	if !data.OK {
		return core.E("bundle.Save", "marshal bundle", resultError(data))
	}
	if result := core.WriteFile(path, data.Value.([]byte), 0o600); !result.OK {
		return core.E("bundle.Save", "write bundle", resultError(result))
	}
	return nil
}

// SaveCompact writes the bundle as newlineless JSON for cold storage.
//
//	if err := b.SaveCompact(path); err != nil { … }
//
// Wire-identical to Save — same field order, same value encoding, same
// `Load` round-trips both forms. The only difference is whitespace:
// `Save` emits `{\n  "version": 1,\n  ...}` (~75% whitespace on a typical
// bundle); `SaveCompact` emits `{"version":1,...}`. Pair with State
// container packaging (.mp4 chunks embedding bundle headers) or any
// archive tier where on-disk footprint dominates human-debug ergonomics.
// Load auto-detects both — no SaveCompact-specific reader needed.
func (b *Bundle) SaveCompact(path string) error {
	if err := b.Validate(); err != nil {
		return err
	}
	data := core.JSONMarshal(b)
	if !data.OK {
		return core.E("bundle.SaveCompact", "marshal bundle", resultError(data))
	}
	if result := core.WriteFile(path, data.Value.([]byte), 0o600); !result.OK {
		return core.E("bundle.SaveCompact", "write bundle", resultError(result))
	}
	return nil
}

// Load reads a bundle saved by (*Bundle).Save or (*Bundle).SaveCompact.
//
//	b, err := bundle.Load(path)
func Load(path string) (*Bundle, error) {
	read := core.ReadFile(path)
	if !read.OK {
		return nil, core.E("bundle.Load", "read bundle", resultError(read))
	}
	data, ok := read.Value.([]byte)
	if !ok {
		return nil, core.E("bundle.Load", "read bundle returned non-byte data", nil)
	}
	var b Bundle
	if result := core.JSONUnmarshal(data, &b); !result.OK {
		return nil, core.E("bundle.Load", "parse bundle", resultError(result))
	}
	if err := b.Validate(); err != nil {
		return nil, err
	}
	return &b, nil
}

// Snapshot returns a defensive kv.Snapshot copy, loading KVPath when needed.
//
//	snap, err := b.Snapshot()
func (b *Bundle) Snapshot() (*kv.Snapshot, error) {
	if b == nil {
		return nil, errBundleNil
	}
	if b.KV != nil {
		return b.KV.Clone(), nil
	}
	if b.KVPath == "" {
		return nil, errBundleNoSnapshot
	}
	snapshot, err := kv.Load(b.KVPath)
	if err != nil {
		return nil, err
	}
	if b.KVHash != "" {
		got, hashErr := kv.HashSnapshot(snapshot)
		if hashErr != nil {
			return nil, hashErr
		}
		if got != b.KVHash {
			return nil, errBundleKVHash
		}
	}
	return snapshot, nil
}

// SnapshotFromState resolves a State-backed KV snapshot.
//
//	snap, err := b.SnapshotFromState(ctx, store)
func (b *Bundle) SnapshotFromState(ctx context.Context, store state.Store) (*kv.Snapshot, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if b == nil {
		return nil, errBundleNil
	}
	if b.KV != nil || b.KVPath != "" {
		return b.Snapshot()
	}
	ref, ok := b.stateRef()
	if !ok {
		return nil, errBundleNoStateKVSnapshot
	}
	snapshot, err := kv.LoadFromState(ctx, store, ref)
	if err != nil {
		return nil, err
	}
	if b.KVHash != "" {
		got, hashErr := kv.HashSnapshot(snapshot)
		if hashErr != nil {
			return nil, hashErr
		}
		if got != b.KVHash {
			return nil, errBundleKVHash
		}
	}
	return snapshot, nil
}

// SnapshotFromMemvid resolves an old memvid-backed KV snapshot.
//
// Deprecated: use SnapshotFromState.
func (b *Bundle) SnapshotFromMemvid(ctx context.Context, store state.Store) (*kv.Snapshot, error) {
	return b.SnapshotFromState(ctx, store)
}

func (b *Bundle) stateRef() (state.ChunkRef, bool) {
	if b == nil {
		return state.ChunkRef{}, false
	}
	refs := b.Refs
	for i := range refs {
		ref := &refs[i]
		switch ref.Kind {
		case RefState:
			// State refs prefer the typed State field; fall back to the
			// older Memvid field for migrated bundles.
			if ref.State.ChunkID != 0 {
				return ref.State, true
			}
			if ref.Memvid.ChunkID != 0 {
				return ref.Memvid, true
			}
		case RefMemvid:
			return ref.Memvid, true
		}
	}
	return state.ChunkRef{}, false
}

// Validate checks schema version, kind, and embedded KV hash integrity.
//
//	if err := b.Validate(); err != nil { … }
func (b *Bundle) Validate() error {
	if b == nil {
		return errBundleNil
	}
	if b.Version <= 0 || b.Version > Version {
		return errBundleUnsupportedVersion
	}
	if b.Kind != Kind {
		return errBundleInvalidKind
	}
	if b.KV == nil && b.KVPath == "" {
		if _, ok := b.stateRef(); !ok {
			return errBundleNoSnapshot
		}
		return nil
	}
	if b.KV != nil && b.KVHash != "" {
		got, err := kv.HashSnapshot(b.KV)
		if err != nil {
			return err
		}
		if got != b.KVHash {
			return errBundleKVHash
		}
	}
	return nil
}

// CheckCompatibility verifies that a loaded model can safely restore a bundle.
//
//	if err := bundle.CheckCompatibility(modelInfo, b); err != nil { … }
func CheckCompatibility(info ModelInfo, b *Bundle) error {
	if b == nil {
		return errBundleNil
	}
	if err := b.Validate(); err != nil {
		return err
	}
	if b.Model.Architecture != "" && info.Architecture != "" && b.Model.Architecture != info.Architecture {
		return errBundleArchMismatch
	}
	if b.Model.NumLayers > 0 && info.NumLayers > 0 && b.Model.NumLayers != info.NumLayers {
		return errBundleLayerMismatch
	}
	return checkAdapterCompatibility(info.Adapter, b.Adapter)
}

// fileHashStreamThreshold gates the buffer-load vs streaming fast-path
// inside FileHash. Files smaller than the threshold are read whole into
// an exactly-sized buffer (1 alloc of file_size) and hashed via
// `core.SHA256Hex`, which is cheaper than the stdlib `io.Copy` 32KB
// scratch path for sub-32KB inputs. Files at or above the threshold are
// streamed, capping per-call allocation at ~33KB regardless of file
// size — the dominant win on 1MB tokenizer shards and 10MB+ LoRA
// adapter weights. Threshold sits at the stdlib `io.Copy` default
// scratch size so the streaming path is only chosen when its scratch is
// genuinely smaller than the file would be.
const fileHashStreamThreshold = 32 * 1024

// FileHash hashes an external file for strict bundle metadata.
//
//	hash, err := bundle.FileHash(path)
//
// Single Open + fd-Stat, then size-conditional. The earlier shape did a
// standalone `core.Stat` (2 allocs — `os.Stat` plus the Result interface
// box) *before* reading, pure overhead on the small path that reads the
// file anyway and a redundant second namei walk on the large path that
// re-opens. Opening once and stat-ing the descriptor collapses both:
// small files (<32KB chat-templates, license blobs) read fully into an
// exactly-sized buffer and hash via `core.SHA256Hex`; large files
// (≥32KB tokenizer shards, LoRA adapter weights) stream through SHA-256
// via a fixed scratch, capping per-call allocation at ~33KB regardless
// of file size. Bit-exact with the legacy buffer-load path for any size
// — see `TestFileHash_StreamMatchesBufferLoad_Good`.
//
// `crypto/sha256` is reached for directly here because the SPOR
// `core.SHA256*` helpers operate on a complete []byte (i.e. the very
// load-the-whole-file path we are eliminating on large files). A
// streaming SHA-256 primitive belongs in `external/go/hash.go` — see
// W10-AG forward note — but until that lands upstream the local fix
// preserves bundle's streaming guarantee.
func FileHash(path string) (string, error) {
	opened := core.Open(path)
	if !opened.OK {
		return "", core.E("bundle.FileHash", "open file", resultError(opened))
	}
	file, ok := opened.Value.(*core.OSFile)
	if !ok {
		return "", core.E("bundle.FileHash", "open file returned non-file", nil)
	}
	defer file.Close()
	stat, err := file.Stat()
	if err != nil {
		return "", core.E("bundle.FileHash", "stat file", err)
	}
	if stat.Size() < fileHashStreamThreshold {
		// Exactly-sized buffer + ReadFull — one alloc of file_size, no
		// standalone Stat syscall and no io.Copy 32KB scratch. ReadFull on a
		// zero-length buffer is a no-op, so the empty-file boundary hashes
		// the empty digest like the legacy path.
		buf := make([]byte, stat.Size())
		if _, rerr := io.ReadFull(file, buf); rerr != nil {
			return "", core.E("bundle.FileHash", "read file", rerr)
		}
		return core.SHA256Hex(buf), nil
	}
	hasher := sha256.New()
	if r := core.Copy(hasher, file); !r.OK {
		return "", core.E("bundle.FileHash", "stream into hasher", resultError(r))
	}
	// Stack-resident digest scratch defeats hash.Sum's nil-path
	// 32-byte heap alloc; HexEncode still allocates the 64-byte
	// output string backing (unavoidable string return).
	var sum [sha256.Size]byte
	return core.HexEncode(hasher.Sum(sum[:0])), nil
}

// NormaliseTokenizer fills missing Tokenizer hash fields based on
// Path / ChatTemplate values.
//
//	t := bundle.NormaliseTokenizer(t)
func NormaliseTokenizer(tokenizer Tokenizer) Tokenizer {
	if tokenizer.Hash == "" && tokenizer.Path != "" {
		tokenizer.Hash = HashString(tokenizer.Path)
	}
	if tokenizer.ChatTemplateHash == "" && tokenizer.ChatTemplate != "" {
		tokenizer.ChatTemplateHash = HashString(tokenizer.ChatTemplate)
	}
	return tokenizer
}

// AdapterEmpty reports whether the adapter has no meaningful fields set.
//
//	if bundle.AdapterEmpty(a) { … }
func AdapterEmpty(adapter Adapter) bool {
	return adapter.Name == "" && adapter.Path == "" && adapter.Hash == "" && adapter.Rank == 0 && adapter.Alpha == 0 && adapter.Scale == 0 && len(adapter.TargetKeys) == 0
}

// AdapterFromInfo lifts a lora.AdapterInfo into an Adapter.
//
//	a := bundle.AdapterFromInfo(info)
func AdapterFromInfo(info lora.AdapterInfo) Adapter {
	return Adapter{
		Name:       info.Name,
		Path:       info.Path,
		Hash:       info.Hash,
		Rank:       info.Rank,
		Alpha:      info.Alpha,
		Scale:      info.Scale,
		TargetKeys: core.SliceClone(info.TargetKeys),
	}
}

// AdapterToInfo lowers an Adapter to a lora.AdapterInfo.
//
//	info := bundle.AdapterToInfo(a)
func AdapterToInfo(adapter Adapter) lora.AdapterInfo {
	return lora.AdapterInfo{
		Name:       adapter.Name,
		Path:       adapter.Path,
		Hash:       adapter.Hash,
		Rank:       adapter.Rank,
		Alpha:      adapter.Alpha,
		Scale:      adapter.Scale,
		TargetKeys: core.SliceClone(adapter.TargetKeys),
	}
}

// HashString returns the SHA-256 hex of a string, or empty for empty input.
//
//	h := bundle.HashString("hello")
func HashString(value string) string {
	if value == "" {
		return ""
	}
	return core.SHA256HexString(value)
}

// StateURI renders a State chunk reference as a state:// URI.
//
//	uri := bundle.StateURI(ref)
func StateURI(ref state.ChunkRef) string {
	// Hand-built — avoids Sprintf's interface boxing of segment and chunk
	// ID. Two branches, both single-allocation.
	if ref.Segment != "" {
		buf := make([]byte, 0, 8+len(ref.Segment)+7+20)
		buf = append(buf, "state://"...)
		buf = append(buf, ref.Segment...)
		buf = append(buf, "#chunk="...)
		buf = strconv.AppendInt(buf, int64(ref.ChunkID), 10)
		return core.AsString(buf)
	}
	buf := make([]byte, 0, 14+20)
	buf = append(buf, "state://chunk/"...)
	buf = strconv.AppendInt(buf, int64(ref.ChunkID), 10)
	return core.AsString(buf)
}

func buildModel(snapshot *kv.Snapshot, opts Options) Model {
	src := opts.Source
	arch := src.Architecture
	if arch == "" && snapshot != nil {
		arch = snapshot.Architecture
	}
	numLayers := src.NumLayers
	if numLayers == 0 && snapshot != nil {
		numLayers = snapshot.NumLayers
	}
	model := Model{
		Name:          opts.Model,
		Path:          opts.ModelPath,
		Architecture:  arch,
		VocabSize:     src.VocabSize,
		NumLayers:     numLayers,
		HiddenSize:    src.HiddenSize,
		QuantBits:     src.QuantBits,
		QuantGroup:    src.QuantGroup,
		ContextLength: src.ContextLength,
	}
	// Hand-built hash payload — avoids 4× Sprintf("%d") boxing and a
	// 7-arg Join intermediate slice. Stack-buffer fast-path: dynamic
	// `make([]byte, 0, n)` heap-allocates even when escape analysis says
	// the buffer does not escape (size is unknown at compile time, so the
	// compiler can't reserve stack space). A fixed-size stack array slid
	// into via `stackBuf[:0]` IS stack-allocated. The buf is consumed
	// in-function via `HashString(core.AsString(buf))` and never escapes,
	// so the stack fast-path is safe; the `make` fallback covers oversized
	// model.Name / model.Path / model.Architecture inputs.
	var stackBuf [256]byte
	needed := len(model.Name) + len(model.Path) + len(model.Architecture) + 48
	var buf []byte
	if needed <= len(stackBuf) {
		buf = stackBuf[:0]
	} else {
		buf = make([]byte, 0, needed)
	}
	buf = append(buf, model.Name...)
	buf = append(buf, '\n')
	buf = append(buf, model.Path...)
	buf = append(buf, '\n')
	buf = append(buf, model.Architecture...)
	buf = append(buf, '\n')
	buf = strconv.AppendInt(buf, int64(model.VocabSize), 10)
	buf = append(buf, '\n')
	buf = strconv.AppendInt(buf, int64(model.NumLayers), 10)
	buf = append(buf, '\n')
	buf = strconv.AppendInt(buf, int64(model.QuantBits), 10)
	buf = append(buf, '\n')
	buf = strconv.AppendInt(buf, int64(model.ContextLength), 10)
	model.Hash = HashString(core.AsString(buf))
	return model
}

func normaliseRuntime(runtime Runtime) Runtime {
	if runtime.Name == "" {
		runtime.Name = "go-mlx"
	}
	return runtime
}

func buildAdapter(adapter Adapter, adapterPath string, info lora.AdapterInfo) Adapter {
	// Track whether TargetKeys was supplied by AdapterFromInfo — that path
	// already SliceClones from info.TargetKeys, so the defensive clone at
	// function-end would be a redundant second copy. Caller-supplied
	// adapter.TargetKeys still aliases user-owned memory and must clone.
	keysFromInfo := false
	if AdapterEmpty(adapter) && !info.IsEmpty() {
		adapter = AdapterFromInfo(info)
		keysFromInfo = true
	}
	if adapter.Path == "" {
		adapter.Path = adapterPath
	}
	// Fast-skip the hash computation when the adapter is fully empty —
	// the final all-zero check at the end would clear the freshly-built
	// hash anyway, so building it is wasted SHA + alloc on every
	// adapter-less bundle.New.
	allEmpty := adapter.Path == "" && adapter.Name == "" && adapter.Rank == 0 && adapter.Alpha == 0 && adapter.Scale == 0 && len(adapter.TargetKeys) == 0
	if adapter.Hash == "" && !allEmpty {
		// Hand-built hash payload — avoids Sprintf("%d") + 2× Sprintf("%f")
		// boxing and a 6-arg Join intermediate. Float formatting matches
		// fmt's default %f precision (6 decimals).
		keyCommas := 0
		if n := len(adapter.TargetKeys); n > 1 {
			keyCommas = n - 1
		}
		keyBytes := 0
		for _, key := range adapter.TargetKeys {
			keyBytes += len(key)
		}
		// Stack-buffer fast-path — see buildModel for the rationale on why
		// `make([]byte, 0, n)` heap-allocates despite escape analysis saying
		// no-escape. Typical LoRA adapter hash payloads (Name + Path +
		// 4 target keys × 8 chars + scalars) land well under 256 bytes;
		// oversized inputs fall back to the heap `make`.
		var stackBuf [256]byte
		needed := len(adapter.Name) + len(adapter.Path) + keyBytes + keyCommas + 48
		var buf []byte
		if needed <= len(stackBuf) {
			buf = stackBuf[:0]
		} else {
			buf = make([]byte, 0, needed)
		}
		buf = append(buf, adapter.Name...)
		buf = append(buf, '\n')
		buf = append(buf, adapter.Path...)
		buf = append(buf, '\n')
		buf = strconv.AppendInt(buf, int64(adapter.Rank), 10)
		buf = append(buf, '\n')
		buf = strconv.AppendFloat(buf, float64(adapter.Alpha), 'f', 6, 32)
		buf = append(buf, '\n')
		buf = strconv.AppendFloat(buf, float64(adapter.Scale), 'f', 6, 32)
		buf = append(buf, '\n')
		for i, key := range adapter.TargetKeys {
			if i > 0 {
				buf = append(buf, ',')
			}
			buf = append(buf, key...)
		}
		adapter.Hash = HashString(core.AsString(buf))
	}
	// `allEmpty` is the byte-for-byte same predicate as the final clear
	// check below, so reuse it instead of re-walking the seven field
	// compares + the TargetKeys-len recheck.
	if allEmpty {
		adapter.Hash = ""
	}
	if !keysFromInfo {
		adapter.TargetKeys = core.SliceClone(adapter.TargetKeys)
	}
	return adapter
}

func checkAdapterCompatibility(active lora.AdapterInfo, expected Adapter) error {
	if AdapterEmpty(expected) {
		return nil
	}
	if active.IsEmpty() {
		return errBundleNeedsLoRA
	}
	want := AdapterToInfo(expected)
	if want.Hash != "" && active.Hash != "" && want.Hash != active.Hash {
		return errBundleLoRAHash
	}
	if want.Path != "" && active.Path != "" && want.Path != active.Path && (want.Hash == "" || active.Hash == "") {
		return errBundleLoRAPath
	}
	if want.Rank > 0 && active.Rank > 0 && want.Rank != active.Rank {
		return errBundleLoRARank
	}
	if want.Alpha != 0 && active.Alpha != 0 && want.Alpha != active.Alpha {
		return errBundleLoRAAlpha
	}
	return nil
}

// MemvidURI renders an old memvid chunk reference as a memvid:// URI.
//
// Deprecated: use StateURI.
func MemvidURI(ref state.ChunkRef) string {
	// Hand-built — same pattern as StateURI; no Sprintf boxing.
	if ref.Segment != "" {
		buf := make([]byte, 0, 9+len(ref.Segment)+7+20)
		buf = append(buf, "memvid://"...)
		buf = append(buf, ref.Segment...)
		buf = append(buf, "#chunk="...)
		buf = strconv.AppendInt(buf, int64(ref.ChunkID), 10)
		return core.AsString(buf)
	}
	buf := make([]byte, 0, 15+20)
	buf = append(buf, "memvid://chunk/"...)
	buf = strconv.AppendInt(buf, int64(ref.ChunkID), 10)
	return core.AsString(buf)
}

// joinChunkRefs returns a single allocation containing primary first
// then fallback. Replaces the `append(append(nil, A...), B...)` pattern
// which allocates twice and grows on the second append. When only one
// input has entries we alias it — the sole caller (buildRefs) only
// reads the result, so the read-only aliasing is safe.
func joinChunkRefs(primary, fallback []state.ChunkRef) []state.ChunkRef {
	switch {
	case len(primary) == 0 && len(fallback) == 0:
		return nil
	case len(fallback) == 0:
		return primary
	case len(primary) == 0:
		return fallback
	}
	out := make([]state.ChunkRef, 0, len(primary)+len(fallback))
	out = append(out, primary...)
	out = append(out, fallback...)
	return out
}

func buildRefs(refs []Ref, stateRefs []state.ChunkRef) []Ref {
	if len(refs) == 0 && len(stateRefs) == 0 {
		return nil
	}
	out := make([]Ref, 0, len(refs)+len(stateRefs))
	out = append(out, refs...)
	for _, ref := range stateRefs {
		uri := StateURI(ref)
		out = append(out, Ref{
			Kind:  RefState,
			URI:   uri,
			Hash:  HashString(uri),
			State: ref,
		})
	}
	return out
}

func cloneMeta(meta map[string]string) map[string]string {
	// core.MapClone wraps maps.Clone, which returns a fresh empty map for
	// an empty input. cloneMeta has always returned nil for both nil and
	// zero-length input — keep that contract so JSON marshal omits the
	// field via `omitempty` instead of emitting "{}".
	if len(meta) == 0 {
		return nil
	}
	return core.MapClone(meta)
}

func resultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	if text, ok := result.Value.(string); ok {
		return core.NewError(text)
	}
	return errCoreResultFailed
}
