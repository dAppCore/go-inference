// SPDX-Licence-Identifier: EUPL-1.2

package agent

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference/kv"
	"dappco.re/go/inference/kv/memory"
	"dappco.re/go/inference/model/bundle"
	state "dappco.re/go/inference/model/state"
)

// WakeOptions selects a durable KV prefix to restore into a live
// session. EntryURI is optional when the index has exactly one natural first
// entry.
type WakeOptions struct {
	Index                  *StateIndex
	IndexURI               string
	EntryURI               string
	Tokenizer              bundle.Tokenizer
	LoadOptions            kv.LoadOptions
	SkipCompatibilityCheck bool
}

// WakeReport describes the restored durable prefix.
type WakeReport struct {
	IndexURI        string `json:"index_uri,omitempty"`
	EntryURI        string `json:"entry_uri,omitempty"`
	BundleURI       string `json:"bundle_uri,omitempty"`
	Title           string `json:"title,omitempty"`
	PrefixTokens    int    `json:"prefix_tokens,omitempty"`
	BundleTokens    int    `json:"bundle_tokens,omitempty"`
	BlockSize       int    `json:"block_size,omitempty"`
	BlocksRead      int    `json:"blocks_read,omitempty"`
	RestoreStrategy string `json:"restore_strategy,omitempty"`
	IndexHash       string `json:"index_hash,omitempty"`
	SnapshotHash    string `json:"snapshot_hash,omitempty"`
}

// SleepOptions controls how a live session is streamed to durable
// KV block storage.
type SleepOptions struct {
	EntryURI          string
	BundleURI         string
	IndexURI          string
	ParentEntryURI    string
	ParentBundleURI   string
	ParentIndexURI    string
	Title             string
	Model             string
	ModelPath         string
	ModelInfo         memory.ModelInfo
	Tokenizer         bundle.Tokenizer
	ReuseParentPrefix bool
	// ReuseParentPrefixTrusted declares the parent prefix identical by
	// construction (append-only session sleeping over its own prior sleep) —
	// parent blocks graft by reference with no re-capture or re-hash.
	ReuseParentPrefixTrusted bool
	BlockOptions             kv.StateBlockOptions
	Labels                   []string
	Meta                     map[string]string
}

// SleepReport describes the durable state written by Sleep.
type SleepReport struct {
	IndexURI        string         `json:"index_uri,omitempty"`
	EntryURI        string         `json:"entry_uri,omitempty"`
	BundleURI       string         `json:"bundle_uri,omitempty"`
	ParentEntryURI  string         `json:"parent_entry_uri,omitempty"`
	ParentBundleURI string         `json:"parent_bundle_uri,omitempty"`
	ParentIndexURI  string         `json:"parent_index_uri,omitempty"`
	Title           string         `json:"title,omitempty"`
	TokenCount      int            `json:"token_count,omitempty"`
	BlockSize       int            `json:"block_size,omitempty"`
	BlocksWritten   int            `json:"blocks_written,omitempty"`
	BlocksReused    int            `json:"blocks_reused,omitempty"`
	KVEncoding      kv.Encoding    `json:"kv_encoding,omitempty"`
	IndexHash       string         `json:"index_hash,omitempty"`
	SnapshotHash    string         `json:"snapshot_hash,omitempty"`
	BundleRef       state.ChunkRef `json:"bundle_ref"`
	IndexRef        state.ChunkRef `json:"index_ref"`
}

type WakePlan struct {
	Index  *StateIndex
	Entry  StateIndexEntry
	Bundle *kv.StateBlockBundle
	Report *WakeReport
}

func LoadWakeSnapshot(ctx context.Context, store state.Store, opts WakeOptions, info memory.ModelInfo) (*kv.Snapshot, *WakeReport, error) {
	plan, err := PlanWake(ctx, store, opts, info)
	if err != nil {
		return nil, nil, err
	}
	snapshot, err := kv.LoadPrefixFromStateBlocksWithOptions(ctx, store, plan.Bundle, plan.Entry.PrefixTokens(), opts.LoadOptions)
	if err != nil {
		return nil, nil, err
	}
	return snapshot, plan.Report, nil
}

func PlanWake(ctx context.Context, store state.Store, opts WakeOptions, info memory.ModelInfo) (*WakePlan, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return nil, errStateStoreNil
	}
	// When compat check is enabled it runs its own Validate; skip the
	// duplicate loadIndex-side validation in that case.
	index, err := loadIndex(ctx, store, opts, opts.SkipCompatibilityCheck)
	if err != nil {
		return nil, err
	}
	if !opts.SkipCompatibilityCheck {
		if err := CheckStateIndexCompatibility(info, opts.Tokenizer, index); err != nil {
			return nil, err
		}
	}
	entryURI := core.Trim(opts.EntryURI)
	if entryURI == "" && len(index.Entries) > 0 {
		entryURI = index.Entries[0].URI
	}
	entry, ok := index.Entry(entryURI)
	if !ok {
		return nil, errStateIndexEntryNotFound
	}
	bundleURI := core.FirstNonBlank(entry.BundleURI, index.BundleURI)
	bundle, err := kv.LoadStateBlockBundle(ctx, store, bundleURI)
	if err != nil {
		return nil, err
	}
	prefixTokens := entry.PrefixTokens()
	if prefixTokens <= 0 || prefixTokens > bundle.TokenCount {
		return nil, errStateIndexPrefixInvalid
	}
	report := &WakeReport{
		IndexURI:     opts.IndexURI,
		EntryURI:     entry.URI,
		BundleURI:    bundleURI,
		Title:        entry.Title,
		PrefixTokens: prefixTokens,
		BundleTokens: bundle.TokenCount,
		BlockSize:    bundle.BlockSize,
		BlocksRead:   blocksNeededForPrefix(bundle, prefixTokens),
		IndexHash:    index.Hash,
		SnapshotHash: bundle.SnapshotHash,
	}
	return &WakePlan{
		Index:  index,
		Entry:  entry,
		Bundle: bundle,
		Report: report,
	}, nil
}

func loadIndex(ctx context.Context, store state.Store, opts WakeOptions, mustValidate bool) (*StateIndex, error) {
	if opts.Index != nil {
		if mustValidate {
			if err := opts.Index.Validate(); err != nil {
				return nil, err
			}
		}
		return opts.Index, nil
	}
	if core.Trim(opts.IndexURI) == "" {
		return nil, errStateIndexURIRequired
	}
	// LoadStateIndex always validates the loaded payload before returning,
	// so the mustValidate signal only matters for the in-memory opts.Index
	// branch above.
	return LoadStateIndex(ctx, store, opts.IndexURI)
}

func SleepURIs(opts SleepOptions) (entryURI, bundleURI, indexURI string, err error) {
	entryURI = core.Trim(opts.EntryURI)
	bundleURI = core.Trim(opts.BundleURI)
	indexURI = core.Trim(opts.IndexURI)
	if entryURI == "" {
		switch {
		case bundleURI != "":
			entryURI = bundleURI
		case indexURI != "":
			entryURI = indexURI
		default:
			entryURI = "mlx://state/latest"
		}
	}
	if bundleURI == "" {
		bundleURI = entryURI + "/bundle"
	}
	if indexURI == "" {
		indexURI = entryURI + "/index"
	}
	if entryURI == "" || bundleURI == "" || indexURI == "" {
		return "", "", "", errStateURIRequired
	}
	return entryURI, bundleURI, indexURI, nil
}

func SleepBlockOptions(opts SleepOptions, bundleURI string) kv.StateBlockOptions {
	blockOpts := opts.BlockOptions
	if opts.ReuseParentPrefixTrusted {
		blockOpts.ReusePrefixTrusted = true
	}
	if blockOpts.KVEncoding == "" {
		blockOpts.KVEncoding = kv.EncodingNative
	}
	if blockOpts.URI == "" {
		blockOpts.URI = bundleURI + "/blocks"
	}
	if blockOpts.Title == "" {
		blockOpts.Title = core.FirstNonBlank(opts.Title, "go-mlx State")
	}
	labels := make([]string, len(blockOpts.Labels), len(blockOpts.Labels)+1)
	copy(labels, blockOpts.Labels)
	blockOpts.Labels = append(labels, "state")
	return blockOpts
}

func NewSleepIndex(bundle *kv.StateBlockBundle, opts SleepOptions, entryURI, bundleURI string) (*StateIndex, error) {
	// Validate the bundle up front so a nil/invalid bundle returns the
	// canonical errBundleNil (matching NewStateIndex's contract) instead
	// of panicking on the bundle.TokenCount read below. NewStateIndex
	// re-validates, but it is only reached after that dereference.
	if err := kv.ValidateStateBlockBundle(bundle); err != nil {
		return nil, err
	}
	// Labels + Meta: NewStateIndex below takes a shallow per-entry copy
	// (cloneIndexEntries: make+copy, no per-entry clone), so it aliases
	// these reference fields rather than cloning them. That is safe here
	// because both are freshly owned and never mutated after this call:
	// opts.Labels comes straight from the caller's SleepOptions and is
	// not retained downstream, and sleepEntryMeta returns a fresh map.
	// No defensive clone is needed on either side.
	entry := StateIndexEntry{
		URI:        entryURI,
		BundleURI:  bundleURI,
		Title:      opts.Title,
		TokenStart: 0,
		TokenCount: bundle.TokenCount,
		Labels:     opts.Labels,
		Meta:       sleepEntryMeta(opts),
	}
	if entry.Title == "" {
		entry.Title = "State"
	}
	return NewStateIndex(bundle, StateIndexOptions{
		BundleURI: bundleURI,
		Title:     opts.Title,
		Model:     opts.Model,
		ModelPath: opts.ModelPath,
		ModelInfo: opts.ModelInfo,
		Tokenizer: opts.Tokenizer,
		Entries:   []StateIndexEntry{entry},
	})
}

func sleepEntryMeta(opts SleepOptions) map[string]string {
	meta := cloneStringMap(opts.Meta)
	if opts.ParentEntryURI != "" {
		if meta == nil {
			meta = map[string]string{}
		}
		meta["parent_entry_uri"] = opts.ParentEntryURI
	}
	if opts.ParentBundleURI != "" {
		if meta == nil {
			meta = map[string]string{}
		}
		meta["parent_bundle_uri"] = opts.ParentBundleURI
	}
	if opts.ParentIndexURI != "" {
		if meta == nil {
			meta = map[string]string{}
		}
		meta["parent_index_uri"] = opts.ParentIndexURI
	}
	return meta
}

func NewSleepReport(index *StateIndex, bundle *kv.StateBlockBundle, opts SleepOptions, entryURI, bundleURI, indexURI string, bundleRef, indexRef state.ChunkRef) *SleepReport {
	return &SleepReport{
		IndexURI:        indexURI,
		EntryURI:        entryURI,
		BundleURI:       bundleURI,
		ParentEntryURI:  opts.ParentEntryURI,
		ParentBundleURI: opts.ParentBundleURI,
		ParentIndexURI:  opts.ParentIndexURI,
		Title:           opts.Title,
		TokenCount:      bundle.TokenCount,
		BlockSize:       bundle.BlockSize,
		BlocksWritten:   len(bundle.Blocks),
		BlocksReused:    bundle.ReusedBlocks,
		KVEncoding:      bundle.KVEncoding,
		IndexHash:       index.Hash,
		SnapshotHash:    bundle.SnapshotHash,
		BundleRef:       bundleRef,
		IndexRef:        indexRef,
	}
}

func WakeReportFromSleep(report *SleepReport) *WakeReport {
	if report == nil {
		return nil
	}
	return &WakeReport{
		IndexURI:     report.IndexURI,
		EntryURI:     report.EntryURI,
		BundleURI:    report.BundleURI,
		Title:        report.Title,
		PrefixTokens: report.TokenCount,
		BundleTokens: report.TokenCount,
		BlockSize:    report.BlockSize,
		BlocksRead:   0,
		IndexHash:    report.IndexHash,
		SnapshotHash: report.SnapshotHash,
	}
}

func CloneWakeReport(report *WakeReport) *WakeReport {
	if report == nil {
		return nil
	}
	cloned := *report
	return &cloned
}

func blocksNeededForPrefix(bundle *kv.StateBlockBundle, prefixTokens int) int {
	if bundle == nil || prefixTokens <= 0 {
		return 0
	}
	count := 0
	blocks := bundle.Blocks
	for i := range blocks {
		tokenStart := blocks[i].TokenStart
		if tokenStart >= prefixTokens {
			break
		}
		count++
		if tokenStart+blocks[i].TokenCount >= prefixTokens {
			break
		}
	}
	return count
}
