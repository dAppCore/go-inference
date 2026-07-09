// SPDX-Licence-Identifier: EUPL-1.2

// Package artifact exports compact session-state records — KV provenance,
// optional binary KV snapshots, and SAMI visualisation data — that can be
// archived to State stores or local files.
//
//	record, err := artifact.Export(ctx, snapshot, artifact.Options{
//	    Model: "gemma3-1b",
//	    Store: store,
//	    URI:   "mlx://session/trace-1",
//	})
package artifact

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference/kv"
	"dappco.re/go/inference/model/bundle"
	state "dappco.re/go/inference/model/state"
)

// Kind labels session-state artifacts written by this package.
const Kind = "go-mlx/session-state"

// errSnapshotNil is the sentinel returned when Export is invoked without
// a KV snapshot. Hoisted to a package var so the nil-guard at the top
// of Export does not allocate a fresh *Err on every call.
var errSnapshotNil = core.NewError("artifact: KV snapshot is nil")

// cachedFeatureLabels is the package-once-cached result of kv.FeatureLabels.
// kv.FeatureLabels allocates a fresh slice every call (currently 7 strings);
// Export embeds the slice once per Record so the labels alloc fires on
// every Export call. The label list is invariant — kv exposes it as the
// stable order matching Features — so it is safe to compute once at
// package init and share across all Exports. Callers must NOT mutate the
// slice (none currently do; Records that travel to JSON only ever read).
var cachedFeatureLabels = kv.FeatureLabels()

// Options controls local model-state artifact export.
type Options struct {
	Model    string
	Prompt   string
	Analysis *kv.Analysis
	KVPath   string
	Store    state.Writer
	URI      string
	Title    string
	Kind     string
	Track    string
	Tags     map[string]string
	Labels   []string
}

// Record is the compact JSON payload written into a State chunk.
type Record struct {
	Version       int               `json:"version"`
	Kind          string            `json:"kind"`
	Model         string            `json:"model"`
	Prompt        string            `json:"prompt"`
	Snapshot      Snapshot          `json:"snapshot"`
	Analysis      *kv.Analysis      `json:"analysis"`
	Features      []float64         `json:"features"`
	FeatureLabels []string          `json:"feature_labels"`
	SAMI          bundle.SAMIResult `json:"sami"`
	KVPath        string            `json:"kv_path,omitempty"`
	ChunkRef      state.ChunkRef    `json:"chunk_ref"`
}

// Snapshot is the lightweight tensor provenance stored in text chunks.
type Snapshot struct {
	Architecture  string `json:"architecture"`
	TokenCount    int    `json:"token_count"`
	NumLayers     int    `json:"num_layers"`
	NumHeads      int    `json:"num_heads"`
	SeqLen        int    `json:"seq_len"`
	HeadDim       int    `json:"head_dim"`
	NumQueryHeads int    `json:"num_query_heads"`
}

// payload is the go-mlx-specific analysis bundle carried as
// state.Artifact's opaque Payload field. Kept unexported: callers get
// typed access through Record, which Export reshapes from the delegated
// state.Artifact plus this payload.
type payload struct {
	Snapshot      Snapshot          `json:"snapshot"`
	Analysis      *kv.Analysis      `json:"analysis"`
	Features      []float64         `json:"features"`
	FeatureLabels []string          `json:"feature_labels"`
	SAMI          bundle.SAMIResult `json:"sami"`
}

// Export writes optional KV binary data and optional State JSON for the
// supplied KV snapshot.
//
// Delegates the versioned envelope, the optional local-path side-save, and
// the marshal+Store.Put archival step onto inference/state.ExportArtifact
// — the generalised form of this exact export shape shared across engines.
// The KV/SAMI/analysis payload stays go-mlx-specific and travels as the
// opaque Payload state.ExportArtifact carries without inspecting.
//
//	record, err := artifact.Export(ctx, snapshot, artifact.Options{KVPath: "/tmp/state.kv"})
func Export(ctx context.Context, snapshot *kv.Snapshot, opts Options) (*Record, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}
	if snapshot == nil {
		return nil, errSnapshotNil
	}
	analysis := opts.Analysis
	if analysis == nil {
		analysis = kv.Analyze(snapshot)
	}
	p := payload{
		Snapshot: Snapshot{
			Architecture:  snapshot.Architecture,
			TokenCount:    len(snapshot.Tokens),
			NumLayers:     snapshot.NumLayers,
			NumHeads:      snapshot.NumHeads,
			SeqLen:        snapshot.SeqLen,
			HeadDim:       snapshot.HeadDim,
			NumQueryHeads: snapshot.NumQueryHeads,
		},
		Analysis:      analysis,
		Features:      kv.Features(analysis),
		FeatureLabels: cachedFeatureLabels,
		SAMI:          bundle.SAMIFromKV(snapshot, analysis, bundle.SAMIOptions{Model: opts.Model, Prompt: opts.Prompt}),
	}
	// Save only when KVPath is set — state.ExportArtifact invokes Save
	// unconditionally whenever it is non-nil, so leave it nil rather than
	// passing snapshot.Save and relying on an empty path no-op.
	var save func(path string) error
	if opts.KVPath != "" {
		save = snapshot.Save
	}
	artifact, err := state.ExportArtifact(ctx, p, state.ArtifactOptions{
		Model:     opts.Model,
		Prompt:    opts.Prompt,
		Kind:      Kind,
		LocalPath: opts.KVPath,
		Save:      save,
		Store:     opts.Store,
		Put: state.PutOptions{
			URI:    opts.URI,
			Title:  opts.Title,
			Kind:   opts.Kind,
			Track:  opts.Track,
			Tags:   opts.Tags,
			Labels: opts.Labels,
		},
	})
	if err != nil {
		return nil, err
	}
	return &Record{
		Version:       artifact.Version,
		Kind:          artifact.Kind,
		Model:         artifact.Model,
		Prompt:        artifact.Prompt,
		Snapshot:      p.Snapshot,
		Analysis:      p.Analysis,
		Features:      p.Features,
		FeatureLabels: p.FeatureLabels,
		SAMI:          p.SAMI,
		KVPath:        artifact.LocalPath,
		ChunkRef:      artifact.ChunkRef,
	}, nil
}
