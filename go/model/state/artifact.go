// SPDX-Licence-Identifier: EUPL-1.2

package state

import (
	"context"

	core "dappco.re/go"
)

// ArtifactVersion is the schema version stamped onto every Artifact
// produced by ExportArtifact.
const ArtifactVersion = 1

// errArtifactPayloadNil is the sentinel returned when ExportArtifact is
// invoked without a payload. Hoisted to a package var so the nil-guard at
// the top of ExportArtifact does not allocate a fresh error on every call.
var errArtifactPayloadNil = core.NewError("state: artifact payload is nil")

// Artifact is a versioned, engine-supplied diagnostic or analysis
// snapshot packaged for archival. Payload carries whatever
// engine-specific structure the caller wants to preserve — KV-cache
// coherence metrics, visualisation summaries, profiling counters, or any
// other JSON-marshalable analysis result; state treats it as opaque data
// and never inspects its shape.
//
// Artifact is deliberately lighter than Bundle: Bundle is a restorable
// state envelope meant to be woken from later, while Artifact is a
// one-way export for observability, dashboards, and research trails.
type Artifact struct {
	Version   int      `json:"version"`
	Kind      string   `json:"kind"`
	Model     string   `json:"model"`
	Prompt    string   `json:"prompt"`
	Payload   any      `json:"payload"`
	LocalPath string   `json:"local_path,omitempty"`
	ChunkRef  ChunkRef `json:"chunk_ref"`
}

// ArtifactOptions controls ExportArtifact.
type ArtifactOptions struct {
	// Model and Prompt are caller-owned provenance copied verbatim onto
	// the returned Artifact.
	Model  string
	Prompt string
	// Kind identifies the engine-specific schema of Payload (e.g.
	// "go-mlx/session-state"). It is stamped onto the Artifact and, when
	// Put.Kind is empty, also classifies the archived chunk.
	Kind string
	// LocalPath and Save optionally persist a heavier side-payload (e.g.
	// a raw KV-cache binary) outside the JSON record. Save is invoked
	// with LocalPath when set; the engine owns the actual encoding, so
	// state never touches a filesystem itself.
	LocalPath string
	Save      func(path string) error
	// Store archives the marshaled Artifact when set. Put carries the
	// same routing/tagging metadata accepted by Writer.Put; an empty
	// Put.Kind defaults to Kind.
	Store Writer
	Put   PutOptions
}

// ExportArtifact packages payload into a versioned Artifact and, when
// opts.Store is set, archives it as JSON via Writer.Put — backfilling the
// returned ChunkRef onto the Artifact so callers never have to splice it
// back in by hand.
//
//	rec, err := state.ExportArtifact(ctx, coherence, state.ArtifactOptions{
//	    Model: "gemma3-1b",
//	    Kind:  "go-mlx/session-state",
//	    Store: store,
//	    Put:   state.PutOptions{URI: "mlx://session/trace-1"},
//	})
func ExportArtifact(ctx context.Context, payload any, opts ArtifactOptions) (*Artifact, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}
	if payload == nil {
		return nil, errArtifactPayloadNil
	}
	if opts.Save != nil {
		if err := opts.Save(opts.LocalPath); err != nil {
			return nil, err
		}
	}
	record := &Artifact{
		Version:   ArtifactVersion,
		Kind:      opts.Kind,
		Model:     opts.Model,
		Prompt:    opts.Prompt,
		Payload:   payload,
		LocalPath: opts.LocalPath,
	}
	if opts.Store != nil {
		data := core.JSONMarshalIndent(record, "", "  ")
		if !data.OK {
			return nil, core.E("state.ExportArtifact", "marshal artifact", data.Err())
		}
		marshalled := data.Bytes()
		putOpts := opts.Put
		if putOpts.Kind == "" {
			putOpts.Kind = opts.Kind
		}
		ref, err := opts.Store.Put(ctx, core.AsString(marshalled), putOpts)
		if err != nil {
			return nil, err
		}
		record.ChunkRef = ref
	}
	return record, nil
}
