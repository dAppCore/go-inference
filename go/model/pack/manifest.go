// SPDX-Licence-Identifier: EUPL-1.2

// Package pack wraps an unpacked model pack (the directory shape walked by
// inference.ModelPackInspector) into a Trix container with magic "MDL1",
// and round-trips back to disk.
//
// Container layout (delegated to forge.lthn.ai/Snider/Enchantrix/pkg/trix):
//
//	[Magic "MDL1" (4)] [Version (1)] [Header Length (4)] [JSON Header] [Payload]
//
// Header is the JSON-marshalled Manifest. Payload is a deterministic tar of the
// source pack directory, optionally followed by an embedded vindex blob at the
// offset/length declared in Manifest.Vindex.
//
//	r := pack.Pack(c, "/path/to/gemma-4-26b-a4b-it", "out.model", pack.PackOptions{})
//	if !r.OK { return r }
package pack

import (
	iofs "io/fs"

	"dappco.re/go/inference"
)

// Magic is the 4-byte Trix magic for a .model container.
const Magic = "MDL1"

// Manifest is the JSON header carried inside a .model Trix container.
// It mirrors the shape of inference.ModelPackInspection for the contained
// pack, plus packaging-specific metadata (lineage, vindex placement,
// producer attribution, signatures).
type Manifest struct {
	// Model is the portable model identity for the contained pack.
	Model inference.ModelIdentity `json:"model"`

	// Tokenizer is the portable tokenizer identity for the contained pack.
	Tokenizer inference.TokenizerIdentity `json:"tokenizer"`

	// SourceFormat names the on-disk shape of the model bytes inside the
	// payload tar — currently "safetensors" or "gguf".
	SourceFormat string `json:"source_format"`

	// Capabilities are the per-pack capabilities reported by the inspector.
	Capabilities []inference.Capability `json:"capabilities,omitempty"`

	// Lineage points back at the source .train this .model was derived
	// from. Optional — top-level training runs derived without a prior
	// .train may omit it.
	Lineage *Lineage `json:"lineage,omitempty"`

	// Vindex describes an embedded LARQL vindex blob. When Vindex is nil
	// the .model carries only the model pack tar; LQL operations that
	// require a vindex must EXTRACT one first.
	Vindex *VindexRef `json:"vindex,omitempty"`

	// Producer records who emitted the .model.
	Producer Producer `json:"producer"`

	// Signatures are detached signatures over the payload bytes.
	// Verification is handled at consumer layer; this package only
	// round-trips the slice.
	Signatures []Signature `json:"signatures,omitempty"`
}

// Lineage records the source .train file the .model was derived from.
type Lineage struct {
	TrainURI string `json:"train_uri"`
	TrainSHA string `json:"train_sha,omitempty"`
}

// VindexRef points at an embedded vindex blob inside the payload.
type VindexRef struct {
	// Embedded is always true for .model files where Vindex != nil — the
	// flag exists so external readers don't need to introspect Offset/Length
	// to know whether to expect a payload-side vindex.
	Embedded bool `json:"embedded"`

	// Offset is the byte offset (within the Trix payload) at which the
	// vindex blob starts.
	Offset uint64 `json:"offset"`

	// Length is the vindex blob length in bytes.
	Length uint64 `json:"length"`

	// Format names the vindex serialisation. "msgpack" is the LARQL
	// .larql.bin form.
	Format string `json:"format,omitempty"`
}

// Producer records the tool that emitted the .model.
type Producer struct {
	Name    string `json:"name"`
	Commit  string `json:"commit,omitempty"`
	Created string `json:"created"` // RFC3339 UTC
}

// Signature is a detached signature over the Trix payload bytes.
type Signature struct {
	KeyID string `json:"key_id"`
	Alg   string `json:"alg"` // e.g. "ed25519"
	Sig   string `json:"sig"` // base64 standard encoding
}

// PackOptions controls Pack behaviour.
type PackOptions struct {
	// Manifest is the manifest to embed in the Trix header. If
	// Manifest.Producer.Created is empty, Pack fills it with the current
	// UTC RFC3339 timestamp.
	Manifest Manifest

	// VindexBlob, when non-nil, requests an embedded vindex. NOT yet
	// implemented — passing a non-nil value causes Pack to return an
	// explicit "vindex embedding not yet implemented" Result so the seam
	// is honest rather than silently dropping the blob.
	VindexBlob []byte
}

// UnpackOptions controls Unpack behaviour.
type UnpackOptions struct {
	// Overwrite allows Unpack to write into a non-empty destination dir.
	// Default false — Unpack refuses if the destination already contains
	// files.
	Overwrite bool
}

// Entry is one tar entry inside a .model payload — the shape List
// returns. Path, Size, and Mode are surfaced; content is not read.
type Entry struct {
	Path string        `json:"path"`
	Size int64         `json:"size"`
	Mode iofs.FileMode `json:"mode"`
}

// IdentityFingerprint is the deterministic identity projection of a
// Manifest — the subset of fields that, together, mean "these two .model
// files describe the same logical model artefact". Timestamps, signatures,
// and lineage URIs are deliberately excluded — they are provenance, not
// identity.
type IdentityFingerprint struct {
	Model        inference.ModelIdentity     `json:"model"`
	Tokenizer    inference.TokenizerIdentity `json:"tokenizer"`
	SourceFormat string                      `json:"source_format"`
	Capabilities []inference.Capability      `json:"capabilities,omitempty"`
	VindexHash   string                      `json:"vindex_hash,omitempty"`
}

// Identity returns the identity projection of this Manifest — the
// fields that decide "is this the same logical model?".
//
//	id := manifest.Identity()
//	_ = id.Model.Architecture
func (m Manifest) Identity() IdentityFingerprint {
	return IdentityFingerprint{
		Model:        m.Model,
		Tokenizer:    m.Tokenizer,
		SourceFormat: m.SourceFormat,
		Capabilities: m.Capabilities,
		// VindexHash left empty until vindex embedding lands and the
		// hash of the embedded blob is known at fingerprint time.
	}
}
