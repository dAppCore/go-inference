// SPDX-Licence-Identifier: EUPL-1.2

package pack

import (
	"archive/tar"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"io"
	iofs "io/fs"
	"sort"
	"sync"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"

	"forge.lthn.ai/Snider/Enchantrix/pkg/trix"
)

// sharedFs returns a process-wide cached unrestricted Fs handle.
//
// Pre-cache, every Hash/Pack/Unpack/List/Inspect call paid a fresh
// (&core.Fs{}).NewUnrestricted() construction — measurement on the
// Hash hot path showed ~50 allocs of the ~116 total came from this
// repeated init. The Fs is stateless (no per-call context, no auth
// scope mutation), so a single cached handle serves every call.
//
// Same shape as the sync.Once Core cache landed in pkg-level
// Discover (commit 5f29441) — that fix cut Discover by 46 allocs
// across every variant. Pack/Hash/List/Inspect should see a
// similar transfer.
var (
	sharedFsOnce sync.Once
	sharedFsHdl  *core.Fs
)

func sharedFs() *core.Fs {
	sharedFsOnce.Do(func() {
		sharedFsHdl = (&core.Fs{}).NewUnrestricted()
	})
	return sharedFsHdl
}

// Pack reads an unpacked model pack at srcDir and writes a .model Trix
// container to dest. Payload is a deterministic tar of srcDir contents.
// Manifest is embedded as the Trix header.
//
//	r := pack.Pack("/models/gemma-4-26b-a4b-it", "out.model", pack.PackOptions{
//	    Manifest: pack.Manifest{
//	        Model:        inference.ModelIdentity{Architecture: "gemma-4-26b-a4b-it", QuantBits: 4},
//	        Tokenizer:    inference.TokenizerIdentity{Kind: "sentencepiece"},
//	        SourceFormat: "safetensors",
//	        Producer:     pack.Producer{Name: "go-mlx"},
//	    },
//	})
//	if !r.OK { return r }
func Pack(srcDir, dest string, opts PackOptions) core.Result {
	if !dirExists(srcDir) {
		return core.Fail(core.E("pack.Pack", core.Sprintf("srcDir %q is not a directory", srcDir), nil))
	}
	if opts.VindexBlob != nil {
		return core.Fail(core.E("pack.Pack", "vindex embedding not yet implemented", nil))
	}

	manifest := opts.Manifest
	if manifest.Producer.Created == "" {
		manifest.Producer.Created = time.Now().UTC().Format(time.RFC3339)
	}
	if manifest.Model.Hash == "" {
		// Auto-populate the canonical pack hash so consumers never
		// see a .model with an empty Model.Hash. Caller can pre-fill
		// it to skip this step when a cached value is already known.
		h, hr := Hash(srcDir)
		if !hr.OK {
			return hr
		}
		manifest.Model.Hash = h
	}

	tarBytes, tr := buildTar(srcDir)
	if !tr.OK {
		return tr
	}

	headerMap, hr := manifestToHeaderMap(manifest)
	if !hr.OK {
		return hr
	}

	container := &trix.Trix{
		Header:  headerMap,
		Payload: tarBytes,
	}

	encoded, err := trix.Encode(container, Magic, nil)
	if err != nil {
		return core.Fail(core.E("pack.Pack", "trix.Encode failed", err))
	}

	if wr := core.WriteFile(dest, encoded, 0o644); !wr.OK {
		return wr
	}
	return core.Ok(nil)
}

// Unpack reads a .model Trix container at src and writes its contained
// model pack to destDir. destDir must not exist, must be empty, or
// UnpackOptions.Overwrite must be true.
//
//	r := pack.Unpack("out.model", "/tmp/extracted", pack.UnpackOptions{})
//	if !r.OK { return r }
func Unpack(src, destDir string, opts UnpackOptions) core.Result {
	rr := core.ReadFile(src)
	if !rr.OK {
		return rr
	}
	data := rr.Value.([]byte)

	container, err := trix.Decode(data, Magic, nil)
	if err != nil {
		return core.Fail(core.E("pack.Unpack", "trix.Decode failed", err))
	}

	if dr := assertDestDirWritable(destDir, opts.Overwrite); !dr.OK {
		return dr
	}
	if mr := core.MkdirAll(destDir, 0o755); !mr.OK {
		return mr
	}
	return extractTar(container.Payload, destDir)
}

// List reads a .model Trix container and returns the payload tar's
// entries (path, size, mode) without extracting file contents. Useful
// for tree-view UI without paying the full extract cost.
//
//	entries, manifest, r := pack.List("gemma.model")
//	if !r.OK { return r }
//	for _, e := range entries { core.Println(e.Path) }
func List(src string) ([]Entry, *Manifest, core.Result) {
	rr := core.ReadFile(src)
	if !rr.OK {
		return nil, nil, rr
	}
	data := rr.Value.([]byte)

	container, err := trix.Decode(data, Magic, nil)
	if err != nil {
		return nil, nil, core.Fail(core.E("pack.List", "trix.Decode failed", err))
	}

	manifest, mr := headerMapToManifest(container.Header)
	if !mr.OK {
		return nil, nil, mr
	}

	tr := tar.NewReader(bytes.NewReader(container.Payload))
	var entries []Entry
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, nil, core.Fail(core.E("pack.List", "tar.Next failed", err))
		}
		if hdr.Typeflag != tar.TypeReg {
			continue
		}
		entries = append(entries, Entry{
			Path: hdr.Name,
			Size: hdr.Size,
			Mode: iofs.FileMode(hdr.Mode),
		})
	}
	return entries, manifest, core.Ok(nil)
}

// Inspect reads a .model Trix container header (no payload extraction)
// and returns the Manifest plus a synthesised ModelPackInspection.
//
//	manifest, inspection, r := pack.Inspect("out.model")
//	if !r.OK { return r }
//	core.Println(inspection.Model.Architecture)
func Inspect(src string) (*Manifest, *inference.ModelPackInspection, core.Result) {
	rr := core.ReadFile(src)
	if !rr.OK {
		return nil, nil, rr
	}
	data := rr.Value.([]byte)

	container, err := trix.Decode(data, Magic, nil)
	if err != nil {
		return nil, nil, core.Fail(core.E("pack.Inspect", "trix.Decode failed", err))
	}

	manifest, mr := headerMapToManifest(container.Header)
	if !mr.OK {
		return nil, nil, mr
	}

	inspection := &inference.ModelPackInspection{
		Path:         src,
		Format:       manifest.SourceFormat,
		Model:        manifest.Model,
		Tokenizer:    manifest.Tokenizer,
		Supported:    true,
		Capabilities: manifest.Capabilities,
	}
	return manifest, inspection, core.Ok(nil)
}

// Hash computes the canonical model-pack hash for an unwrapped pack
// directory: SHA-256 of sorted content of the small metadata files
// (config.json, tokenizer.json, chat_template.jinja, adapter_config.json)
// concatenated with sorted file sizes of the *.safetensors blobs.
//
// Lightweight — doesn't read tensor bytes. Captures everything that
// affects behaviour without forcing a full content scan. Mirrors the
// shape inference.ModelPackInspector reads on the go-mlx side, so the
// hash from a packed .model and the hash from re-running InspectModelPack
// on the unwrapped dir agree byte-for-byte.
//
//	h, r := pack.Hash("/models/gemma-3-4b-it")
//	if !r.OK { return r }
//	manifest.Model.Hash = h
//
// Missing optional files (chat_template.jinja, adapter_config.json) are
// simply skipped — their absence is part of the pack's identity.
func Hash(srcDir string) (string, core.Result) {
	if !dirExists(srcDir) {
		return "", core.Fail(core.E("pack.Hash", core.Sprintf("srcDir %q is not a directory", srcDir), nil))
	}

	metaCandidates := []string{
		"config.json",
		"tokenizer.json",
		"chat_template.jinja",
		"adapter_config.json",
	}
	type metaFile struct {
		name    string
		content []byte
	}
	metas := make([]metaFile, 0, len(metaCandidates))
	fs := sharedFs()
	for _, name := range metaCandidates {
		path := core.JoinPath(srcDir, name)
		if !fs.IsFile(path) {
			continue
		}
		rr := core.ReadFile(path)
		if !rr.OK {
			return "", rr
		}
		metas = append(metas, metaFile{name: name, content: rr.Value.([]byte)})
	}
	sort.Slice(metas, func(i, j int) bool { return metas[i].name < metas[j].name })

	var safetensorSizes []int64
	for e, err := range fs.WalkSeq(srcDir) {
		if err != nil {
			return "", core.Fail(core.E("pack.Hash", "walk failed", err))
		}
		if e.IsDir {
			continue
		}
		if !core.HasSuffix(e.Path, ".safetensors") {
			continue
		}
		statR := core.Stat(core.JoinPath(srcDir, e.Path))
		if !statR.OK {
			return "", statR
		}
		info, ok := statR.Value.(iofs.FileInfo)
		if !ok {
			return "", core.Fail(core.E("pack.Hash", core.Sprintf("unexpected Stat shape for %q", e.Path), nil))
		}
		safetensorSizes = append(safetensorSizes, info.Size())
	}
	sort.Slice(safetensorSizes, func(i, j int) bool { return safetensorSizes[i] < safetensorSizes[j] })

	h := sha256.New()
	for _, m := range metas {
		h.Write([]byte(m.name))
		h.Write([]byte{0})
		h.Write(m.content)
		h.Write([]byte{0})
	}
	h.Write([]byte("safetensors_sizes"))
	h.Write([]byte{0})
	var sizeBuf [8]byte
	for _, sz := range safetensorSizes {
		u := uint64(sz)
		for i := 0; i < 8; i++ {
			sizeBuf[i] = byte(u >> (8 * i))
		}
		h.Write(sizeBuf[:])
	}
	return hex.EncodeToString(h.Sum(nil)), core.Ok(nil)
}

// Fingerprint returns the SHA-256 hex digest of a Manifest's Identity
// projection. Stable across machines and across re-packs of the same
// logical model. Useful for "is this the same logical artefact?" without
// reading the payload.
//
//	if pack.Fingerprint(a) == pack.Fingerprint(b) { /* same logical model */ }
func Fingerprint(m Manifest) string {
	r := core.JSONMarshal(m.Identity())
	if !r.OK {
		return ""
	}
	sum := sha256.Sum256(r.Value.([]byte))
	return hex.EncodeToString(sum[:])
}

// buildTar walks srcDir and produces a deterministic tar of all regular
// files. Entries are sorted by relative path; timestamps, uid/gid are
// zeroed so byte output is reproducible for identical input trees.
func buildTar(srcDir string) ([]byte, core.Result) {
	fs := sharedFs()

	type entry struct {
		rel  string
		abs  string
		mode iofs.FileMode
	}
	var entries []entry
	for e, err := range fs.WalkSeq(srcDir) {
		if err != nil {
			return nil, core.Fail(core.E("pack.buildTar", "walk failed", err))
		}
		if e.IsDir {
			continue
		}
		entries = append(entries, entry{
			rel:  e.Path,
			abs:  core.JoinPath(srcDir, e.Path),
			mode: e.Mode,
		})
	}

	sort.Slice(entries, func(i, j int) bool { return entries[i].rel < entries[j].rel })

	var buf bytes.Buffer
	tw := tar.NewWriter(&buf)
	for _, e := range entries {
		rr := core.ReadFile(e.abs)
		if !rr.OK {
			return nil, rr
		}
		content := rr.Value.([]byte)

		hdr := &tar.Header{
			Name:     e.rel,
			Mode:     int64(e.mode.Perm()),
			Size:     int64(len(content)),
			Typeflag: tar.TypeReg,
		}
		if err := tw.WriteHeader(hdr); err != nil {
			return nil, core.Fail(core.E("pack.buildTar", core.Sprintf("write header for %q", e.rel), err))
		}
		if _, err := tw.Write(content); err != nil {
			return nil, core.Fail(core.E("pack.buildTar", core.Sprintf("write content for %q", e.rel), err))
		}
	}
	if err := tw.Close(); err != nil {
		return nil, core.Fail(core.E("pack.buildTar", "tar.Close failed", err))
	}
	return buf.Bytes(), core.Ok(nil)
}

// extractTar reads a tar stream and writes each regular-file entry under
// destDir. Path-traversal entries (containing "..") are rejected.
func extractTar(payload []byte, destDir string) core.Result {
	tr := tar.NewReader(bytes.NewReader(payload))
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return core.Fail(core.E("pack.extractTar", "tar.Next failed", err))
		}
		if hdr.Typeflag != tar.TypeReg {
			continue
		}
		if !safeRelPath(hdr.Name) {
			return core.Fail(core.E("pack.extractTar", core.Sprintf("unsafe entry path %q", hdr.Name), nil))
		}
		out := core.JoinPath(destDir, hdr.Name)
		if mr := core.MkdirAll(core.PathDir(out), 0o755); !mr.OK {
			return mr
		}
		content := make([]byte, hdr.Size)
		if _, err := io.ReadFull(tr, content); err != nil {
			return core.Fail(core.E("pack.extractTar", core.Sprintf("read content for %q", hdr.Name), err))
		}
		if wr := core.WriteFile(out, content, iofs.FileMode(hdr.Mode)); !wr.OK {
			return wr
		}
	}
	return core.Ok(nil)
}

// manifestToHeaderMap marshals a Manifest to JSON and back into a
// map[string]interface{} suitable for trix.Trix.Header.
func manifestToHeaderMap(m Manifest) (map[string]interface{}, core.Result) {
	jr := core.JSONMarshal(m)
	if !jr.OK {
		return nil, jr
	}
	data := jr.Value.([]byte)
	var out map[string]interface{}
	if ur := core.JSONUnmarshal(data, &out); !ur.OK {
		return nil, ur
	}
	return out, core.Ok(nil)
}

// headerMapToManifest is the inverse — marshals the Trix header map back
// to JSON, then unmarshals into a typed Manifest.
func headerMapToManifest(h map[string]interface{}) (*Manifest, core.Result) {
	jr := core.JSONMarshal(h)
	if !jr.OK {
		return nil, jr
	}
	data := jr.Value.([]byte)
	var out Manifest
	if ur := core.JSONUnmarshal(data, &out); !ur.OK {
		return nil, ur
	}
	return &out, core.Ok(nil)
}

// dirExists reports whether p exists and is a directory.
func dirExists(p string) bool {
	fs := sharedFs()
	return fs.IsDir(p)
}

// assertDestDirWritable returns a failing Result if destDir exists, is a
// directory, contains entries, and overwrite is false. Missing destDir is
// fine (caller MkdirAll's it).
func assertDestDirWritable(destDir string, overwrite bool) core.Result {
	fs := sharedFs()
	if !fs.Exists(destDir) {
		return core.Ok(nil)
	}
	if !fs.IsDir(destDir) {
		return core.Fail(core.E("pack.Unpack", core.Sprintf("destDir %q exists but is not a directory", destDir), nil))
	}
	if overwrite {
		return core.Ok(nil)
	}
	lr := fs.List(destDir)
	if !lr.OK {
		return lr
	}
	if entries, ok := lr.Value.([]iofs.DirEntry); ok && len(entries) > 0 {
		return core.Fail(core.E("pack.Unpack", core.Sprintf("destDir %q is not empty (set UnpackOptions.Overwrite to allow)", destDir), nil))
	}
	return core.Ok(nil)
}

// safeRelPath rejects tar entries that would escape the destination via
// path traversal or absolute paths.
func safeRelPath(p string) bool {
	if p == "" || core.HasPrefix(p, "/") {
		return false
	}
	// Reject any ".." segment — guards against tar slip vulnerabilities.
	for _, seg := range splitSegments(p) {
		if seg == ".." {
			return false
		}
	}
	return true
}

// splitSegments splits a slash-separated path into its segments without
// importing path/filepath or strings.
func splitSegments(p string) []string {
	var out []string
	start := 0
	for i := 0; i < len(p); i++ {
		if p[i] == '/' {
			if i > start {
				out = append(out, p[start:i])
			}
			start = i + 1
		}
	}
	if start < len(p) {
		out = append(out, p[start:])
	}
	return out
}
