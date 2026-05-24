// SPDX-Licence-Identifier: EUPL-1.2

package pack_test

import (
	"crypto/sha256"
	"encoding/hex"
	iofs "io/fs"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/model/pack"
)

// fixtureFile is one synthetic file written into the fixture pack dir.
type fixtureFile struct {
	relPath string
	content []byte
	mode    iofs.FileMode
}

// buildFixturePack writes a small but realistic Gemma-4-shaped pack into
// dir — config.json + tokenizer.json + chat_template.jinja + a small
// model.safetensors with a valid header. Tests use this as the round-trip
// source.
func buildFixturePack(t *testing.T, dir string, extras ...fixtureFile) {
	t.Helper()

	if mr := core.MkdirAll(dir, 0o755); !mr.OK {
		t.Fatalf("MkdirAll %q: %v", dir, mr.Value)
	}

	defaults := []fixtureFile{
		{
			relPath: "config.json",
			content: []byte(`{"model_type":"gemma","architectures":["GemmaForCausalLM"],"hidden_size":2304,"num_hidden_layers":26,"num_attention_heads":8,"vocab_size":262144}`),
			mode:    0o644,
		},
		{
			relPath: "tokenizer.json",
			content: []byte(`{"version":"1.0","tokenizer":{"type":"sentencepiece"},"bos_token":"<bos>","eos_token":"<eos>"}`),
			mode:    0o644,
		},
		{
			relPath: "chat_template.jinja",
			content: []byte(`{% for m in messages %}{{m.role}}: {{m.content}}{% endfor %}`),
			mode:    0o644,
		},
		{
			relPath: "model.safetensors",
			content: synthSafetensors(),
			mode:    0o644,
		},
	}

	for _, ff := range append(defaults, extras...) {
		path := core.JoinPath(dir, ff.relPath)
		if dirPath := core.PathDir(path); dirPath != dir {
			if mr := core.MkdirAll(dirPath, 0o755); !mr.OK {
				t.Fatalf("MkdirAll %q: %v", dirPath, mr.Value)
			}
		}
		if wr := core.WriteFile(path, ff.content, ff.mode); !wr.OK {
			t.Fatalf("WriteFile %q: %v", path, wr.Value)
		}
	}
}

// synthSafetensors emits a valid-shape safetensors file: 8-byte little-
// endian header length + JSON header + zero-byte tensor payload. Loader
// won't read tensors so empty payload is fine.
func synthSafetensors() []byte {
	header := []byte(`{"__metadata__":{"format":"pt"}}`)
	// 8-byte little-endian length prefix
	out := make([]byte, 8+len(header))
	n := uint64(len(header))
	for i := 0; i < 8; i++ {
		out[i] = byte(n >> (8 * i))
	}
	copy(out[8:], header)
	return out
}

// fileTreeHash returns a single SHA-256 over a sorted (relPath || sha256(content))
// of every regular file under dir, suitable for byte-level tree equality
// assertions.
func fileTreeHash(t *testing.T, dir string) string {
	t.Helper()
	fs := (&core.Fs{}).NewUnrestricted()
	type entry struct {
		rel  string
		hash [32]byte
	}
	var entries []entry
	for e, err := range fs.WalkSeq(dir) {
		if err != nil {
			t.Fatalf("WalkSeq %q: %v", dir, err)
		}
		if e.IsDir {
			continue
		}
		rr := core.ReadFile(core.JoinPath(dir, e.Path))
		if !rr.OK {
			t.Fatalf("ReadFile %q: %v", e.Path, rr.Value)
		}
		entries = append(entries, entry{
			rel:  e.Path,
			hash: sha256.Sum256(rr.Value.([]byte)),
		})
	}
	// Sort
	for i := 0; i < len(entries); i++ {
		for j := i + 1; j < len(entries); j++ {
			if entries[j].rel < entries[i].rel {
				entries[i], entries[j] = entries[j], entries[i]
			}
		}
	}
	h := sha256.New()
	for _, e := range entries {
		h.Write([]byte(e.rel))
		h.Write([]byte{0})
		h.Write(e.hash[:])
		h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil))
}

func sampleManifest() pack.Manifest {
	return pack.Manifest{
		Model: inference.ModelIdentity{
			ID:            "google/gemma-3-4b-it",
			Architecture:  "gemma",
			QuantBits:     4,
			ContextLength: 8192,
			NumLayers:     26,
			HiddenSize:    2304,
			VocabSize:     262144,
		},
		Tokenizer: inference.TokenizerIdentity{
			Kind:         "sentencepiece",
			ChatTemplate: "gemma",
		},
		SourceFormat: "safetensors",
		Producer: pack.Producer{
			Name:   "go-mlx",
			Commit: "abc123",
		},
	}
}

func TestPack_Roundtrip_Good(t *testing.T) {
	tempRoot := (&core.Fs{}).NewUnrestricted().TempDir("pack-roundtrip-good-")
	defer core.RemoveAll(tempRoot)

	srcDir := core.JoinPath(tempRoot, "src")
	dest := core.JoinPath(tempRoot, "out.model")
	outDir := core.JoinPath(tempRoot, "out")

	buildFixturePack(t, srcDir)
	srcHash := fileTreeHash(t, srcDir)

	if r := pack.Pack(srcDir, dest, pack.PackOptions{Manifest: sampleManifest()}); !r.OK {
		t.Fatalf("Pack: %v", r.Value)
	}

	// Verify dest starts with Trix magic "MDL1".
	data := readBytes(t, dest)
	if string(data[:4]) != pack.Magic {
		t.Fatalf("expected magic %q at offset 0, got %q", pack.Magic, string(data[:4]))
	}

	if r := pack.Unpack(dest, outDir, pack.UnpackOptions{}); !r.OK {
		t.Fatalf("Unpack: %v", r.Value)
	}
	outHash := fileTreeHash(t, outDir)

	if srcHash != outHash {
		t.Fatalf("file tree hash mismatch:\n  src: %s\n  out: %s", srcHash, outHash)
	}
}

func TestPack_Inspect_Good(t *testing.T) {
	tempRoot := (&core.Fs{}).NewUnrestricted().TempDir("pack-inspect-good-")
	defer core.RemoveAll(tempRoot)

	srcDir := core.JoinPath(tempRoot, "src")
	dest := core.JoinPath(tempRoot, "out.model")

	buildFixturePack(t, srcDir)

	if r := pack.Pack(srcDir, dest, pack.PackOptions{Manifest: sampleManifest()}); !r.OK {
		t.Fatalf("Pack: %v", r.Value)
	}

	manifest, inspection, r := pack.Inspect(dest)
	if !r.OK {
		t.Fatalf("Inspect: %v", r.Value)
	}
	if manifest.Model.Architecture != "gemma" {
		t.Errorf("expected Architecture gemma, got %q", manifest.Model.Architecture)
	}
	if manifest.Model.QuantBits != 4 {
		t.Errorf("expected QuantBits 4, got %d", manifest.Model.QuantBits)
	}
	if manifest.SourceFormat != "safetensors" {
		t.Errorf("expected SourceFormat safetensors, got %q", manifest.SourceFormat)
	}
	if manifest.Producer.Created == "" {
		t.Errorf("expected Producer.Created to be auto-filled, was empty")
	}
	if inspection.Path != dest {
		t.Errorf("expected inspection.Path %q, got %q", dest, inspection.Path)
	}
	if inspection.Format != "safetensors" {
		t.Errorf("expected inspection.Format safetensors, got %q", inspection.Format)
	}
	if inspection.Model.Architecture != "gemma" {
		t.Errorf("expected inspection.Model.Architecture gemma, got %q", inspection.Model.Architecture)
	}
}

func TestPack_Roundtrip_Bad(t *testing.T) {
	// Truncated .model file must return a failing Result, never panic.
	tempRoot := (&core.Fs{}).NewUnrestricted().TempDir("pack-bad-")
	defer core.RemoveAll(tempRoot)

	srcDir := core.JoinPath(tempRoot, "src")
	dest := core.JoinPath(tempRoot, "out.model")
	outDir := core.JoinPath(tempRoot, "out")

	buildFixturePack(t, srcDir)

	if r := pack.Pack(srcDir, dest, pack.PackOptions{Manifest: sampleManifest()}); !r.OK {
		t.Fatalf("Pack: %v", r.Value)
	}

	// Truncate dest to half its size — payload is now corrupt.
	full := readBytes(t, dest)
	half := full[:len(full)/2]
	if wr := core.WriteFile(dest, half, 0o644); !wr.OK {
		t.Fatalf("WriteFile (truncate): %v", wr.Value)
	}

	r := pack.Unpack(dest, outDir, pack.UnpackOptions{})
	if r.OK {
		t.Fatalf("expected Unpack to fail on truncated input, got OK")
	}
}

func TestPack_Roundtrip_Ugly(t *testing.T) {
	// Unusual but valid file names — spaces and unicode — must round-trip
	// intact.
	tempRoot := (&core.Fs{}).NewUnrestricted().TempDir("pack-ugly-")
	defer core.RemoveAll(tempRoot)

	srcDir := core.JoinPath(tempRoot, "src")
	dest := core.JoinPath(tempRoot, "out.model")
	outDir := core.JoinPath(tempRoot, "out")

	extras := []fixtureFile{
		{relPath: "notes with spaces.txt", content: []byte("hello"), mode: 0o644},
		{relPath: "papierość.bin", content: []byte{0x00, 0x01, 0x02, 0xFF}, mode: 0o644},
		{relPath: "subdir/nested.json", content: []byte(`{"k":"v"}`), mode: 0o644},
	}
	buildFixturePack(t, srcDir, extras...)
	srcHash := fileTreeHash(t, srcDir)

	if r := pack.Pack(srcDir, dest, pack.PackOptions{Manifest: sampleManifest()}); !r.OK {
		t.Fatalf("Pack: %v", r.Value)
	}
	if r := pack.Unpack(dest, outDir, pack.UnpackOptions{}); !r.OK {
		t.Fatalf("Unpack: %v", r.Value)
	}
	if outHash := fileTreeHash(t, outDir); outHash != srcHash {
		t.Fatalf("ugly tree hash mismatch:\n  src: %s\n  out: %s", srcHash, outHash)
	}
}

func TestPack_VindexOption_Bad(t *testing.T) {
	// Seam-honesty: VindexBlob != nil must return an explicit
	// "not yet implemented" failure so callers know the embedding seam
	// exists but isn't wired.
	tempRoot := (&core.Fs{}).NewUnrestricted().TempDir("pack-vindex-bad-")
	defer core.RemoveAll(tempRoot)

	srcDir := core.JoinPath(tempRoot, "src")
	dest := core.JoinPath(tempRoot, "out.model")

	buildFixturePack(t, srcDir)

	r := pack.Pack(srcDir, dest, pack.PackOptions{
		Manifest:   sampleManifest(),
		VindexBlob: []byte("not real msgpack but non-nil"),
	})
	if r.OK {
		t.Fatalf("expected Pack to fail when VindexBlob is non-nil, got OK")
	}
}

func TestPack_List_Good(t *testing.T) {
	tempRoot := (&core.Fs{}).NewUnrestricted().TempDir("pack-list-good-")
	defer core.RemoveAll(tempRoot)

	srcDir := core.JoinPath(tempRoot, "src")
	dest := core.JoinPath(tempRoot, "out.model")

	buildFixturePack(t, srcDir)

	if r := pack.Pack(srcDir, dest, pack.PackOptions{Manifest: sampleManifest()}); !r.OK {
		t.Fatalf("Pack: %v", r.Value)
	}

	entries, manifest, r := pack.List(dest)
	if !r.OK {
		t.Fatalf("List: %v", r.Value)
	}
	if manifest.SourceFormat != "safetensors" {
		t.Errorf("expected manifest.SourceFormat safetensors, got %q", manifest.SourceFormat)
	}

	want := map[string]bool{
		"config.json":         false,
		"tokenizer.json":      false,
		"chat_template.jinja": false,
		"model.safetensors":   false,
	}
	for _, e := range entries {
		if _, ok := want[e.Path]; !ok {
			t.Errorf("unexpected entry %q", e.Path)
			continue
		}
		want[e.Path] = true
		if e.Size <= 0 {
			t.Errorf("entry %q has non-positive size %d", e.Path, e.Size)
		}
	}
	for name, seen := range want {
		if !seen {
			t.Errorf("expected entry %q not present in List output", name)
		}
	}
}

func TestPack_List_Bad(t *testing.T) {
	tempRoot := (&core.Fs{}).NewUnrestricted().TempDir("pack-list-bad-")
	defer core.RemoveAll(tempRoot)

	srcDir := core.JoinPath(tempRoot, "src")
	dest := core.JoinPath(tempRoot, "out.model")

	buildFixturePack(t, srcDir)
	if r := pack.Pack(srcDir, dest, pack.PackOptions{Manifest: sampleManifest()}); !r.OK {
		t.Fatalf("Pack: %v", r.Value)
	}

	full := readBytes(t, dest)
	if wr := core.WriteFile(dest, full[:len(full)/2], 0o644); !wr.OK {
		t.Fatalf("WriteFile (truncate): %v", wr.Value)
	}

	if _, _, r := pack.List(dest); r.OK {
		t.Fatalf("expected List to fail on truncated input, got OK")
	}
}

func TestPack_Deterministic_Good(t *testing.T) {
	// Same source tree + same Manifest (Producer.Created pinned) must
	// produce byte-identical .model output, twice in a row. The property
	// `.model` is content-addressable depends on it: same input → same
	// SHA-256 → cache hits, lineage chains, registry dedup all work.
	tempRoot := (&core.Fs{}).NewUnrestricted().TempDir("pack-deterministic-good-")
	defer core.RemoveAll(tempRoot)

	srcDir := core.JoinPath(tempRoot, "src")
	dest1 := core.JoinPath(tempRoot, "out1.model")
	dest2 := core.JoinPath(tempRoot, "out2.model")

	buildFixturePack(t, srcDir, fixtureFile{
		relPath: "extras/zeta.bin",
		content: []byte("trailing-entry-to-stress-sort-order"),
		mode:    0o644,
	}, fixtureFile{
		relPath: "extras/alpha.bin",
		content: []byte("leading-entry-to-stress-sort-order"),
		mode:    0o644,
	})

	manifest := sampleManifest()
	manifest.Producer.Created = "2026-01-01T00:00:00Z" // pin so the only delta source is the algorithm itself

	if r := pack.Pack(srcDir, dest1, pack.PackOptions{Manifest: manifest}); !r.OK {
		t.Fatalf("Pack #1: %v", r.Value)
	}
	if r := pack.Pack(srcDir, dest2, pack.PackOptions{Manifest: manifest}); !r.OK {
		t.Fatalf("Pack #2: %v", r.Value)
	}

	b1 := readBytes(t, dest1)
	b2 := readBytes(t, dest2)

	h1 := sha256.Sum256(b1)
	h2 := sha256.Sum256(b2)

	if hex.EncodeToString(h1[:]) != hex.EncodeToString(h2[:]) {
		t.Fatalf("Pack non-deterministic:\n  size1=%d sha=%s\n  size2=%d sha=%s\nFirst differing byte index: %d",
			len(b1), hex.EncodeToString(h1[:]),
			len(b2), hex.EncodeToString(h2[:]),
			firstDiffIndex(b1, b2),
		)
	}
}

func firstDiffIndex(a, b []byte) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	if len(a) != len(b) {
		return n
	}
	return -1
}

func TestPack_Fingerprint_TimestampOrthogonal_Good(t *testing.T) {
	// Two manifests differing only in Producer.Created (provenance) +
	// Lineage (provenance) + Signatures (orthogonal) must produce the
	// same identity fingerprint.
	a := sampleManifest()
	a.Producer.Created = "2026-01-01T00:00:00Z"
	a.Lineage = &pack.Lineage{TrainURI: "file:///a.train", TrainSHA: "deadbeef"}
	a.Signatures = []pack.Signature{{KeyID: "k1", Alg: "ed25519", Sig: "sigA"}}

	b := sampleManifest()
	b.Producer.Created = "2027-06-15T12:34:56Z"
	b.Producer.Commit = "different-commit"
	b.Lineage = &pack.Lineage{TrainURI: "file:///somewhere/else.train", TrainSHA: "beefcafe"}
	b.Signatures = []pack.Signature{{KeyID: "k2", Alg: "ed25519", Sig: "sigB"}}

	if pack.Fingerprint(a) != pack.Fingerprint(b) {
		t.Fatalf("expected fingerprints equal under provenance-only delta:\n  a=%s\n  b=%s",
			pack.Fingerprint(a), pack.Fingerprint(b))
	}
}

func TestPack_Fingerprint_IdentityDelta_Ugly(t *testing.T) {
	// Each identity-shaping field, varied independently, must change the
	// fingerprint. If any of these doesn't change it, identity has a hole.
	base := sampleManifest()
	baseFP := pack.Fingerprint(base)

	cases := []struct {
		name   string
		mutate func(*pack.Manifest)
	}{
		{"Model.Architecture", func(m *pack.Manifest) { m.Model.Architecture = "llama" }},
		{"Model.QuantBits", func(m *pack.Manifest) { m.Model.QuantBits = 8 }},
		{"Model.NumLayers", func(m *pack.Manifest) { m.Model.NumLayers = 99 }},
		{"Model.VocabSize", func(m *pack.Manifest) { m.Model.VocabSize = 100000 }},
		{"Tokenizer.Kind", func(m *pack.Manifest) { m.Tokenizer.Kind = "gpt2-bpe" }},
		{"Tokenizer.ChatTemplate", func(m *pack.Manifest) { m.Tokenizer.ChatTemplate = "llama" }},
		{"SourceFormat", func(m *pack.Manifest) { m.SourceFormat = "gguf" }},
	}
	for _, tc := range cases {
		m := sampleManifest()
		tc.mutate(&m)
		got := pack.Fingerprint(m)
		if got == baseFP {
			t.Errorf("mutating %s did not change fingerprint (still %s)", tc.name, got)
		}
	}
}

func TestPack_Fingerprint_HexShape_Good(t *testing.T) {
	// Sanity: fingerprint is hex sha256 (64 chars, lower-case hex).
	fp := pack.Fingerprint(sampleManifest())
	if len(fp) != 64 {
		t.Errorf("expected 64-char fingerprint, got %d (%q)", len(fp), fp)
	}
	for _, r := range fp {
		switch {
		case r >= '0' && r <= '9':
		case r >= 'a' && r <= 'f':
		default:
			t.Errorf("non-hex character %q in fingerprint %q", r, fp)
		}
	}
}

// readBytes is a small test helper that reads a file via core.ReadFile.
func readBytes(t *testing.T, path string) []byte {
	t.Helper()
	rr := core.ReadFile(path)
	if !rr.OK {
		t.Fatalf("ReadFile %q: %v", path, rr.Value)
	}
	return rr.Value.([]byte)
}
