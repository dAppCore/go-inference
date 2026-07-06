// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"

	"dappco.re/go/inference/model/safetensors"
)

// TestProbeModelTypes covers the model_type probe: the top-level id and the multimodal-wrapper's nested
// text_config.model_type, which the reactive loader resolves against the alias registry.
func TestProbeModelTypes(t *testing.T) {
	cases := []struct{ json, wantMT, wantText string }{
		{`{"model_type":"arch3"}`, "arch3", ""},
		{`{"model_type":"wrap_unified","text_config":{"model_type":"wrap_text"}}`, "wrap_unified", "wrap_text"},
		{`{"text_config":{"model_type":"wrap_text"}}`, "", "wrap_text"},
		{`{}`, "", ""},
	}
	for _, c := range cases {
		mt, text := probeModelTypes([]byte(c.json))
		if mt != c.wantMT || text != c.wantText {
			t.Fatalf("probeModelTypes(%s) = (%q,%q), want (%q,%q)", c.json, mt, text, c.wantMT, c.wantText)
		}
	}
}

// TestLoadUnregisteredModelType covers the dispatch-miss path: a config.json whose model_type has no
// registered ArchSpec is a clean error (read + probe + LookupArch), before any mmap. The full-load
// success path is exercised end-to-end by the native gate once a backend delegates to model.Load.
func TestLoadUnregisteredModelType(t *testing.T) {
	dir := t.TempDir()
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), `{"model_type":"nope4"}`); err != nil {
		t.Fatalf("write config: %v", err)
	}
	if _, _, err := Load(dir); err == nil {
		t.Fatal("expected an error for an unregistered model_type")
	}
}

// fakeLoadArchConfig is a minimal ArchConfig for exercising Load end-to-end: a fixed,
// zero-layer Arch (so Assemble's layer loop never runs), no dim inference needed.
type fakeLoadArchConfig struct{ hidden int }

func (c fakeLoadArchConfig) InferFromWeights(map[string]safetensors.Tensor) {}
func (c fakeLoadArchConfig) Arch() (Arch, error)                            { return Arch{Hidden: c.hidden}, nil }

// writeLoadFixture writes a hermetic checkpoint dir: config.json declaring modelType,
// plus a single model.safetensors carrying an "embed.weight" [vocab,hidden] tensor and a
// "norm.weight" [hidden] tensor — exactly what a zero-layer Arch's Assemble needs
// (Embed + FinalNorm, the only always-required weights with no layers).
func writeLoadFixture(t testing.TB, modelType string) string {
	t.Helper()
	dir := t.TempDir()
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), `{"model_type":"`+modelType+`"}`); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	const vocab, hidden = 8, 4
	infos := map[string]safetensors.SafetensorsTensorInfo{
		"embed.weight": {Dtype: "F32", Shape: []int{vocab, hidden}},
		"norm.weight":  {Dtype: "F32", Shape: []int{hidden}},
	}
	data := map[string][]byte{
		"embed.weight": make([]byte, vocab*hidden*4),
		"norm.weight":  make([]byte, hidden*4),
	}
	if r := safetensors.WriteSafetensors(core.PathJoin(dir, "model.safetensors"), infos, data); !r.OK {
		t.Fatalf("WriteSafetensors: %s", r.Error())
	}
	return dir
}

// TestLoad_Load_Good covers the full reactive load: read config.json, probe model_type,
// dispatch to the registered ArchSpec, mmap the checkpoint, assemble the LoadedModel.
func TestLoad_Load_Good(t *testing.T) {
	RegisterArch(ArchSpec{
		ModelTypes: []string{"loadtest-good"},
		Parse:      func([]byte) (ArchConfig, error) { return fakeLoadArchConfig{hidden: 4}, nil },
		Weights:    WeightNames{Embed: "embed", FinalNorm: "norm.weight"},
	})
	dir := writeLoadFixture(t, "loadtest-good")
	m, dm, err := Load(dir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer func() { _ = dm.Close() }()
	if m == nil || m.Embed == nil || m.FinalNorm == nil {
		t.Fatalf("Load result missing Embed/FinalNorm: %+v", m)
	}
	if m.Embed.OutDim != 8 {
		t.Fatalf("Embed.OutDim = %d, want 8 (the vocab, read from the tensor shape)", m.Embed.OutDim)
	}
}

// TestLoad_Load_Bad covers the dispatch-miss path: an unregistered model_type is a
// clean error before any mmap — the same case TestLoadUnregisteredModelType names.
func TestLoad_Load_Bad(t *testing.T) {
	dir := t.TempDir()
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), `{"model_type":"loadtest-bad-unregistered"}`); err != nil {
		t.Fatalf("write config: %v", err)
	}
	if _, _, err := Load(dir); err == nil {
		t.Fatal("Load with an unregistered model_type: expected an error")
	}
}

// TestLoad_Load_Ugly covers a downstream failure AFTER successful registration + mmap: a
// checkpoint missing the always-required embed tensor surfaces Assemble's error, and the
// DirMapping is closed on that path (no leaked mmap) rather than propagating a partial result.
func TestLoad_Load_Ugly(t *testing.T) {
	RegisterArch(ArchSpec{
		ModelTypes: []string{"loadtest-ugly"},
		Parse:      func([]byte) (ArchConfig, error) { return fakeLoadArchConfig{hidden: 4}, nil },
		Weights:    WeightNames{Embed: "embed", FinalNorm: "norm.weight"},
	})
	dir := t.TempDir()
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), `{"model_type":"loadtest-ugly"}`); err != nil {
		t.Fatalf("write config: %v", err)
	}
	// only norm.weight present — embed.weight (always required) is missing.
	infos := map[string]safetensors.SafetensorsTensorInfo{"norm.weight": {Dtype: "F32", Shape: []int{4}}}
	data := map[string][]byte{"norm.weight": make([]byte, 16)}
	if r := safetensors.WriteSafetensors(core.PathJoin(dir, "model.safetensors"), infos, data); !r.OK {
		t.Fatalf("WriteSafetensors: %s", r.Error())
	}
	if _, _, err := Load(dir); err == nil {
		t.Fatal("Load with no embed tensor: expected an Assemble error")
	}
}

// TestLoad_ProbeDirArch_Good covers the front-door check: it returns the top-level
// model_type plus the raw config bytes, for a backend whose loader isn't the reactive
// Assemble path (e.g. a recurrent SSM with its own loader).
func TestLoad_ProbeDirArch_Good(t *testing.T) {
	dir := t.TempDir()
	const cfg = `{"model_type":"probearch-good","hidden_size":4}`
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), cfg); err != nil {
		t.Fatalf("write config: %v", err)
	}
	mt, raw, err := ProbeDirArch(dir)
	if err != nil {
		t.Fatalf("ProbeDirArch: %v", err)
	}
	if mt != "probearch-good" {
		t.Fatalf("ProbeDirArch modelType = %q, want %q", mt, "probearch-good")
	}
	if string(raw) != cfg {
		t.Fatalf("ProbeDirArch configJSON = %q, want the raw file contents %q", raw, cfg)
	}
}

// TestLoad_ProbeDirArch_Bad covers a directory with no config.json: a clean read error.
func TestLoad_ProbeDirArch_Bad(t *testing.T) {
	if _, _, err := ProbeDirArch(t.TempDir()); err == nil {
		t.Fatal("ProbeDirArch with no config.json: expected an error")
	}
}

// TestLoad_ProbeDirArch_Ugly covers malformed config.json: probeModelTypes silently
// ignores the unmarshal error (`_ = core.JSONUnmarshal`), so ProbeDirArch returns an
// EMPTY model_type and nil error rather than surfacing the parse failure — a documented
// edge worth pinning so a future change to that ignore doesn't go unnoticed.
func TestLoad_ProbeDirArch_Ugly(t *testing.T) {
	dir := t.TempDir()
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), `not json`); err != nil {
		t.Fatalf("write config: %v", err)
	}
	mt, raw, err := ProbeDirArch(dir)
	if err != nil {
		t.Fatalf("ProbeDirArch(malformed json): %v, want nil error (probe failure is silently ignored)", err)
	}
	if mt != "" {
		t.Fatalf("ProbeDirArch(malformed json) modelType = %q, want empty", mt)
	}
	if string(raw) != "not json" {
		t.Fatalf("ProbeDirArch(malformed json) configJSON = %q, want the raw bytes returned regardless", raw)
	}
}

// TestLoad_ProbeModelTypes_Good covers the exported front door onto probeModelTypes: a
// multimodal wrapper's top-level id AND its nested text_config.model_type both resolve.
func TestLoad_ProbeModelTypes_Good(t *testing.T) {
	mt, text := ProbeModelTypes([]byte(`{"model_type":"wrap_unified","text_config":{"model_type":"wrap_text"}}`))
	if mt != "wrap_unified" || text != "wrap_text" {
		t.Fatalf("ProbeModelTypes = (%q,%q), want (%q,%q)", mt, text, "wrap_unified", "wrap_text")
	}
}

// TestLoad_ProbeModelTypes_Bad covers a flat (non-wrapper) config: only the top-level
// model_type is set, textModelType is empty.
func TestLoad_ProbeModelTypes_Bad(t *testing.T) {
	mt, text := ProbeModelTypes([]byte(`{"model_type":"arch3"}`))
	if mt != "arch3" || text != "" {
		t.Fatalf("ProbeModelTypes = (%q,%q), want (%q,\"\")", mt, text, "arch3")
	}
}

// TestLoad_ProbeModelTypes_Ugly covers the degenerate empty JSON object: both ids come
// back empty, never an error (the caller decides what "no model_type" means).
func TestLoad_ProbeModelTypes_Ugly(t *testing.T) {
	mt, text := ProbeModelTypes([]byte(`{}`))
	if mt != "" || text != "" {
		t.Fatalf("ProbeModelTypes({}) = (%q,%q), want (\"\",\"\")", mt, text)
	}
}
