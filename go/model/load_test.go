// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
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
