// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"testing"

	core "dappco.re/go"
)

func TestMetadata_Good(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "model.gguf")
	writeTestGGUF(t, path, []ggufMetaSpec{
		{Key: "general.architecture", ValueType: ValueTypeString, Value: "phi"},
		{Key: "general.file_type", ValueType: ValueTypeUint32, Value: uint32(1)},
	}, nil)

	meta, err := Metadata(path)
	if err != nil {
		t.Fatalf("Metadata: %v", err)
	}
	if meta["general.architecture"] != "phi" {
		t.Errorf("architecture = %v, want phi", meta["general.architecture"])
	}
	if meta["general.file_type"] != uint32(1) {
		t.Errorf("file_type = %v, want 1", meta["general.file_type"])
	}
	if len(meta) != 2 {
		t.Errorf("len(meta) = %d, want 2", len(meta))
	}
}

func TestMetadata_Bad(t *testing.T) {
	_, err := Metadata(core.PathJoin(t.TempDir(), "missing.gguf"))
	if err == nil {
		t.Fatalf("Metadata(missing file): want error, got nil")
	}
}
