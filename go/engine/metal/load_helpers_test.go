// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

func TestMamba2EpsFromConfig(t *testing.T) {
	tests := []struct {
		name string
		cfg  string
		want float32
	}{
		{name: "top-level", cfg: `{"rms_norm_eps":0.000001}`, want: 0.000001},
		{name: "nested text config", cfg: `{"text_config":{"rms_norm_eps":0.000002}}`, want: 0.000002},
		{name: "default", cfg: `{}`, want: 1e-5},
		{name: "invalid json default", cfg: `{`, want: 1e-5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := mamba2EpsFromConfig([]byte(tt.cfg)); got != tt.want {
				t.Fatalf("mamba2EpsFromConfig = %g, want %g", got, tt.want)
			}
		})
	}
}

func TestNativeTokenModelVocab(t *testing.T) {
	model := &NativeTokenModel{vocab: 32000}
	if got := model.Vocab(); got != 32000 {
		t.Fatalf("Vocab = %d, want 32000", got)
	}
}

func TestNativeTokenModelSpecialLoaderErrors(t *testing.T) {
	if _, err := loadMamba2TokenModel(t.TempDir(), []byte(`{}`)); err == nil {
		t.Fatal("loadMamba2TokenModel(empty dir) error = nil")
	}
	if _, err := loadComposedTokenModel(t.TempDir(), []byte(`{}`)); err == nil {
		t.Fatal("loadComposedTokenModel(empty dir) error = nil")
	}
}

func TestLoadTokenModelDirRoutesSpecialArchitectures(t *testing.T) {
	for _, modelType := range []string{"mamba2", "qwen3_6", "composed"} {
		t.Run(modelType, func(t *testing.T) {
			dir := t.TempDir()
			cfg := core.Sprintf(`{"model_type":%q}`, modelType)
			if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), cfg); err != nil {
				t.Fatalf("write config: %v", err)
			}
			if _, err := LoadTokenModelDir(dir, 4); err == nil {
				t.Fatal("LoadTokenModelDir(special architecture without weights) error = nil")
			}
		})
	}
}
