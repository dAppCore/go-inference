// SPDX-Licence-Identifier: EUPL-1.2

package rwkv7

import "testing"

func TestConfig_epsFromConfig_Good(t *testing.T) {
	if got := epsFromConfig([]byte(`{"norm_eps": 2.5e-4}`)); got != 2.5e-4 {
		t.Fatalf("epsFromConfig = %v, want 2.5e-4", got)
	}
}

// TestConfig_epsFromConfig_Bad proves an absent norm_eps falls back to RWKV7Config's own default
// (1e-5), the value every released RWKV-7 checkpoint declares.
func TestConfig_epsFromConfig_Bad(t *testing.T) {
	if got := epsFromConfig([]byte(`{}`)); got != 1e-5 {
		t.Fatalf("epsFromConfig({}) = %v, want default 1e-5", got)
	}
}

// TestConfig_epsFromConfig_Ugly proves malformed JSON degrades to the same default rather than panicking.
func TestConfig_epsFromConfig_Ugly(t *testing.T) {
	if got := epsFromConfig([]byte(`not json`)); got != 1e-5 {
		t.Fatalf("epsFromConfig(malformed) = %v, want default 1e-5", got)
	}
}

// TestConfig_checkUnsupportedConfig_Good proves the real checkpoint's config (attn:null, hidden_act
// absent or "sqrelu") passes.
func TestConfig_checkUnsupportedConfig_Good(t *testing.T) {
	if err := checkUnsupportedConfig([]byte(`{"attn": null, "hidden_act": "sqrelu"}`)); err != nil {
		t.Fatalf("real-shaped config rejected: %v", err)
	}
	if err := checkUnsupportedConfig([]byte(`{}`)); err != nil {
		t.Fatalf("empty config rejected: %v", err)
	}
}

// TestConfig_checkUnsupportedConfig_Bad rejects a non-null "attn" (a hybrid softmax-attention config this
// port does not implement).
func TestConfig_checkUnsupportedConfig_Bad(t *testing.T) {
	if err := checkUnsupportedConfig([]byte(`{"attn": {"layers": [0], "num_heads": 4}}`)); err == nil {
		t.Fatal("hybrid attn config accepted")
	}
}

// TestConfig_checkUnsupportedConfig_Ugly rejects a hidden_act other than "sqrelu" — a distinct refusal
// reason from _Bad's hybrid-attn case.
func TestConfig_checkUnsupportedConfig_Ugly(t *testing.T) {
	if err := checkUnsupportedConfig([]byte(`{"hidden_act": "gelu"}`)); err == nil {
		t.Fatal("unsupported hidden_act accepted")
	}
}
