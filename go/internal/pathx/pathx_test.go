// SPDX-Licence-Identifier: EUPL-1.2

package pathx

import (
	"runtime"
	"testing"
)

// TestPathx_Base_Good pins the ordinary shapes: a '/'-separated absolute path,
// a Hugging Face style ref, and a bare name with no separator at all. All three
// answer identically on every platform, which is the point of the helper —
// core.PathBase hands back the whole string for these on Windows.
func TestPathx_Base_Good(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"/models/gemma3-1b", "gemma3-1b"},
		{"/models/gemma4/model.gguf", "model.gguf"},
		{"org/model-a", "model-a"},
		{"model.gguf", "model.gguf"},
	} {
		if got := Base(tc.in); got != tc.want {
			t.Errorf("Base(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

// TestPathx_Base_Bad covers the inputs that name no element: empty, a lone
// separator, and a run of them.
func TestPathx_Base_Bad(t *testing.T) {
	for _, in := range []string{"", "/", "///"} {
		if got := Base(in); got != "" {
			t.Errorf("Base(%q) = %q, want %q", in, got, "")
		}
	}
}

// TestPathx_Base_Ugly covers the surprising-but-valid cases: trailing
// separators name the element before them, and a backslash is a separator only
// where the platform agrees — on POSIX it is a legitimate filename character
// and must survive intact.
func TestPathx_Base_Ugly(t *testing.T) {
	if got := Base("/models/gemma3-1b/"); got != "gemma3-1b" {
		t.Errorf("Base(trailing slash) = %q, want %q", got, "gemma3-1b")
	}
	if got := Base("/models/gemma3-1b//"); got != "gemma3-1b" {
		t.Errorf("Base(trailing slashes) = %q, want %q", got, "gemma3-1b")
	}

	backslashed := `C:\models\gemma3-1b`
	got := Base(backslashed)
	if runtime.GOOS == "windows" {
		if got != "gemma3-1b" {
			t.Errorf("Base(%q) = %q, want %q", backslashed, got, "gemma3-1b")
		}
		return
	}
	if got != backslashed {
		t.Errorf("Base(%q) = %q, want it unchanged — '\\' is a filename character on %s",
			backslashed, got, runtime.GOOS)
	}
}

// TestPathx_Join_Good pins the ordinary join and the trailing-separator
// collapse, both of which keep the separator the caller already used.
func TestPathx_Join_Good(t *testing.T) {
	for _, tc := range []struct{ dir, child, want string }{
		{"/models/my-lora", "adapter_config.json", "/models/my-lora/adapter_config.json"},
		{"/models/my-lora/", "adapter_config.json", "/models/my-lora/adapter_config.json"},
		{"/", "adapter_config.json", "/adapter_config.json"},
	} {
		if got := Join(tc.dir, tc.child); got != tc.want {
			t.Errorf("Join(%q, %q) = %q, want %q", tc.dir, tc.child, got, tc.want)
		}
	}
}

// TestPathx_Join_Bad covers the empty dir, which yields the bare child rather
// than an accidentally root-anchored path.
func TestPathx_Join_Bad(t *testing.T) {
	if got := Join("", "adapter_config.json"); got != "adapter_config.json" {
		t.Errorf("Join(empty dir) = %q, want %q", got, "adapter_config.json")
	}
}

// TestPathx_Join_Ugly covers a separator-less dir, which takes the platform's
// own separator — the only case where the result's shape depends on the host.
func TestPathx_Join_Ugly(t *testing.T) {
	got := Join(".", "adapter_config.json")
	want := "." + separatorOf(".") + "adapter_config.json"
	if got != want {
		t.Errorf("Join(%q, %q) = %q, want %q", ".", "adapter_config.json", got, want)
	}
	if runtime.GOOS != "windows" && got != "./adapter_config.json" {
		t.Errorf("Join(%q, ...) = %q, want %q on %s", ".", got, "./adapter_config.json", runtime.GOOS)
	}
}

// TestPathx_Dir_Good pins the parent of a nested path and of a path anchored
// directly at the root.
func TestPathx_Dir_Good(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"/models/my-lora/adapter.safetensors", "/models/my-lora"},
		{"/models/my-lora", "/models"},
		{"/models", "/"},
		{"org/model-a", "org"},
	} {
		if got := Dir(tc.in); got != tc.want {
			t.Errorf("Dir(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

// TestPathx_Dir_Bad covers the inputs with no parent to report. Unlike
// core.PathDir these answer "" rather than ".", so a caller can tell "no
// directory component" from "the current directory".
func TestPathx_Dir_Bad(t *testing.T) {
	for _, in := range []string{"", "adapter.safetensors"} {
		if got := Dir(in); got != "" {
			t.Errorf("Dir(%q) = %q, want %q", in, got, "")
		}
	}
}

// TestPathx_Dir_Ugly covers trailing separators — "/models/x/" has the same
// parent as "/models/x" — and the platform-dependent backslash.
func TestPathx_Dir_Ugly(t *testing.T) {
	if got := Dir("/models/my-lora/"); got != "/models" {
		t.Errorf("Dir(trailing slash) = %q, want %q", got, "/models")
	}
	if got := Dir("/"); got != "/" {
		t.Errorf("Dir(root) = %q, want %q — the root is its own parent, as filepath.Dir has it", got, "/")
	}

	backslashed := `C:\models\my-lora\adapter.safetensors`
	got := Dir(backslashed)
	if runtime.GOOS == "windows" {
		if got != `C:\models\my-lora` {
			t.Errorf("Dir(%q) = %q, want %q", backslashed, got, `C:\models\my-lora`)
		}
		return
	}
	if got != "" {
		t.Errorf("Dir(%q) = %q, want %q — '\\' is a filename character on %s",
			backslashed, got, "", runtime.GOOS)
	}
}
