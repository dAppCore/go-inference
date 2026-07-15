// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"testing"

	core "dappco.re/go"
)

// TestRunServeCommand_Rejects covers the parse guards that return before the
// serve loop: -h exits 0, an unknown flag exits 2.
func TestRunServeCommand_Rejects(t *testing.T) {
	for _, tc := range []struct {
		name     string
		args     []string
		wantCode int
	}{
		{"help", []string{"--help"}, 0},
		{"unknown flag", []string{"--nonsense"}, 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			if code := runServeCommand(context.Background(), tc.args, &stdout, &stderr); code != tc.wantCode {
				t.Fatalf("exit %d, want %d; stderr=%s", code, tc.wantCode, stderr.String())
			}
		})
	}
}

// adminTokenFile is the path AdminTokenPath resolves to under a given HOME.
func adminTokenFile(home string) string {
	return filepath.Join(home, "Lethean", "lem", "admin.token")
}

// TestRunServeCommand_AdminToken drives the two token-management subcommands that
// return before the serve loop, with HOME redirected so the real token is never
// touched. print generates a fresh token then reads the same one back
// idempotently; rotate replaces it with a different one.
func TestRunServeCommand_AdminToken(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	tokenFile := adminTokenFile(home)

	// First print generates the token (the file did not exist).
	var stdout, stderr bytes.Buffer
	if code := runServeCommand(context.Background(), []string{"--print-admin-token"}, &stdout, &stderr); code != 0 {
		t.Fatalf("print exit %d; stderr=%s", code, stderr.String())
	}
	first := readToken(t, tokenFile)
	if !core.HasPrefix(first, "lthn-mlx_") {
		t.Errorf("token %q missing lthn-mlx_ prefix", first)
	}
	if !core.Contains(stderr.String(), first) {
		t.Errorf("print did not echo the token to stderr; got %q", stderr.String())
	}

	// A second print reads the existing token — same value, no regeneration.
	var out2, err2 bytes.Buffer
	if code := runServeCommand(context.Background(), []string{"--print-admin-token"}, &out2, &err2); code != 0 {
		t.Fatalf("second print exit %d; stderr=%s", code, err2.String())
	}
	if again := readToken(t, tokenFile); again != first {
		t.Errorf("print regenerated the token: %q != %q", again, first)
	}

	// Rotate replaces it with a fresh, different token.
	var out3, err3 bytes.Buffer
	if code := runServeCommand(context.Background(), []string{"--rotate-admin-token"}, &out3, &err3); code != 0 {
		t.Fatalf("rotate exit %d; stderr=%s", code, err3.String())
	}
	rotated := readToken(t, tokenFile)
	if rotated == first {
		t.Errorf("rotate did not change the token (%q)", rotated)
	}
	if !core.HasPrefix(rotated, "lthn-mlx_") {
		t.Errorf("rotated token %q missing lthn-mlx_ prefix", rotated)
	}
}

// TestRunServeCommand_AdminTokenFailClosed proves the token subcommand exits 1
// (fail-closed) when the token file cannot be written — HOME points at a
// regular file, so the ~/Lethean/lem parent directory cannot be created.
func TestRunServeCommand_AdminTokenFailClosed(t *testing.T) {
	blocker := filepath.Join(t.TempDir(), "home-is-a-file")
	if r := core.WriteFile(blocker, []byte("not a directory"), 0o644); !r.OK {
		t.Fatalf("write blocker: %v", r.Err())
	}
	t.Setenv("HOME", blocker)

	var stdout, stderr bytes.Buffer
	if code := runServeCommand(context.Background(), []string{"--print-admin-token"}, &stdout, &stderr); code != 1 {
		t.Fatalf("exit %d, want 1 (fail-closed); stderr=%s", code, stderr.String())
	}
}

// readToken reads the trimmed admin token from path, failing the test on error.
func readToken(t *testing.T, path string) string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read token file %s: %v", path, err)
	}
	return core.Trim(string(data))
}
