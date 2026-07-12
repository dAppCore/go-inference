// SPDX-Licence-Identifier: EUPL-1.2

// main() itself is not driven here: it reads core.Args() and ends in core.Exit,
// which would terminate the test binary. Its logic beyond that is the thin
// commandName assignment; the routable surface is runCommand, tested directly.
package main

import (
	"bytes"
	"context"
	"path/filepath"
	"testing"

	core "dappco.re/go"
)

// TestRunCommand_Dispatch drives every switch arm of the command router through
// the cheapest side-effect-free path per verb: no-args/help print usage and
// exit 0, an unknown command exits 2 with the usage banner, and each real verb
// is reached via a guard (empty args -> its own exit 2, or -h/-o -> 0) so the
// route is exercised without loading a model.
func TestRunCommand_Dispatch(t *testing.T) {
	specOut := filepath.Join(t.TempDir(), "openapi.json")
	for _, tc := range []struct {
		name     string
		args     []string
		wantCode int
		wantHelp bool
	}{
		{"no args", nil, 0, true},
		{"help word", []string{"help"}, 0, true},
		{"help flag", []string{"-h"}, 0, true},
		{"unknown command", []string{"frobnicate"}, 2, true},
		{"serve route", []string{"serve", "-h"}, 0, false},
		{"generate route", []string{"generate"}, 2, false},
		{"ssd route", []string{"ssd"}, 2, false},
		{"sft route", []string{"sft"}, 2, false},
		{"tune route", []string{"tune"}, 2, false},
		{"pack route", []string{"pack"}, 2, false},
		{"quant route", []string{"quant"}, 2, false},
		{"ebook route", []string{"ebook"}, 2, false},
		{"spec route", []string{"spec", "-o", specOut}, 0, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			code := runCommand(context.Background(), tc.args, &stdout, &stderr)
			if code != tc.wantCode {
				t.Fatalf("exit %d, want %d; stderr=%s", code, tc.wantCode, stderr.String())
			}
			if tc.wantHelp && !core.Contains(stdout.String()+stderr.String(), "Usage:") {
				t.Errorf("usage banner missing; stdout=%q stderr=%q", stdout.String(), stderr.String())
			}
		})
	}
}

// TestPrintUsage proves the top-level usage lists every verb group, so a verb
// dropping out of the banner is caught.
func TestPrintUsage(t *testing.T) {
	var buf bytes.Buffer
	printUsage(&buf)
	out := buf.String()
	for _, verb := range []string{"serve", "generate", "ssd", "sft", "tune", "pack", "ebook", "quant", "spec"} {
		if !core.Contains(out, verb) {
			t.Errorf("usage missing verb %q", verb)
		}
	}
}

// TestCliName proves the invoked-name resolver falls back to "lem" for an empty
// or whitespace commandName and otherwise returns the trimmed name — so a
// renamed copy of the binary prints its own name.
func TestCliName(t *testing.T) {
	orig := commandName
	defer func() { commandName = orig }()
	for _, tc := range []struct {
		name, set, want string
	}{
		{"custom", "mylem", "mylem"},
		{"empty falls back", "", "lem"},
		{"whitespace falls back", "   ", "lem"},
		{"trimmed", "  lem  ", "lem"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			commandName = tc.set
			if got := cliName(); got != tc.want {
				t.Errorf("cliName() with commandName=%q = %q, want %q", tc.set, got, tc.want)
			}
		})
	}
}

// TestCliCommandName proves the usage-prefix builder returns the bare name for an
// empty command and "<name> <command>" otherwise.
func TestCliCommandName(t *testing.T) {
	orig := commandName
	defer func() { commandName = orig }()
	commandName = "lem"
	if got := cliCommandName(""); got != "lem" {
		t.Errorf("cliCommandName(\"\") = %q, want lem", got)
	}
	if got := cliCommandName("serve"); got != "lem serve" {
		t.Errorf("cliCommandName(\"serve\") = %q, want \"lem serve\"", got)
	}
}
