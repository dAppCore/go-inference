// SPDX-Licence-Identifier: EUPL-1.2

// main() itself is not driven here: it reads core.Args() and ends in core.Exit,
// which would terminate the test binary. Its logic beyond that is the thin
// commandName assignment; the routable surface is runCommand, tested directly.
package main

import (
	"bytes"
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
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
		{"help flag", []string{"--help"}, 0, true},
		{"unknown command", []string{"frobnicate"}, 2, true},
		{"serve route", []string{"serve", "--help"}, 0, false},
		{"generate route", []string{"generate"}, 2, false},
		{"ssd route", []string{"ssd"}, 2, false},
		{"sft route", []string{"sft"}, 2, false},
		{"tune route", []string{"tune"}, 2, false},
		{"pack route", []string{"pack"}, 2, false},
		{"quant route", []string{"quant"}, 2, false},
		{"data route", []string{"data"}, 2, false},
		{"ebook route", []string{"ebook"}, 2, false},
		{"spec route", []string{"spec", "--output", specOut}, 0, false},
		{"update route", []string{"update", "--help"}, 0, false},
		{"version route", []string{"version"}, 0, false},
		{"version flag", []string{"--version"}, 0, false},
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

// TestHelpPresentsLongFlagsOnly locks the presentation contract: every verb's
// help renders GNU-style --long options only — no single-dash flag may appear
// in a flag listing or in prose. (Go's flag parser still accepts one dash for
// the same names; what this pins is what we PRESENT.)
func TestHelpPresentsLongFlagsOnly(t *testing.T) {
	singleDash := regexp.MustCompile(`(^|[\s("` + "`" + `])-[a-z][a-z0-9-]*`)
	for _, argv := range [][]string{
		{"--help"},
		{"serve", "--help"},
		{"generate", "--help"},
		{"ssd", "--help"},
		{"sft", "--help"},
		{"tune", "--help"},
		{"pack", "--help"},
		{"pack", "create", "--help"},
		{"pack", "inspect", "--help"},
		{"pack", "list", "--help"},
		{"pack", "extract", "--help"},
		{"quant", "--help"},
		{"data", "--help"},
		{"data", "create", "--help"},
		{"data", "list", "--help"},
		{"data", "stats", "--help"},
		{"data", "import", "--help"},
		{"data", "score", "--help"},
		{"data", "export", "--help"},
		{"data", "archive", "--help"},
		{"data", "review", "--help"},
		{"spec", "--help"},
		{"ebook", "--help"},
		{"update", "--help"},
		{"tui", "--help"},
	} {
		t.Run(strings.Join(argv, "_"), func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			if code := runCommand(context.Background(), argv, &stdout, &stderr); code != 0 {
				t.Fatalf("exit %d; stderr=%s", code, stderr.String())
			}
			help := stdout.String() + stderr.String()
			if help == "" {
				t.Fatal("no help output")
			}
			if m := singleDash.FindString(help); m != "" {
				t.Fatalf("single-dash flag %q leaked into help:\n%s", strings.TrimSpace(m), help)
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
	for _, verb := range []string{"serve", "generate", "ssd", "sft", "tune", "pack", "ebook", "quant", "data", "spec", "update"} {
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

// TestMain_main_Helper isolates main because it terminates through core.Exit.
func TestMain_main_Helper(t *testing.T) {
	if os.Getenv("LEM_MAIN_HELPER") != "1" {
		return
	}
	args := os.Getenv("LEM_MAIN_ARGS")
	if args == "" {
		os.Args = []string{"lem"}
	} else {
		os.Args = append([]string{"lem"}, strings.Split(args, "\x1f")...)
	}
	main()
}

func runMainProcess(t *testing.T, args ...string) (int, string) {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=^TestMain_main_Helper$")
	cmd.Env = append(os.Environ(), "LEM_MAIN_HELPER=1", "LEM_MAIN_ARGS="+strings.Join(args, "\x1f"))
	out, err := cmd.CombinedOutput()
	if err == nil {
		return 0, string(out)
	}
	exit, ok := err.(*exec.ExitError)
	if !ok {
		t.Fatalf("run main helper: %v; output=%s", err, out)
	}
	return exit.ExitCode(), string(out)
}

// TestMain_main_Good_NoArguments checks the actual binary prints its command
// usage and exits successfully when no command was selected.
func TestMain_main_Good_NoArguments(t *testing.T) {
	code, output := runMainProcess(t)
	if code != 0 {
		t.Fatalf("main exit %d, want 0; output=%s", code, output)
	}
	if !core.Contains(output, "Usage: lem") {
		t.Errorf("main usage missing from output %q", output)
	}
}

// TestMain_main_Bad_UnknownCommand checks the actual binary returns the CLI
// error exit status and names the rejected command.
func TestMain_main_Bad_UnknownCommand(t *testing.T) {
	code, output := runMainProcess(t, "unknown")
	if code != 2 {
		t.Fatalf("main exit %d, want 2; output=%s", code, output)
	}
	if !core.Contains(output, `unknown command "unknown"`) {
		t.Errorf("unknown-command diagnostic missing from %q", output)
	}
}
