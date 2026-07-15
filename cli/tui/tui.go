// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"context"
	"flag"
	"io"

	tea "github.com/charmbracelet/bubbletea"

	core "dappco.re/go"
)

// Run is the `lem tui` verb: pick a model (or take -model), then chat with
// streaming tokens, the thinking channel rendered live, esc-to-cancel and
// per-turn decode tok/s. Returns a process exit code.
func Run(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("tui", flag.ContinueOnError)
	fs.SetOutput(stderr)
	modelPath := fs.String("model", "", "model snapshot directory; empty opens the picker")
	ctxLen := fs.Int("context", 0, "context length override; 0 uses the model default")
	maxTokens := fs.Int("max-tokens", 4096, "per-reply token budget (thinking spends from it)")
	check := fs.Bool("check", false, "render one frame to stdout and exit (no TTY needed — the smoke receipt)")
	fs.Usage = func() {
		core.WriteString(stderr, "Usage: tui [flags]\n")
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Chat with a model in the terminal: tabs for Chat, Models, Service (host the\n")
		core.WriteString(stderr, "HTTP API on the loaded model), Settings, Tools and Modes.\n")
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Flags:\n")
		// the same GNU-style block the lem verbs print (long options only)
		fs.VisitAll(func(f *flag.Flag) {
			if f.DefValue == "" {
				core.WriteString(stderr, core.Sprintf("  --%s\n\t%s\n", f.Name, f.Usage))
				return
			}
			core.WriteString(stderr, core.Sprintf("  --%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
		})
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}

	a := newApp(*modelPath, *ctxLen, *maxTokens)

	if *check {
		// A real render of the first frame, no terminal required: the picker
		// (or loading) view straight to stdout. Proves the UI constructs and
		// renders without a display — the CI-shaped receipt.
		frame, _ := a.Update(tea.WindowSizeMsg{Width: 80, Height: 24})
		// run the discover pass synchronously so the receipt shows the real
		// Models tab content, not an empty list
		frame, _ = frame.Update(discoverModels())
		core.WriteString(stdout, frame.View()+"\n")
		return 0
	}

	p := tea.NewProgram(a, tea.WithAltScreen(), tea.WithContext(ctx))
	if _, err := p.Run(); err != nil {
		core.WriteString(stderr, "tui: "+err.Error()+"\n")
		return 1
	}
	return 0
}
