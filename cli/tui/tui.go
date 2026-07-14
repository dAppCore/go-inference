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
	if err := fs.Parse(args); err != nil {
		return 2
	}

	a := newApp(*modelPath, *ctxLen, *maxTokens)

	if *check {
		// A real render of the first frame, no terminal required: the picker
		// (or loading) view straight to stdout. Proves the UI constructs and
		// renders without a display — the CI-shaped receipt.
		frame, _ := a.Update(tea.WindowSizeMsg{Width: 80, Height: 24})
		if a.state == statePick {
			// run the discover pass synchronously so the receipt shows the
			// real picker content, not an empty list
			frame, _ = frame.Update(discoverModels())
		}
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
