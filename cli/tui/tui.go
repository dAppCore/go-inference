// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"context"
	"flag"
	"io"

	tea "dappco.re/go/render/display/tui"

	core "dappco.re/go"
)

// Run is the `lem tui` verb. It opens the durable ~/.lem workspace before
// starting Bubble Tea and closes every workspace resource before returning.
func Run(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	return runWithWorkspace(ctx, args, stdout, stderr, workspaceLoaders{
		Normal: func() core.Result { return loadDefaultWorkspaceContext(ctx) },
		Check:  func() core.Result { return loadDefaultCheckWorkspaceContext(ctx) },
	})
}

type workspaceLoaders struct {
	Normal func() core.Result
	Check  func() core.Result
}

func loadDefaultWorkspace() core.Result {
	return loadDefaultWorkspaceContext(context.Background())
}

func loadDefaultWorkspaceContext(ctx context.Context) core.Result {
	if preflight := nativeWorkspacePreflight(ctx); !preflight.OK {
		return preflight
	}
	pathsResult := defaultAppPaths()
	if !pathsResult.OK {
		return pathsResult
	}
	paths, ok := pathsResult.Value.(appPaths)
	if !ok {
		return core.Fail(core.E("tui.loadDefaultWorkspace", "invalid application paths result", nil))
	}
	return openWorkspaceContext(ctx, paths.Root, workspaceOpeners{Agent: openNativeWorkspaceAgent})
}

func loadDefaultCheckWorkspace() core.Result {
	return loadDefaultCheckWorkspaceContext(context.Background())
}

func loadDefaultCheckWorkspaceContext(ctx context.Context) core.Result {
	pathsResult := defaultAppPaths()
	if !pathsResult.OK {
		return pathsResult
	}
	paths, ok := pathsResult.Value.(appPaths)
	if !ok {
		return core.Fail(core.E("tui.loadDefaultCheckWorkspace", "invalid application paths result", nil))
	}
	return openWorkspaceContext(ctx, paths.Root, workspaceOpeners{Agent: openReadOnlyWorkspaceAgent})
}

func runWithWorkspace(
	ctx context.Context,
	args []string,
	stdout io.Writer,
	stderr io.Writer,
	loaders workspaceLoaders,
) int {
	fs := flag.NewFlagSet("tui", flag.ContinueOnError)
	fs.SetOutput(stderr)
	modelPath := fs.String("model", "", "model snapshot directory; empty opens the picker")
	ctxLen := fs.Int("context", 0, "context length override; 0 uses the model default")
	maxTokens := fs.Int("max-tokens", 4096, "per-reply token budget (thinking spends from it)")
	check := fs.Bool("check", false, "open ~/.lem, render one 100x30 frame, close it, and exit")
	fs.Usage = func() {
		core.WriteString(stderr, "Usage: tui [flags]\n")
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "A persistent terminal workspace with Chat, Work, Models, and Service panels.\n")
		core.WriteString(stderr, "Settings, tools, modes, knowledge, and runtime detail live in the inspector.\n")
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Flags:\n")
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

	loader := loaders.Normal
	if *check {
		loader = loaders.Check
	}
	a := newWorkspaceApp(*modelPath, *ctxLen, *maxTokens, loader)
	if *check {
		return runWorkspaceCheck(a, stdout, stderr)
	}

	// Cell-motion mouse reporting feeds the panel bar's teabox hit-testing
	// (click a tab to switch panels) and the transcript's wheel scrolling.
	program := tea.NewProgram(a, tea.WithContext(ctx))
	finalModel, runErr := program.Run()
	shutdownResult := shutdownProgramModel(a, finalModel)
	if runErr != nil {
		core.WriteString(stderr, "tui: "+runErr.Error()+"\n")
		return 1
	}
	if !shutdownResult.OK {
		core.WriteString(stderr, "tui: "+shutdownResult.Error()+"\n")
		return 1
	}
	return 0
}

// newDataReviewApp builds the workspace app RunDataReview drives — focused
// on the Data panel and, when slug is non-empty, pre-filtered to that one
// dataset (attachData consumes dataInitialSlug once the workspace
// connects and the dataset store opens). Split out from RunDataReview so
// the state setup itself is unit-testable via runWorkspaceCheck without a
// live terminal — RunDataReview's own tea.NewProgram().Run() call is,
// like Run's own interactive branch, exercised by hand rather than by an
// automated test: every existing tui_test.go test drives only the
// --check headless path (see TestRunWithWorkspace_Good and neighbours),
// which is the established, precedented boundary this mirrors.
func newDataReviewApp(ctx context.Context, slug string) app {
	a := newWorkspaceApp("", 0, 4096, func() core.Result { return loadDefaultWorkspaceContext(ctx) })
	a.activePanel = panelData
	a.dataInitialSlug = core.Trim(slug)
	return a
}

// RunDataReview is the `lem data review [slug]` entry point (tracker #43,
// plan Task 8) — opens the same persistent workspace program as Run, but
// focused on the Data panel. Falls back to Run's own honest error path
// (the "tui: <err>" stderr line, same as any other program-start
// failure) when Bubble Tea cannot start; cli/data.go's caller adds its
// own headless fallback pointer on a non-zero exit (see runDataReview),
// keeping that message rather than promising a broken TUI.
func RunDataReview(ctx context.Context, slug string, stdout, stderr io.Writer) int {
	a := newDataReviewApp(ctx, slug)
	program := tea.NewProgram(a, tea.WithContext(ctx))
	finalModel, runErr := program.Run()
	shutdownResult := shutdownProgramModel(a, finalModel)
	if runErr != nil {
		core.WriteString(stderr, "tui: "+runErr.Error()+"\n")
		return 1
	}
	if !shutdownResult.OK {
		core.WriteString(stderr, "tui: "+shutdownResult.Error()+"\n")
		return 1
	}
	return 0
}

func shutdownProgramModel(initial app, final tea.Model) core.Result {
	if finalApp, ok := final.(app); ok {
		return finalApp.shutdown()
	}
	return initial.shutdown()
}

func runWorkspaceCheck(a app, stdout, stderr io.Writer) int {
	model, _ := a.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
	a = model.(app)
	model, _ = a.Update(workspaceBootstrap(a.workspaceLoader)())
	a = model.(app)
	if a.boot.phase == bootReady {
		model, _ = a.Update(discoverModels())
		a = model.(app)
	}
	core.WriteString(stdout, a.View().Content+"\n")
	if a.boot.phase == bootFailed {
		_ = a.shutdown()
		return 1
	}
	if result := a.shutdown(); !result.OK {
		core.WriteString(stderr, "tui: "+result.Error()+"\n")
		return 1
	}
	return 0
}
