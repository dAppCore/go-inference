// SPDX-License-Identifier: EUPL-1.2

package workspace

import (
	"context"

	core "dappco.re/go"
	process "dappco.re/go/process"
)

func ExampleProcessRunner_Run() {
	runner := ProcessRunner{run: func(_ context.Context, options process.RunOptions) core.Result {
		return core.Ok(core.Join(" ", append([]string{options.Command}, options.Args...)...))
	}}
	result := runner.Run(context.Background(), Command{Executable: "git", Args: []string{"status", "--short"}})
	core.Println(result.Value)
	// Output: git status --short
}

func ExampleProcessRunner_RunDetailed() {
	runner := ProcessRunner{run: func(context.Context, process.RunOptions) core.Result {
		return core.Fail(core.NewError("command exited"))
	}}
	result := runner.RunDetailed(context.Background(), Command{Executable: "git", Args: []string{"rev-parse", "--show-toplevel"}})
	outcome := result.Value.(CommandOutcome)
	core.Println(outcome.ExitCode, outcome.Failure != nil)
	// Output: -1 true
}

func ExampleCommandOutcome() {
	outcome := CommandOutcome{Output: "fatal: not a git repository", ExitCode: 128}
	core.Println(outcome.ExitCode, outcome.Output)
	// Output: 128 fatal: not a git repository
}
