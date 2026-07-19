// SPDX-License-Identifier: EUPL-1.2

package workspace

import (
	"context"
	"testing"

	core "dappco.re/go"
	process "dappco.re/go/process"
)

func TestGit_ProcessRunner_Run_Good(t *testing.T) {
	var received process.RunOptions
	runner := ProcessRunner{run: func(_ context.Context, options process.RunOptions) core.Result {
		received = options
		return core.Ok("main\n")
	}}
	command := Command{
		Dir:         "/tmp/project with spaces",
		Executable:  "git",
		Args:        []string{"rev-parse", "--show-toplevel"},
		Environment: []string{"GIT_CONFIG_NOSYSTEM=1"},
	}

	result := runner.Run(context.Background(), command)
	core.AssertTrue(t, result.OK, result.Error())
	core.AssertEqual(t, "main\n", result.Value.(string))
	core.AssertEqual(t, command.Dir, received.Dir)
	core.AssertEqual(t, command.Executable, received.Command)
	core.AssertEqual(t, command.Args, received.Args)
	core.AssertEqual(t, command.Environment, received.Env)

	command.Args[0] = "mutated"
	command.Environment[0] = "MUTATED=1"
	core.AssertEqual(t, "rev-parse", received.Args[0])
	core.AssertEqual(t, "GIT_CONFIG_NOSYSTEM=1", received.Env[0])
}

func TestGit_ProcessRunner_Run_Bad(t *testing.T) {
	runner := ProcessRunner{}
	core.AssertFalse(t, runner.Run(nil, Command{Executable: "git"}).OK)
	core.AssertFalse(t, runner.Run(context.Background(), Command{}).OK)
	core.AssertFalse(t, runner.Run(context.Background(), Command{Executable: "git", Dir: "relative"}).OK)
	core.AssertFalse(t, runner.Run(context.Background(), Command{Executable: "git"}).OK)
}

func TestGit_ProcessRunner_Run_Ugly(t *testing.T) {
	runner := ProcessRunner{run: func(context.Context, process.RunOptions) core.Result {
		return core.Fail(core.NewError("injected process failure"))
	}}
	result := runner.Run(context.Background(), Command{Executable: "git", Args: []string{"status"}})
	core.AssertFalse(t, result.OK)
	core.AssertContains(t, result.Error(), "injected")

	runner.run = func(context.Context, process.RunOptions) core.Result { return core.Ok(17) }
	core.AssertFalse(t, runner.Run(context.Background(), Command{Executable: "git"}).OK)
}

func TestGit_ProcessRunner_RunDetailed_Good(t *testing.T) {
	runner := ProcessRunner{run: func(context.Context, process.RunOptions) core.Result {
		return core.Ok("main\n")
	}}
	result := runner.RunDetailed(context.Background(), Command{Executable: "git", Args: []string{"branch", "--show-current"}})
	core.AssertTrue(t, result.OK, result.Error())
	outcome := result.Value.(CommandOutcome)
	core.AssertEqual(t, "main\n", outcome.Output)
	core.AssertEqual(t, 0, outcome.ExitCode)
	core.AssertTrue(t, outcome.Failure == nil)
}

func TestGit_ProcessRunner_RunDetailed_Bad(t *testing.T) {
	runner := ProcessRunner{}
	core.AssertFalse(t, runner.RunDetailed(nil, Command{Executable: "git"}).OK)
	core.AssertFalse(t, runner.RunDetailed(context.Background(), Command{}).OK)
	core.AssertFalse(t, runner.RunDetailed(context.Background(), Command{Executable: "git", Dir: "relative"}).OK)
}

func TestGit_ProcessRunner_RunDetailed_Ugly(t *testing.T) {
	runner := ProcessRunner{run: func(context.Context, process.RunOptions) core.Result {
		return core.Fail(core.NewError("injected process failure"))
	}}
	result := runner.RunDetailed(context.Background(), Command{Executable: "git", Args: []string{"status"}})
	core.AssertTrue(t, result.OK, result.Error())
	outcome := result.Value.(CommandOutcome)
	core.AssertEqual(t, -1, outcome.ExitCode)
	core.AssertContains(t, outcome.Failure.Error(), "injected process failure")
}
