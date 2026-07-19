// SPDX-License-Identifier: EUPL-1.2

package workspace

import (
	"context"

	core "dappco.re/go"
	process "dappco.re/go/process"
	commandexec "dappco.re/go/process/exec"
)

// CommandOutcome preserves output and process status even when a command exits unsuccessfully.
type CommandOutcome struct {
	Output   string
	ExitCode int
	Failure  error
}

type detailedRunner interface {
	RunDetailed(context.Context, Command) core.Result
}

// Command is one explicit executable invocation with no shell interpolation.
type Command struct {
	Dir         string
	Executable  string
	Args        []string
	Environment []string
}

// Runner executes explicit command argument vectors for workspace Git operations.
type Runner interface {
	Run(context.Context, Command) core.Result
}

// ProcessRunner executes commands through the shared core/process service.
type ProcessRunner struct {
	run func(context.Context, process.RunOptions) core.Result
}

// RunDetailed retains command output and exit status for callers that must classify failures.
func (runner ProcessRunner) RunDetailed(ctx context.Context, command Command) core.Result {
	if runner.run != nil {
		result := runner.Run(ctx, command)
		if result.OK {
			return core.Ok(CommandOutcome{Output: result.Value.(string), ExitCode: 0})
		}
		return core.Ok(CommandOutcome{ExitCode: -1, Failure: result.Err()})
	}
	if ctx == nil {
		return core.Fail(core.NewError("agent workspace command context is required"))
	}
	command.Executable = core.Trim(command.Executable)
	if command.Executable == "" {
		return core.Fail(core.NewError("agent workspace command executable is required"))
	}
	if command.Dir != "" && !core.PathIsAbs(command.Dir) {
		return core.Fail(core.NewError("agent workspace command directory must be absolute"))
	}
	stdout := core.NewBuffer()
	stderr := core.NewBuffer()
	run := commandexec.Command(ctx, command.Executable, command.Args...).
		WithDir(command.Dir).
		WithEnv(command.Environment).
		WithStdout(stdout).
		WithStderr(stderr).
		Run()
	outcome := CommandOutcome{Output: core.Concat(stdout.String(), stderr.String()), ExitCode: 0}
	if run.OK {
		return core.Ok(outcome)
	}
	outcome.ExitCode = -1
	outcome.Failure = run.Err()
	var exit interface{ ExitCode() int }
	if core.As(run.Err(), &exit) {
		outcome.ExitCode = exit.ExitCode()
	}
	return core.Ok(outcome)
}

// Run validates and executes one explicit command argument vector.
func (runner ProcessRunner) Run(ctx context.Context, command Command) core.Result {
	if ctx == nil {
		return core.Fail(core.NewError("agent workspace command context is required"))
	}
	command.Executable = core.Trim(command.Executable)
	if command.Executable == "" {
		return core.Fail(core.NewError("agent workspace command executable is required"))
	}
	if command.Dir != "" && !core.PathIsAbs(command.Dir) {
		return core.Fail(core.NewError("agent workspace command directory must be absolute"))
	}
	options := process.RunOptions{
		Command: command.Executable,
		Args:    append([]string(nil), command.Args...),
		Dir:     command.Dir,
		Env:     append([]string(nil), command.Environment...),
	}
	run := runner.run
	if run == nil {
		run = process.RunWithOptions
	}
	result := run(ctx, options)
	if !result.OK {
		return core.Fail(core.E("workspace.ProcessRunner.Run", core.Concat("failed to run ", command.Executable), result.Err()))
	}
	output, ok := result.Value.(string)
	if !ok {
		return core.Fail(core.Errorf("agent workspace command returned %T instead of string output", result.Value))
	}
	return core.Ok(output)
}
