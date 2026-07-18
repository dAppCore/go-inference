// SPDX-License-Identifier: EUPL-1.2

package workspace

import (
	"context"

	core "dappco.re/go"
	process "dappco.re/go/process"
)

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
