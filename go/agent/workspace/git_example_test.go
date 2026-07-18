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
