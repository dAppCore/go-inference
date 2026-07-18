// SPDX-License-Identifier: EUPL-1.2

package provider

func buildOpenCodeArgs(configuration Config, launch Launch, prompt string) []string {
	args := []string{
		"run",
		"--format", "json",
		"--pure",
		"--dir", launch.Worktree,
	}
	model := launch.Model
	if model == "" {
		model = configuration.DefaultModel
	}
	if model != "" {
		args = append(args, "--model", model)
	}
	args = append(args, configuration.Flags...)
	args = append(args, launch.UnsafeFlags...)
	return append(args, prompt)
}

func parseOpenCodeLine(stream, line string) []Output {
	return parseProviderLine("opencode", stream, line)
}
