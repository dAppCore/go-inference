// SPDX-License-Identifier: EUPL-1.2

package provider

func buildCodexArgs(configuration Config, launch Launch, prompt string) []string {
	args := []string{
		"--ask-for-approval", "never",
		"--sandbox", "workspace-write",
		"--cd", launch.Worktree,
		"exec", "--json", "--color", "never",
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

func parseCodexLine(stream, line string) []Output {
	return parseProviderLine("codex", stream, line)
}
