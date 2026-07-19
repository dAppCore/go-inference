// SPDX-License-Identifier: EUPL-1.2

package provider

func buildClaudeArgs(configuration Config, launch Launch, prompt string) []string {
	args := []string{
		"--print",
		"--output-format", "stream-json",
		"--permission-mode", "acceptEdits",
		"--no-session-persistence",
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

func parseClaudeLine(stream, line string) []Output {
	return parseProviderLine("claude", stream, line)
}
