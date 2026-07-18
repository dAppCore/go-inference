// SPDX-License-Identifier: EUPL-1.2

package workspace

import (
	"testing"

	core "dappco.re/go"
)

var workspaceBenchmarkResult core.Result

func BenchmarkWorkspacePath(b *testing.B) {
	manager := &Manager{root: "/tmp/lem/workspaces"}
	b.ReportAllocs()
	b.ResetTimer()
	for index := 0; index < b.N; index++ {
		workspaceBenchmarkResult = manager.internalPath("project-1", "runs", "run-9", "worktree")
	}
}

func BenchmarkRunBranch(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for index := 0; index < b.N; index++ {
		workspaceBenchmarkResult = runBranch("work / alpha", 9)
	}
}
