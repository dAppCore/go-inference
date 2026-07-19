// SPDX-License-Identifier: EUPL-1.2

package orchestrator

import (
	core "dappco.re/go"
	"dappco.re/go/inference/agent/work"
)

func ExampleCommit() {
	run := work.Run{ID: "run-1", Status: work.RunQueued}
	commit := Commit{Run: &run, CreateRun: true}
	core.Println(commit.CreateRun, commit.Run.Status)
	// Output: true queued
}
