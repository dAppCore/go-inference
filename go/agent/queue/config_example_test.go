// SPDX-License-Identifier: EUPL-1.2

package queue

import (
	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

func ExampleLoadPolicy() {
	medium := coreio.NewMemoryMedium()
	if err := medium.Write("agents.yaml", "version: 1\ndispatch:\n  default_agent: codex\n"); err != nil {
		core.Println(err)
		return
	}
	result := LoadPolicy(medium, "agents.yaml")
	if !result.OK {
		core.Println(result.Error())
		return
	}
	policy := result.Value.(Policy)
	core.Println(policy.Dispatch.DefaultAgent, policy.Dispatch.TimeoutMinutes)
	// Output: codex 60
}
