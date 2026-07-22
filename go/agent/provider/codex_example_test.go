// SPDX-License-Identifier: EUPL-1.2

package provider

import (
	core "dappco.re/go"
)

func ExampleDefaultRegistry_codex() {
	registry := providerExampleRegistry(nil)
	adapter := registry.Adapter("codex").Value.(Adapter)
	command := adapter.Build(providerExampleLaunch()).Value.(Command)
	core.Println(command.Executable, command.Args[:6])
	// Output: codex [--ask-for-approval never --sandbox workspace-write --cd /tmp/lem/workspaces/work-7]
}
