// SPDX-License-Identifier: EUPL-1.2

package provider

import (
	core "dappco.re/go"
)

func ExampleDefaultRegistry_claude() {
	registry := providerExampleRegistry(nil)
	adapter := registry.Adapter("claude").Value.(Adapter)
	command := adapter.Build(providerExampleLaunch()).Value.(Command)
	core.Println(command.Executable, command.Args[:5])
	// Output: claude [--print --output-format stream-json --permission-mode acceptEdits]
}
