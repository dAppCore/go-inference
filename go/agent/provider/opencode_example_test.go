// SPDX-License-Identifier: EUPL-1.2

package provider

import (
	core "dappco.re/go"
)

func ExampleDefaultRegistry_opencode() {
	registry := providerExampleRegistry(nil)
	adapter := registry.Adapter("opencode").Value.(Adapter)
	command := adapter.Build(providerExampleLaunch()).Value.(Command)
	core.Println(command.Executable, command.Args[:6])
	// Output: opencode [run --format json --pure --dir /tmp/lem/workspaces/work-7]
}
