// SPDX-License-Identifier: EUPL-1.2

package orchestrator

import (
	"context"

	core "dappco.re/go"
	coreprocess "dappco.re/go/process"
)

func ExampleNewNativeLauncher() {
	app := core.New()
	serviceResult := coreprocess.NewService(coreprocess.Options{})(app)
	if !serviceResult.OK {
		core.Println(false)
		return
	}
	service := serviceResult.Value.(*coreprocess.Service)
	result := NewNativeLauncher(service, []string{"PATH", "HOME"})
	core.Println(result.OK)
	if result.OK {
		closed := result.Value.(Launcher).Close()
		if !closed.OK {
			core.Println(false)
		}
	}
	shutdown := service.OnShutdown(context.Background())
	if !shutdown.OK {
		core.Println(false)
	}
	// Output: true
}
