package mcp

import (
	"context"

	core "dappco.re/go"
)

func ExampleService_ServeUnix() {
	service := core.MustCast[*Service](New(WithWorkspaceRoot("")))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	socketPath := core.PathJoin("/tmp", core.Sprintf("mcp-example-%d.sock", core.Getpid()))

	err := service.ServeUnix(ctx, socketPath)
	core.Println(err.OK)
	// Output:
	// true
}
