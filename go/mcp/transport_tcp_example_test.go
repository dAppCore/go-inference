package mcp

import (
	"context"

	core "dappco.re/go"
)

func ExampleService_ServeTCP() {
	service := core.MustCast[*Service](New(WithWorkspaceRoot("")))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := service.ServeTCP(ctx, "127.0.0.1:0")
	core.Println(err.OK)
	// Output:
	// true
}
