package mcp

import (
	"context"

	core "dappco.re/go"
)

func ExampleService_ServeStdio() {
	service := core.MustCast[*Service](New(WithWorkspaceRoot("")))
	oldReader, oldWriter := stdioReader, stdioWriter
	defer func() { stdioReader, stdioWriter = oldReader, oldWriter }()

	out := core.NewBuffer()
	stdioReader = core.NewReader("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}\n")
	stdioWriter = out
	err := service.ServeStdio(context.Background())

	core.Println(err.OK)
	core.Println(core.Contains(out.String(), `"tools"`))
	// Output:
	// true
	// true
}

func ExampleService_Run() {
	service := core.MustCast[*Service](New(WithWorkspaceRoot("")))
	oldReader, oldWriter := stdioReader, stdioWriter
	oldGetenv := mcpGetenv
	defer func() {
		stdioReader, stdioWriter = oldReader, oldWriter
		mcpGetenv = oldGetenv
	}()
	out := core.NewBuffer()
	mcpGetenv = func(string) string { return "" }
	stdioReader = core.NewReader("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}\n")
	stdioWriter = out

	err := service.Run(context.Background())
	core.Println(err.OK)
	core.Println(core.Contains(out.String(), `"tools"`))
	// Output:
	// true
	// true
}
