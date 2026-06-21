package mcp

import (
	"context"
	"io"

	core "dappco.re/go"
)

var (
	stdioReader io.Reader = core.Stdin()
	stdioWriter io.Writer = core.Stdout()
	mcpGetenv             = core.Getenv
)

// ServeStdio serves newline-delimited MCP JSON-RPC over stdin/stdout.
func (s *Service) ServeStdio(ctx context.Context) core.Result {
	return serveReaderWriter(ctx, stdioReader, stdioWriter, s.HandleFrame)
}

// Run starts the transport selected by MCP_UNIX_SOCKET or MCP_ADDR. With no
// environment configured it serves stdio.
func (s *Service) Run(ctx context.Context) core.Result {
	if socketPath := mcpGetenv("MCP_UNIX_SOCKET"); socketPath != "" {
		return s.ServeUnix(ctx, socketPath)
	}
	if addr := mcpGetenv("MCP_ADDR"); addr != "" {
		return s.ServeTCP(ctx, addr)
	}
	return s.ServeStdio(ctx)
}
