package mcp

import (
	"context"
	"net"

	core "dappco.re/go"
)

// DefaultUnixSocket is used when ServeUnix is called with an empty path.
const DefaultUnixSocket = "/tmp/core-mcp.sock"

// ServeUnix serves newline-delimited MCP JSON-RPC over a Unix domain socket.
func (s *Service) ServeUnix(ctx context.Context, socketPath string) core.Result {
	if socketPath == "" {
		socketPath = DefaultUnixSocket
	}
	if r := core.MkdirAll(osPathDir(socketPath), 0o755); !r.OK {
		return r
	}
	if r := core.Remove(socketPath); !r.OK {
		err, _ := r.Value.(error)
		if !core.IsNotExist(err) {
			return r
		}
	}

	listener, err := net.Listen("unix", socketPath)
	if err != nil {
		return core.Fail(err)
	}
	defer func() {
		if err := listener.Close(); err != nil && !core.Is(err, net.ErrClosed) {
			core.Print(core.Stderr(), "MCP Unix listener close error: %v\n", err)
		}
		if r := core.Remove(socketPath); !r.OK {
			err, _ := r.Value.(error)
			if !core.IsNotExist(err) {
				core.Print(core.Stderr(), "MCP Unix socket cleanup error: %s\n", r.Error())
			}
		}
	}()

	go func() {
		<-ctx.Done()
		if err := listener.Close(); err != nil && !core.Is(err, net.ErrClosed) {
			core.Print(core.Stderr(), "MCP Unix listener close error: %v\n", err)
		}
	}()

	for {
		conn, err := listener.Accept()
		if err != nil {
			select {
			case <-ctx.Done():
				return core.Ok(nil)
			default:
				core.Print(core.Stderr(), "MCP Unix accept error: %v\n", err)
				continue
			}
		}
		go s.serveConn(ctx, conn)
	}
}
