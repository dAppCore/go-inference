package mcp

import (
	"context"
	"net"

	core "dappco.re/go"
)

// DefaultTCPAddr is the default TCP MCP listen address.
const DefaultTCPAddr = "127.0.0.1:9100"

// ServeTCP serves newline-delimited MCP JSON-RPC over TCP.
func (s *Service) ServeTCP(ctx context.Context, addr string) core.Result {
	addr = normalizeTCPAddr(addr)
	host, port, err := net.SplitHostPort(addr)
	if err == nil && host == "" {
		addr = net.JoinHostPort("127.0.0.1", port)
	}
	if err == nil && host == "0.0.0.0" {
		core.Print(core.Stderr(), "WARNING: MCP TCP server binding to all interfaces (%s). Use 127.0.0.1 for local-only access.\n", addr)
	}

	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return core.Fail(err)
	}
	defer listener.Close()

	go func() {
		<-ctx.Done()
		if err := listener.Close(); err != nil && !core.Is(err, net.ErrClosed) {
			core.Print(core.Stderr(), "MCP TCP listener close error: %v\n", err)
		}
	}()

	for {
		conn, err := listener.Accept()
		if err != nil {
			select {
			case <-ctx.Done():
				return core.Ok(nil)
			default:
				core.Print(core.Stderr(), "MCP TCP accept error: %v\n", err)
				continue
			}
		}
		go s.serveConn(ctx, conn)
	}
}

func normalizeTCPAddr(addr string) string {
	if addr == "" {
		return DefaultTCPAddr
	}
	host, port, err := net.SplitHostPort(addr)
	if err == nil && host == "" {
		return net.JoinHostPort("127.0.0.1", port)
	}
	return addr
}

func (s *Service) serveConn(ctx context.Context, conn net.Conn) {
	defer conn.Close()
	go func() {
		<-ctx.Done()
		if err := conn.Close(); err != nil && !core.Is(err, net.ErrClosed) {
			core.Print(core.Stderr(), "MCP TCP connection close error: %v\n", err)
		}
	}()
	if r := serveReaderWriter(ctx, conn, conn, s.HandleFrame); !r.OK {
		err, _ := resultError(r).(error)
		if core.Is(err, net.ErrClosed) {
			return
		}
		core.Print(core.Stderr(), "MCP TCP connection error: %v\n", err)
	}
}
