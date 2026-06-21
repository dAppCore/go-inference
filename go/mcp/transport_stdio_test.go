package mcp

import (
	core "dappco.re/go"
)

// --- AX-7 canonical triplets ---

func TestTransportStdio_Service_ServeStdio_Good(t *core.T) {
	service := core.MustCast[*Service](New(WithWorkspaceRoot(t.TempDir())))
	oldReader, oldWriter := stdioReader, stdioWriter
	defer func() { stdioReader, stdioWriter = oldReader, oldWriter }()

	output := core.NewBuffer()
	stdioReader = core.NewReader(`{"jsonrpc":"2.0","id":1,"method":"tools/list"}` + "\n")
	stdioWriter = output
	r := service.ServeStdio(core.Background())

	core.AssertTrue(t, r.OK)
	core.AssertContains(t, output.String(), `"tools"`)
}

func TestTransportStdio_Service_ServeStdio_Bad(t *core.T) {
	service := core.MustCast[*Service](New(WithWorkspaceRoot(t.TempDir())))
	oldReader, oldWriter := stdioReader, stdioWriter
	defer func() { stdioReader, stdioWriter = oldReader, oldWriter }()

	output := core.NewBuffer()
	stdioReader = core.NewReader("{bad json\n")
	stdioWriter = output
	r := service.ServeStdio(core.Background())

	core.AssertTrue(t, r.OK)
	core.AssertContains(t, output.String(), "parse error")
}

func TestTransportStdio_Service_ServeStdio_Ugly(t *core.T) {
	service := core.MustCast[*Service](New(WithWorkspaceRoot(t.TempDir())))
	oldReader, oldWriter := stdioReader, stdioWriter
	defer func() { stdioReader, stdioWriter = oldReader, oldWriter }()

	stdioReader = core.NewReader("")
	stdioWriter = core.NewBuffer()
	r := service.ServeStdio(core.Background())

	core.AssertTrue(t, r.OK)
	core.AssertEqual(t, []string{}, []string{})
}

func TestTransportStdio_Service_Run_Good(t *core.T) {
	service := core.MustCast[*Service](New(WithWorkspaceRoot(t.TempDir())))
	oldReader, oldWriter := stdioReader, stdioWriter
	defer func() { stdioReader, stdioWriter = oldReader, oldWriter }()

	output := core.NewBuffer()
	stdioReader = core.NewReader(`{"jsonrpc":"2.0","id":1,"method":"ping"}` + "\n")
	stdioWriter = output
	r := service.Run(core.Background())

	core.AssertTrue(t, r.OK)
	core.AssertContains(t, output.String(), `"result"`)
}

func TestTransportStdio_Service_Run_Bad(t *core.T) {
	oldGetenv := mcpGetenv
	defer func() { mcpGetenv = oldGetenv }()
	mcpGetenv = func(key string) string {
		if key == "MCP_ADDR" {
			return "127.0.0.1:bad"
		}
		return ""
	}
	service := core.MustCast[*Service](New(WithWorkspaceRoot(t.TempDir())))

	r := service.Run(core.Background())
	core.AssertFalse(t, r.OK)
	core.AssertContains(t, r.Error(), "listen")
}

func TestTransportStdio_Service_Run_Ugly(t *core.T) {
	socketPath := core.Path(t.TempDir(), "socket-name-that-is-intentionally-too-long-for-a-unix-domain-socket-path-because-the-kernel-limit-is-small")
	oldGetenv := mcpGetenv
	defer func() { mcpGetenv = oldGetenv }()
	mcpGetenv = func(key string) string {
		if key == "MCP_UNIX_SOCKET" {
			return socketPath
		}
		return ""
	}
	service := core.MustCast[*Service](New(WithWorkspaceRoot(t.TempDir())))

	r := service.Run(core.Background())
	core.AssertFalse(t, r.OK)
	core.AssertContains(t, r.Error(), "invalid")
}
