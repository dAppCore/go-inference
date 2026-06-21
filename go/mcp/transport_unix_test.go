package mcp

import (
	core "dappco.re/go"
)

// --- AX-7 canonical triplets ---

func TestTransportUnix_Service_ServeUnix_Good(t *core.T) {
	service := core.MustCast[*Service](New(WithWorkspaceRoot(t.TempDir())))
	socketPath := core.PathJoin("/tmp", core.Sprintf("mcp-%d-good.sock", core.Getpid()))
	ctx, cancel := core.WithCancel(core.Background())

	errCh := make(chan core.Result, 1)
	go func() { errCh <- service.ServeUnix(ctx, socketPath) }()
	waitForUnix(t, socketPath)
	cancel()
	core.AssertTrue(t, (<-errCh).OK)
}

func TestTransportUnix_Service_ServeUnix_Bad(t *core.T) {
	service := core.MustCast[*Service](New(WithWorkspaceRoot(t.TempDir())))
	r := service.ServeUnix(core.Background(), "\x00")

	core.AssertFalse(t, r.OK)
	core.AssertContains(t, r.Error(), "invalid")
}

func TestTransportUnix_Service_ServeUnix_Ugly(t *core.T) {
	service := core.MustCast[*Service](New(WithWorkspaceRoot(t.TempDir())))
	socketPath := core.PathJoin("/tmp", core.Sprintf("mcp-%d-ugly.sock", core.Getpid()))
	core.AssertTrue(t, core.WriteFile(socketPath, []byte("stale socket"), 0o600).OK)
	ctx, cancel := core.WithCancel(core.Background())

	errCh := make(chan core.Result, 1)
	go func() { errCh <- service.ServeUnix(ctx, socketPath) }()
	waitForUnix(t, socketPath)
	cancel()
	core.AssertTrue(t, (<-errCh).OK)
}
