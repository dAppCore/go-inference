package mcp

import (
	core "dappco.re/go"
)

// --- AX-7 canonical triplets ---

func TestTransportTcp_Service_ServeTCP_Good(t *core.T) {
	service := core.MustCast[*Service](New(WithWorkspaceRoot(t.TempDir())))
	addr := reserveTCPAddr(t)
	ctx, cancel := core.WithCancel(core.Background())

	errCh := make(chan core.Result, 1)
	go func() { errCh <- service.ServeTCP(ctx, addr) }()
	waitForTCP(t, addr)
	cancel()
	core.AssertTrue(t, (<-errCh).OK)
}

func TestTransportTcp_Service_ServeTCP_Bad(t *core.T) {
	service := core.MustCast[*Service](New(WithWorkspaceRoot(t.TempDir())))
	r := service.ServeTCP(core.Background(), "127.0.0.1:bad")

	core.AssertFalse(t, r.OK)
	core.AssertContains(t, r.Error(), "listen")
}

func TestTransportTcp_Service_ServeTCP_Ugly(t *core.T) {
	service := core.MustCast[*Service](New(WithWorkspaceRoot(t.TempDir())))
	r := service.ServeTCP(core.Background(), "256.256.256.256:1")

	core.AssertFalse(t, r.OK)
	core.AssertContains(t, r.Error(), "listen")
}
