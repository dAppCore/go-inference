package mcp

import (
	core "dappco.re/go"
)

// --- AX-7 canonical triplets ---

func TestJsonrpc_Service_HandleFrame_Good(t *core.T) {
	service := core.MustCast[*Service](New(WithWorkspaceRoot(t.TempDir())))
	responseResult := service.HandleFrame(core.Background(), []byte(`{"jsonrpc":"2.0","id":1,"method":"ping"}`))
	response := responseResult.Value.([]byte)

	core.AssertTrue(t, responseResult.OK)
	core.AssertContains(t, string(response), `"result"`)
}

func TestJsonrpc_Service_HandleFrame_Bad(t *core.T) {
	service := core.MustCast[*Service](New(WithWorkspaceRoot(t.TempDir())))
	responseResult := service.HandleFrame(core.Background(), []byte(`{bad json`))
	response := responseResult.Value.([]byte)

	core.AssertTrue(t, responseResult.OK)
	core.AssertContains(t, string(response), "parse error")
}

func TestJsonrpc_Service_HandleFrame_Ugly(t *core.T) {
	service := core.MustCast[*Service](New(WithWorkspaceRoot(t.TempDir())))
	responseResult := service.HandleFrame(core.Background(), []byte(`{"jsonrpc":"2.0","method":"notifications/initialized"}`))
	response := responseResult.Value.([]byte)

	core.AssertTrue(t, responseResult.OK)
	core.AssertNil(t, response)
}
