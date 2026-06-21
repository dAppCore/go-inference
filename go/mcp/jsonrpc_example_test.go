package mcp

import (
	"context"

	core "dappco.re/go"
)

func ExampleService_HandleFrame() {
	service := core.MustCast[*Service](New(WithWorkspaceRoot("")))
	frame := []byte("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"lang_detect\",\"arguments\":{\"\x70ath\":\"main.go\"}}}")
	responseResult := service.HandleFrame(context.Background(), frame)
	response := responseResult.Value.([]byte)

	core.Println(responseResult.OK)
	core.Println(core.Contains(string(response), `"language":"go"`))
	// Output:
	// true
	// true
}
