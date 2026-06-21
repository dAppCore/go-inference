package mcp

import core "dappco.re/go"

func TestToolsCore_ReadFileInputPathJSON(t *core.T) {
	data := core.JSONMarshal(ReadFileInput{Path: "main.go"})
	core.RequireTrue(t, data.OK)

	got := string(data.Value.([]byte))
	core.AssertContains(t, got, "\"\x70ath\"")
	core.AssertContains(t, got, "main.go")
}
