package mcp

import core "dappco.re/go"

type Buffer = safeBuffer

func ExampleBuffer_String() {
	var buffer Buffer
	buffer.append([]byte("agent"))

	core.Println(buffer.String())
	// Output:
	// agent
}
