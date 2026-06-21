package mcp

import core "dappco.re/go"

func ExampleReadFileInput() {
	input := ReadFileInput{Path: "main.go"}

	core.Println(input.Path)
	// Output:
	// main.go
}
