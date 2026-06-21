package lab

import (
	core "dappco.re/go"
)

func ExampleAddLabCommands() {
	root := core.New()
	r := AddLabCommands(root)
	cmd := root.Command("lab/serve")

	core.Println(r.OK && cmd.OK)
	core.Println(cmd.Value.(*core.Command).Name)
	// Output:
	// true
	// serve
}

func ExampleRunServe() {
	r := RunServe(CommandOptions{Bind: "0.0.0.0:8080"})

	core.Println(!r.OK)
	core.Println(core.Contains(r.Error(), "non-loopback"))
	// Output:
	// true
	// true
}

func ExampleValidateBindAddress() {
	r := ValidateBindAddress("127.0.0.1:8080", false)

	core.Println(r.OK)
	// Output:
	// true
}

func ExampleIsLoopbackBindAddress() {
	core.Println(IsLoopbackBindAddress("localhost:8080"))
	// Output:
	// true
}

func ExampleValidateRemoteAuth() {
	r := ValidateRemoteAuth(true, "")

	core.Println(!r.OK)
	core.Println(core.Contains(r.Error(), "CORE_LAB_API_TOKEN"))
	// Output:
	// true
	// true
}
