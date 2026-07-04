package hip

import core "dappco.re/go"

func ExampleModelInfo() {
	info := ModelInfo{Name: "demo", Architecture: "llama"}
	core.Println(info.Name, info.Architecture)
	// Output: demo llama
}

func ExampleVRAMInfo() {
	info := VRAMInfo{Total: 8, Used: 3, Free: 5}
	core.Println(info.Total, info.Free)
	// Output: 8 5
}
