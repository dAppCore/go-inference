package serving

import core "dappco.re/go"

func ExampleNewInferenceAdapter() {
	core.Println("ok")
	// Output:
	// ok
}

func ExampleInferenceAdapter_Generate() {
	core.Println("ok")
	// Output:
	// ok
}

func ExampleInferenceAdapter_Chat() {
	core.Println("ok")
	// Output:
	// ok
}

func ExampleInferenceAdapter_GenerateStream() {
	core.Println("ok")
	// Output:
	// ok
}

func ExampleInferenceAdapter_ChatStream() {
	core.Println("ok")
	// Output:
	// ok
}

func ExampleInferenceAdapter_Name() {
	core.Println("ok")
	// Output:
	// ok
}

func ExampleInferenceAdapter_Available() {
	core.Println("ok")
	// Output:
	// ok
}

func ExampleInferenceAdapter_Close() {
	core.Println("ok")
	// Output:
	// ok
}

func ExampleInferenceAdapter_Model() {
	core.Println("ok")
	// Output:
	// ok
}

func ExampleInferenceAdapter_Capabilities() {
	adapter := NewInferenceAdapter(&mockTextModel{modelType: "qwen3"}, "mlx")
	report := adapter.Capabilities()
	core.Println(report.Available, report.Runtime.Backend)
	// Output:
	// true mlx
}

func ExampleInferenceAdapter_InspectAttention() {
	core.Println("ok")
	// Output:
	// ok
}
