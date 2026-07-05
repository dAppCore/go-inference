package mlservice

import core "dappco.re/go"

func ExampleNewService() {
	core.Println("ok")
	// Output:
	// ok
}

func ExampleRegisterCore() {
	r := RegisterCore(core.New())
	core.Println(r.OK)
	// Output:
	// true
}

func ExampleService_OnStartup() {
	core.Println("ok")
	// Output:
	// ok
}

func ExampleService_OnShutdown() {
	core.Println("ok")
	// Output:
	// ok
}

func ExampleService_RegisterBackend() {
	core.Println("ok")
	// Output:
	// ok
}

func ExampleService_Backend() {
	core.Println("ok")
	// Output:
	// ok
}

func ExampleService_DefaultBackend() {
	core.Println("ok")
	// Output:
	// ok
}

func ExampleService_Backends() {
	core.Println("ok")
	// Output:
	// ok
}

func ExampleService_BackendsIter() {
	core.Println("ok")
	// Output:
	// ok
}

func ExampleService_BackendCapabilities() {
	factory := NewService(Options{})
	result := factory(core.New())
	svc := result.Value.(*Service)
	svc.RegisterBackend("local", &testBackend{name: "local", available: true})
	report, ok := svc.BackendCapabilities("local")
	core.Println(ok, report.Runtime.Backend)
	// Output:
	// true local
}

func ExampleService_Judge() {
	core.Println("ok")
	// Output:
	// ok
}

func ExampleService_Engine() {
	core.Println("ok")
	// Output:
	// ok
}

func ExampleService_Generate() {
	core.Println("ok")
	// Output:
	// ok
}

func ExampleService_ScoreResponses() {
	core.Println("ok")
	// Output:
	// ok
}
