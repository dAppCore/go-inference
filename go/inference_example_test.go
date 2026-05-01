package inference

import (
	core "dappco.re/go"
)

func resetExampleBackends() {
	backendsMu.Lock()
	defer backendsMu.Unlock()
	backends = map[string]Backend{}
}

func ExampleAttentionSnapshot_HasQueries() {
	snap := &AttentionSnapshot{Queries: [][][]float32{{{1, 2, 3}}}}
	core.Println(snap.HasQueries())
	// Output: true
}

func ExampleRegister() {
	resetExampleBackends()
	Register(&stubBackend{name: "example", available: true})

	core.Println(List())
	// Output: [example]
}

func ExampleGet() {
	resetExampleBackends()
	Register(&stubBackend{name: "metal", available: true})

	backend, ok := Get("metal")
	core.Println(ok, backend.Name())
	// Output: true metal
}

func ExampleList() {
	resetExampleBackends()
	Register(&stubBackend{name: "beta", available: true})
	Register(&stubBackend{name: "alpha", available: true})

	core.Println(List())
	// Output: [alpha beta]
}

func ExampleAll() {
	resetExampleBackends()
	Register(&stubBackend{name: "alpha", available: true})
	Register(&stubBackend{name: "beta", available: false})

	for name, backend := range All() {
		core.Println(name, backend.Available())
	}
	// Output:
	// alpha true
	// beta false
}

func ExampleDefault() {
	resetExampleBackends()
	Register(&stubBackend{name: "metal", available: true})

	result := Default()
	backend := result.Value.(Backend)
	core.Println(result.OK, backend.Name())
	// Output: true metal
}

func ExampleLoadModel() {
	resetExampleBackends()
	Register(&stubBackend{name: "metal", available: true})

	result := LoadModel("/models/gemma3")
	model := result.Value.(TextModel)
	core.Println(result.OK, model.ModelType())
	model.Close()
	// Output: true stub
}
