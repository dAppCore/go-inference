package mcp

import (
	"context"

	core "dappco.re/go"
)

type exampleSubsystem struct{}

func (exampleSubsystem) Name() string { return "example" }

func (exampleSubsystem) RegisterTools(s *Service) {
	s.RegisterToolFunc("example", "example_echo", "Echo example", func(context.Context, RawMessage) core.Result {
		return core.Ok(map[string]string{"ok": "true"})
	})
}

func ExampleNew() {
	result := New(Options{Unrestricted: true})
	service := result.Value.(*Service)

	core.Println(result.OK)
	core.Println(len(service.Tools()) > 0)
	// Output:
	// true
	// true
}

func ExampleWithWorkspaceRoot() {
	result := New(WithWorkspaceRoot(""))
	service := result.Value.(*Service)

	core.Println(result.OK)
	core.Println(service.WorkspaceRoot() == "")
	// Output:
	// true
	// true
}

func ExampleWithProcessService() {
	marker := struct{ Name string }{Name: "process"}
	result := New(WithProcessService(marker))
	service := result.Value.(*Service)

	core.Println(result.OK)
	core.Println(service.processService == marker)
	// Output:
	// true
	// true
}

func ExampleWithWSHub() {
	marker := struct{ Name string }{Name: "hub"}
	result := New(WithWSHub(marker))
	service := result.Value.(*Service)

	core.Println(result.OK)
	core.Println(service.wsHub == marker)
	// Output:
	// true
	// true
}

func ExampleWithInferenceModel() {
	result := New(WithInferenceModel(&generateModel{}, "openai", "gpt-test"))
	service := result.Value.(*Service)

	core.Println(result.OK)
	core.Println(service.mlBackend)
	core.Println(service.mlModelName)
	// Output:
	// true
	// openai
	// gpt-test
}

func ExampleWithSubsystem() {
	result := New(WithSubsystem(exampleSubsystem{}))
	service := result.Value.(*Service)

	core.Println(result.OK)
	core.Println(core.Contains(core.Join(",", service.ToolNames()...), "example_echo"))
	// Output:
	// true
	// true
}

func ExampleService_WorkspaceRoot() {
	service := core.MustCast[*Service](New(WithWorkspaceRoot("")))

	core.Println(service.WorkspaceRoot() == "")
	// Output:
	// true
}

func ExampleService_Tools() {
	service := core.MustCast[*Service](New(WithWorkspaceRoot("")))

	core.Println(len(service.Tools()) > 0)
	// Output:
	// true
}

func ExampleService_ToolNames() {
	service := core.MustCast[*Service](New(WithWorkspaceRoot("")))

	core.Println(core.Contains(core.Join(",", service.ToolNames()...), "file_read"))
	// Output:
	// true
}

func ExampleService_RegisterTool() {
	service := core.MustCast[*Service](New(WithWorkspaceRoot("")))
	err := service.RegisterTool(Tool{Name: "example_tool", Handler: func(context.Context, RawMessage) core.Result {
		return core.Ok(map[string]bool{"ok": true})
	}})

	core.Println(err.OK)
	core.Println(core.Contains(core.Join(",", service.ToolNames()...), "example_tool"))
	// Output:
	// true
	// true
}

func ExampleService_RegisterToolFunc() {
	service := core.MustCast[*Service](New(WithWorkspaceRoot("")))
	err := service.RegisterToolFunc("example", "example_func", "Example func", func(context.Context, RawMessage) core.Result {
		return core.Ok(map[string]bool{"ok": true})
	})

	core.Println(err.OK)
	core.Println(service.tools["example_func"].Group)
	// Output:
	// true
	// example
}

func ExampleService_Shutdown() {
	service := core.MustCast[*Service](New(WithWorkspaceRoot("")))
	err := service.Shutdown(context.Background())

	core.Println(err.OK)
	// Output:
	// true
}
