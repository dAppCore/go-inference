package mcp

import (
	"bufio"
	"context"
	"net"
	"testing"
	"time"

	core "dappco.re/go"
)

func mustNewService(t *testing.T, args ...any) *Service {
	t.Helper()
	result := New(args...)
	if !result.OK {
		t.Fatalf("New: %s", result.Error())
	}
	return result.Value.(*Service)
}

func TestService_RegisterTool_Good(t *testing.T) {
	s := &Service{tools: map[string]Tool{}}

	r := s.RegisterTool(Tool{
		Name:        "custom_tool",
		Description: "Custom tool",
		Handler: func(ctx context.Context, raw RawMessage) core.Result {
			return core.Ok(map[string]bool{"ok": true})
		},
	})
	if !r.OK {
		t.Fatalf("RegisterTool failed: %s", r.Error())
	}
	if got := s.ToolNames(); len(got) != 1 || got[0] != "custom_tool" {
		t.Fatalf("ToolNames() = %v, want [custom_tool]", got)
	}
}

func TestService_RegisterTool_Bad(t *testing.T) {
	s := &Service{tools: map[string]Tool{}}
	if r := s.RegisterTool(Tool{Name: "", Handler: func(context.Context, RawMessage) core.Result { return core.Ok(nil) }}); r.OK {
		t.Fatal("expected missing name to fail")
	}
	if r := s.RegisterTool(Tool{Name: "missing_handler"}); r.OK {
		t.Fatal("expected missing handler to fail")
	}
	if r := s.RegisterTool(Tool{Name: "dup", Handler: func(context.Context, RawMessage) core.Result { return core.Ok(nil) }}); !r.OK {
		t.Fatalf("first duplicate setup failed: %s", r.Error())
	}
	if r := s.RegisterTool(Tool{Name: "dup", Handler: func(context.Context, RawMessage) core.Result { return core.Ok(nil) }}); r.OK {
		t.Fatal("expected duplicate registration to fail")
	}
}

func TestService_HandleFrame_Good(t *testing.T) {
	s := mustNewService(t, WithWorkspaceRoot(t.TempDir()))

	frame := []byte("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"lang_detect\",\"arguments\":{\"\x70ath\":\"main.go\"}}}")
	responseResult := s.HandleFrame(context.Background(), frame)
	if !responseResult.OK {
		t.Fatalf("HandleFrame failed: %s", responseResult.Error())
	}
	response := responseResult.Value.([]byte)
	var decoded struct {
		Result struct {
			StructuredContent DetectLanguageOutput `json:"structuredContent"`
		} `json:"result"`
	}
	if r := core.JSONUnmarshal(response, &decoded); !r.OK {
		t.Fatalf("decode response: %v", r.Error())
	}
	if decoded.Result.StructuredContent.Language != "go" {
		t.Fatalf("language = %q, want go", decoded.Result.StructuredContent.Language)
	}
}

func TestService_HandleFrame_Bad(t *testing.T) {
	s := mustNewService(t, WithWorkspaceRoot(t.TempDir()))

	responseResult := s.HandleFrame(context.Background(), []byte(`{"jsonrpc":"2.0","id":1,"method":"missing"}`))
	if !responseResult.OK {
		t.Fatalf("HandleFrame failed: %s", responseResult.Error())
	}
	response := responseResult.Value.([]byte)
	var decoded struct {
		Error *rpcError `json:"error"`
	}
	if r := core.JSONUnmarshal(response, &decoded); !r.OK {
		t.Fatalf("decode error response: %v", r.Error())
	}
	if decoded.Error == nil || decoded.Error.Code != -32601 {
		t.Fatalf("error = %+v, want method-not-found", decoded.Error)
	}
}

func TestServeStdio_Good(t *testing.T) {
	s := mustNewService(t, WithWorkspaceRoot(t.TempDir()))

	oldReader, oldWriter := stdioReader, stdioWriter
	defer func() {
		stdioReader, stdioWriter = oldReader, oldWriter
	}()

	out := core.NewBuffer()
	stdioReader = core.NewReader(`{"jsonrpc":"2.0","id":1,"method":"tools/list"}` + "\n")
	stdioWriter = out

	if r := s.ServeStdio(context.Background()); !r.OK {
		t.Fatalf("ServeStdio: %s", r.Error())
	}
	if !core.Contains(out.String(), `"tools"`) {
		t.Fatalf("stdio output %q missing tools list", out.String())
	}
}

func TestServeTCP_Good(t *testing.T) {
	s := mustNewService(t, WithWorkspaceRoot(t.TempDir()))

	addr := reserveTCPAddr(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errCh := make(chan core.Result, 1)
	go func() {
		errCh <- s.ServeTCP(ctx, addr)
	}()
	waitForTCP(t, addr)

	conn, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	defer conn.Close()
	if _, err := conn.Write([]byte("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"lang_detect\",\"arguments\":{\"\x70ath\":\"x.py\"}}}\n")); err != nil {
		t.Fatalf("write request: %v", err)
	}
	line, err := bufio.NewReader(conn).ReadString('\n')
	if err != nil {
		t.Fatalf("read response: %v", err)
	}
	if !core.Contains(line, `"language":"python"`) {
		t.Fatalf("response %q missing python language", line)
	}

	cancel()
	if r := <-errCh; !r.OK {
		t.Fatalf("ServeTCP returned %s", r.Error())
	}
}

func TestServeUnix_Good(t *testing.T) {
	s := mustNewService(t, WithWorkspaceRoot(t.TempDir()))

	socketPath := core.PathJoin("/tmp", core.Sprintf("mcp-%d-service.sock", core.Getpid()))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errCh := make(chan core.Result, 1)
	go func() {
		errCh <- s.ServeUnix(ctx, socketPath)
	}()
	waitForUnix(t, socketPath)

	conn, err := net.Dial("unix", socketPath)
	if err != nil {
		t.Fatalf("Dial unix: %v", err)
	}
	defer conn.Close()
	if _, err := conn.Write([]byte(`{"jsonrpc":"2.0","id":1,"method":"tools/list"}` + "\n")); err != nil {
		t.Fatalf("write request: %v", err)
	}
	line, err := bufio.NewReader(conn).ReadString('\n')
	if err != nil {
		t.Fatalf("read response: %v", err)
	}
	if !core.Contains(line, `"file_read"`) {
		t.Fatalf("response %q missing file_read", line)
	}

	cancel()
	if r := <-errCh; !r.OK {
		t.Fatalf("ServeUnix returned %s", r.Error())
	}
	if r := core.Stat(socketPath); r.OK {
		t.Fatalf("socket file still exists")
	} else if statErr, _ := resultError(r).(error); !core.IsNotExist(statErr) {
		t.Fatalf("socket stat failed unexpectedly: %v", statErr)
	}
}

func TestServiceToolInventoryCount(t *testing.T) {
	s := mustNewService(t, WithWorkspaceRoot(t.TempDir()))
	if got, want := len(s.Tools()), 49; got != want {
		t.Fatalf("tool count = %d, want %d", got, want)
	}
}

func reserveTCPAddr(t *testing.T) string {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("reserve tcp addr: %v", err)
	}
	addr := listener.Addr().String()
	if err := listener.Close(); err != nil {
		t.Fatalf("close reserved listener: %v", err)
	}
	return addr
}

func waitForTCP(t *testing.T, addr string) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		conn, err := net.DialTimeout("tcp", addr, 50*time.Millisecond)
		if err == nil {
			conn.Close()
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for tcp %s", addr)
}

func waitForUnix(t *testing.T, socketPath string) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		conn, err := net.DialTimeout("unix", socketPath, 50*time.Millisecond)
		if err == nil {
			conn.Close()
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for unix socket %s", socketPath)
}

// --- AX-7 canonical triplets ---

type testSubsystem struct {
	called *bool
	err    error
}

func (s testSubsystem) Name() string { return "ax7" }

func (s testSubsystem) RegisterTools(*Service) {
	if s.called != nil {
		*s.called = true
	}
}

func (s testSubsystem) Shutdown(context.Context) error {
	if s.called != nil {
		*s.called = true
	}
	return s.err
}

func TestService_New_Good(t *core.T) {
	result := New(WithWorkspaceRoot(t.TempDir()))
	service := result.Value.(*Service)
	names := service.ToolNames()

	core.AssertTrue(t, result.OK)
	core.AssertTrue(t, len(names) > 0)
}

func TestService_New_Bad(t *core.T) {
	result := New(42)
	got := result.Error()

	core.AssertFalse(t, result.OK)
	core.AssertContains(t, got, "unsupported")
}

func TestService_New_Ugly(t *core.T) {
	result := New(Options{Unrestricted: true})
	service := result.Value.(*Service)
	root := service.WorkspaceRoot()

	core.AssertTrue(t, result.OK)
	core.AssertEqual(t, "", root)
}

func TestService_WithWorkspaceRoot_Good(t *core.T) {
	service := &Service{}
	option := WithWorkspaceRoot(t.TempDir())
	result := option(service)

	core.AssertTrue(t, result.OK)
	core.AssertNotEqual(t, "", service.WorkspaceRoot())
}

func TestService_WithWorkspaceRoot_Bad(t *core.T) {
	service := &Service{workspaceRoot: "before"}
	option := WithWorkspaceRoot("")
	result := option(service)

	core.AssertTrue(t, result.OK)
	core.AssertEqual(t, "", service.WorkspaceRoot())
}

func TestService_WithWorkspaceRoot_Ugly(t *core.T) {
	service := &Service{}
	option := WithWorkspaceRoot(".")
	result := option(service)

	core.AssertTrue(t, result.OK)
	core.AssertTrue(t, service.WorkspaceRoot() != ".")
}

func TestService_WithProcessService_Good(t *core.T) {
	service := &Service{}
	option := WithProcessService("process")
	result := option(service)

	core.AssertTrue(t, result.OK)
	core.AssertEqual(t, "process", service.processService)
}

func TestService_WithProcessService_Bad(t *core.T) {
	service := &Service{processService: "before"}
	option := WithProcessService(nil)
	result := option(service)

	core.AssertTrue(t, result.OK)
	core.AssertNil(t, service.processService)
}

func TestService_WithProcessService_Ugly(t *core.T) {
	service := &Service{}
	payload := map[string]bool{"ok": true}
	result := WithProcessService(payload)(service)

	core.AssertTrue(t, result.OK)
	core.AssertEqual(t, payload, service.processService)
}

func TestService_WithWSHub_Good(t *core.T) {
	service := &Service{}
	option := WithWSHub("hub")
	result := option(service)

	core.AssertTrue(t, result.OK)
	core.AssertEqual(t, "hub", service.wsHub)
}

func TestService_WithWSHub_Bad(t *core.T) {
	service := &Service{wsHub: "before"}
	option := WithWSHub(nil)
	result := option(service)

	core.AssertTrue(t, result.OK)
	core.AssertNil(t, service.wsHub)
}

func TestService_WithWSHub_Ugly(t *core.T) {
	service := &Service{}
	payload := map[string]bool{"connected": true}
	result := WithWSHub(payload)(service)

	core.AssertTrue(t, result.OK)
	core.AssertEqual(t, payload, service.wsHub)
}

func TestService_WithInferenceModel_Good(t *core.T) {
	service := &Service{}
	model := &generateModel{}
	result := WithInferenceModel(model, "openai", "gpt-test")(service)

	core.AssertTrue(t, result.OK)
	core.AssertEqual(t, model, service.mlModel)
	core.AssertEqual(t, "openai", service.mlBackend)
	core.AssertEqual(t, "gpt-test", service.mlModelName)
}

func TestService_WithInferenceModel_Bad(t *core.T) {
	service := &Service{mlBackend: "before", mlModelName: "before"}
	result := WithInferenceModel(nil, "", "")(service)

	core.AssertTrue(t, result.OK)
	core.AssertNil(t, service.mlModel)
	core.AssertEqual(t, "", service.mlBackend)
	core.AssertEqual(t, "", service.mlModelName)
}

func TestService_WithInferenceModel_Ugly(t *core.T) {
	service := &Service{}
	model := &generateModel{}
	result := WithInferenceModel(model, "  backend  ", "  model  ")(service)

	core.AssertTrue(t, result.OK)
	core.AssertEqual(t, "backend", service.mlBackend)
	core.AssertEqual(t, "model", service.mlModelName)
}

func TestService_WithSubsystem_Good(t *core.T) {
	service := &Service{}
	sub := testSubsystem{}
	result := WithSubsystem(sub)(service)

	core.AssertTrue(t, result.OK)
	core.AssertLen(t, service.subsystems, 1)
}

func TestService_WithSubsystem_Bad(t *core.T) {
	service := &Service{}
	result := WithSubsystem(nil)(service)
	got := len(service.subsystems)

	core.AssertTrue(t, result.OK)
	core.AssertEqual(t, 0, got)
}

func TestService_WithSubsystem_Ugly(t *core.T) {
	service := &Service{}
	first := testSubsystem{}
	second := testSubsystem{}

	core.AssertTrue(t, WithSubsystem(first)(service).OK)
	core.AssertTrue(t, WithSubsystem(second)(service).OK)
	core.AssertLen(t, service.subsystems, 2)
}

func TestService_Service_WorkspaceRoot_Good(t *core.T) {
	service := &Service{workspaceRoot: "/repo"}
	got := service.WorkspaceRoot()
	want := "/repo"

	core.AssertEqual(t, want, got)
	core.AssertNotEqual(t, "", got)
}

func TestService_Service_WorkspaceRoot_Bad(t *core.T) {
	service := &Service{}
	got := service.WorkspaceRoot()
	want := ""

	core.AssertEqual(t, want, got)
	core.AssertEmpty(t, got)
}

func TestService_Service_WorkspaceRoot_Ugly(t *core.T) {
	service := &Service{workspaceRoot: ""}
	got := service.WorkspaceRoot()
	unrestricted := got == ""

	core.AssertTrue(t, unrestricted)
	core.AssertEqual(t, "", got)
}

func TestService_Service_Tools_Good(t *core.T) {
	handler := typedHandler(func(context.Context, struct{}) core.Result { return core.Ok(map[string]bool{"ok": true}) })
	service := &Service{tools: map[string]Tool{"x": {Name: "x", InputSchema: objectSchema(), Handler: handler}}, toolOrder: []string{"x"}}
	records := service.Tools()

	core.AssertLen(t, records, 1)
	core.AssertEqual(t, "x", records[0].Name)
}

func TestService_Service_Tools_Bad(t *core.T) {
	service := &Service{tools: map[string]Tool{}, toolOrder: nil}
	records := service.Tools()
	got := len(records)

	core.AssertEqual(t, 0, got)
	core.AssertEmpty(t, records)
}

func TestService_Service_Tools_Ugly(t *core.T) {
	handler := typedHandler(func(context.Context, struct{}) core.Result { return core.Ok(map[string]bool{"ok": true}) })
	service := &Service{tools: map[string]Tool{"x": {Name: "x", InputSchema: objectSchema(), Handler: handler}}, toolOrder: []string{"x"}}
	records := service.Tools()

	records[0].InputSchema["mutated"] = true
	core.AssertNil(t, service.tools["x"].InputSchema["mutated"])
}

func TestService_Service_ToolNames_Good(t *core.T) {
	service := &Service{toolOrder: []string{"a", "b"}}
	names := service.ToolNames()
	got := core.Join(",", names...)

	core.AssertEqual(t, "a,b", got)
	core.AssertLen(t, names, 2)
}

func TestService_Service_ToolNames_Bad(t *core.T) {
	service := &Service{}
	names := service.ToolNames()
	got := len(names)

	core.AssertEqual(t, 0, got)
	core.AssertEmpty(t, names)
}

func TestService_Service_ToolNames_Ugly(t *core.T) {
	service := &Service{toolOrder: []string{"a"}}
	names := service.ToolNames()
	names[0] = "mutated"

	core.AssertEqual(t, []string{"a"}, service.ToolNames())
	core.AssertEqual(t, []string{"mutated"}, names)
}

func TestService_Service_RegisterTool_Good(t *core.T) {
	handler := typedHandler(func(context.Context, struct{}) core.Result { return core.Ok(map[string]bool{"ok": true}) })
	service := &Service{tools: map[string]Tool{}}
	r := service.RegisterTool(Tool{Name: "custom", Handler: handler})

	core.AssertTrue(t, r.OK)
	core.AssertEqual(t, []string{"custom"}, service.ToolNames())
}

func TestService_Service_RegisterTool_Bad(t *core.T) {
	service := &Service{tools: map[string]Tool{}}
	r := service.RegisterTool(Tool{Name: "", Handler: typedHandler(func(context.Context, struct{}) core.Result { return core.Ok(nil) })})
	got := r.Error()

	core.AssertFalse(t, r.OK)
	core.AssertContains(t, got, "name is required")
}

func TestService_Service_RegisterTool_Ugly(t *core.T) {
	handler := typedHandler(func(context.Context, struct{}) core.Result { return core.Ok(map[string]bool{"ok": true}) })
	service := &Service{tools: map[string]Tool{}}
	r := service.RegisterTool(Tool{Name: "custom", Handler: handler})

	core.AssertTrue(t, r.OK)
	core.AssertEqual(t, "object", service.tools["custom"].InputSchema["type"])
}

func TestService_Service_RegisterToolFunc_Good(t *core.T) {
	handler := typedHandler(func(context.Context, struct{}) core.Result { return core.Ok(map[string]bool{"ok": true}) })
	service := &Service{tools: map[string]Tool{}}
	r := service.RegisterToolFunc("group", "custom", "Custom tool", handler)

	core.AssertTrue(t, r.OK)
	core.AssertEqual(t, "group", service.tools["custom"].Group)
}

func TestService_Service_RegisterToolFunc_Bad(t *core.T) {
	service := &Service{tools: map[string]Tool{}}
	r := service.RegisterToolFunc("group", "", "Custom tool", nil)
	got := r.Error()

	core.AssertFalse(t, r.OK)
	core.AssertContains(t, got, "name is required")
}

func TestService_Service_RegisterToolFunc_Ugly(t *core.T) {
	handler := typedHandler(func(context.Context, struct{}) core.Result { return core.Ok(map[string]bool{"ok": true}) })
	service := &Service{tools: map[string]Tool{}}
	r := service.RegisterToolFunc("", "custom", "", handler)

	core.AssertTrue(t, r.OK)
	core.AssertEqual(t, "", service.tools["custom"].Group)
}

func TestService_Service_Shutdown_Good(t *core.T) {
	called := false
	service := &Service{subsystems: []Subsystem{testSubsystem{called: &called}}}
	r := service.Shutdown(core.Background())

	core.AssertTrue(t, r.OK)
	core.AssertTrue(t, called)
}

func TestService_Service_Shutdown_Bad(t *core.T) {
	service := &Service{subsystems: []Subsystem{testSubsystem{err: core.AnError}}}
	r := service.Shutdown(core.Background())
	got := r.Error()

	core.AssertFalse(t, r.OK)
	core.AssertContains(t, got, core.AnError.Error())
}

func TestService_Service_Shutdown_Ugly(t *core.T) {
	service := &Service{processes: map[string]*managedProcess{}}
	r := service.Shutdown(core.Background())
	got := len(service.processes)

	core.AssertTrue(t, r.OK)
	core.AssertEqual(t, 0, got)
}
