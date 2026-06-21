package mcp

import (
	"bufio"
	"bytes"
	"context"
	"io"
	"net/http"
	"slices"
	"sync"
	"sync/atomic"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

const (
	serverName        = "core-cli"
	serverVersion     = "0.1.0"
	maxMCPMessageSize = 10 * 1024 * 1024
)

var (
	errInvalidRequest = core.NewError("invalid JSON-RPC request")
	errInvalidParams  = core.NewError("invalid JSON-RPC params")
)

// Option configures a Service before tools are registered.
type Option func(*Service) core.Result

// Options is accepted by New for compatibility with callers that prefer a struct.
type Options struct {
	WorkspaceRoot  string
	Unrestricted   bool
	ProcessService any
	WSHub          any
	Subsystems     []Subsystem
}

// Subsystem registers additional MCP tools at startup.
type Subsystem interface {
	Name() string
	RegisterTools(*Service)
}

// SubsystemWithShutdown extends Subsystem with graceful cleanup.
type SubsystemWithShutdown interface {
	Subsystem
	Shutdown(context.Context) error
}

// RawMessage preserves raw JSON arguments without requiring a direct
// encoding/json import in MCP surface types.
type RawMessage []byte

// ToolHandler receives the raw JSON arguments from tools/call and returns a
// JSON-serialisable structured response.
type ToolHandler func(context.Context, RawMessage) core.Result

// Tool describes one MCP tool.
type Tool struct {
	Name        string
	Description string
	Group       string
	InputSchema map[string]any
	Handler     ToolHandler
}

// ToolRecord is the public, immutable view of a registered tool.
type ToolRecord struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Group       string         `json:"group,omitempty"`
	InputSchema map[string]any `json:"inputSchema,omitempty"`
}

// Service is the central MCP server state.
type Service struct {
	workspaceRoot string
	tools         map[string]Tool
	toolOrder     []string
	subsystems    []Subsystem

	processMu      sync.Mutex
	processSeq     atomic.Uint64
	processes      map[string]*managedProcess
	wsMu           sync.Mutex
	wsServer       *http.Server
	wsAddr         string
	webviewMu      sync.Mutex
	webviewState   webviewSession
	startedAt      time.Time
	processService any
	wsHub          any
	mlModel        inference.TextModel
	mlBackend      string
	mlModelName    string
}

// New constructs a Service and registers the built-in 49-tool inventory.
//
// Supported call forms:
//
//	mcp.New(mcp.WithWorkspaceRoot("/repo"))
//	mcp.New(mcp.Options{WorkspaceRoot: "/repo"})
func New(args ...any) core.Result {
	rootResult := core.Getwd()
	if !rootResult.OK {
		return core.Fail(core.Errorf("mcp: get working directory: %s", rootResult.Error()))
	}
	root := rootResult.Value.(string)
	absResult := core.PathAbs(root)
	if !absResult.OK {
		return core.Fail(core.Errorf("mcp: resolve working directory: %s", absResult.Error()))
	}
	root = absResult.Value.(string)

	s := &Service{
		workspaceRoot: root,
		tools:         make(map[string]Tool),
		processes:     make(map[string]*managedProcess),
		startedAt:     time.Now(),
	}

	for _, arg := range args {
		switch v := arg.(type) {
		case nil:
			continue
		case Option:
			if r := v(s); !r.OK {
				return r
			}
		case Options:
			if r := applyOptionsStruct(s, v); !r.OK {
				return r
			}
		default:
			return core.Fail(core.Errorf("mcp: unsupported New option %T", arg))
		}
	}

	if r := s.registerBuiltInTools(); !r.OK {
		return r
	}
	for _, sub := range s.subsystems {
		if sub != nil {
			sub.RegisterTools(s)
		}
	}

	return core.Ok(s)
}

func applyOptionsStruct(s *Service, opts Options) core.Result {
	if opts.Unrestricted {
		if r := WithWorkspaceRoot("")(s); !r.OK {
			return r
		}
	} else if opts.WorkspaceRoot != "" {
		if r := WithWorkspaceRoot(opts.WorkspaceRoot)(s); !r.OK {
			return r
		}
	}
	if opts.ProcessService != nil {
		s.processService = opts.ProcessService
	}
	if opts.WSHub != nil {
		s.wsHub = opts.WSHub
	}
	for _, sub := range opts.Subsystems {
		if sub != nil {
			s.subsystems = append(s.subsystems, sub)
		}
	}
	return core.Ok(nil)
}

// WithWorkspaceRoot restricts file operations to root. Passing an empty string
// disables sandboxing and lets file tools operate on cleaned OS paths.
func WithWorkspaceRoot(root string) Option {
	return func(s *Service) core.Result {
		if root == "" {
			s.workspaceRoot = ""
			return core.Ok(nil)
		}
		abs := core.PathAbs(root)
		if !abs.OK {
			return core.Fail(core.Errorf("mcp: resolve workspace root: %s", abs.Error()))
		}
		s.workspaceRoot = abs.Value.(string)
		return core.Ok(nil)
	}
}

// WithProcessService records an externally supplied process service. The
// in-module process tools still provide a local fallback when this is nil.
func WithProcessService(ps any) Option {
	return func(s *Service) core.Result {
		s.processService = ps
		return core.Ok(nil)
	}
}

// WithWSHub records an externally supplied WebSocket hub.
func WithWSHub(hub any) Option {
	return func(s *Service) core.Result {
		s.wsHub = hub
		return core.Ok(nil)
	}
}

// WithInferenceModel routes the ml_generate tool through a configured
// inference.TextModel.
func WithInferenceModel(model inference.TextModel, backendName, modelName string) Option {
	return func(s *Service) core.Result {
		s.mlModel = model
		s.mlBackend = core.Trim(backendName)
		s.mlModelName = core.Trim(modelName)
		return core.Ok(nil)
	}
}

// WithSubsystem appends a subsystem plugin.
func WithSubsystem(sub Subsystem) Option {
	return func(s *Service) core.Result {
		if sub != nil {
			s.subsystems = append(s.subsystems, sub)
		}
		return core.Ok(nil)
	}
}

// WorkspaceRoot returns the configured filesystem sandbox root. An empty value
// means unrestricted filesystem access.
func (s *Service) WorkspaceRoot() string {
	return s.workspaceRoot
}

// Tools returns registered tools in registration order.
func (s *Service) Tools() []ToolRecord {
	records := make([]ToolRecord, 0, len(s.toolOrder))
	for _, name := range s.toolOrder {
		tool := s.tools[name]
		records = append(records, ToolRecord{
			Name:        tool.Name,
			Description: tool.Description,
			Group:       tool.Group,
			InputSchema: cloneStringAnyMap(tool.InputSchema),
		})
	}
	return records
}

// ToolNames returns registered tool names in registration order.
func (s *Service) ToolNames() []string {
	return slices.Clone(s.toolOrder)
}

// RegisterTool adds a tool to the service.
func (s *Service) RegisterTool(tool Tool) core.Result {
	tool.Name = core.Trim(tool.Name)
	if tool.Name == "" {
		return core.Fail(core.Errorf("mcp: tool name is required"))
	}
	if tool.Handler == nil {
		return core.Fail(core.Errorf("mcp: handler is required for tool %q", tool.Name))
	}
	if _, exists := s.tools[tool.Name]; exists {
		return core.Fail(core.Errorf("mcp: tool %q already registered", tool.Name))
	}
	if tool.InputSchema == nil {
		tool.InputSchema = objectSchema()
	}
	s.tools[tool.Name] = tool
	s.toolOrder = append(s.toolOrder, tool.Name)
	return core.Ok(nil)
}

// RegisterToolFunc adds a tool with a raw JSON argument handler.
func (s *Service) RegisterToolFunc(group, name, description string, handler ToolHandler) core.Result {
	return s.RegisterTool(Tool{
		Name:        name,
		Description: description,
		Group:       group,
		Handler:     handler,
	})
}

// Shutdown gracefully stops subsystems, local WebSocket serving, and managed processes.
func (s *Service) Shutdown(ctx context.Context) core.Result {
	var errs []error
	for _, sub := range s.subsystems {
		if sh, ok := sub.(SubsystemWithShutdown); ok {
			if err := sh.Shutdown(ctx); err != nil {
				errs = append(errs, err)
			}
		}
	}

	s.wsMu.Lock()
	wsServer := s.wsServer
	s.wsMu.Unlock()
	if wsServer != nil {
		if err := wsServer.Shutdown(ctx); err != nil {
			errs = append(errs, err)
		}
	}

	s.processMu.Lock()
	processes := make([]*managedProcess, 0, len(s.processes))
	for _, proc := range s.processes {
		processes = append(processes, proc)
	}
	s.processMu.Unlock()
	for _, proc := range processes {
		if proc.isRunning() && proc.cmd.Process != nil {
			if err := proc.cmd.Process.Kill(); err != nil {
				errs = append(errs, err)
			}
		}
	}

	if err := core.ErrorJoin(errs...); err != nil {
		return core.Fail(err)
	}
	return core.Ok(nil)
}

type typedToolFunc[I any] func(context.Context, I) core.Result

func typedHandler[I any](fn typedToolFunc[I]) ToolHandler {
	return func(ctx context.Context, raw RawMessage) core.Result {
		var input I
		// bytes.TrimSpace returns a subslice — zero alloc, vs the
		// previous []byte→string→Trim→[]byte round-trip which allocated
		// two strings plus a fresh byte slice per typed-tool invocation.
		// `string(raw) == "null"` compiles to a byte compare without
		// allocating the temporary string.
		raw = RawMessage(bytes.TrimSpace(raw))
		if len(raw) == 0 || string(raw) == "null" {
			raw = RawMessage("{}")
		}
		if r := core.JSONUnmarshal([]byte(raw), &input); !r.OK {
			return core.Fail(core.Errorf("%w: %s", errInvalidParams, r.Error()))
		}
		return fn(ctx, input)
	}
}

func objectSchema() map[string]any {
	return map[string]any{
		"type":                 "object",
		"additionalProperties": true,
	}
}

func cloneStringAnyMap(input map[string]any) map[string]any {
	if input == nil {
		return nil
	}
	out := make(map[string]any, len(input))
	for k, v := range input {
		out[k] = v
	}
	return out
}

func serveReaderWriter(ctx context.Context, r io.Reader, w io.Writer, handle func(context.Context, []byte) core.Result) core.Result {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 64*1024), maxMCPMessageSize)
	for scanner.Scan() {
		select {
		case <-ctx.Done():
			return core.Ok(nil)
		default:
		}

		result := handle(ctx, scanner.Bytes())
		if !result.OK {
			return result
		}
		response, _ := result.Value.([]byte)
		if len(response) == 0 {
			continue
		}
		if _, err := w.Write(append(response, '\n')); err != nil {
			return core.Fail(err)
		}
	}
	if err := scanner.Err(); err != nil {
		return core.Fail(err)
	}
	return core.Ok(nil)
}
