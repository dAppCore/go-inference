package serving

import (
	"context"
	"net/http"
	"time"

	"dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/process"
)

// LlamaBackend manages a llama-server process and delegates HTTP calls to it.
type LlamaBackend struct {
	processSvc *process.Service
	procID     string
	port       int
	http       *HTTPBackend
	modelPath  string
	loraPath   string
	llamaPath  string
}

// LlamaOpts configures the llama-server backend.
type LlamaOpts struct {
	// LlamaPath is the path to the llama-server binary.
	LlamaPath string
	// ModelPath is the path to the GGUF model file.
	ModelPath string
	// LoraPath is the optional path to a GGUF LoRA adapter file.
	LoraPath string
	// Port is the HTTP port for llama-server (default: 18090).
	Port int
}

// NewLlamaBackend creates a backend that manages a llama-server process.
//
// The constructor accepts the current process-service form used by the codebase
// as well as the RFC's simpler model-path form:
//
//	NewLlamaBackend(processSvc, LlamaOpts{ModelPath: "model.gguf"})
//	NewLlamaBackend("model.gguf")
func NewLlamaBackend(args ...any) *LlamaBackend {
	var processSvc *process.Service
	opts := LlamaOpts{}

	for _, arg := range args {
		switch v := arg.(type) {
		case *process.Service:
			processSvc = v
		case LlamaOpts:
			if v.LlamaPath != "" {
				opts.LlamaPath = v.LlamaPath
			}
			if v.ModelPath != "" {
				opts.ModelPath = v.ModelPath
			}
			if v.LoraPath != "" {
				opts.LoraPath = v.LoraPath
			}
			if v.Port != 0 {
				opts.Port = v.Port
			}
		case string:
			if opts.ModelPath == "" {
				opts.ModelPath = v
			}
		case nil:
			// Allow the existing nil placeholder used by tests and callers.
		}
	}

	if opts.Port == 0 {
		opts.Port = 18090
	}
	if opts.LlamaPath == "" {
		opts.LlamaPath = "llama-server"
	}

	baseURL := core.Sprintf("http://127.0.0.1:%d", opts.Port)
	return &LlamaBackend{
		processSvc: processSvc,
		port:       opts.Port,
		modelPath:  opts.ModelPath,
		loraPath:   opts.LoraPath,
		llamaPath:  opts.LlamaPath,
		http:       NewHTTPBackend(baseURL, ""),
	}
}

// Name returns "llama".
func (b *LlamaBackend) Name() string { return "llama" }

// SetMaxTokens sets the maximum token count forwarded to the managed
// llama-server for subsequent Generate/Chat calls. Matches spec §2.4.
//
//	backend := serving.NewLlamaBackend(process, serving.LlamaOpts{...})
//	backend.SetMaxTokens(2048)
func (b *LlamaBackend) SetMaxTokens(n int) {
	if b.http != nil {
		b.http.SetMaxTokens(n)
	}
}

// LoadModel satisfies inference.Backend by wrapping the managed llama-server
// as an inference.TextModel. The path argument is ignored — the GGUF path is
// supplied at construction time via LlamaOpts.ModelPath. Spec §2.4.
//
//	backend := serving.NewLlamaBackend(svc, serving.LlamaOpts{ModelPath: "model.gguf"})
//	result := backend.LoadModel("dummy")
//	for tok := range model.Generate(ctx, "hello") {
//	    fmt.Print(tok.Text)
//	}
func (b *LlamaBackend) LoadModel(_ string, _ ...inference.LoadOption) core.Result {
	if b.http == nil {
		return core.Fail(core.E("serving.LlamaBackend.LoadModel", "HTTP shim not configured", nil))
	}
	return core.Ok(NewLlamaTextModel(b))
}

// Available checks if the llama-server is responding to health checks.
func (b *LlamaBackend) Available() bool {
	if b.procID == "" {
		return false
	}
	url := core.Sprintf("http://127.0.0.1:%d/health", b.port)
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

// Start launches the llama-server process.
//
//	r := b.Start(ctx)
//	if !r.OK { return r }
func (b *LlamaBackend) Start(ctx context.Context) core.Result {
	if b.processSvc == nil {
		return core.Fail(core.E("serving.LlamaBackend.Start", "process service not configured", nil))
	}

	args := []string{
		"-m", b.modelPath,
		"--port", core.Sprintf("%d", b.port),
		"--host", "127.0.0.1",
	}
	if b.loraPath != "" {
		args = append(args, "--lora", b.loraPath)
	}

	startResult := b.processSvc.StartWithOptions(ctx, process.RunOptions{
		Command: b.llamaPath,
		Args:    args,
	})
	if !startResult.OK {
		return core.Fail(core.E("serving.LlamaBackend.Start", "failed to start llama-server: "+startResult.Error(), nil))
	}
	proc, ok := startResult.Value.(*process.Process)
	if !ok || proc == nil {
		return core.Fail(core.E("serving.LlamaBackend.Start", "process service returned an invalid process handle", nil))
	}
	b.procID = proc.ID

	// Wait for health check (up to 30 seconds).
	deadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) {
		if b.Available() {
			return core.Ok(nil)
		}
		time.Sleep(500 * time.Millisecond)
	}

	return core.Fail(core.E("serving.LlamaBackend.Start", "llama-server did not become healthy within 30s", nil))
}

// Stop terminates the llama-server process.
//
//	r := b.Stop()
//	if !r.OK { return r }
func (b *LlamaBackend) Stop() core.Result {
	if b.procID == "" {
		return core.Ok(nil)
	}
	if b.processSvc == nil {
		return core.Fail(core.E("serving.LlamaBackend.Stop", "process service not configured", nil))
	}
	return core.ResultOf(nil, b.processSvc.Kill(b.procID))
}

// Generate sends a prompt to the managed llama-server.
//
//	r := b.Generate(ctx, "hello", serving.DefaultGenOpts())
//	if !r.OK { return r }
//	resp := r.Value.(serving.Result)
func (b *LlamaBackend) Generate(ctx context.Context, prompt string, opts GenOpts) core.Result {
	if !b.Available() {
		return core.Fail(core.E("serving.LlamaBackend.Generate", "llama-server not available", nil))
	}
	if b.http == nil {
		return core.Fail(core.E("serving.LlamaBackend.Generate", "HTTP shim not configured", nil))
	}
	return b.http.Generate(ctx, prompt, opts)
}

// Chat sends a conversation to the managed llama-server.
//
//	r := b.Chat(ctx, messages, serving.DefaultGenOpts())
//	if !r.OK { return r }
//	resp := r.Value.(serving.Result)
func (b *LlamaBackend) Chat(ctx context.Context, messages []Message, opts GenOpts) core.Result {
	if !b.Available() {
		return core.Fail(core.E("serving.LlamaBackend.Chat", "llama-server not available", nil))
	}
	if b.http == nil {
		return core.Fail(core.E("serving.LlamaBackend.Chat", "HTTP shim not configured", nil))
	}
	return b.http.Chat(ctx, messages, opts)
}
