package mlservice

import (
	"context"
	"iter"
	"slices"
	"sync"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/capability"
	"dappco.re/go/inference/eval/score"
	"dappco.re/go/inference/serving"
)

// Service manages ML inference backends and scoring with Core lifecycle.
type Service struct {
	*core.ServiceRuntime[Options]

	backends map[string]serving.Backend
	mu       sync.RWMutex
	engine   *score.Engine
	judge    *score.Judge
}

// Options configures the ML service.
type Options struct {
	// DefaultBackend is the name of the default inference backend.
	DefaultBackend string

	// LlamaPath is the path to the llama-server binary.
	LlamaPath string

	// ModelDir is the directory containing model files.
	ModelDir string

	// OllamaURL is the Ollama API base URL.
	OllamaURL string

	// JudgeURL is the judge model API URL.
	JudgeURL string

	// JudgeModel is the judge model name.
	JudgeModel string

	// InfluxURL is the InfluxDB URL for metrics.
	InfluxURL string

	// InfluxDB is the InfluxDB database name.
	InfluxDB string

	// Concurrency is the number of concurrent scoring workers.
	Concurrency int

	// Suites is a comma-separated list of scoring suites to enable.
	Suites string
}

// NewService creates an ML service factory for Core registration.
// Usage example:
//
//	core.New(
//	    core.WithService(mlservice.NewService(mlservice.Options{})),
//	)
func NewService(opts Options) func(*core.Core) core.Result {
	return func(c *core.Core) core.Result {
		if opts.Concurrency == 0 {
			opts.Concurrency = 4
		}
		if opts.Suites == "" {
			opts.Suites = "all"
		}

		svc := &Service{
			ServiceRuntime: core.NewServiceRuntime(c, opts),
			backends:       make(map[string]serving.Backend),
		}
		return core.Ok(svc)
	}
}

// RegisterCore registers the ML service with default options on a Core runtime.
//
//	r := mlservice.RegisterCore(core.New())
//	if !r.OK { return r }
func RegisterCore(c *core.Core) core.Result {
	return core.WithService(NewService(Options{}))(c)
}

// OnStartup initializes backends and scoring engine.
//
//	r := svc.OnStartup(ctx)
//	if !r.OK { return r }
func (s *Service) OnStartup(ctx context.Context) core.Result {
	opts := s.Options()

	// Register Ollama backend if URL provided.
	if opts.OllamaURL != "" {
		s.RegisterBackend("ollama", serving.NewHTTPBackend(opts.OllamaURL, opts.JudgeModel))
	}

	// Set up judge if judge URL is provided.
	if opts.JudgeURL != "" {
		judgeBackend := serving.NewHTTPBackend(opts.JudgeURL, opts.JudgeModel)
		s.judge = score.NewJudge(judgeBackend)
		s.engine = score.NewEngine(s.judge, opts.Concurrency, opts.Suites)
	}

	return core.Ok(nil)
}

// OnShutdown cleans up resources.
//
//	r := svc.OnShutdown(ctx)
//	if !r.OK { return r }
func (s *Service) OnShutdown(ctx context.Context) core.Result {
	return core.Ok(nil)
}

// RegisterBackend adds or replaces a named inference backend.
func (s *Service) RegisterBackend(name string, backend serving.Backend) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.backends[name] = backend
}

// serving.Backend returns a named backend, or nil if not found.
func (s *Service) Backend(name string) serving.Backend {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.backends[name]
}

// DefaultBackend returns the configured default backend.
func (s *Service) DefaultBackend() serving.Backend {
	name := s.Options().DefaultBackend
	if name == "" {
		name = "ollama"
	}
	return s.Backend(name)
}

// Backends returns the names of all registered backends.
func (s *Service) Backends() []string {
	s.mu.RLock()
	if len(s.backends) == 0 {
		s.mu.RUnlock()
		return nil
	}
	names := make([]string, 0, len(s.backends))
	for name := range s.backends {
		names = append(names, name)
	}
	s.mu.RUnlock()
	slices.Sort(names)
	return names
}

// BackendsIter returns an iterator over the names of all registered backends.
func (s *Service) BackendsIter() iter.Seq[string] {
	return func(yield func(string) bool) {
		s.mu.RLock()
		defer s.mu.RUnlock()
		for name := range s.backends {
			if !yield(name) {
				return
			}
		}
	}
}

// BackendCapabilities returns the shared capability report for a registered backend.
func (s *Service) BackendCapabilities(name string) (inference.CapabilityReport, bool) {
	backend := s.Backend(name)
	if backend == nil {
		return inference.CapabilityReport{}, false
	}
	return capability.CapabilityReportForBackend(name, backend), true
}

// score.Judge returns the configured judge, or nil if not set up.
func (s *Service) Judge() *score.Judge {
	return s.judge
}

// score.Engine returns the scoring engine, or nil if not set up.
func (s *Service) Engine() *score.Engine {
	return s.engine
}

// Generate generates text using the named backend (or default).
//
//	r := svc.Generate(ctx, "ollama", "hello", mlservice.DefaultGenOpts())
//	if !r.OK { return r }
//	resp := r.Value.(mlservice.Result)
func (s *Service) Generate(ctx context.Context, backendName, prompt string, opts serving.GenOpts) core.Result {
	b := s.Backend(backendName)
	if b == nil {
		b = s.DefaultBackend()
	}
	if b == nil {
		return core.Fail(core.E("mlservice.Service.Generate", core.Sprintf("no backend available (requested: %q)", backendName), nil))
	}
	return b.Generate(ctx, prompt, opts)
}

// ScoreResponses scores a batch of responses using the configured engine.
//
//	r := svc.ScoreResponses(ctx, responses)
//	if !r.OK { return r }
//	scores := r.Value.(map[string][]mlservice.PromptScore)
func (s *Service) ScoreResponses(ctx context.Context, responses []score.Response) core.Result {
	if s.engine == nil {
		return core.Fail(core.E("mlservice.Service.ScoreResponses", "scoring engine not configured (set JudgeURL and JudgeModel)", nil))
	}
	return core.Ok(s.engine.ScoreAll(ctx, responses))
}
