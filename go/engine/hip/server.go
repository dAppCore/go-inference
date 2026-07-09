//go:build linux && amd64 && rocm_legacy_server

package hip

import (
	"context"
	// Note: intrinsic - net.Listener for the HTTP server; no core equivalent.
	"net"
	// Note: intrinsic - numeric parsing from ROCm output; core has no ParseInt/Atoi.
	"strconv"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/engine/hip/internal/llamacpp"
)

var (
	serverStartupTimeout    = 60 * time.Second
	serverReadyPollInterval = 100 * time.Millisecond
	serverPortAllocator     = newDeterministicPortAllocator(serverPortRangeStart, serverPortRangeCount)
	// listenLocalTCP lets tests stub port probing without opening real sockets.
	listenLocalTCP = net.Listen
)

const (
	serverProcessOutputLimit       = 32 << 10
	serverProcessOutputSummarySize = 1024
	serverPortRangeStart           = 38080
	serverPortRangeCount           = 256
)

// server manages a llama-server subprocess.
type server struct {
	processCommand   *core.Cmd
	port             int
	llamaClient      *llamacpp.Client
	processExited    chan struct{}
	processExitError error // safe to read only after <-processExited
	processOutput    *processOutputCapture
}

// serverStartConfig keeps llama-server startup settings named instead of positional.
type serverStartConfig struct {
	BinaryPath        string
	ModelPath         string
	GPULayerCount     int
	ContextSize       int
	ParallelSlotCount int
}

// alive reports whether the llama-server process is still running.
func (s *server) alive() bool {
	if s == nil || s.processExited == nil {
		return false
	}
	select {
	case <-s.processExited:
		return false
	default:
		return true
	}
}

// findLlamaServer locates the llama-server binary.
// Checks ROCM_LLAMA_SERVER_PATH first, then PATH.
func findLlamaServer() (
	string,
	error,
) {
	if p := core.Getenv("ROCM_LLAMA_SERVER_PATH"); p != "" {
		return validateLlamaServerPath(p)
	}
	for _, dir := range core.Split(core.Getenv("PATH"), string(core.PathListSeparator)) {
		p := core.PathJoin(dir, "llama-server")
		if _, err := validateLlamaServerPath(p); err == nil {
			return p, nil
		}
	}
	return "", core.E("rocm.findLlamaServer", "llama-server not found in PATH", nil)
}

func validateLlamaServerPath(path string) (
	string,
	error,
) {
	infoResult := core.Stat(path)
	if !infoResult.OK {
		return "", core.E("rocm.findLlamaServer", "llama-server not found at ROCM_LLAMA_SERVER_PATH="+path, infoResult.Value.(error))
	}
	info := infoResult.Value.(core.FsFileInfo)
	if info.IsDir() {
		return "", core.E("rocm.findLlamaServer", "ROCM_LLAMA_SERVER_PATH must point to a file", nil)
	}
	if info.Mode().Perm()&0o111 == 0 {
		return "", core.E("rocm.findLlamaServer", "llama-server is not executable at ROCM_LLAMA_SERVER_PATH="+path, nil)
	}
	return path, nil
}

// freePort walks a deterministic localhost port range and returns the first
// currently-bindable port.
func freePort() (
	int,
	error,
) {
	return serverPortAllocator.NextAvailablePort()
}

// serverEnv returns the environment for the llama-server subprocess.
// Filters any existing HIP_* settings and sets HIP_VISIBLE_DEVICES=0 to mask
// the iGPU. This is critical — the Ryzen 9 iGPU crashes llama-server if not
// masked, and inherited HIP variables can re-expose multi-GPU state.
func serverEnv() []string {
	environ := core.Environ()
	env := make([]string, 0, len(environ)+1)
	for _, e := range environ {
		if core.HasPrefix(e, "HIP_") {
			continue
		}
		env = append(env, e)
	}
	env = append(env, "HIP_VISIBLE_DEVICES=0")
	return env
}

// startServer spawns llama-server and waits for it to become ready.
// It selects a free port automatically, retrying up to 3 times if startup
// fails before the health endpoint becomes ready.
func startServer(startConfig serverStartConfig) (
	*server,
	error,
) {
	gpuLayerCount := startConfig.GPULayerCount
	if gpuLayerCount < 0 {
		gpuLayerCount = 999
	}

	const maxAttempts = 3
	var lastStartupError error

	for attempt := 0; attempt < maxAttempts; attempt++ {
		port, err := freePort()
		if err != nil {
			return nil, core.E("rocm.startServer", "find free port", err)
		}

		commandArguments := llamaServerArguments(startConfig, port, gpuLayerCount)

		outputCapture := newProcessOutputCapture(serverProcessOutputLimit)
		processCommand := &core.Cmd{Path: startConfig.BinaryPath, Args: append([]string{startConfig.BinaryPath}, commandArguments...)}
		processCommand.Env = serverEnv()
		processCommand.Stdout = outputCapture
		processCommand.Stderr = outputCapture

		if err := processCommand.Start(); err != nil {
			return nil, core.E("rocm.startServer", "start llama-server", err)
		}

		s := &server{
			processCommand: processCommand,
			port:           port,
			llamaClient:    llamacpp.NewClient(core.Sprintf("http://127.0.0.1:%d", port)),
			processExited:  make(chan struct{}),
			processOutput:  outputCapture,
		}

		go func() {
			s.processExitError = processCommand.Wait()
			close(s.processExited)
		}()

		ctx, cancel := context.WithTimeout(context.Background(), serverStartupTimeout)
		err = s.waitReady(ctx)
		cancel()
		if err == nil {
			return s, nil
		}

		if stopErr := s.stop(); stopErr != nil {
			core.Warn("llama-server cleanup after failed startup returned error", "attempt", attempt+1, "err", stopErr)
		}
		lastStartupError = core.E("rocm.startServer", core.Sprintf("attempt %d", attempt+1), err)
		if attempt < maxAttempts-1 {
			core.Warn("llama-server startup failed; retrying", "attempt", attempt+1, "max_attempts", maxAttempts, "err", lastStartupError)
		}
	}

	return nil, core.E("rocm.startServer", core.Sprintf("server failed after %d attempts", maxAttempts), lastStartupError)
}

func llamaServerArguments(startConfig serverStartConfig, port, gpuLayerCount int) []string {
	commandArguments := []string{
		"--model", startConfig.ModelPath,
		"--host", "127.0.0.1",
		"--port", strconv.Itoa(port),
		"--n-gpu-layers", strconv.Itoa(gpuLayerCount),
	}
	if startConfig.ContextSize > 0 {
		commandArguments = append(commandArguments, "--ctx-size", strconv.Itoa(startConfig.ContextSize))
	}
	if startConfig.ParallelSlotCount > 0 {
		commandArguments = append(commandArguments, "--parallel", strconv.Itoa(startConfig.ParallelSlotCount))
	}
	return commandArguments
}

// waitReady polls the health endpoint until the server is ready.
func (s *server) waitReady(ctx context.Context) rocmFailure {
	ticker := time.NewTicker(serverReadyPollInterval)
	defer ticker.Stop()

	var lastHealthError error

	for {
		select {
		case <-ctx.Done():
			if lastHealthError != nil {
				return core.E("server.waitReady", s.messageWithProcessOutput("timeout waiting for llama-server"), lastHealthError)
			}
			return core.E("server.waitReady", s.messageWithProcessOutput("timeout waiting for llama-server"), ctx.Err())
		case <-s.processExited:
			return s.processFailure("server.waitReady", "llama-server exited before becoming ready", s.processExitError)
		case <-ticker.C:
			if err := s.llamaClient.Health(ctx); err == nil {
				return nil
			} else {
				lastHealthError = err
			}
		}
	}
}

// stop sends SIGTERM and waits up to 5s, then SIGKILL. Exit caused by those
// signals is treated as a successful caller-initiated shutdown.
func (s *server) stop() rocmFailure {
	if s == nil || s.processCommand == nil || s.processCommand.Process == nil {
		return nil
	}

	// Already exited?
	select {
	case <-s.processExited:
		if isExpectedStopExitFailure(s.processExitError) {
			return nil
		}
		return s.processFailure("server.stop", "llama-server already exited", s.processExitError)
	default:
	}

	// Send SIGTERM for graceful shutdown.
	if err := s.processCommand.Process.Signal(syscall.SIGTERM); err != nil {
		return core.E("server.stop", "sigterm llama-server", err)
	}

	// Wait up to 5 seconds for clean exit.
	select {
	case <-s.processExited:
		if isExpectedStopExitFailure(s.processExitError) {
			return nil
		}
		return s.processFailure("server.stop", "llama-server exited after sigterm", s.processExitError)
	case <-time.After(5 * time.Second):
		// Force kill.
		if err := s.processCommand.Process.Kill(); err != nil {
			return core.E("server.stop", "kill llama-server", err)
		}
		<-s.processExited
		if isExpectedStopExitFailure(s.processExitError) {
			return nil
		}
		return s.processFailure("server.stop", "llama-server exited after sigkill", s.processExitError)
	}
}

func isExpectedStopExitFailure(err error) bool {
	if err == nil {
		return false
	}

	text := err.Error()
	return core.Contains(text, syscall.SIGTERM.String()) || core.Contains(text, syscall.SIGKILL.String()) ||
		core.Contains(text, "terminated") || core.Contains(text, "killed")
}

func (s *server) messageWithProcessOutput(message string) string {
	if s == nil || s.processOutput == nil {
		return message
	}
	output := s.processOutput.Summary()
	if output == "" {
		return message
	}
	return message + " (llama-server output: " + output + ")"
}

func (s *server) processFailure(
	op string,
	message string,
	err error,
) error {
	if err == nil {
		return nil
	}
	return core.E(op, s.messageWithProcessOutput(message), err)
}

type deterministicPortAllocator struct {
	basePort  int
	portCount int
	nextPort  atomic.Uint64
}

func newDeterministicPortAllocator(basePort, portCount int) *deterministicPortAllocator {
	return &deterministicPortAllocator{
		basePort:  basePort,
		portCount: portCount,
	}
}

func (allocator *deterministicPortAllocator) NextAvailablePort() (
	int,
	error,
) {
	if allocator == nil || allocator.portCount <= 0 {
		return 0, core.E("rocm.freePort", "port allocator is not configured", nil)
	}

	lastPort := allocator.basePort + allocator.portCount - 1
	if allocator.basePort <= 0 || lastPort > 65535 {
		return 0, core.E("rocm.freePort", core.Sprintf("invalid port range %d-%d", allocator.basePort, lastPort), nil)
	}

	startIndex := allocator.nextPort.Add(1) - 1
	for scanned := 0; scanned < allocator.portCount; scanned++ {
		portIndex := int((startIndex + uint64(scanned)) % uint64(allocator.portCount))
		port := allocator.basePort + portIndex
		address := net.JoinHostPort("127.0.0.1", strconv.Itoa(port))

		listener, err := listenLocalTCP("tcp", address)
		if err != nil {
			continue
		}
		listener.Close()

		allocator.advancePast(startIndex + uint64(scanned) + 1)
		return port, nil
	}

	return 0, core.E("rocm.freePort", core.Sprintf("no free port in deterministic range %d-%d", allocator.basePort, lastPort), nil)
}

func (allocator *deterministicPortAllocator) advancePast(candidate uint64) {
	for {
		current := allocator.nextPort.Load()
		if current >= candidate {
			return
		}
		if allocator.nextPort.CompareAndSwap(current, candidate) {
			return
		}
	}
}

type processOutputCapture struct {
	maxBytes int

	mu        sync.Mutex
	buffer    []byte
	truncated bool
}

func newProcessOutputCapture(maxBytes int) *processOutputCapture {
	return &processOutputCapture{maxBytes: maxBytes}
}

func (c *processOutputCapture) Write(p []byte) (
	int,
	error,
) {
	c.mu.Lock()
	defer c.mu.Unlock()

	written := len(p)
	if c.maxBytes <= 0 || written == 0 {
		return written, nil
	}

	c.buffer = append(c.buffer, p...)
	if len(c.buffer) > c.maxBytes {
		c.buffer = append([]byte(nil), c.buffer[len(c.buffer)-c.maxBytes:]...)
		c.truncated = true
	}

	return written, nil
}

func (c *processOutputCapture) Summary() string {
	c.mu.Lock()
	defer c.mu.Unlock()

	output := core.Trim(string(c.buffer))
	if output == "" {
		return ""
	}

	lines := core.Split(output, "\n")
	parts := make([]string, 0, len(lines))
	for _, line := range lines {
		line = core.Trim(line)
		if line == "" {
			continue
		}
		parts = append(parts, line)
	}

	output = core.Join(" | ", parts...)
	if len(output) > serverProcessOutputSummarySize {
		output = output[:serverProcessOutputSummarySize] + "..."
	}
	if c.truncated {
		return "..." + output
	}
	return output
}
