package agent

import (
	"context"
	"strconv"
	"time"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	goexec "dappco.re/go/process/exec"
)

// RemoteTransport abstracts remote command execution and file transfer.
// Implementations may use SSH/SCP, Docker exec, or in-memory fakes for testing.
type RemoteTransport interface {
	// Run executes a command on the remote host and returns combined output.
	//
	//	r := t.Run(ctx, "ls /tmp")
	//	if !r.OK { return r }
	//	out := r.String()
	Run(ctx context.Context, cmd string) core.Result

	// CopyFrom copies a file from the remote host to a local path.
	//
	//	r := t.CopyFrom(ctx, "/remote/path", "/local/path")
	//	if !r.OK { return r }
	CopyFrom(ctx context.Context, remote, local string) core.Result

	// CopyTo copies a local file to the remote host.
	//
	//	r := t.CopyTo(ctx, "/local/path", "/remote/path")
	//	if !r.OK { return r }
	CopyTo(ctx context.Context, local, remote string) core.Result
}

// SSHTransport implements RemoteTransport using the ssh and scp binaries.
type SSHTransport struct {
	Host    string
	User    string
	KeyPath string
	Port    string
	Timeout time.Duration
}

// SSHOption configures an SSHTransport.
type SSHOption func(*SSHTransport)

// WithPort sets a non-default SSH port.
func WithPort(port string) SSHOption {
	return func(t *SSHTransport) {
		t.Port = port
	}
}

// WithTimeout sets the SSH connection timeout.
func WithTimeout(d time.Duration) SSHOption {
	return func(t *SSHTransport) {
		t.Timeout = d
	}
}

// NewSSHTransport creates an SSHTransport with the given credentials and options.
func NewSSHTransport(host, user, keyPath string, opts ...SSHOption) *SSHTransport {
	t := &SSHTransport{
		Host:    host,
		User:    user,
		KeyPath: keyPath,
		Port:    "22",
		Timeout: 10 * time.Second,
	}
	for _, o := range opts {
		o(t)
	}
	return t
}

// commonArgs returns the shared SSH options for both ssh and scp.
func (t *SSHTransport) commonArgs() []string {
	timeout := int(t.Timeout.Seconds())
	if timeout < 1 {
		timeout = 10
	}
	args := []string{
		"-o", core.Sprintf("ConnectTimeout=%d", timeout),
		"-o", "BatchMode=yes",
		"-o", "StrictHostKeyChecking=no",
		"-i", t.KeyPath,
	}
	if t.Port != "" && t.Port != "22" {
		args = append(args, "-P", t.Port)
	}
	return args
}

// sshPortArgs returns the port flag for ssh (uses -p, not -P).
func (t *SSHTransport) sshPortArgs() []string {
	timeout := int(t.Timeout.Seconds())
	if timeout < 1 {
		timeout = 10
	}
	args := []string{
		"-o", core.Sprintf("ConnectTimeout=%d", timeout),
		"-o", "BatchMode=yes",
		"-o", "StrictHostKeyChecking=no",
		"-i", t.KeyPath,
	}
	if t.Port != "" && t.Port != "22" {
		args = append(args, "-p", t.Port)
	}
	return args
}

// Run executes a command on the remote host via ssh.
//
//	r := t.Run(ctx, "ls /tmp")
//	if !r.OK { return r }
//	out := r.String()
func (t *SSHTransport) Run(ctx context.Context, cmd string) core.Result {
	args := t.sshPortArgs()
	args = append(args, core.Sprintf("%s@%s", t.User, t.Host), cmd)

	c := goexec.Command(ctx, "ssh", args...)
	result := c.CombinedOutput()
	if !result.OK {
		return core.Fail(core.E("agent.SSHTransport.Run", core.Sprintf("ssh %q: %s", cmd, result.Error()), nil))
	}
	return core.Ok(string(result.Bytes()))
}

// CopyFrom copies a file from the remote host to a local path via scp.
//
//	r := t.CopyFrom(ctx, "/remote/model.gguf", "/local/model.gguf")
//	if !r.OK { return r }
func (t *SSHTransport) CopyFrom(ctx context.Context, remote, local string) core.Result {
	coreio.Local.EnsureDir(core.PathDir(local))
	args := t.commonArgs()
	args = append(args, core.Sprintf("%s@%s:%s", t.User, t.Host, remote), local)

	c := goexec.Command(ctx, "scp", args...)
	result := c.CombinedOutput()
	if !result.OK {
		return core.Fail(core.E("agent.SSHTransport.CopyFrom", core.Sprintf("scp %s: %s", remote, result.Error()), nil))
	}
	return core.Ok(nil)
}

// CopyTo copies a local file to the remote host via scp.
//
//	r := t.CopyTo(ctx, "/local/adapter.safetensors", "/remote/adapter.safetensors")
//	if !r.OK { return r }
func (t *SSHTransport) CopyTo(ctx context.Context, local, remote string) core.Result {
	args := t.commonArgs()
	args = append(args, local, core.Sprintf("%s@%s:%s", t.User, t.Host, remote))

	c := goexec.Command(ctx, "scp", args...)
	result := c.CombinedOutput()
	if !result.OK {
		return core.Fail(core.E("agent.SSHTransport.CopyTo", core.Sprintf("scp to %s: %s", remote, result.Error()), nil))
	}
	return core.Ok(nil)
}

// SSHCommand executes a command on M3 via SSH.
// Deprecated: Use AgentConfig.Transport.Run() instead.
//
//	r := agent.SSHCommand(cfg, "ls /tmp")
//	if !r.OK { return r }
//	out := r.String()
func SSHCommand(cfg *AgentConfig, cmd string) core.Result {
	return cfg.transport().Run(context.Background(), cmd)
}

// SCPFrom copies a file from M3 to a local path.
// Deprecated: Use AgentConfig.Transport.CopyFrom() instead.
//
//	r := agent.SCPFrom(cfg, "/remote/model.gguf", "/local/model.gguf")
//	if !r.OK { return r }
func SCPFrom(cfg *AgentConfig, remotePath, localPath string) core.Result {
	return cfg.transport().CopyFrom(context.Background(), remotePath, localPath)
}

// SCPTo copies a local file to M3.
// Deprecated: Use AgentConfig.Transport.CopyTo() instead.
//
//	r := agent.SCPTo(cfg, "/local/adapter.safetensors", "/remote/adapter.safetensors")
//	if !r.OK { return r }
func SCPTo(cfg *AgentConfig, localPath, remotePath string) core.Result {
	return cfg.transport().CopyTo(context.Background(), localPath, remotePath)
}

// fileBase returns the last component of a path.
func fileBase(path string) string {
	if core.Contains(path, "\\") {
		path = core.Replace(path, "\\", "/")
	}
	return core.PathBase(path)
}

// EnvOr returns the environment variable value or a fallback.
func EnvOr(key, fallback string) string {
	if v := core.Env(key); v != "" {
		return v
	}
	return fallback
}

// IntEnvOr returns the integer environment variable value or a fallback.
func IntEnvOr(key string, fallback int) int {
	v := core.Env(key)
	if v == "" {
		return fallback
	}
	n, err := strconv.Atoi(v)
	if err != nil || n == 0 {
		return fallback
	}
	return n
}

// ExpandHome expands ~ to the user's home directory.
func ExpandHome(path string) string {
	if core.HasPrefix(path, "~/") {
		home := core.Env("DIR_HOME")
		if home != "" {
			return core.JoinPath(home, path[2:])
		}
	}
	return path
}
