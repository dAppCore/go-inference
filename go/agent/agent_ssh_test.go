package agent

import (
	"context"
	"time"

	core "dappco.re/go"
)

func TestAgentSsh_WithPort_Good(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	transport := NewSSHTransport("host", "user", "key", WithPort("2222"))
	core.AssertEqual(t, "2222", transport.Port)
}

func TestAgentSsh_WithPort_Bad(t *core.T) {
	opt := WithPort("")
	transport := NewSSHTransport("host", "user", "key", opt)
	core.AssertEqual(t, "", transport.Port)
}

func TestAgentSsh_WithPort_Ugly(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	transport := NewSSHTransport("host", "user", "key", WithPort("22"))
	core.AssertEqual(t, "22", transport.Port)
}

func TestAgentSsh_WithTimeout_Good(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	transport := NewSSHTransport("host", "user", "key", WithTimeout(time.Second))
	core.AssertEqual(t, time.Second, transport.Timeout)
}

func TestAgentSsh_WithTimeout_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	transport := NewSSHTransport("host", "user", "key", WithTimeout(0))
	core.AssertEqual(t, time.Duration(0), transport.Timeout)
}

func TestAgentSsh_WithTimeout_Ugly(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	transport := NewSSHTransport("host", "user", "key", WithTimeout(time.Nanosecond))
	core.AssertEqual(t, time.Nanosecond, transport.Timeout)
}

func TestAgentSsh_NewSSHTransport_Good(t *core.T) {
	transport := NewSSHTransport("host", "user", "key")
	core.AssertEqual(t, "host", transport.Host)
	core.AssertEqual(t, "22", transport.Port)
}

func TestAgentSsh_NewSSHTransport_Bad(t *core.T) {
	transport := NewSSHTransport("", "", "")
	core.AssertEqual(t, "", transport.Host)
	core.AssertEqual(t, "", transport.User)
}

func TestAgentSsh_NewSSHTransport_Ugly(t *core.T) {
	transport := NewSSHTransport("host", "user", "key", WithPort("2200"), WithTimeout(time.Millisecond))
	core.AssertEqual(t, "2200", transport.Port)
	core.AssertEqual(t, time.Millisecond, transport.Timeout)
}

func TestAgentSsh_SSHTransport_Run_Good(t *core.T) {
	transport := NewSSHTransport("127.0.0.1", "nobody", "/missing", WithTimeout(time.Millisecond))
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	r := transport.Run(ctx, "true")
	assertResultError(t, r)

	// A non-default port exercises the "-p" branch of sshPortArgs, which the
	// default-port transport above never reaches.
	portTransport := NewSSHTransport("127.0.0.1", "nobody", "/missing", WithPort("2222"), WithTimeout(time.Millisecond))
	ctx2, cancel2 := context.WithTimeout(context.Background(), time.Second)
	defer cancel2()
	r2 := portTransport.Run(ctx2, "true")
	assertResultError(t, r2)
}

func TestAgentSsh_SSHTransport_Run_Bad(t *core.T) {
	transport := NewSSHTransport("", "", "", WithTimeout(time.Millisecond))
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	r := transport.Run(ctx, "true")
	assertResultError(t, r)
}

func TestAgentSsh_SSHTransport_Run_Ugly(t *core.T) {
	transport := &SSHTransport{Host: "127.0.0.1", User: "nobody", KeyPath: "/missing", Timeout: -1}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	r := transport.Run(ctx, "true")
	assertResultError(t, r)
}

func TestAgentSsh_SSHTransport_CopyFrom_Good(t *core.T) {
	transport := NewSSHTransport("127.0.0.1", "nobody", "/missing", WithTimeout(time.Millisecond))
	r := transport.CopyFrom(context.Background(), "/remote/file", core.JoinPath(t.TempDir(), "local"))
	assertResultError(t, r)

	// A non-default port exercises the "-P" branch of commonArgs, which the
	// default-port transport above never reaches.
	portTransport := NewSSHTransport("127.0.0.1", "nobody", "/missing", WithPort("2222"), WithTimeout(time.Millisecond))
	r2 := portTransport.CopyFrom(context.Background(), "/remote/file", core.JoinPath(t.TempDir(), "local"))
	assertResultError(t, r2)
}

func TestAgentSsh_SSHTransport_CopyFrom_Bad(t *core.T) {
	transport := NewSSHTransport("", "", "", WithTimeout(time.Millisecond))
	r := transport.CopyFrom(context.Background(), "", core.JoinPath(t.TempDir(), "local"))
	assertResultError(t, r)
}

func TestAgentSsh_SSHTransport_CopyFrom_Ugly(t *core.T) {
	transport := &SSHTransport{Host: "127.0.0.1", User: "nobody", KeyPath: "/missing", Timeout: -1}
	r := transport.CopyFrom(context.Background(), "/remote/file", core.JoinPath(t.TempDir(), "local"))
	assertResultError(t, r)
}

func TestAgentSsh_SSHTransport_CopyTo_Good(t *core.T) {
	transport := NewSSHTransport("127.0.0.1", "nobody", "/missing", WithTimeout(time.Millisecond))
	r := transport.CopyTo(context.Background(), core.JoinPath(t.TempDir(), "local"), "/remote/file")
	assertResultError(t, r)
}

func TestAgentSsh_SSHTransport_CopyTo_Bad(t *core.T) {
	transport := NewSSHTransport("", "", "", WithTimeout(time.Millisecond))
	r := transport.CopyTo(context.Background(), "", "")
	assertResultError(t, r)
}

func TestAgentSsh_SSHTransport_CopyTo_Ugly(t *core.T) {
	transport := &SSHTransport{Host: "127.0.0.1", User: "nobody", KeyPath: "/missing", Timeout: -1}
	r := transport.CopyTo(context.Background(), core.JoinPath(t.TempDir(), "local"), "/remote/file")
	assertResultError(t, r)
}

func TestAgentSsh_SSHCommand_Good(t *core.T) {
	ft := newFakeTransport()
	ft.On("echo ok", "ok\n", nil)
	r := SSHCommand(&AgentConfig{Transport: ft}, "echo ok")
	requireResultOK(t, r)
	out := r.Value.(string)
	core.AssertEqual(t, "ok\n", out)
}

func TestAgentSsh_SSHCommand_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	r := SSHCommand(&AgentConfig{Transport: newFakeTransport()}, "missing")
	assertResultError(t, r)
}

func TestAgentSsh_SSHCommand_Ugly(t *core.T) {
	ft := newFakeTransport()
	ft.On("fail", "", core.AnError)
	r := SSHCommand(&AgentConfig{Transport: ft}, "fail")
	assertResultError(t, r)
}

func TestAgentSsh_SCPFrom_Good(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	r := SCPFrom(&AgentConfig{Transport: newFakeTransport()}, "/remote", core.JoinPath(t.TempDir(), "local"))
	assertResultOK(t, r)
}

func TestAgentSsh_SCPFrom_Bad(t *core.T) {
	ft := newFakeTransport()
	r := SCPFrom(&AgentConfig{Transport: ft}, "", "")
	assertResultOK(t, r)
}

func TestAgentSsh_SCPFrom_Ugly(t *core.T) {
	cfg := &AgentConfig{Transport: newFakeTransport()}
	r := SCPFrom(cfg, "/remote", core.JoinPath(t.TempDir(), "local"))
	assertResultOK(t, r)
}

func TestAgentSsh_SCPTo_Good(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	r := SCPTo(&AgentConfig{Transport: newFakeTransport()}, core.JoinPath(t.TempDir(), "local"), "/remote")
	assertResultOK(t, r)
}

func TestAgentSsh_SCPTo_Bad(t *core.T) {
	ft := newFakeTransport()
	r := SCPTo(&AgentConfig{Transport: ft}, "", "")
	assertResultOK(t, r)
}

func TestAgentSsh_SCPTo_Ugly(t *core.T) {
	cfg := &AgentConfig{Transport: newFakeTransport()}
	r := SCPTo(cfg, core.JoinPath(t.TempDir(), "local"), "/remote")
	assertResultOK(t, r)
}

func TestAgentSsh_EnvOr_Good(t *core.T) {
	t.Setenv("ML_TEST_ENV", "value")
	got := EnvOr("ML_TEST_ENV", "fallback")
	core.AssertEqual(t, "value", got)
}

func TestAgentSsh_EnvOr_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	got := EnvOr("ML_TEST_MISSING", "fallback")
	core.AssertEqual(t, "fallback", got)
}

func TestAgentSsh_EnvOr_Ugly(t *core.T) {
	t.Setenv("ML_TEST_EMPTY", "")
	got := EnvOr("ML_TEST_EMPTY", "fallback")
	core.AssertEqual(t, "fallback", got)
}

func TestAgentSsh_IntEnvOr_Good(t *core.T) {
	t.Setenv("ML_TEST_INT", "7")
	got := IntEnvOr("ML_TEST_INT", 1)
	core.AssertEqual(t, 7, got)
}

func TestAgentSsh_IntEnvOr_Bad(t *core.T) {
	t.Setenv("ML_TEST_INT_BAD", "not-number")
	got := IntEnvOr("ML_TEST_INT_BAD", 3)
	core.AssertEqual(t, 3, got)

	// Unset entirely (empty core.Env read) short-circuits before strconv.Atoi.
	got2 := IntEnvOr("ML_TEST_INT_UNSET", 9)
	core.AssertEqual(t, 9, got2)
}

func TestAgentSsh_IntEnvOr_Ugly(t *core.T) {
	t.Setenv("ML_TEST_INT_ZERO", "0")
	got := IntEnvOr("ML_TEST_INT_ZERO", 5)
	core.AssertEqual(t, 5, got)
}

func TestAgentSsh_ExpandHome_Good(t *core.T) {
	t.Setenv("DIR_HOME", "/home/tester")
	got := ExpandHome("~/models")
	core.AssertContains(t, got, "models")
}

func TestAgentSsh_ExpandHome_Bad(t *core.T) {
	t.Setenv("DIR_HOME", "")
	got := ExpandHome("~/models")
	core.AssertContains(t, got, "models")
}

func TestAgentSsh_ExpandHome_Ugly(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	got := ExpandHome("/absolute/models")
	core.AssertEqual(t, "/absolute/models", got)
}

func TestAgentSsh_fileBase_Good(t *core.T) {
	got := fileBase("/data/adapters-27b/0001000_adapters.safetensors")
	core.AssertEqual(t, "0001000_adapters.safetensors", got)
}

func TestAgentSsh_fileBase_Bad(t *core.T) {
	got := fileBase("bare.safetensors")
	core.AssertEqual(t, "bare.safetensors", got)
}

func TestAgentSsh_fileBase_Ugly(t *core.T) {
	// Windows-style separators are normalised to "/" before basename
	// extraction.
	got := fileBase(`C:\data\adapters-27b\0001000_adapters.safetensors`)
	core.AssertEqual(t, "0001000_adapters.safetensors", got)
}
