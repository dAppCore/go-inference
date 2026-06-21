package lab

import (
	core "dappco.re/go"
)

// --- AX-7 canonical triplets ---

func TestCmd_AddLabCommands_Good(t *core.T) {
	root := core.New()
	r := AddLabCommands(root)
	cmd := root.Command("lab")

	core.AssertTrue(t, r.OK)
	core.AssertTrue(t, cmd.OK)
	core.AssertEqual(t, "lab", cmd.Value.(*core.Command).Name)
}

func TestCmd_AddLabCommands_Bad(t *core.T) {
	root := core.New()
	AddLabCommands(root)
	AddLabCommands(root)

	core.AssertLen(t, root.Commands(), 2)
	core.AssertEqual(t, "lab", root.Commands()[0])
}

func TestCmd_AddLabCommands_Ugly(t *core.T) {
	root := core.New()
	root.Command("lab", core.Command{Description: "pre-existing"})
	AddLabCommands(root)

	core.AssertLen(t, root.Commands(), 2)
	core.AssertEqual(t, "lab", root.Commands()[0])
}

func TestCmd_RunServe_Good(t *core.T) {
	t.Setenv("CORE_LAB_API_TOKEN", "")
	r := RunServe(CommandOptions{Bind: "0.0.0.0:8080"})
	got := r.Error()

	core.AssertFalse(t, r.OK)
	core.AssertContains(t, got, "non-loopback")
}

func TestCmd_RunServe_Bad(t *core.T) {
	t.Setenv("CORE_LAB_API_TOKEN", "")
	r := RunServe(CommandOptions{Bind: "127.0.0.1:8080", AllowRemote: true})
	got := r.Error()

	core.AssertFalse(t, r.OK)
	core.AssertContains(t, got, "CORE_LAB_API_TOKEN")
}

func TestCmd_RunServe_Ugly(t *core.T) {
	t.Setenv("CORE_LAB_API_TOKEN", "")
	r := RunServe(CommandOptions{Bind: "not-a-host", AllowRemote: false})
	got := r.Error()

	core.AssertFalse(t, r.OK)
	core.AssertContains(t, got, "non-loopback")
}

func TestCmd_ValidateBindAddress_Good(t *core.T) {
	r := ValidateBindAddress("127.0.0.1:8080", false)
	got := IsLoopbackBindAddress("127.0.0.1:8080")
	want := true

	core.AssertTrue(t, r.OK)
	core.AssertEqual(t, want, got)
}

func TestCmd_ValidateBindAddress_Bad(t *core.T) {
	r := ValidateBindAddress("0.0.0.0:8080", false)
	got := r.Error()
	want := "non-loopback"

	core.AssertFalse(t, r.OK)
	core.AssertContains(t, got, want)
}

func TestCmd_ValidateBindAddress_Ugly(t *core.T) {
	r := ValidateBindAddress(":8080", true)
	got := IsLoopbackBindAddress(":8080")
	want := false

	core.AssertTrue(t, r.OK)
	core.AssertEqual(t, want, got)
}

func TestCmd_IsLoopbackBindAddress_Good(t *core.T) {
	got := IsLoopbackBindAddress("localhost:8080")
	ipv4 := IsLoopbackBindAddress("127.0.0.1:8080")
	ipv6 := IsLoopbackBindAddress("[::1]:8080")

	core.AssertTrue(t, got)
	core.AssertTrue(t, ipv4)
	core.AssertTrue(t, ipv6)
}

func TestCmd_IsLoopbackBindAddress_Bad(t *core.T) {
	got := IsLoopbackBindAddress("0.0.0.0:8080")
	wildcard := IsLoopbackBindAddress(":8080")
	remote := IsLoopbackBindAddress("example.com:8080")

	core.AssertFalse(t, got)
	core.AssertFalse(t, wildcard)
	core.AssertFalse(t, remote)
}

func TestCmd_IsLoopbackBindAddress_Ugly(t *core.T) {
	empty := IsLoopbackBindAddress("")
	malformed := IsLoopbackBindAddress("::notanaddr:8080")
	missingPort := IsLoopbackBindAddress("localhost")

	core.AssertFalse(t, empty)
	core.AssertFalse(t, malformed)
	core.AssertFalse(t, missingPort)
}

func TestCmd_ValidateRemoteAuth_Good(t *core.T) {
	r := ValidateRemoteAuth(false, "")
	remote := ValidateRemoteAuth(true, "token")
	want := true

	core.AssertTrue(t, r.OK)
	core.AssertTrue(t, remote.OK)
	core.AssertTrue(t, want)
}

func TestCmd_ValidateRemoteAuth_Bad(t *core.T) {
	r := ValidateRemoteAuth(true, "")
	got := r.Error()
	want := "CORE_LAB_API_TOKEN"

	core.AssertFalse(t, r.OK)
	core.AssertContains(t, got, want)
}

func TestCmd_ValidateRemoteAuth_Ugly(t *core.T) {
	r := ValidateRemoteAuth(true, "  ")
	got := r.Error()
	want := "--allow-remote"

	core.AssertFalse(t, r.OK)
	core.AssertContains(t, got, want)
}
