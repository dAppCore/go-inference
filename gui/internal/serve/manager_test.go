// SPDX-Licence-Identifier: EUPL-1.2

package serve

import (
	core "dappco.re/go"
)

// fakeLem writes an executable stand-in for the lem binary to a temp dir: it
// ignores its args and sleeps, so Start spawns a live child that a default
// SIGTERM (from Stop) terminates. Returns the binary path.
func fakeLem(t *core.T) string {
	dir := t.TempDir()
	path := core.PathJoin(dir, "lem")
	if r := core.WriteFile(path, []byte("#!/bin/sh\nsleep 300\n"), 0o755); !r.OK {
		t.Fatal("write fake lem: " + r.Error())
	}
	return path
}

func TestManager_NewManager_Good(t *core.T) {
	m := NewManager("lem", ":36911")

	core.AssertEqual(t, "lem", m.binary)
	core.AssertEqual(t, ":36911", m.Addr())
}

func TestManager_NewManager_Bad(t *core.T) {
	m := NewManager("", "")

	core.AssertEqual(t, "lem", m.binary)
	core.AssertEqual(t, ":36911", m.Addr())
}

func TestManager_NewManager_Ugly(t *core.T) {
	m := NewManager("/opt/lem/bin/lem", ":0")

	core.AssertEqual(t, "/opt/lem/bin/lem", m.binary)
	core.AssertEqual(t, ":0", m.Addr())
}

func TestManager_Manager_Start_Good(t *core.T) {
	m := NewManager(fakeLem(t), ":0")
	defer m.Stop()

	r := m.Start("")

	core.AssertTrue(t, r.OK)
	core.AssertTrue(t, m.Managed())
}

func TestManager_Manager_Start_Bad(t *core.T) {
	m := NewManager("/nonexistent/lem-binary", ":0")

	r := m.Start("")

	core.AssertFalse(t, r.OK)
	core.AssertFalse(t, m.Managed())
	core.AssertContains(t, r.Error(), "serve")
}

func TestManager_Manager_Start_Ugly(t *core.T) {
	m := NewManager(fakeLem(t), ":0")
	defer m.Stop()

	first := m.Start("")
	second := m.Start("/models/other") // already managing → idempotent no-op

	core.AssertTrue(t, first.OK)
	core.AssertTrue(t, second.OK)
	core.AssertTrue(t, m.Managed())
}

func TestManager_Manager_Stop_Good(t *core.T) {
	m := NewManager(fakeLem(t), ":0")
	m.Start("")

	r := m.Stop()

	core.AssertTrue(t, r.OK)
	core.AssertFalse(t, m.Managed())
}

func TestManager_Manager_Stop_Bad(t *core.T) {
	m := NewManager("lem", ":0") // nothing spawned

	r := m.Stop()

	core.AssertTrue(t, r.OK)
	core.AssertFalse(t, m.Managed())
}

func TestManager_Manager_Stop_Ugly(t *core.T) {
	m := NewManager(fakeLem(t), ":0")
	m.Start("")

	first := m.Stop()
	second := m.Stop() // double stop → no-op

	core.AssertTrue(t, first.OK)
	core.AssertTrue(t, second.OK)
	core.AssertFalse(t, m.Managed())
}

func TestManager_Manager_Managed_Good(t *core.T) {
	m := NewManager(fakeLem(t), ":0")
	defer m.Stop()
	m.Start("")

	core.AssertTrue(t, m.Managed())
}

func TestManager_Manager_Managed_Bad(t *core.T) {
	m := NewManager("lem", ":0")

	core.AssertFalse(t, m.Managed())
}

func TestManager_Manager_Managed_Ugly(t *core.T) {
	m := NewManager(fakeLem(t), ":0")
	m.Start("")
	m.Stop()

	core.AssertFalse(t, m.Managed())
}
