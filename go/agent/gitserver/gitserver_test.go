// SPDX-License-Identifier: EUPL-1.2

package gitserver

import (
	"testing"
	"time"

	core "dappco.re/go"
)

func TestGitserver_DefaultOptions_Good(t *testing.T) {
	dataPath := t.TempDir()
	result := DefaultOptions(dataPath)
	core.AssertTrue(t, result.OK, result.Error())
	options := result.Value.(Options)
	core.AssertEqual(t, dataPath, options.DataPath)
	core.AssertEqual(t, "127.0.0.1:23231", options.ListenAddress)
	core.AssertEqual(t, "ssh://127.0.0.1:23231", options.PublicURL)
	core.AssertTrue(t, options.ShutdownTimeout > 0)
	core.AssertTrue(t, options.PID() > 0)
}

func TestGitserver_DefaultOptions_Bad(t *testing.T) {
	result := DefaultOptions("relative/soft-serve")
	core.AssertFalse(t, result.OK)
	core.AssertContains(t, result.Error(), "absolute")
}

func TestGitserver_DefaultOptions_Ugly(t *testing.T) {
	core.AssertFalse(t, DefaultOptions("").OK)
	core.AssertFalse(t, DefaultOptions(" \t ").OK)
	core.AssertFalse(t, DefaultOptions(".").OK)
}

func TestGitserverOptionsIsolation(t *testing.T) {
	result := DefaultOptions(t.TempDir())
	core.AssertTrue(t, result.OK, result.Error())
	options := result.Value.(Options)
	core.AssertEqual(t, 5*time.Second, options.ShutdownTimeout)
	core.AssertFalse(t, options.ProcessAlive(0))
}
