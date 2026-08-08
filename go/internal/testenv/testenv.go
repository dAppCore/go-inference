// SPDX-Licence-Identifier: EUPL-1.2

// Package testenv sets the environment a test wants to isolate, in the form
// each platform actually reads.
//
// It exists because "set HOME to a temp dir" is a POSIX-only idiom that fails
// SILENTLY elsewhere — the isolation looks applied, the test carries on, and
// the assertions run against the real machine.
package testenv

import (
	"runtime"
	"testing"
)

// SetHome points the user's home directory at dir for the duration of the
// test, restoring the previous value on cleanup.
//
// os.UserHomeDir reads $HOME on POSIX but %USERPROFILE% on Windows, so a test
// setting only HOME keeps resolving the runner's real profile there. That is
// worse than an outright failure: an "isolated" test then asserts against
// C:\Users\runneradmin, so a case expecting no resolvable home finds one, and
// a case expecting a file beneath the temp home looks for it in the wrong
// place entirely.
//
// Pass "" to make the home directory unresolvable, which is how the
// no-home-directory failure paths are exercised.
func SetHome(t *testing.T, dir string) {
	t.Helper()
	t.Setenv("HOME", dir)
	if runtime.GOOS == "windows" {
		// USERPROFILE is what os.UserHomeDir consults; HOME is set alongside
		// it because code reading the variable directly still expects it.
		t.Setenv("USERPROFILE", dir)
	}
}
