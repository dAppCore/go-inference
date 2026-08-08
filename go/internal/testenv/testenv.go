// SPDX-Licence-Identifier: EUPL-1.2

// Package testenv sets the environment a test wants to isolate, in the form
// each platform actually reads.
//
// It exists because "set HOME to a temp dir" is a POSIX-only idiom that fails
// SILENTLY elsewhere — the isolation looks applied, the test carries on, and
// the assertions run against the real machine.
package testenv

import (
	"os"
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
// It takes a testing.TB so benchmarks get the same isolation as tests; a
// benchmark reading the real home directory is as wrong as a test doing it.
func SetHome(tb testing.TB, dir string) {
	tb.Helper()
	tb.Setenv("HOME", dir)
	if runtime.GOOS == "windows" {
		// USERPROFILE is what os.UserHomeDir consults; HOME is set alongside
		// it because code reading the variable directly still expects it.
		tb.Setenv("USERPROFILE", dir)
	}
}

// SetHomeDir is SetHome for callers with no testing.TB — Example functions,
// which the testing package runs without one.
//
// Examples were the gap in the first sweep of this: it replaced every
// t.Setenv("HOME", …) and left the core.Setenv form untouched, so the
// isolation stayed broken in exactly the files that double as documentation.
//
// Returns a function restoring the previous values, for defer.
func SetHomeDir(dir string) func() {
	previousHome, hadHome := os.LookupEnv("HOME")
	previousProfile, hadProfile := os.LookupEnv("USERPROFILE")
	setOrUnset("HOME", dir, true)
	if runtime.GOOS == "windows" {
		setOrUnset("USERPROFILE", dir, true)
	}
	return func() {
		setOrUnset("HOME", previousHome, hadHome)
		if runtime.GOOS == "windows" {
			setOrUnset("USERPROFILE", previousProfile, hadProfile)
		}
	}
}

// setOrUnset restores a variable to exactly its previous state, including
// having been absent — setting it to "" instead would leave a home directory
// that resolves to the empty string rather than one that does not resolve.
func setOrUnset(name, value string, present bool) {
	if !present {
		_ = os.Unsetenv(name)
		return
	}
	_ = os.Setenv(name, value)
}
