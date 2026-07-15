// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"
)

func TestMain(m *testing.M) {
	if os.Getenv(MetallibPathEnv) == "" {
		os.Exit(0)
	}
	os.Exit(m.Run())
}
