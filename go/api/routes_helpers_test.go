// SPDX-Licence-Identifier: EUPL-1.2

package api_test

import (
	"testing"

	core "dappco.re/go"
)

func mustJSONUnmarshalBytes(t testing.TB, data []byte, out any) {
	t.Helper()
	if r := core.JSONUnmarshal(data, out); !r.OK {
		t.Fatalf("unmarshal error: %v", r.Value.(error))
	}
}
