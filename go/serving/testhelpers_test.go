// SPDX-Licence-Identifier: EUPL-1.2

package serving

import (
	"io"
	"net/http"
	"testing"

	core "dappco.re/go"
)

func requireResultOK(t testing.TB, r core.Result) {
	t.Helper()
	if !r.OK {
		t.Fatalf("unexpected result error: %s", r.Error())
	}
}

func assertResultOK(t testing.TB, r core.Result) {
	t.Helper()
	if !r.OK {
		t.Errorf("unexpected result error: %s", r.Error())
	}
}

func assertResultError(t testing.TB, r core.Result, contains ...string) {
	t.Helper()
	if r.OK {
		t.Fatalf("expected result error, got OK value %#v", r.Value)
	}
	if len(contains) > 0 && contains[0] != "" && !core.Contains(r.Error(), contains[0]) {
		t.Fatalf("expected result error containing %q, got %q", contains[0], r.Error())
	}
}

func mustJSONUnmarshalBytes(t testing.TB, data []byte, out any) {
	t.Helper()
	if r := core.JSONUnmarshal(data, out); !r.OK {
		t.Fatalf("unmarshal error: %v", r.Value.(error))
	}
}

func mustReadJSONRequest(t testing.TB, r *http.Request, out any) {
	t.Helper()
	body, err := io.ReadAll(r.Body)
	if err != nil {
		t.Fatalf("read request body: %v", err)
	}
	mustJSONUnmarshalBytes(t, body, out)
}

func mustWriteJSONResponse(t testing.TB, w io.Writer, v any) {
	t.Helper()
	if _, err := io.WriteString(w, core.JSONMarshalString(v)); err != nil {
		t.Fatalf("write json response: %v", err)
	}
}
