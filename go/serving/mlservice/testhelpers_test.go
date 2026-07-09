// SPDX-Licence-Identifier: EUPL-1.2

package mlservice

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/eval/datapipe"
	"dappco.re/go/inference/serving"
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

type fakeInfluxRecorder struct {
	mu     sync.Mutex
	writes []string
}

func (r *fakeInfluxRecorder) writeCount() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.writes)
}

func newFakeInflux(t testing.TB, queries map[string][]map[string]any, writeStatus int) (*datapipe.InfluxClient, *fakeInfluxRecorder) {
	t.Helper()
	rec := &fakeInfluxRecorder{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/v3/write_lp":
			rBody := readAll(r.Body)
			body := []byte{}
			if rBody.OK {
				body = rBody.Value.([]byte)
			}
			rec.mu.Lock()
			rec.writes = append(rec.writes, string(body))
			rec.mu.Unlock()
			if writeStatus == 0 {
				w.WriteHeader(http.StatusNoContent)
				return
			}
			w.WriteHeader(writeStatus)
		case "/api/v3/query_sql":
			rBody := readAll(r.Body)
			body := []byte{}
			if rBody.OK {
				body = rBody.Value.([]byte)
			}
			sql := string(body)
			rows := []map[string]any{}
			for key, value := range queries {
				if core.Contains(sql, key) {
					rows = value
					break
				}
			}
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(core.JSONMarshalString(rows)))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	t.Cleanup(server.Close)
	return datapipe.NewInfluxClient(server.URL, "test"), rec
}

// testBackend is a fake serving.Backend for exercising the service facade.
type testBackend struct {
	name      string
	available bool
	result    serving.Result
	err       error
}

func (b *testBackend) Name() string {
	if b.name == "" {
		return "test"
	}
	return b.name
}

func (b *testBackend) Available() bool { return b.available }

func (b *testBackend) Generate(_ context.Context, prompt string, _ serving.GenOpts) core.Result {
	if b.err != nil {
		return core.Fail(b.err)
	}
	if b.result.Text != "" {
		return core.Ok(b.result)
	}
	return core.Ok(serving.Result{Text: prompt})
}

func (b *testBackend) Chat(_ context.Context, messages []serving.Message, _ serving.GenOpts) core.Result {
	if b.err != nil {
		return core.Fail(b.err)
	}
	if b.result.Text != "" {
		return core.Ok(b.result)
	}
	if len(messages) == 0 {
		return core.Ok(serving.Result{})
	}
	return core.Ok(serving.Result{Text: messages[len(messages)-1].Content})
}
