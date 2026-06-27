// SPDX-Licence-Identifier: EUPL-1.2

package datapipe

import (
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/store"
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

func newFakeInflux(t testing.TB, queries map[string][]map[string]any, writeStatus int) (*InfluxClient, *fakeInfluxRecorder) {
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
	return &InfluxClient{url: server.URL, db: "test"}, rec
}

func (r *fakeInfluxRecorder) writeCount() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.writes)
}

func newTestDB(t testing.TB) *DB {
	t.Helper()
	rDB := OpenDBReadWrite(core.JoinPath(t.TempDir(), "test.duckdb"))
	requireResultOK(t, rDB)
	db := rDB.Value.(*DB)
	t.Cleanup(func() { _ = db.Close() })
	return db
}

func newStoreDuckDB(t testing.TB) *store.DuckDB {
	t.Helper()
	db, err := store.OpenDuckDBReadWrite(core.JoinPath(t.TempDir(), "store.duckdb"))
	requireResultOK(t, err)
	t.Cleanup(func() { _ = db.Close() })
	return db
}

func seedGoldenStoreDB(t *core.T) *store.DuckDB {
	t.Helper()
	db := newStoreDuckDB(t)
	requireResultOK(t, db.Exec(`CREATE TABLE golden_set (
		idx INTEGER, seed_id VARCHAR, domain VARCHAR, voice VARCHAR,
		gen_time DOUBLE, char_count INTEGER
	)`))
	return db
}

func toInt(v any) int {
	switch n := v.(type) {
	case int64:
		return int(n)
	case int32:
		return int(n)
	case float64:
		return int(n)
	default:
		return 0
	}
}
