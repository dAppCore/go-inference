// SPDX-Licence-Identifier: EUPL-1.2

package modelmgmt

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/eval/datapipe"
	"dappco.re/go/inference/serving"
	coreio "dappco.re/go/io"
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

func mustJSONUnmarshalString(t testing.TB, data string, out any) {
	t.Helper()
	if r := core.JSONUnmarshalString(data, out); !r.OK {
		t.Fatalf("unmarshal error: %v", r.Value.(error))
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

func newTestDB(t testing.TB) *datapipe.DB {
	t.Helper()
	rDB := datapipe.OpenDBReadWrite(core.JoinPath(t.TempDir(), "test.duckdb"))
	requireResultOK(t, rDB)
	db := rDB.Value.(*datapipe.DB)
	t.Cleanup(func() { _ = db.Close() })
	return db
}

func seedMLDB(t *core.T) *datapipe.DB {
	t.Helper()
	db := newTestDB(t)
	requireResultOK(t, db.Exec(`CREATE TABLE golden_set (
		idx INTEGER, seed_id VARCHAR, domain VARCHAR, voice VARCHAR,
		prompt VARCHAR, response VARCHAR, gen_time DOUBLE, char_count INTEGER
	)`))
	requireResultOK(t, db.Exec(`INSERT INTO golden_set VALUES (1,'s1','ethics','calm','p','long response',1.5,13)`))
	requireResultOK(t, db.Exec(`CREATE TABLE expansion_prompts (
		idx BIGINT, seed_id VARCHAR, region VARCHAR, domain VARCHAR, language VARCHAR,
		prompt VARCHAR, prompt_en VARCHAR, priority INTEGER, status VARCHAR
	)`))
	requireResultOK(t, db.Exec(`INSERT INTO expansion_prompts VALUES (1,'s1','en','ethics','en','p','',2,'pending')`))
	return db
}

func writeSafetensorsFixture(t testing.TB) (string, string) {
	t.Helper()
	dir := t.TempDir()
	key := "model.layers.0.self_attn.q_proj.lora_a"
	sf := core.JoinPath(dir, "adapter_model.safetensors")
	cfg := core.JoinPath(dir, "adapter_config.json")
	tensors := map[string]SafetensorsTensorInfo{
		key: {Dtype: "F32", Shape: []int{1, 1}},
	}
	data := map[string][]byte{
		key: {1, 2, 3, 4},
	}
	requireResultOK(t, WriteSafetensors(sf, tensors, data))
	core.RequireNoError(t, coreio.Local.Write(cfg, `{"lora_parameters":{"rank":2,"scale":3,"dropout":0.1}}`))
	return sf, cfg
}

// testBackend is a fake serving.Backend for exercising the expansion path.
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
