// SPDX-Licence-Identifier: EUPL-1.2

package score

import (
	"context"
	"io"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/serving"
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

func mustWriteJSONResponse(t testing.TB, w io.Writer, v any) {
	t.Helper()
	if _, err := io.WriteString(w, core.JSONMarshalString(v)); err != nil {
		t.Fatalf("write json response: %v", err)
	}
}

func newStoreDuckDB(t testing.TB) *store.DuckDB {
	t.Helper()
	rOpen := store.OpenDuckDBReadWrite(core.JoinPath(t.TempDir(), "store.duckdb"))
	requireResultOK(t, rOpen)
	db := rOpen.Value.(*store.DuckDB)
	t.Cleanup(func() { _ = db.Close() })
	return db
}

// testBackend is a fake serving.Backend for exercising the Judge and Engine
// without a live model.
type testBackend struct {
	name      string
	available bool
	result    serving.Result
	err       error
}

var _ serving.Backend = (*testBackend)(nil)

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
