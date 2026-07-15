package datapipe

import (
	"net/http"
	"net/http/httptest"

	"dappco.re/go"
)

func TestInflux_NewInfluxClient_Good(t *core.T) {
	client := NewInfluxClient("http://127.0.0.1:8181", "db")
	core.AssertEqual(t, "http://127.0.0.1:8181", client.url)
	core.AssertEqual(t, "db", client.db)
}

func TestInflux_NewInfluxClient_Bad(t *core.T) {
	client := NewInfluxClient("", "")
	core.AssertEqual(t, "training", client.db)
	core.AssertContains(t, client.url, "http")
}

func TestInflux_NewInfluxClient_Ugly(t *core.T) {
	client := NewInfluxClient("http://127.0.0.1:1", "edge")
	core.AssertEqual(t, "edge", client.db)
	core.AssertEqual(t, "http://127.0.0.1:1", client.url)
}

func TestInflux_InfluxClient_WriteLp_Good(t *core.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusNoContent) }))
	defer srv.Close()
	client := NewInfluxClient(srv.URL, "db")
	assertResultOK(t, client.WriteLp([]string{"m value=1i"}))
}

func TestInflux_InfluxClient_WriteLp_Bad(t *core.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusBadRequest) }))
	defer srv.Close()
	client := NewInfluxClient(srv.URL, "db")
	assertResultError(t, client.WriteLp([]string{"bad"}))
}

func TestInflux_InfluxClient_WriteLp_Ugly(t *core.T) {
	client := NewInfluxClient("http://127.0.0.1:1", "db")
	err := client.WriteLp(nil)
	core.AssertError(t, err)
	core.AssertEqual(t, "db", client.db)
}

func TestInflux_InfluxClient_QuerySQL_Good(t *core.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { core.WriteString(w, `[{"n":1}]`) }))
	defer srv.Close()
	client := NewInfluxClient(srv.URL, "db")
	rRows := client.QuerySQL("select 1")
	requireResultOK(t, rRows)
	rows := rRows.Value.([]map[string]any)
	core.AssertLen(t, rows, 1)
}

func TestInflux_InfluxClient_QuerySQL_Bad(t *core.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusInternalServerError) }))
	defer srv.Close()
	client := NewInfluxClient(srv.URL, "db")
	rRows := client.QuerySQL("select 1")
	assertResultError(t, rRows)
}

func TestInflux_InfluxClient_QuerySQL_Ugly(t *core.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { core.WriteString(w, `not rows`) }))
	defer srv.Close()
	client := NewInfluxClient(srv.URL, "db")
	rRows := client.QuerySQL("")
	assertResultError(t, rRows)
}

func TestInflux_EscapeLp_Good(t *core.T) {
	got := EscapeLp("a b,c=d")
	core.AssertEqual(t, `a\ b\,c\=d`, got)
	core.AssertContains(t, got, `\ `)
}

func TestInflux_EscapeLp_Bad(t *core.T) {
	got := EscapeLp("")
	core.AssertEqual(t, "", got)
	core.AssertLen(t, got, 0)
}

func TestInflux_EscapeLp_Ugly(t *core.T) {
	got := EscapeLp("plain")
	core.AssertEqual(t, "plain", got)
	core.AssertNotContains(t, got, `\`)
}
