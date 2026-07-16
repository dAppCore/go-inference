package modelmgmt

import (
	"net/http"
	"net/http/httptest"

	"dappco.re/go"
	coreio "dappco.re/go/io"
)

// hfTestServer stands up a fake HuggingFace Hub upload endpoint and repoints
// hfHubBase at it for the duration of the test. status is the HTTP code the
// upload PUT returns; a >=300 code exercises the failure branch.
func hfTestServer(t *core.T, status int) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPut {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		if status >= 300 {
			w.WriteHeader(status)
			_, _ = w.Write([]byte("upload rejected"))
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	prev := hfHubBase
	hfHubBase = server.URL
	t.Cleanup(func() {
		hfHubBase = prev
		server.Close()
	})
	return server
}

func TestPublish_Publish_Good(t *core.T) {
	dir := t.TempDir()
	core.RequireNoError(t, coreio.Local.Write(core.JoinPath(dir, "train.parquet"), "data"))
	buf := core.NewBuffer(nil)
	requireResultOK(t, Publish(PublishConfig{InputDir: dir, Repo: "owner/repo", DryRun: true, Public: true}, buf))
	core.AssertContains(t, buf.String(), "Dry run")
}

func TestPublish_Publish_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	assertResultError(t, Publish(PublishConfig{}, core.NewBuffer(nil)))
}

// TestPublish_Publish_Ugly covers both an empty directory and one holding a
// non-Parquet file — collectUploadFiles matches by exact filename
// (train/valid/test.parquet), so an unrelated file does not satisfy it.
func TestPublish_Publish_Ugly(t *core.T) {
	dir := t.TempDir()
	assertResultError(t, Publish(PublishConfig{InputDir: dir, Repo: "owner/repo", DryRun: true}, core.NewBuffer(nil)), "no Parquet")
	core.RequireNoError(t, coreio.Local.Write(core.JoinPath(dir, "notes.txt"), "not parquet"))
	assertResultError(t, Publish(PublishConfig{InputDir: dir, Repo: "owner/repo", DryRun: true}, core.NewBuffer(nil)), "no Parquet")
}

// TestPublish_Publish_Live_Good drives the non-dry-run upload path end to end
// against a fake Hub, exercising the loop that dry-run skips.
func TestPublish_Publish_Live_Good(t *core.T) {
	hfTestServer(t, http.StatusOK)
	dir := t.TempDir()
	core.RequireNoError(t, coreio.Local.Write(core.JoinPath(dir, "train.parquet"), "data"))
	buf := core.NewBuffer(nil)
	requireResultOK(t, Publish(PublishConfig{InputDir: dir, Repo: "owner/repo", Token: "tok"}, buf))
	core.AssertContains(t, buf.String(), "Published to")
}

// TestPublish_Publish_Live_Ugly makes the Hub reject the upload so the live
// path returns the wrapped upload failure.
func TestPublish_Publish_Live_Ugly(t *core.T) {
	hfTestServer(t, http.StatusInternalServerError)
	dir := t.TempDir()
	core.RequireNoError(t, coreio.Local.Write(core.JoinPath(dir, "train.parquet"), "data"))
	assertResultError(t, Publish(PublishConfig{InputDir: dir, Repo: "owner/repo", Token: "tok"}, core.NewBuffer(nil)), "upload")
}

func TestPublish_uploadFileToHF_Good(t *core.T) {
	hfTestServer(t, http.StatusOK)
	dir := t.TempDir()
	local := core.JoinPath(dir, "train.parquet")
	core.RequireNoError(t, coreio.Local.Write(local, "data"))
	requireResultOK(t, uploadFileToHF("tok", "owner/repo", local, "data/train.parquet"))
}

// TestPublish_uploadFileToHF_Bad covers two distinct local-read failures: a
// missing file and a path that is a directory, not a file.
func TestPublish_uploadFileToHF_Bad(t *core.T) {
	hfTestServer(t, http.StatusOK)
	assertResultError(t, uploadFileToHF("tok", "owner/repo", core.JoinPath(t.TempDir(), "missing.parquet"), "data/train.parquet"), "read")
	assertResultError(t, uploadFileToHF("tok", "owner/repo", t.TempDir(), "data/train.parquet"), "read")
}

func TestPublish_uploadFileToHF_Ugly(t *core.T) {
	hfTestServer(t, http.StatusForbidden)
	dir := t.TempDir()
	local := core.JoinPath(dir, "train.parquet")
	core.RequireNoError(t, coreio.Local.Write(local, "data"))
	assertResultError(t, uploadFileToHF("tok", "owner/repo", local, "data/train.parquet"), "HTTP 403")
}

// TestPublish_resolveHFToken_Good documents that an explicit token always
// wins, even when the HF_TOKEN env fallback is also set to something else.
func TestPublish_resolveHFToken_Good(t *core.T) {
	core.AssertEqual(t, "explicit-tok", resolveHFToken("explicit-tok"))
	t.Setenv("HF_TOKEN", "env-tok")
	core.AssertEqual(t, "explicit-tok", resolveHFToken("explicit-tok"))
}

func TestPublish_resolveHFToken_Bad(t *core.T) {
	t.Setenv("HF_TOKEN", "")
	t.Setenv("DIR_HOME", "")
	core.AssertEqual(t, "", resolveHFToken(""))
}

func TestPublish_resolveHFToken_Ugly(t *core.T) {
	// With no explicit token, HF_TOKEN env is consulted next.
	t.Setenv("HF_TOKEN", "env-tok")
	core.AssertEqual(t, "env-tok", resolveHFToken(""))

	// Unlike the ~/.huggingface/token fallback (core.Trim'd), the HF_TOKEN
	// env value is returned exactly as read — no whitespace trimming.
	t.Setenv("HF_TOKEN", "  spaced-tok  ")
	core.AssertEqual(t, "  spaced-tok  ", resolveHFToken(""))

	// The ~/.huggingface/token fallback is an environment boundary: core.Env
	// reads a pre-populated DIR_HOME cache that Setenv cannot override, so
	// reaching that branch would mean writing to the real home dir. Left
	// uncovered deliberately.
}
