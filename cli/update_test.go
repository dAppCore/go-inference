// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"archive/zip"
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"runtime"
	"testing"

	core "dappco.re/go"
	updater "dappco.re/go/update"
)

// --- deriveUpdateChannel -----------------------------------------------

func TestDeriveUpdateChannel_Good(t *testing.T) {
	for _, tc := range []struct{ version, want string }{
		{"v1.2.3", "stable"},
		{"1.2.3", "stable"},
		{"v1.2.3-alpha.2", "alpha"},
		{"v1.2.3-beta.1", "beta"},
	} {
		if got := deriveUpdateChannel(tc.version); got != tc.want {
			t.Errorf("deriveUpdateChannel(%q) = %q, want %q", tc.version, got, tc.want)
		}
	}
}

func TestDeriveUpdateChannel_Bad(t *testing.T) {
	// The rolling shapes: a bare "dev" fallback and a real dev-<sha12> build.
	for _, tc := range []struct{ version, want string }{
		{"dev", "dev"},
		{"dev-c65fa359abcd", "dev"},
		{"DEV-C65FA359ABCD", "dev"},
	} {
		if got := deriveUpdateChannel(tc.version); got != tc.want {
			t.Errorf("deriveUpdateChannel(%q) = %q, want %q", tc.version, got, tc.want)
		}
	}
}

func TestDeriveUpdateChannel_Ugly(t *testing.T) {
	// A hyphenated prerelease suffix that names neither alpha nor beta still
	// falls back to beta (mirroring go-update's own isPreRelease-with-no-
	// alpha/beta-substring -> beta rule); an empty version has no hyphen and
	// defaults to stable.
	for _, tc := range []struct{ version, want string }{
		{"v2.0.0-rc.1", "beta"},
		{"", "stable"},
	} {
		if got := deriveUpdateChannel(tc.version); got != tc.want {
			t.Errorf("deriveUpdateChannel(%q) = %q, want %q", tc.version, got, tc.want)
		}
	}
}

// --- updateAssetPrefix ---------------------------------------------------

func TestUpdateAssetPrefix_Good(t *testing.T) {
	for _, tc := range []struct {
		name                           string
		goos, goarch, backend, invoked string
		wantPrefix, wantRole           string
	}{
		{"native macos metal", "darwin", "arm64", "metal", "lem", "macos-aarch64-lem-metal-", "native"},
		{"native linux cpu amd64", "linux", "amd64", "cpu", "lem", "linux-x86_64-lem-cpu-", "native"},
		{"driver linux amd", "linux", "amd64", "amd", "lem-amd", "linux-x86_64-lem-driver-amd-", "driver"},
		{"driver macos metal", "darwin", "arm64", "metal", "lem-metal", "macos-aarch64-lem-driver-metal-", "driver"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			prefix, role, err := updateAssetPrefix(tc.goos, tc.goarch, tc.backend, tc.invoked)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if prefix != tc.wantPrefix {
				t.Errorf("prefix = %q, want %q", prefix, tc.wantPrefix)
			}
			if role != tc.wantRole {
				t.Errorf("role = %q, want %q", role, tc.wantRole)
			}
		})
	}
}

func TestUpdateAssetPrefix_Bad(t *testing.T) {
	// A renamed or symlinked binary: the invoked name matches neither the
	// Native nor the Driver shape for its own backend.
	_, _, err := updateAssetPrefix("linux", "amd64", "cpu", "myrenamed-lem")
	if err == nil {
		t.Fatal("expected an error for an unrecognised invoked name")
	}
	if !core.Contains(err.Error(), "myrenamed-lem") {
		t.Errorf("error %q does not name the unrecognised invoked name", err.Error())
	}
}

func TestUpdateAssetPrefix_Ugly(t *testing.T) {
	// An unstamped binary (backend == "") cannot select an asset at all —
	// this must fail loudly rather than guess a backend.
	_, _, err := updateAssetPrefix("linux", "amd64", "", "lem")
	if err == nil {
		t.Fatal("expected an error for an unstamped backend")
	}
	if !core.Contains(err.Error(), "backend") {
		t.Errorf("error %q does not mention the missing backend", err.Error())
	}
}

// --- releaseOSName / releaseArchName -------------------------------------

func TestReleaseOSName_Good(t *testing.T) {
	for _, tc := range []struct{ goos, want string }{
		{"linux", "linux"},
		{"windows", "windows"},
	} {
		if got := releaseOSName(tc.goos); got != tc.want {
			t.Errorf("releaseOSName(%q) = %q, want %q", tc.goos, got, tc.want)
		}
	}
}

func TestReleaseOSName_Ugly(t *testing.T) {
	// darwin is the one GOOS this repo's grid spells differently.
	if got := releaseOSName("darwin"); got != "macos" {
		t.Errorf("releaseOSName(darwin) = %q, want macos", got)
	}
}

func TestReleaseArchName_Good(t *testing.T) {
	for _, tc := range []struct{ goarch, want string }{
		{"amd64", "x86_64"},
		{"arm64", "aarch64"},
	} {
		if got := releaseArchName(tc.goarch); got != tc.want {
			t.Errorf("releaseArchName(%q) = %q, want %q", tc.goarch, got, tc.want)
		}
	}
}

func TestReleaseArchName_Ugly(t *testing.T) {
	// An unmapped arch passes through unchanged — it will simply never
	// prefix-match a real asset, the correct loud failure via
	// selectReleaseAsset rather than a silent wrong guess.
	if got := releaseArchName("386"); got != "386" {
		t.Errorf("releaseArchName(386) = %q, want 386 (pass-through)", got)
	}
}

// --- selectReleaseAsset ---------------------------------------------------

func TestSelectReleaseAsset_Good(t *testing.T) {
	assets := []updater.ReleaseAsset{
		{Name: "linux-x86_64-lem-cpu-v1.2.3.zip", DownloadURL: "https://example.com/a"},
		{Name: "linux-x86_64-lem-driver-cpu-v1.2.3.zip", DownloadURL: "https://example.com/b"},
	}

	got, err := selectReleaseAsset(assets, "linux-x86_64-lem-cpu-")

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Name != assets[0].Name {
		t.Errorf("got %q, want %q", got.Name, assets[0].Name)
	}
}

func TestSelectReleaseAsset_Bad(t *testing.T) {
	// No asset for this platform at all.
	_, err := selectReleaseAsset([]updater.ReleaseAsset{{Name: "windows-x86_64-lem-cpu-v1.2.3.zip"}}, "linux-x86_64-lem-cpu-")

	if err == nil {
		t.Fatal("expected an error for zero matches")
	}
	if !core.Contains(err.Error(), "linux-x86_64-lem-cpu-") {
		t.Errorf("error %q does not name the prefix that was looked for", err.Error())
	}
}

func TestSelectReleaseAsset_Ugly(t *testing.T) {
	// Two assets both matching the prefix is also a loud failure, naming
	// what matched — never a silent first-pick.
	assets := []updater.ReleaseAsset{
		{Name: "linux-x86_64-lem-cpu-v1.2.3.zip"},
		{Name: "linux-x86_64-lem-cpu-v1.2.4.zip"},
	}

	_, err := selectReleaseAsset(assets, "linux-x86_64-lem-cpu-")

	if err == nil {
		t.Fatal("expected an error for multiple matches")
	}
	if !core.Contains(err.Error(), "2 release assets") {
		t.Errorf("error %q does not report the match count", err.Error())
	}
}

// --- fetchUpdateRelease ---------------------------------------------------

// updateFetchTestClient is a configurable updater.GithubClient stub —
// mirroring go-update's own flowsTestClient/updaterTestClient pattern — used
// to drive fetchUpdateRelease and runUpdateCommand without touching the
// network.
type updateFetchTestClient struct {
	latest    *updater.Release
	latestErr string
	tagged    *updater.Release
	taggedErr string
}

func (c updateFetchTestClient) GetPublicRepos(ctx context.Context, userOrOrg string) core.Result {
	return core.Ok([]string{})
}

func (c updateFetchTestClient) GetLatestRelease(ctx context.Context, owner, repo, channel string) core.Result {
	if c.latestErr != "" {
		return core.Fail(core.E("updateFetchTestClient.GetLatestRelease", c.latestErr, nil))
	}
	return core.Ok(c.latest)
}

func (c updateFetchTestClient) GetReleaseByPullRequest(ctx context.Context, owner, repo string, prNumber int) core.Result {
	return core.Ok((*updater.Release)(nil))
}

func (c updateFetchTestClient) GetReleaseByTag(ctx context.Context, owner, repo, tag string) core.Result {
	if c.taggedErr != "" {
		return core.Fail(core.E("updateFetchTestClient.GetReleaseByTag", c.taggedErr, nil))
	}
	return core.Ok(c.tagged)
}

func withUpdateGithubClient(client updater.GithubClient) func() {
	original := updater.NewGithubClient
	updater.NewGithubClient = func() updater.GithubClient { return client }
	return func() { updater.NewGithubClient = original }
}

func withUpdateVersion(v string) func() {
	original := updater.Version
	updater.Version = v
	return func() { updater.Version = original }
}

func TestFetchUpdateRelease_Good(t *testing.T) {
	// A semver channel: CheckForNewerVersion's own comparison decides
	// availability, unpacked through the VersionCheckResult interface.
	defer withUpdateVersion("1.0.0")()
	defer withUpdateGithubClient(updateFetchTestClient{latest: &updater.Release{TagName: "v1.1.0"}})()

	release, available, err := fetchUpdateRelease(context.Background(), "stable")

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !available {
		t.Error("expected available = true (1.0.0 -> v1.1.0)")
	}
	if release == nil || release.TagName != "v1.1.0" {
		t.Errorf("release = %+v, want TagName v1.1.0", release)
	}
}

func TestFetchUpdateRelease_Bad(t *testing.T) {
	defer withUpdateVersion("1.0.0")()
	defer withUpdateGithubClient(updateFetchTestClient{latestErr: "github down"})()

	_, _, err := fetchUpdateRelease(context.Background(), "stable")

	if err == nil {
		t.Fatal("expected an error")
	}
	if !core.Contains(err.Error(), "failed to check for updates") {
		t.Errorf("error %q missing the expected wrap message", err.Error())
	}
}

func TestFetchUpdateRelease_Ugly(t *testing.T) {
	// The dev channel: fetched by exact tag, never through the semver path;
	// availability is always reported false here — the caller compares
	// against the selected asset's own embedded version instead (see
	// runUpdateCommand's dev-channel handling).
	defer withUpdateGithubClient(updateFetchTestClient{tagged: &updater.Release{TagName: "dev"}})()

	release, available, err := fetchUpdateRelease(context.Background(), "dev")

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if available {
		t.Error("dev channel must never report availability from fetchUpdateRelease itself")
	}
	if release == nil || release.TagName != "dev" {
		t.Errorf("release = %+v, want TagName dev", release)
	}
}

// --- extractZipEntry -------------------------------------------------------

func buildTestZip(t *testing.T, entries map[string][]byte) []byte {
	t.Helper()
	var buf bytes.Buffer
	w := zip.NewWriter(&buf)
	for name, data := range entries {
		f, err := w.Create(name)
		if err != nil {
			t.Fatalf("create zip entry %q: %v", name, err)
		}
		if _, err := f.Write(data); err != nil {
			t.Fatalf("write zip entry %q: %v", name, err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatalf("close zip writer: %v", err)
	}
	return buf.Bytes()
}

func TestExtractZipEntry_Good(t *testing.T) {
	zipBytes := buildTestZip(t, map[string][]byte{"lem": []byte("binary-bytes")})

	data, err := extractZipEntry(zipBytes, "lem")

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if string(data) != "binary-bytes" {
		t.Errorf("data = %q, want %q", data, "binary-bytes")
	}
}

func TestExtractZipEntry_Bad(t *testing.T) {
	zipBytes := buildTestZip(t, map[string][]byte{"lem-metal": []byte("other")})

	_, err := extractZipEntry(zipBytes, "lem")

	if err == nil {
		t.Fatal("expected an error for a missing entry")
	}
	if !core.Contains(err.Error(), `"lem"`) {
		t.Errorf("error %q does not name the missing entry", err.Error())
	}
}

func TestExtractZipEntry_Ugly(t *testing.T) {
	_, err := extractZipEntry([]byte("not a zip"), "lem")

	if err == nil {
		t.Fatal("expected an error for a corrupt archive")
	}
	if !core.Contains(err.Error(), "not a valid zip") {
		t.Errorf("error %q does not report the archive as invalid", err.Error())
	}
}

// --- applyReleaseAsset ------------------------------------------------------

func withUpdateDoUpdateFromReader(fn func(core.Reader) core.Result) func() {
	original := updater.DoUpdateFromReader
	updater.DoUpdateFromReader = fn
	return func() { updater.DoUpdateFromReader = original }
}

func TestApplyReleaseAsset_Good(t *testing.T) {
	zipBytes := buildTestZip(t, map[string][]byte{"lem": []byte("new-binary-bytes")})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(zipBytes)
	}))
	defer server.Close()

	var applied []byte
	defer withUpdateDoUpdateFromReader(func(r core.Reader) core.Result {
		data, _ := io.ReadAll(r)
		applied = data
		return core.Ok(nil)
	})()

	err := applyReleaseAsset(context.Background(), "lem", updater.ReleaseAsset{DownloadURL: server.URL})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if string(applied) != "new-binary-bytes" {
		t.Errorf("applied bytes = %q, want %q", applied, "new-binary-bytes")
	}
}

func TestApplyReleaseAsset_Bad(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	err := applyReleaseAsset(context.Background(), "lem", updater.ReleaseAsset{DownloadURL: server.URL})

	if err == nil {
		t.Fatal("expected an error for a failed download")
	}
	if !core.Contains(err.Error(), "download failed") {
		t.Errorf("error %q missing the expected download-failure message", err.Error())
	}
}

func TestApplyReleaseAsset_Ugly(t *testing.T) {
	// The archive downloads fine but doesn't contain the entry we're looking
	// for — e.g. a Driver zip fetched while running as Native.
	zipBytes := buildTestZip(t, map[string][]byte{"lem-metal": []byte("wrong-binary")})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(zipBytes)
	}))
	defer server.Close()

	err := applyReleaseAsset(context.Background(), "lem", updater.ReleaseAsset{DownloadURL: server.URL})

	if err == nil {
		t.Fatal("expected an error for a missing zip entry")
	}
	if !core.Contains(err.Error(), `"lem"`) {
		t.Errorf("error %q does not name the missing entry", err.Error())
	}
}

// --- runUpdateCommand (end to end) -----------------------------------------

// withUpdateIdentity pins the build-identity package vars runUpdateCommand
// reads (version, backend, commandName) for the duration of a test.
func withUpdateIdentity(v, b, name string) func() {
	origVersion, origBackend, origCommandName := version, backend, commandName
	version, backend, commandName = v, b, name
	return func() { version, backend, commandName = origVersion, origBackend, origCommandName }
}

func TestRunUpdateCommand_Good(t *testing.T) {
	// A stable channel with a real update available applies end to end:
	// stubbed GithubClient -> real HTTP download of a real in-memory zip ->
	// real unzip -> stubbed DoUpdateFromReader (never the real
	// selfupdate.Apply, which would touch the test binary).
	defer withUpdateIdentity("1.0.0", "cpu", "lem")()

	zipBytes := buildTestZip(t, map[string][]byte{"lem": []byte("new-bytes")})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(zipBytes)
	}))
	defer server.Close()

	assetName := core.Sprintf("%s-%s-lem-cpu-v1.1.0.zip", releaseOSName(runtime.GOOS), releaseArchName(runtime.GOARCH))
	defer withUpdateGithubClient(updateFetchTestClient{latest: &updater.Release{
		TagName: "v1.1.0",
		Assets:  []updater.ReleaseAsset{{Name: assetName, DownloadURL: server.URL}},
	}})()
	var applied []byte
	defer withUpdateDoUpdateFromReader(func(r core.Reader) core.Result {
		data, _ := io.ReadAll(r)
		applied = data
		return core.Ok(nil)
	})()

	var stdout, stderr bytes.Buffer
	code := runUpdateCommand(context.Background(), nil, &stdout, &stderr)

	if code != 0 {
		t.Fatalf("exit %d, want 0; stderr=%s", code, stderr.String())
	}
	if string(applied) != "new-bytes" {
		t.Errorf("applied = %q, want %q", applied, "new-bytes")
	}
	if !core.Contains(stdout.String(), "Updated to v1.1.0") {
		t.Errorf("stdout missing the updated-to message: %s", stdout.String())
	}
}

func TestRunUpdateCommand_Bad(t *testing.T) {
	// An unrecognised --channel value is rejected before any network call —
	// no GithubClient stub is even installed, so a real call would fail loud.
	var stdout, stderr bytes.Buffer
	code := runUpdateCommand(context.Background(), []string{"--channel", "nightly"}, &stdout, &stderr)

	if code != 2 {
		t.Fatalf("exit %d, want 2; stderr=%s", code, stderr.String())
	}
	if !core.Contains(stderr.String(), `unknown channel "nightly"`) {
		t.Errorf("stderr missing the unknown-channel message: %s", stderr.String())
	}
}

func TestRunUpdateCommand_Ugly(t *testing.T) {
	// --check reports an available update but never downloads or applies it.
	defer withUpdateIdentity("1.0.0", "cpu", "lem")()

	assetName := core.Sprintf("%s-%s-lem-cpu-v1.1.0.zip", releaseOSName(runtime.GOOS), releaseArchName(runtime.GOARCH))
	defer withUpdateGithubClient(updateFetchTestClient{latest: &updater.Release{
		TagName: "v1.1.0",
		Assets:  []updater.ReleaseAsset{{Name: assetName, DownloadURL: "https://updates.example.invalid/must-not-be-fetched"}},
	}})()
	applyCalled := false
	defer withUpdateDoUpdateFromReader(func(r core.Reader) core.Result {
		applyCalled = true
		return core.Ok(nil)
	})()

	var stdout, stderr bytes.Buffer
	code := runUpdateCommand(context.Background(), []string{"--check"}, &stdout, &stderr)

	if code != 0 {
		t.Fatalf("exit %d, want 0; stderr=%s", code, stderr.String())
	}
	if applyCalled {
		t.Error("--check must never apply the update")
	}
	if !core.Contains(stdout.String(), "Update available") {
		t.Errorf("stdout missing the update-available message: %s", stdout.String())
	}
}

func TestRunUpdateCommand_AlreadyUpToDate(t *testing.T) {
	defer withUpdateIdentity("1.1.0", "cpu", "lem")()
	defer withUpdateGithubClient(updateFetchTestClient{latest: &updater.Release{TagName: "v1.1.0"}})()

	var stdout, stderr bytes.Buffer
	code := runUpdateCommand(context.Background(), nil, &stdout, &stderr)

	if code != 0 {
		t.Fatalf("exit %d, want 0; stderr=%s", code, stderr.String())
	}
	if !core.Contains(stdout.String(), "Already up to date") {
		t.Errorf("stdout missing the up-to-date message: %s", stdout.String())
	}
}

func TestRunUpdateCommand_NoRelease(t *testing.T) {
	defer withUpdateIdentity("1.0.0", "cpu", "lem")()
	defer withUpdateGithubClient(updateFetchTestClient{latest: nil})()

	var stdout, stderr bytes.Buffer
	code := runUpdateCommand(context.Background(), nil, &stdout, &stderr)

	if code != 0 {
		t.Fatalf("exit %d, want 0; stderr=%s", code, stderr.String())
	}
	if !core.Contains(stdout.String(), "No release found") {
		t.Errorf("stdout missing the no-release message: %s", stdout.String())
	}
}

func TestRunUpdateCommand_DevChannel(t *testing.T) {
	// --channel=dev forces the rolling channel regardless of the stamped
	// version, and compares the running version string against the asset's
	// own embedded version — never a semver compare on a rolling tag.
	defer withUpdateIdentity("dev-aaaaaaaaaaaa", "cpu", "lem")()

	assetName := core.Sprintf("%s-%s-lem-cpu-dev-bbbbbbbbbbbb.zip", releaseOSName(runtime.GOOS), releaseArchName(runtime.GOARCH))
	defer withUpdateGithubClient(updateFetchTestClient{tagged: &updater.Release{
		TagName: "dev",
		Assets:  []updater.ReleaseAsset{{Name: assetName}},
	}})()

	var stdout, stderr bytes.Buffer
	code := runUpdateCommand(context.Background(), []string{"--channel", "dev", "--check"}, &stdout, &stderr)

	if code != 0 {
		t.Fatalf("exit %d, want 0; stderr=%s", code, stderr.String())
	}
	if !core.Contains(stdout.String(), "dev-aaaaaaaaaaaa -> dev-bbbbbbbbbbbb") {
		t.Errorf("stdout missing the version-diff message: %s", stdout.String())
	}
}

func TestRunUpdateCommand_UnstampedBackend(t *testing.T) {
	// An unstamped build (backend == "") cannot select an asset — fails
	// loudly after a successful release fetch, rather than guessing.
	defer withUpdateIdentity("1.0.0", "", "lem")()
	defer withUpdateGithubClient(updateFetchTestClient{latest: &updater.Release{TagName: "v1.1.0"}})()

	var stdout, stderr bytes.Buffer
	code := runUpdateCommand(context.Background(), nil, &stdout, &stderr)

	if code != 1 {
		t.Fatalf("exit %d, want 1; stderr=%s", code, stderr.String())
	}
	if !core.Contains(stderr.String(), "backend") {
		t.Errorf("stderr missing the missing-backend message: %s", stderr.String())
	}
}
