// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"archive/zip"
	"bytes"
	"context"
	"flag"
	"io"
	"runtime"

	core "dappco.re/go"
	updater "dappco.re/go/update"
)

// The GitHub repository lem's own releases live in, for go-update's
// GithubClient primitives (see docs/release-artifacts.md for the asset grid
// those releases carry).
const (
	updateRepoOwner = "dAppCore"
	updateRepoName  = "go-inference"
	updateDevTag    = "dev"
)

// runUpdateCommand implements `lem update`: self-update from GitHub releases.
//
// The channel defaults to whatever the running binary's own version implies
// (deriveUpdateChannel) rather than being a separate setting the user
// tracks — a v1.2.3 build follows stable, v1.2.3-beta.1 follows beta,
// dev-<sha12> follows the rolling dev prerelease. --channel overrides it.
//
//	lem update                    # check + apply, channel from the binary's version
//	lem update --check             # report only, never download
//	lem update --channel=dev       # follow the rolling dev prerelease regardless
func runUpdateCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("update"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	channelFlag := fs.String("channel", "", "release channel: stable, beta, alpha, or dev (default: derived from the binary's own version)")
	checkOnly := fs.Bool("check", false, "only check for an available update, do not apply it")
	fs.Usage = func() {
		core.WriteString(stderr, core.Sprintf("Usage: %s update [flags]\n", cliCommandName("update")))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Self-update lem from GitHub releases. The channel defaults to whatever\n")
		core.WriteString(stderr, "the running binary's own version implies: a vX.Y.Z build follows stable,\n")
		core.WriteString(stderr, "vX.Y.Z-beta.N follows beta, vX.Y.Z-alpha.N follows alpha, and a\n")
		core.WriteString(stderr, "dev-<sha12> build follows the rolling dev prerelease.\n")
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Flags:\n")
		printFlagBlock(stderr, fs)
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}

	channel := core.Trim(*channelFlag)
	if channel == "" {
		channel = deriveUpdateChannel(version)
	}
	switch channel {
	case "stable", "beta", "alpha", "dev":
	default:
		core.Print(stderr, "%s: unknown channel %q (want stable, beta, alpha, or dev)", cliCommandName("update"), channel)
		return 2
	}

	// CheckForNewerVersion/CheckOnly (reached via fetchUpdateRelease below)
	// compare against this package var, not against go-update's own built-in
	// placeholder — every comparison in this command must be relative to
	// lem's own stamped version.
	updater.Version = version

	release, semverAvailable, err := fetchUpdateRelease(ctx, channel)
	if err != nil {
		core.Print(stderr, "%s: %v", cliCommandName("update"), err)
		return 1
	}
	if release == nil {
		core.Print(stdout, "No release found on the %s channel.", channel)
		return 0
	}

	core.Print(stdout, "Current version: %s", version)
	core.Print(stdout, "Channel: %s", channel)

	// For the semver channels, "up to date" is already known from the
	// release tag alone (semverAvailable), with no asset involved — so a
	// platform this release simply has no asset for (e.g. a windows build
	// that was never shipped) is never blocked from reporting "nothing to
	// do" when there genuinely is nothing to do. Asset-selection is only
	// reached when there is a real reason to need the asset: reporting or
	// applying an available update, or (for dev) determining availability
	// at all, since a rolling tag has no semver to decide it from.
	if channel != updateDevTag && !semverAvailable {
		core.Print(stdout, "Already up to date (%s).", release.TagName)
		return 0
	}

	invokedName := core.TrimSuffix(commandName, ".exe")
	prefix, role, err := updateAssetPrefix(runtime.GOOS, runtime.GOARCH, backend, invokedName)
	if err != nil {
		core.Print(stderr, "%s: %v", cliCommandName("update"), err)
		return 1
	}

	asset, err := selectReleaseAsset(release.Assets, prefix)
	if err != nil {
		core.Print(stderr, "%s: %v", cliCommandName("update"), err)
		return 1
	}
	// The version token embedded in the winning asset's own name — e.g.
	// "dev-c65fa359abcd" or "v1.2.3" — trimming the exact prefix we matched
	// and the .zip suffix leaves exactly that token, no separate parse step.
	assetVersion := core.TrimSuffix(core.TrimPrefix(asset.Name, prefix), ".zip")

	// The dev channel has no semver to compare (a rolling tag is not a
	// version), so its availability is a plain string inequality between the
	// running binary's version and the asset it would download — never a
	// semver.Compare on "dev-<sha12>".
	available := semverAvailable
	if channel == updateDevTag {
		available = version != assetVersion
	}

	core.Print(stdout, "Platform: %s/%s (backend=%s, role=%s)", runtime.GOOS, runtime.GOARCH, backend, role)
	core.Print(stdout, "Latest on %s: %s (asset %s)", channel, assetVersion, asset.Name)

	if !available {
		core.Println("Already up to date.")
		return 0
	}

	if *checkOnly {
		core.Print(stdout, "Update available: %s -> %s", version, assetVersion)
		core.Print(stdout, "Run %s update to apply it.", cliName())
		return 0
	}

	core.Println("Downloading update...")
	if err := applyReleaseAsset(ctx, invokedName, asset); err != nil {
		core.Print(stderr, "%s: %v", cliCommandName("update"), err)
		return 1
	}
	core.Print(stdout, "Updated to %s. Run %s again to use it.", assetVersion, cliName())
	return 0
}

// deriveUpdateChannel infers the release channel from a version string —
// the same shape build.yml's own prerelease-honesty check and go-update's
// determineChannel use: alpha/beta substrings win outright, any other
// hyphenated (prerelease-shaped) suffix falls back to beta (mirroring
// determineChannel's "prerelease flag with no alpha/beta substring -> beta"
// rule, with the tag's hyphen standing in for the flag since there is no
// GitHub release object here yet), a "dev" or "dev-*" version is the
// rolling channel go-update's alpha/beta/stable classifier has no concept
// of, and anything left is stable. This is the release-channels-as-a-
// side-effect design: a binary knows which channel it belongs to just by
// reading its own stamped version, with no separate tracked setting.
//
//	deriveUpdateChannel("v1.2.3")            // "stable"
//	deriveUpdateChannel("v1.2.3-beta.1")     // "beta"
//	deriveUpdateChannel("v1.2.3-alpha.2")    // "alpha"
//	deriveUpdateChannel("dev-c65fa359abcd")  // "dev"
func deriveUpdateChannel(v string) string {
	lower := core.Lower(v)
	switch {
	case lower == "dev" || core.HasPrefix(lower, "dev-"):
		return "dev"
	case core.Contains(lower, "alpha"):
		return "alpha"
	case core.Contains(lower, "beta"):
		return "beta"
	case core.Contains(lower, "-"):
		return "beta"
	default:
		return "stable"
	}
}

// fetchUpdateRelease fetches the release for channel through go-update's
// GithubClient and, for the semver channels, whether it is already known to
// be newer than updater.Version.
//
// The dev channel is fetched by its exact tag (GetReleaseByTag) rather than
// through GetLatestRelease's alpha/beta/stable classification: a rolling
// "dev" prerelease republished on every push has no semver to bucket, and
// determineChannel's substring/prerelease-flag rules were never meant to
// target one exact tag string. Its availability is NOT computed here — see
// the dev-channel handling in runUpdateCommand — because that needs the
// asset actually selected, not just the release.
//
// The semver channels go through CheckForNewerVersion and unpack its result
// via the VersionCheckResult interface, so this reuses go-update's own
// semver comparison instead of re-implementing it, while still reaching the
// release's Assets for this repo's own asset-selection (GetDownloadURL's
// automatic matching cannot do that job — see updateAssetPrefix).
func fetchUpdateRelease(ctx context.Context, channel string) (release *updater.Release, available bool, err error) {
	client := updater.NewGithubClient()

	if channel == updateDevTag {
		result := client.GetReleaseByTag(ctx, updateRepoOwner, updateRepoName, updateDevTag)
		if !result.OK {
			return nil, false, core.E("fetchUpdateRelease", "failed to fetch the dev release", core.NewError(result.Error()))
		}
		release, _ = result.Value.(*updater.Release)
		return release, false, nil
	}

	check := updater.CheckForNewerVersion(updateRepoOwner, updateRepoName, channel, true)
	if !check.OK {
		return nil, false, core.E("fetchUpdateRelease", "failed to check for updates", core.NewError(check.Error()))
	}
	vc, ok := check.Value.(updater.VersionCheckResult)
	if !ok {
		return nil, false, core.E("fetchUpdateRelease", "unexpected CheckForNewerVersion result payload", nil)
	}
	return vc.Release(), vc.Available(), nil
}

// updateAssetPrefix builds the exact release-asset-name prefix this binary
// should look for, and reports which packaging (native/driver) that prefix
// implies.
//
// Both packagings ship the SAME compiled bytes: build.yml (docs/release-
// artifacts.md's "build once, package twice") compiles one binary per cell,
// then copies it into two differently-named zips — `lem` for Native,
// `lem-{backend}` for Driver. No ldflags var could distinguish Native from
// Driver, because there is nothing to distinguish at compile time — the
// split happens entirely in post-build packaging. The invoked binary's own
// name is therefore the only signal that exists for this, and it is unsound
// against a deliberate rename or symlink; that is an accepted limitation of
// the two-packagings-one-binary design, not an oversight here.
func updateAssetPrefix(goos, goarch, backend, invokedName string) (prefix, role string, err error) {
	if backend == "" {
		return "", "", core.E("updateAssetPrefix", "binary has no stamped backend (built without -X main.backend=<cpu|metal|amd|cuda>) — cannot select a release asset", nil)
	}
	osName := releaseOSName(goos)
	archName := releaseArchName(goarch)
	switch invokedName {
	case "lem":
		return core.Sprintf("%s-%s-lem-%s-", osName, archName, backend), "native", nil
	case "lem-" + backend:
		return core.Sprintf("%s-%s-lem-driver-%s-", osName, archName, backend), "driver", nil
	default:
		return "", "", core.E("updateAssetPrefix", core.Sprintf(
			"cannot tell whether invoked name %q is the Native or Driver build for backend %q (expected \"lem\" or \"lem-%s\"); self-update cannot pick an asset for a renamed or symlinked binary",
			invokedName, backend, backend), nil)
	}
}

// releaseOSName maps a GOOS to the release grid's os dimension: darwin ->
// macos; linux and windows are spelled the same on both sides.
func releaseOSName(goos string) string {
	if goos == "darwin" {
		return "macos"
	}
	return goos
}

// releaseArchName maps a GOARCH to the release grid's arch dimension.
func releaseArchName(goarch string) string {
	switch goarch {
	case "amd64":
		return "x86_64"
	case "arm64":
		return "aarch64"
	default:
		return goarch
	}
}

// selectReleaseAsset finds the one release asset whose name starts with
// prefix.
//
// go-update's own GetDownloadURL cannot do this job: its automatic
// GOOS/GOARCH matching looks for runtime.GOOS/GOARCH substrings ("darwin",
// "arm64") in the asset name, but this repo's names spell the same
// dimensions differently ("macos", "aarch64" — see releaseOSName /
// releaseArchName), so it would either match nothing (darwin never appears
// in a macos-* name) or match the wrong thing by accident (GOOS "linux"
// happens to be spelled the same, so it would silently pick the first
// linux-flavoured asset regardless of backend or role). Zero or multiple
// matches is a loud, typed error naming exactly what was looked for — and
// for multiple matches, exactly what matched — rather than guessing either
// way.
func selectReleaseAsset(assets []updater.ReleaseAsset, prefix string) (updater.ReleaseAsset, error) {
	var matches []updater.ReleaseAsset
	for _, a := range assets {
		if core.HasPrefix(a.Name, prefix) {
			matches = append(matches, a)
		}
	}
	switch len(matches) {
	case 1:
		return matches[0], nil
	case 0:
		return updater.ReleaseAsset{}, core.E("selectReleaseAsset", core.Sprintf("no release asset found with prefix %q", prefix), nil)
	default:
		names := make([]string, len(matches))
		for i, m := range matches {
			names[i] = m.Name
		}
		return updater.ReleaseAsset{}, core.E("selectReleaseAsset", core.Sprintf("%d release assets match prefix %q (want exactly one): %s", len(matches), prefix, core.Join(", ", names...)), nil)
	}
}

// applyReleaseAsset downloads asset's zip, extracts the single binary named
// wantName, and applies it via go-update's DoUpdateFromReader.
//
// go-update's own DoUpdate cannot be used here: it fetches a URL and feeds
// the raw HTTP response body straight into selfupdate.Apply, which assumes
// that body IS the new binary's bytes. Every asset in this repo's grid is a
// zip holding one binary (docs/release-artifacts.md's zip layout rule) —
// even the plain cpu/metal cells — so DoUpdate would feed selfupdate.Apply a
// zip archive and corrupt the running binary rather than replace it.
// DoUpdateFromReader is the fix upstream: download and unzip HERE, then hand
// the extracted binary bytes to the same apply/rollback machinery DoUpdate
// itself uses internally, instead of re-implementing the binary swap.
func applyReleaseAsset(ctx context.Context, wantName string, asset updater.ReleaseAsset) error {
	client := updater.NewHTTPClient()
	reqResult := core.NewHTTPRequestContext(ctx, "GET", asset.DownloadURL, nil)
	if !reqResult.OK {
		return core.E("applyReleaseAsset", "failed to build download request", core.NewError(reqResult.Error()))
	}
	req, _ := reqResult.Value.(*core.Request)

	resp, err := client.Do(req)
	if err != nil {
		return core.E("applyReleaseAsset", "failed to download release asset", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != core.StatusOK {
		return core.E("applyReleaseAsset", core.Sprintf("download failed: %s", resp.Status), nil)
	}

	bodyResult := core.ReadAll(resp.Body)
	if !bodyResult.OK {
		return core.E("applyReleaseAsset", "failed to read release asset", core.NewError(bodyResult.Error()))
	}

	binary, err := extractZipEntry([]byte(bodyResult.Value.(string)), wantName)
	if err != nil {
		return err
	}

	if r := updater.DoUpdateFromReader(bytes.NewReader(binary)); !r.OK {
		return core.E("applyReleaseAsset", "failed to apply update", core.NewError(r.Error()))
	}
	return nil
}

// extractZipEntry reads the single named entry out of a zip archive held
// entirely in memory. Every release asset in this repo's grid is small
// enough — one binary, occasionally plus a GPU kernel sidecar for amd/cuda
// that this function simply ignores by name — that streaming to a temp file
// first would buy nothing.
func extractZipEntry(zipBytes []byte, wantName string) ([]byte, error) {
	reader, err := zip.NewReader(bytes.NewReader(zipBytes), int64(len(zipBytes)))
	if err != nil {
		return nil, core.E("extractZipEntry", "not a valid zip archive", err)
	}
	for _, f := range reader.File {
		if f.Name != wantName {
			continue
		}
		rc, err := f.Open()
		if err != nil {
			return nil, core.E("extractZipEntry", core.Sprintf("failed to open %q in archive", wantName), err)
		}
		defer rc.Close()
		dataResult := core.ReadAll(rc)
		if !dataResult.OK {
			return nil, core.E("extractZipEntry", core.Sprintf("failed to read %q from archive", wantName), core.NewError(dataResult.Error()))
		}
		return []byte(dataResult.Value.(string)), nil
	}
	return nil, core.E("extractZipEntry", core.Sprintf("no entry named %q in the downloaded archive", wantName), nil)
}
