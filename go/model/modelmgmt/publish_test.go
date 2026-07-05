package modelmgmt

import (
	"dappco.re/go"
	coreio "dappco.re/go/io"
)

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

func TestPublish_Publish_Ugly(t *core.T) {
	dir := t.TempDir()
	assertResultError(t, Publish(PublishConfig{InputDir: dir, Repo: "owner/repo", DryRun: true}, core.NewBuffer(nil)), "no Parquet")
}
