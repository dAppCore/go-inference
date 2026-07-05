package modelmgmt

import (
	"io"
	"net/http"
	"time"

	"dappco.re/go"
	coreio "dappco.re/go/io"
)

// PublishConfig holds options for the publish operation.
type PublishConfig struct {
	InputDir string
	Repo     string
	Public   bool
	Token    string
	DryRun   bool
}

// uploadEntry pairs a local file path with its remote destination.
type uploadEntry struct {
	local  string
	remote string
}

// Publish uploads Parquet files to HuggingFace Hub.
//
// It looks for train.parquet, valid.parquet, and test.parquet in InputDir,
// plus an optional dataset_card.md in the parent directory (uploaded as README.md).
// The token is resolved from PublishConfig.Token, the HF_TOKEN environment variable,
// or ~/.huggingface/token, in that order.
func Publish(cfg PublishConfig, w io.Writer) core.Result {
	if cfg.InputDir == "" {
		return core.Fail(core.E("modelmgmt.Publish", "input directory is required", nil))
	}

	token := resolveHFToken(cfg.Token)
	if token == "" && !cfg.DryRun {
		return core.Fail(core.E("modelmgmt.Publish", "HuggingFace token required (--token, HF_TOKEN env, or ~/.huggingface/token)", nil))
	}

	filesResult := collectUploadFiles(cfg.InputDir)
	if !filesResult.OK {
		return filesResult
	}
	files := filesResult.Value.([]uploadEntry)
	if len(files) == 0 {
		return core.Fail(core.E("modelmgmt.Publish", core.Sprintf("no Parquet files found in %s", cfg.InputDir), nil))
	}

	if cfg.DryRun {
		core.Print(w, "Dry run: would publish to %s", cfg.Repo)
		if cfg.Public {
			core.Print(w, "  Visibility: public")
		} else {
			core.Print(w, "  Visibility: private")
		}
		for _, f := range files {
			info, err := coreio.Local.Stat(f.local)
			if err != nil {
				return core.Fail(core.E("modelmgmt.Publish", core.Sprintf("stat %s", f.local), err))
			}
			sizeMB := float64(info.Size()) / 1024 / 1024
			core.Print(w, "  %s -> %s (%.1f MB)", core.PathBase(f.local), f.remote, sizeMB)
		}
		return core.Ok(nil)
	}

	core.Print(w, "Publishing to https://huggingface.co/datasets/%s", cfg.Repo)

	for _, f := range files {
		if result := uploadFileToHF(token, cfg.Repo, f.local, f.remote); !result.OK {
			return core.Fail(core.E("modelmgmt.Publish", core.Sprintf("upload %s", core.PathBase(f.local)), result.Value.(error)))
		}
		core.Print(w, "  Uploaded %s -> %s", core.PathBase(f.local), f.remote)
	}

	core.Print(w, "")
	core.Print(w, "Published to https://huggingface.co/datasets/%s", cfg.Repo)
	return core.Ok(nil)
}

// resolveHFToken returns a HuggingFace API token from the given value,
// HF_TOKEN env var, or ~/.huggingface/token file.
func resolveHFToken(explicit string) string {
	if explicit != "" {
		return explicit
	}
	if env := core.Env("HF_TOKEN"); env != "" {
		return env
	}
	home := core.Env("DIR_HOME")
	if home == "" {
		return ""
	}
	data, err := coreio.Local.Read(core.JoinPath(home, ".huggingface", "token"))
	if err != nil {
		return ""
	}
	return core.Trim(data)
}

// collectUploadFiles finds Parquet split files and an optional dataset card.
func collectUploadFiles(inputDir string) core.Result {
	splits := []string{"train", "valid", "test"}
	var files []uploadEntry

	for _, split := range splits {
		path := core.JoinPath(inputDir, core.Concat(split, ".parquet"))
		if !coreio.Local.IsFile(path) {
			continue
		}
		files = append(files, uploadEntry{path, core.Sprintf("data/%s.parquet", split)})
	}

	// Check for dataset card in parent directory.
	cardPath := core.JoinPath(inputDir, "..", "dataset_card.md")
	if coreio.Local.IsFile(cardPath) {
		files = append(files, uploadEntry{cardPath, "README.md"})
	}

	return core.Ok(files)
}

// uploadFileToHF uploads a single file to a HuggingFace dataset repo via the Hub API.
func uploadFileToHF(token, repoID, localPath, remotePath string) core.Result {
	raw, err := coreio.Local.Read(localPath)
	if err != nil {
		return core.Fail(core.E("modelmgmt.uploadFileToHF", core.Sprintf("read %s", localPath), err))
	}

	url := core.Sprintf("https://huggingface.co/api/datasets/%s/upload/main/%s", repoID, remotePath)

	req, err := http.NewRequest(http.MethodPut, url, core.NewReader(raw))
	if err != nil {
		return core.Fail(core.E("modelmgmt.uploadFileToHF", "create request", err))
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/octet-stream")

	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return core.Fail(core.E("modelmgmt.uploadFileToHF", "upload request", err))
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		rBody := readAll(resp.Body)
		body := []byte{}
		if rBody.OK {
			body = rBody.Value.([]byte)
		}
		return core.Fail(core.E("modelmgmt.uploadFileToHF", core.Sprintf("upload failed: HTTP %d: %s", resp.StatusCode, string(body)), nil))
	}

	return core.Ok(nil)
}
