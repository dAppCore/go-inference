package modelmgmt

import (
	"bufio"
	"context"
	"io"
	"maps"
	"slices"

	"dappco.re/go"
	coreio "dappco.re/go/io"
	goexec "dappco.re/go/process/exec"
)

// ConsolidateConfig holds options for the consolidate operation.
type ConsolidateConfig struct {
	M3Host    string
	RemoteDir string
	Pattern   string
	OutputDir string
	MergedOut string
}

// Consolidate pulls JSONL response files from M3 via SSH, merges them by idx,
// deduplicates, and writes a single merged JSONL output.
//
//	r := ml.Consolidate(cfg, os.Stdout)
//	if !r.OK { return r }
func Consolidate(cfg ConsolidateConfig, w io.Writer) core.Result {
	if cfg.OutputDir == "" {
		cfg.OutputDir = "responses"
	}
	if err := coreio.Local.EnsureDir(cfg.OutputDir); err != nil {
		return core.Fail(core.E("ml.Consolidate", "create output dir", err))
	}

	// List remote files via SSH.
	core.Print(w, "Pulling responses from remote...")
	listCmd := goexec.Command(context.Background(), "ssh", cfg.M3Host, core.Sprintf("ls %s/%s", cfg.RemoteDir, cfg.Pattern))
	listResult := listCmd.Output()
	if !listResult.OK {
		return core.Fail(core.E("ml.Consolidate", "list remote files: "+listResult.Error(), nil))
	}
	listOutput, _ := listResult.Value.([]byte)

	remoteFiles := core.Split(core.Trim(string(listOutput)), "\n")
	var validFiles []string
	for _, f := range remoteFiles {
		f = core.Trim(f)
		if f != "" {
			validFiles = append(validFiles, f)
		}
	}
	core.Print(w, "  Found %d JSONL files on %s", len(validFiles), cfg.M3Host)

	// Pull each file via SCP.
	for _, rf := range validFiles {
		local := core.JoinPath(cfg.OutputDir, core.PathBase(rf))
		scpCmd := goexec.Command(context.Background(), "scp", core.Sprintf("%s:%s", cfg.M3Host, rf), local)
		if result := scpCmd.Run(); !result.OK {
			core.Print(w, "  warning: failed to pull %s: %v", rf, result.Error())
			continue
		}

		rLines := countLines(local)
		if rLines.OK {
			core.Print(w, "  %s: %d records", core.PathBase(rf), rLines.Value.(int))
		}
	}

	// Merge and deduplicate on idx (first occurrence wins).
	seen := make(map[int]string)
	skipped := 0

	matches := core.PathGlob(core.JoinPath(cfg.OutputDir, cfg.Pattern))
	slices.Sort(matches)

	for _, local := range matches {
		f, err := coreio.Local.Open(local)
		if err != nil {
			continue
		}
		scanner := bufio.NewScanner(f)
		scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
		for scanner.Scan() {
			line := scanner.Text()
			var rec struct {
				Idx *int `json:"idx"`
			}
			if r := core.JSONUnmarshalString(line, &rec); !r.OK {
				skipped++
				continue
			}
			if rec.Idx == nil {
				skipped++
				continue
			}
			if _, exists := seen[*rec.Idx]; !exists {
				seen[*rec.Idx] = line
			}
		}
		f.Close()
	}

	if skipped > 0 {
		core.Print(w, "  Skipped %d records without idx", skipped)
	}

	// Sort by idx and write merged file.
	mergedPath := cfg.MergedOut
	if mergedPath == "" {
		mergedPath = core.JoinPath(cfg.OutputDir, "..", "gold-merged.jsonl")
	}

	idxs := slices.Sorted(maps.Keys(seen))

	out, err := coreio.Local.Create(mergedPath)
	if err != nil {
		return core.Fail(core.E("ml.Consolidate", "create merged file", err))
	}
	defer out.Close()

	bw := bufio.NewWriter(out)
	for _, idx := range idxs {
		bw.WriteString(seen[idx])
		bw.WriteString("\n")
	}
	if err := bw.Flush(); err != nil {
		return core.Fail(core.E("ml.Consolidate", "flush merged file", err))
	}

	core.Print(w, "")
	core.Print(w, "Merged: %d unique examples -> %s", len(seen), mergedPath)
	return core.Ok(nil)
}

// countLines returns the number of lines in a file.
//
//	r := countLines("/data/file.jsonl")
//	if !r.OK { return r }
//	n := r.Value.(int)
func countLines(path string) core.Result {
	f, err := coreio.Local.Open(path)
	if err != nil {
		return core.Fail(err)
	}
	defer f.Close()

	count := 0
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		count++
	}
	return core.ResultOf(count, scanner.Err())
}
