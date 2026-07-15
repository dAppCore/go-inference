package datapipe

import (
	"bufio"

	"dappco.re/go"
	"dappco.re/go/inference"
	coreio "dappco.re/go/io"
	"github.com/parquet-go/parquet-go"
)

// ParquetRow is the schema for exported Parquet files.
type ParquetRow struct {
	Prompt   string `parquet:"prompt"`
	Response string `parquet:"response"`
	System   string `parquet:"system"`
	Messages string `parquet:"messages"`
}

// ExportParquet reads JSONL training splits (train.jsonl, valid.jsonl, test.jsonl)
// from trainingDir and writes Parquet files with snappy compression to outputDir.
// Returns total rows exported in Result.Value.
func ExportParquet(trainingDir, outputDir string) core.Result {
	if outputDir == "" {
		outputDir = core.JoinPath(trainingDir, "parquet")
	}
	if err := coreio.Local.EnsureDir(outputDir); err != nil {
		return core.Fail(core.E("datapipe.ExportParquet", "create output dir", err))
	}

	total := 0
	for _, split := range []string{"train", "valid", "test"} {
		jsonlPath := core.JoinPath(trainingDir, core.Concat(split, ".jsonl"))
		if !coreio.Local.IsFile(jsonlPath) {
			continue
		}

		result := ExportSplitParquet(jsonlPath, outputDir, split)
		if !result.OK {
			return core.Fail(core.E("datapipe.ExportParquet", core.Sprintf("export %s", split), result.Value.(error)))
		}
		total += result.Value.(int)
	}

	return core.Ok(total)
}

// ExportSplitParquet reads a chat JSONL file and writes a Parquet file for the
// given split name. Returns the number of rows written in Result.Value.
func ExportSplitParquet(jsonlPath, outputDir, split string) core.Result {
	f, err := coreio.Local.Open(jsonlPath)
	if err != nil {
		return core.Fail(core.E("datapipe.ExportSplitParquet", core.Sprintf("open %s", jsonlPath), err))
	}
	defer f.Close()

	var rows []ParquetRow
	scanner := bufio.NewScanner(f)
	// 64 KiB initial buffer, grown on demand up to a 1 MiB max line. Avoids
	// allocating a full 1 MiB up front on every call when lines are small
	// (the common case) while still admitting lines up to 1 MiB.
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)

	for scanner.Scan() {
		text := core.Trim(scanner.Text())
		if text == "" {
			continue
		}

		var data struct {
			Messages []inference.Message `json:"messages"`
		}
		if r := core.JSONUnmarshalString(text, &data); !r.OK {
			continue
		}

		var prompt, response, system string
		for _, m := range data.Messages {
			switch m.Role {
			case "user":
				if prompt == "" {
					prompt = m.Content
				}
			case "assistant":
				if response == "" {
					response = m.Content
				}
			case "system":
				if system == "" {
					system = m.Content
				}
			}
		}

		msgsJSON := core.JSONMarshalString(data.Messages)
		rows = append(rows, ParquetRow{
			Prompt:   prompt,
			Response: response,
			System:   system,
			Messages: msgsJSON,
		})
	}

	if err := scanner.Err(); err != nil {
		return core.Fail(core.E("datapipe.ExportSplitParquet", core.Sprintf("scan %s", jsonlPath), err))
	}

	if len(rows) == 0 {
		return core.Ok(0)
	}

	outPath := core.JoinPath(outputDir, core.Concat(split, ".parquet"))

	out, err := coreio.Local.Create(outPath)
	if err != nil {
		return core.Fail(core.E("datapipe.ExportSplitParquet", core.Sprintf("create %s", outPath), err))
	}

	writer := parquet.NewGenericWriter[ParquetRow](out,
		parquet.Compression(&parquet.Snappy),
	)

	if _, err := writer.Write(rows); err != nil {
		out.Close()
		return core.Fail(core.E("datapipe.ExportSplitParquet", "write parquet rows", err))
	}

	if err := writer.Close(); err != nil {
		out.Close()
		return core.Fail(core.E("datapipe.ExportSplitParquet", "close parquet writer", err))
	}

	if err := out.Close(); err != nil {
		return core.Fail(core.E("datapipe.ExportSplitParquet", "close file", err))
	}

	return core.Ok(len(rows))
}
