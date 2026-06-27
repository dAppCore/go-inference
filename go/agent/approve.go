package agent

import (
	"io" // Note: AX-6 intrinsic - io.Writer is the public output surface; core exposes no Writer primitive.

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/modelmgmt"
	coreio "dappco.re/go/io"
	"dappco.re/go/store"
)

// ApproveConfig holds options for the approve operation.
type ApproveConfig struct {
	Output    string
	Threshold float64
}

// ApproveExpansions filters scored expansion responses above the threshold
// and writes approved examples to a training JSONL file.
//
// The query joins expansion_raw with expansion_scores, keeping rows where
// the heuristic passed AND the judge either passed or has not yet scored.
// Each approved row is written as a chat-format JSONL line with user/assistant
// messages.
//
//	r := agent.ApproveExpansions(db, cfg, os.Stdout)
//	if !r.OK { return r }
func ApproveExpansions(db *store.DuckDB, cfg ApproveConfig, w io.Writer) core.Result {
	rows, err := db.Conn().Query(`
		SELECT r.idx, r.seed_id, r.region, r.domain, r.prompt, r.response,
		       r.gen_time, r.model, s.heuristic_score
		FROM expansion_raw r
		JOIN expansion_scores s ON r.idx = s.idx
		WHERE s.heuristic_pass = true
		AND (s.judge_pass = true OR s.judge_pass IS NULL)
		ORDER BY r.idx
	`)
	if err != nil {
		return core.Fail(core.E("agent.ApproveExpansions", "query approved expansions", err))
	}
	defer rows.Close()

	f, err := coreio.Local.Create(cfg.Output)
	if err != nil {
		return core.Fail(core.E("agent.ApproveExpansions", core.Sprintf("create output %s", cfg.Output), err))
	}
	defer f.Close()

	count := 0
	regionSet := make(map[string]bool)
	domainSet := make(map[string]bool)

	for rows.Next() {
		var idx int
		var seedID, region, domain, prompt, response, model string
		var genTime, score float64
		if err := rows.Scan(&idx, &seedID, &region, &domain, &prompt, &response, &genTime, &model, &score); err != nil {
			return core.Fail(core.E("agent.ApproveExpansions", "scan approved row", err))
		}

		example := modelmgmt.TrainingExample{
			Messages: []inference.Message{
				{Role: "user", Content: prompt},
				{Role: "assistant", Content: response},
			},
		}

		if _, err := f.Write([]byte(core.Concat(core.JSONMarshalString(example), "\n"))); err != nil {
			return core.Fail(core.E("agent.ApproveExpansions", "encode example", err))
		}

		regionSet[region] = true
		domainSet[domain] = true
		count++
	}

	if err := rows.Err(); err != nil {
		return core.Fail(core.E("agent.ApproveExpansions", "iterate approved rows", err))
	}

	core.Print(w, "Approved: %d responses (threshold: heuristic > 0)", count)
	core.Print(w, "Exported: %s", cfg.Output)
	core.Print(w, "  Regions: %d, Domains: %d", len(regionSet), len(domainSet))

	return core.Ok(nil)
}
