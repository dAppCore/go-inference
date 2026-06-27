package score

import (
	"io"

	"dappco.re/go"
	"dappco.re/go/store"
)

// NormalizeConfig configures the seed normalization process.
type NormalizeConfig struct {
	MinLength int
}

// NormalizeSeeds deduplicates seeds into the expansion_prompts table.
//
// Steps:
//
//  1. Verify the seeds table exists and report its row count.
//
//  2. Drop and recreate expansion_prompts using deduplicated seeds,
//     excluding prompts already present in the prompts or golden_set tables.
//
//  3. Assign priority based on domain coverage (underrepresented domains
//     receive higher priority via RANK).
//
//  4. Print a region distribution summary.
//
//     r := score.NormalizeSeeds(db, cfg, os.Stdout)
//     if !r.OK { return r }
func NormalizeSeeds(db *store.DuckDB, cfg NormalizeConfig, w io.Writer) core.Result {
	// 1. Check seeds table exists and get count.
	var seedCount int
	if err := db.Conn().QueryRow("SELECT count(*) FROM seeds").Scan(&seedCount); err != nil {
		return core.Fail(core.E("score.NormalizeSeeds", "no seeds table (run import-all first)", err))
	}
	core.Print(w, "Seeds table: %d rows", seedCount)

	if seedCount == 0 {
		return core.Fail(core.E("score.NormalizeSeeds", "seeds table is empty, nothing to normalize", nil))
	}

	// 2. Drop and recreate expansion_prompts.
	if _, err := db.Conn().Exec("DROP TABLE IF EXISTS expansion_prompts"); err != nil {
		return core.Fail(core.E("score.NormalizeSeeds", "drop expansion_prompts", err))
	}

	createSQL := core.Sprintf(`
		CREATE TABLE expansion_prompts AS
		WITH unique_seeds AS (
			SELECT
				ROW_NUMBER() OVER (ORDER BY region, domain, seed_id) AS idx,
				seed_id, region, domain, prompt
			FROM (
				SELECT DISTINCT ON (prompt)
					seed_id, region, domain, prompt
				FROM seeds
				WHERE length(prompt) >= %d
				ORDER BY prompt, seed_id
			)
		),
		existing_prompts AS (
			SELECT prompt FROM prompts
			UNION ALL
			SELECT prompt FROM golden_set
		)
		SELECT
			us.idx, us.seed_id, us.region, us.domain,
			'en' AS language, us.prompt, '' AS prompt_en,
			0 AS priority, 'pending' AS status
		FROM unique_seeds us
		WHERE NOT EXISTS (
			SELECT 1 FROM existing_prompts ep WHERE ep.prompt = us.prompt
		)
	`, cfg.MinLength)

	if _, err := db.Conn().Exec(createSQL); err != nil {
		return core.Fail(core.E("score.NormalizeSeeds", "create expansion_prompts", err))
	}

	var epCount int
	if err := db.Conn().QueryRow("SELECT count(*) FROM expansion_prompts").Scan(&epCount); err != nil {
		return core.Fail(core.E("score.NormalizeSeeds", "count expansion_prompts", err))
	}
	core.Print(w, "Expansion prompts created: %d (min length %d, deduped, excluding existing)", epCount, cfg.MinLength)

	if epCount == 0 {
		core.Print(w, "No new expansion prompts to process.")
		return core.Ok(nil)
	}

	// 3. Assign priority based on domain coverage.
	prioritySQL := `
		UPDATE expansion_prompts SET priority = sub.rnk
		FROM (
			SELECT domain, RANK() OVER (ORDER BY cnt ASC) AS rnk
			FROM (
				SELECT domain, count(*) AS cnt
				FROM expansion_prompts
				GROUP BY domain
			) domain_counts
		) sub
		WHERE expansion_prompts.domain = sub.domain
	`
	if _, err := db.Conn().Exec(prioritySQL); err != nil {
		return core.Fail(core.E("score.NormalizeSeeds", "assign priority", err))
	}
	core.Print(w, "Priority assigned (underrepresented domains ranked higher).")

	// 4. Region distribution summary.
	core.Print(w, "")
	core.Print(w, "Region distribution:")

	rows, err := db.Conn().Query(`
		SELECT
			CASE
				WHEN region LIKE 'cn%' THEN 'cn'
				WHEN region LIKE 'en%' THEN 'en'
				WHEN region LIKE 'ru%' THEN 'ru'
				WHEN region LIKE 'de%' THEN 'de'
				WHEN region LIKE 'es%' THEN 'es'
				WHEN region LIKE 'fr%' THEN 'fr'
				WHEN region LIKE 'latam%' THEN 'latam'
				WHEN region LIKE 'africa%' THEN 'africa'
				WHEN region LIKE 'eu%' THEN 'eu'
				WHEN region LIKE 'me%' THEN 'me'
				ELSE 'other'
			END AS region_group,
			count(*) AS cnt
		FROM expansion_prompts
		GROUP BY region_group
		ORDER BY cnt DESC
	`)
	if err != nil {
		return core.Fail(core.E("score.NormalizeSeeds", "region distribution query", err))
	}
	defer rows.Close()

	var totalFromRegions int
	var lines []string
	for rows.Next() {
		var region string
		var cnt int
		if err := rows.Scan(&region, &cnt); err != nil {
			return core.Fail(core.E("score.NormalizeSeeds", "scan region row", err))
		}
		totalFromRegions += cnt
		lines = append(lines, core.Sprintf("  %-10s %6d", region, cnt))
	}
	if err := rows.Err(); err != nil {
		return core.Fail(core.E("score.NormalizeSeeds", "iterate region rows", err))
	}

	for _, line := range lines {
		core.Print(w, "%s", line)
	}
	core.Print(w, "  %-10s %6d", repeatString("-", 10), totalFromRegions)
	core.Print(w, "  %-10s %6d", "total", totalFromRegions)

	return core.Ok(nil)
}
