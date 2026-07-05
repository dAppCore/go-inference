package score

import (
	"io"

	"dappco.re/go"
	"dappco.re/go/store"
)

// regionRow holds a single row from the region distribution query.
type regionRow struct {
	group   string
	n       int
	domains int
}

// PrintCoverage analyzes seed coverage by region and domain, printing
// a report with bar chart visualization and gap recommendations.
func PrintCoverage(db *store.DuckDB, w io.Writer) core.Result {
	rows, result := db.QueryRows("SELECT count(*) AS total FROM seeds")
	if !result.OK {
		return core.Fail(core.E("score.PrintCoverage", "count seeds", result.Value.(error)))
	}
	if len(rows) == 0 {
		return core.Fail(core.E("score.PrintCoverage", "no seeds table found (run: core ml import-all first)", nil))
	}
	total := toInt(rows[0]["total"])

	core.Print(w, "LEM Seed Coverage Analysis")
	core.Print(w, "==================================================")
	core.Print(w, "")
	core.Print(w, "Total seeds: %d", total)

	// Region distribution.
	regionResult := queryRegionDistribution(db)
	if !regionResult.OK {
		return core.Fail(core.E("score.PrintCoverage", "query regions", regionResult.Value.(error)))
	}
	regionRows := regionResult.Value.([]regionRow)

	core.Print(w, "")
	core.Print(w, "Region distribution (underrepresented first):")
	avg := float64(total) / float64(len(regionRows))
	for _, r := range regionRows {
		barLen := min(int(float64(r.n)/avg*10), 40)
		bar := repeatString("#", barLen)
		gap := ""
		if float64(r.n) < avg*0.5 {
			gap = "  <- UNDERREPRESENTED"
		}
		core.Print(w, "  %-22s %6d  (%4d domains)  %s%s", r.group, r.n, r.domains, bar, gap)
	}

	// Top 10 domains.
	core.Print(w, "")
	core.Print(w, "Top 10 domains (most seeds):")
	topRows, result := db.QueryRows(`
		SELECT domain, count(*) AS n FROM seeds
		WHERE domain != '' GROUP BY domain ORDER BY n DESC LIMIT 10
	`)
	if result.OK {
		for _, row := range topRows {
			domain := strVal(row, "domain")
			n := toInt(row["n"])
			core.Print(w, "  %-40s %5d", domain, n)
		}
	}

	// Bottom 10 domains.
	core.Print(w, "")
	core.Print(w, "Bottom 10 domains (fewest seeds, min 5):")
	bottomRows, result := db.QueryRows(`
		SELECT domain, count(*) AS n FROM seeds
		WHERE domain != '' GROUP BY domain HAVING count(*) >= 5 ORDER BY n ASC LIMIT 10
	`)
	if result.OK {
		for _, row := range bottomRows {
			domain := strVal(row, "domain")
			n := toInt(row["n"])
			core.Print(w, "  %-40s %5d", domain, n)
		}
	}

	core.Print(w, "")
	core.Print(w, "Suggested expansion areas:")
	core.Print(w, "  - Japanese, Korean, Thai, Vietnamese (no seeds found)")
	core.Print(w, "  - Hindi/Urdu, Bengali, Tamil (South Asian)")
	core.Print(w, "  - Swahili, Yoruba, Amharic (Sub-Saharan Africa)")
	core.Print(w, "  - Indigenous languages (Quechua, Nahuatl, Aymara)")

	return core.Ok(nil)
}

func repeatString(part string, count int) string {
	if count <= 0 {
		return ""
	}
	b := core.NewBuilder()
	for range count {
		b.WriteString(part)
	}
	return b.String()
}

// queryRegionDistribution returns seed counts grouped by normalized language
// region, ordered ascending (underrepresented first).
func queryRegionDistribution(db *store.DuckDB) core.Result {
	rows, result := db.QueryRows(`
		SELECT
			CASE
				WHEN region LIKE '%cn%' THEN 'cn (Chinese)'
				WHEN region LIKE '%en-%' OR region LIKE '%en_para%' OR region LIKE '%para%' THEN 'en (English)'
				WHEN region LIKE '%ru%' THEN 'ru (Russian)'
				WHEN region LIKE '%de%' AND region NOT LIKE '%deten%' THEN 'de (German)'
				WHEN region LIKE '%es%' THEN 'es (Spanish)'
				WHEN region LIKE '%fr%' THEN 'fr (French)'
				WHEN region LIKE '%latam%' THEN 'latam (LatAm)'
				WHEN region LIKE '%africa%' THEN 'africa'
				WHEN region LIKE '%eu%' THEN 'eu (European)'
				WHEN region LIKE '%me%' AND region NOT LIKE '%premium%' THEN 'me (MidEast)'
				WHEN region LIKE '%multi%' THEN 'multilingual'
				WHEN region LIKE '%weak%' THEN 'weak-langs'
				ELSE 'other'
			END AS lang_group,
			count(*) AS n,
			count(DISTINCT domain) AS domains
		FROM seeds GROUP BY lang_group ORDER BY n ASC
	`)
	if !result.OK {
		return core.Fail(core.E("score.queryRegionDistribution", "query rows", result.Value.(error)))
	}

	regions := make([]regionRow, 0, len(rows))
	for _, row := range rows {
		regions = append(regions, regionRow{
			group:   strVal(row, "lang_group"),
			n:       toInt(row["n"]),
			domains: toInt(row["domains"]),
		})
	}
	return core.Ok(regions)
}
