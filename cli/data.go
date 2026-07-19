// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"flag"
	"io"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/cli/tui"
	"dappco.re/go/inference/dataset"
	"dappco.re/go/inference/serving/chathistory"
	coreio "dappco.re/go/io"
)

// runDataCommand wires go/dataset's root domain package (Dataset/Item/Score/
// Review/Export, the Store contract, ingest normalisation, heuristic +
// judge-tier scoring, export writers) as the `lem data` verb family: create,
// list, stats, import, score, export, and archive datasets under
// ~/.lem/datasets.duckdb, plus `review` — a pointer at the TUI Data panel
// (Task 8, not this verb's job). Every verb opens the store through
// tui.OpenDatasetStore, never hand-rolling the ~/.lem path itself.
//
//	lem data create evening-vents --title "Evening vents"
//	lem data import evening-vents --jsonl captures.jsonl
//	lem data score evening-vents --auto-approve 'lek>=80'
//	lem data score evening-vents --kind judge:quality --model ~/models/judge-4b
//	lem data export evening-vents --format sft-jsonl --out train.jsonl
func runDataCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		printDataUsage(stderr)
		return 2
	}
	switch args[0] {
	case "create":
		return runDataCreate(args[1:], stdout, stderr)
	case "list":
		return runDataList(args[1:], stdout, stderr)
	case "stats":
		return runDataStats(args[1:], stdout, stderr)
	case "import":
		return runDataImport(args[1:], stdout, stderr)
	case "score":
		return runDataScore(ctx, args[1:], stdout, stderr)
	case "export":
		return runDataExport(args[1:], stdout, stderr)
	case "archive":
		return runDataArchive(args[1:], stdout, stderr)
	case "review":
		return runDataReview(args[1:], stdout, stderr)
	case "-h", "--help", "help":
		printDataUsage(stdout)
		return 0
	default:
		core.Print(stderr, "%s data: unknown subcommand %q", cliName(), args[0])
		printDataUsage(stderr)
		return 2
	}
}

func printDataUsage(w io.Writer) {
	name := cliName()
	core.WriteString(w, core.Sprintf("Usage: %s data <subcommand> [flags]\n", name))
	core.WriteString(w, "\n")
	core.WriteString(w, "The lem-native training-data loop: capture -> score -> review -> export,\n")
	core.WriteString(w, "backed by ~/.lem/datasets.duckdb.\n")
	core.WriteString(w, "\n")
	core.WriteString(w, "Subcommands\n")
	core.WriteString(w, "  create   <slug>   create a new dataset\n")
	core.WriteString(w, "  list              list datasets\n")
	core.WriteString(w, "  stats    <slug>   item counts by kind / source / review status\n")
	core.WriteString(w, "  import   <slug>   ingest rows from --jsonl or --chats\n")
	core.WriteString(w, "  score    <slug>   run the heuristic lek scorer or a judge:<name> template, optionally auto-review by threshold\n")
	core.WriteString(w, "  export   <slug>   write JSONL + a manifest receipt for training\n")
	core.WriteString(w, "  archive  <slug>   flag a dataset archived (never a hard delete)\n")
	core.WriteString(w, "  review   [slug]   how to open the TUI Data panel\n")
	core.WriteString(w, "\n")
	core.WriteString(w, core.Sprintf("Run \"%s data <subcommand> --help\" for sub-action flags.\n", name))
}

// dataSubUsage returns a Usage function in the same synopsis + description +
// flags shape every other lem verb's sub-actions use (see packSubUsage).
func dataSubUsage(fs *flag.FlagSet, w io.Writer, synopsis, desc string) func() {
	return func() {
		core.WriteString(w, core.Sprintf("Usage: %s %s\n", cliName(), synopsis))
		core.WriteString(w, "\n")
		if desc != "" {
			core.WriteString(w, desc)
			core.WriteString(w, "\n\n")
		}
		core.WriteString(w, "Flags:\n")
		printFlagBlock(w, fs)
	}
}

// ---- shared helpers ----

// openDataStore opens the shared dataset store or prints an honest failure
// and returns the exit code callers should return immediately. A nil store
// on success never happens; callers check `if store == nil`.
func openDataStore(stderr io.Writer) (tui.DatasetStore, int) {
	opened := tui.OpenDatasetStore()
	if !opened.OK {
		core.Print(stderr, "%s data: %s", cliName(), opened.Error())
		return nil, 1
	}
	store, ok := opened.Value.(tui.DatasetStore)
	if !ok {
		core.Print(stderr, "%s data: unexpected dataset store result", cliName())
		return nil, 1
	}
	return store, 0
}

// resolveDatasetSlug looks up a dataset by slug, translating the Store's
// core.Result into a plain error for the verb functions' Go-idiom error
// handling.
func resolveDatasetSlug(store dataset.Store, slug string) (dataset.Dataset, error) {
	r := store.DatasetBySlug(slug)
	if !r.OK {
		return dataset.Dataset{}, r.Err()
	}
	ds, ok := r.Value.(dataset.Dataset)
	if !ok {
		return dataset.Dataset{}, core.NewError("dataset: unexpected DatasetBySlug result type")
	}
	return ds, nil
}

// printJSON marshals v compactly and writes it as one line — the shape every
// `--json` read verb emits for scripting.
func printJSON(w io.Writer, v any) {
	data := core.JSONMarshal(v)
	if !data.OK {
		core.Print(w, "error: %s", data.Error())
		return
	}
	core.WriteString(w, core.AsString(data.Value.([]byte)))
	core.WriteString(w, "\n")
}

// splitFilterClause splits a "key=value" --filter clause on the first bare
// '='. A ScoreExpression clause (>=, <=, ==, !=, >, <) never matches: "=="
// and "!=" are excluded by looking at the characters either side of the
// split point, and a bare '>' / '<' clause contains no '=' at all.
func splitFilterClause(clause string) (key, value string, ok bool) {
	idx := core.Index(clause, "=")
	if idx < 0 {
		return "", "", false
	}
	// ">=" / "<=" / "!=" all have one of these immediately before the '=';
	// "==" has a second '=' immediately after it (caught below) — between
	// the two checks, every two-char comparison operator is excluded.
	if idx > 0 {
		switch clause[idx-1] {
		case '>', '<', '!':
			return "", "", false
		}
	}
	if idx+1 < len(clause) && clause[idx+1] == '=' {
		return "", "", false
	}
	return core.Trim(clause[:idx]), core.Trim(clause[idx+1:]), true
}

// parseItemFilter parses --filter's tiny explicit grammar: comma-separated
// clauses, each either "<field>=<value>" (status / kind / source / archived)
// or a bare score expression in dataset.ParseScoreExpression's grammar (e.g.
// "lek>=80", reused rather than reinvented — Task 3 already built and tested
// it). An empty filterExpr is the zero ItemFilter beyond DatasetID — every
// non-archived item.
func parseItemFilter(datasetID, filterExpr string) (dataset.ItemFilter, error) {
	filter := dataset.ItemFilter{DatasetID: datasetID}
	filterExpr = core.Trim(filterExpr)
	if filterExpr == "" {
		return filter, nil
	}
	for _, clause := range core.Split(filterExpr, ",") {
		clause = core.Trim(clause)
		if clause == "" {
			continue
		}
		if key, value, ok := splitFilterClause(clause); ok {
			switch key {
			case "status":
				filter.Status = dataset.ReviewStatus(value)
			case "kind":
				filter.Kind = dataset.ItemKind(value)
			case "source":
				filter.Source = dataset.ItemSource(value)
			case "archived":
				filter.IncludeArchived = value == "true"
			default:
				return filter, core.E("main.parseItemFilter", core.Sprintf("unknown filter field %q", key), nil)
			}
			continue
		}
		exprResult := dataset.ParseScoreExpression(clause)
		if !exprResult.OK {
			return filter, core.E("main.parseItemFilter", core.Sprintf("unrecognised filter clause %q", clause), exprResult.Err())
		}
		parsed := exprResult.Value.(dataset.ScoreExpression)
		filter.Score = &parsed
	}
	return filter, nil
}

// ---- create ----

func runDataCreate(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("data create"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	title := fs.String("title", "", "human title (default: the slug)")
	purpose := fs.String("purpose", "", "one-line purpose note")
	jsonOut := fs.Bool("json", false, "print the created dataset as JSON")
	fs.Usage = dataSubUsage(fs, stderr, "data create [flags] <slug>",
		"Create a new named dataset under ~/.lem/datasets.duckdb. slug must be\n"+
			"lowercase alphanumeric-and-hyphen, e.g. \"evening-vents\".")
	slug, err := parseWithPositional(fs, args)
	if err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		if slug == "" {
			core.Print(stderr, "%s data create: expected exactly one <slug>", cliName())
		} else {
			core.Print(stderr, "%s data create: %s", cliName(), err.Error())
		}
		fs.Usage()
		return 2
	}

	store, code := openDataStore(stderr)
	if store == nil {
		return code
	}
	defer store.Close()

	titleValue := core.Trim(*title)
	if titleValue == "" {
		titleValue = slug
	}
	r := store.DatasetCreate(dataset.Dataset{
		ID: dataset.NewID(), Slug: slug, Title: titleValue, Purpose: *purpose, CreatedAt: time.Now(),
	})
	if !r.OK {
		core.Print(stderr, "%s data create: %s", cliName(), r.Error())
		return 1
	}
	created := r.Value.(dataset.Dataset)
	if *jsonOut {
		printJSON(stdout, created)
		return 0
	}
	core.Print(stdout, "created dataset %q (%s)", created.Slug, created.ID)
	return 0
}

// ---- list ----

func runDataList(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("data list"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	archived := fs.Bool("archived", false, "include archived datasets")
	jsonOut := fs.Bool("json", false, "print datasets as JSON")
	fs.Usage = dataSubUsage(fs, stderr, "data list [flags]", "List every dataset under ~/.lem/datasets.duckdb.")
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 0 {
		core.Print(stderr, "%s data list: unexpected argument %q", cliName(), fs.Arg(0))
		fs.Usage()
		return 2
	}

	store, code := openDataStore(stderr)
	if store == nil {
		return code
	}
	defer store.Close()

	r := store.Datasets(*archived)
	if !r.OK {
		core.Print(stderr, "%s data list: %s", cliName(), r.Error())
		return 1
	}
	list := r.Value.([]dataset.Dataset)
	if *jsonOut {
		printJSON(stdout, list)
		return 0
	}
	if len(list) == 0 {
		core.Print(stdout, "no datasets yet — create one with `%s data create <slug>`", cliName())
		return 0
	}
	for _, d := range list {
		state := ""
		if d.Archived {
			state = "  [archived]"
		}
		core.Print(stdout, "%-24s  %-32s  %s%s", d.Slug, d.Title, d.CreatedAt.Format(time.RFC3339), state)
	}
	return 0
}

// ---- stats ----

// dataStats is the `data stats` report shape — a CLI-local view type (not a
// go/dataset domain type), so it is free to carry the json tags a scripting
// consumer wants.
type dataStats struct {
	Dataset  string         `json:"dataset"`
	Total    int            `json:"total"`
	ByKind   map[string]int `json:"by_kind"`
	BySource map[string]int `json:"by_source"`
	ByStatus map[string]int `json:"by_status"`
}

func runDataStats(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("data stats"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	archived := fs.Bool("archived", false, "include archived items in the counts")
	jsonOut := fs.Bool("json", false, "print the stats as JSON")
	fs.Usage = dataSubUsage(fs, stderr, "data stats [flags] <slug>", "Item counts by kind, source, and latest review status.")
	slug, err := parseWithPositional(fs, args)
	if err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		if slug == "" {
			core.Print(stderr, "%s data stats: expected exactly one <slug>", cliName())
		} else {
			core.Print(stderr, "%s data stats: %s", cliName(), err.Error())
		}
		fs.Usage()
		return 2
	}

	store, code := openDataStore(stderr)
	if store == nil {
		return code
	}
	defer store.Close()
	ds, derr := resolveDatasetSlug(store, slug)
	if derr != nil {
		core.Print(stderr, "%s data stats: %s", cliName(), derr.Error())
		return 1
	}

	itemsResult := store.Items(dataset.ItemFilter{DatasetID: ds.ID, IncludeArchived: *archived})
	if !itemsResult.OK {
		core.Print(stderr, "%s data stats: %s", cliName(), itemsResult.Error())
		return 1
	}
	items := itemsResult.Value.([]dataset.Item)

	stats := dataStats{Dataset: slug, Total: len(items), ByKind: map[string]int{}, BySource: map[string]int{}, ByStatus: map[string]int{}}
	for _, item := range items {
		stats.ByKind[string(item.Kind)]++
		stats.BySource[string(item.Source)]++
		status := string(dataset.StatusPending)
		if reviewResult := store.ReviewLatest(item.ID); reviewResult.OK {
			if rv, ok := reviewResult.Value.(dataset.Review); ok {
				status = string(rv.Status)
			}
		}
		stats.ByStatus[status]++
	}

	if *jsonOut {
		printJSON(stdout, stats)
		return 0
	}
	core.Print(stdout, "dataset %q — %d items", slug, stats.Total)
	for kind, n := range stats.ByKind {
		core.Print(stdout, "  kind=%-10s %d", kind, n)
	}
	for source, n := range stats.BySource {
		core.Print(stdout, "  source=%-16s %d", source, n)
	}
	for status, n := range stats.ByStatus {
		core.Print(stdout, "  status=%-12s %d", status, n)
	}
	return 0
}

// ---- import ----

func runDataImport(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("data import"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonlPath := fs.String("jsonl", "", "ingest a JSONL file — {messages}, {prompt,response}, or CaptureRow rows")
	fingerprint := fs.String("fingerprint", "", "model fingerprint to stamp on --jsonl rows carrying no provenance of their own (default: empty, i.e. human/imported text)")
	chatsUser := fs.String("chats", "", "ingest a user's chathistory (~/Lethean/lem/users/<user>/chats.duckdb)")
	session := fs.String("session", "", "with --chats, import only this one conversation id (default: every conversation)")
	jsonOut := fs.Bool("json", false, "print the import report as JSON")
	fs.Usage = dataSubUsage(fs, stderr, "data import [flags] <slug>",
		"Ingest rows into a dataset — exactly one of --jsonl or --chats. Every\n"+
			"ingest path runs the welfare screen at the door and dedupes by content\n"+
			"hash within the dataset (a duplicate row is a counted no-op, not an error).")
	slug, err := parseWithPositional(fs, args)
	if err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		if slug == "" {
			core.Print(stderr, "%s data import: expected exactly one <slug>", cliName())
		} else {
			core.Print(stderr, "%s data import: %s", cliName(), err.Error())
		}
		fs.Usage()
		return 2
	}

	jsonlSet := core.Trim(*jsonlPath) != ""
	chatsSet := core.Trim(*chatsUser) != ""
	if jsonlSet == chatsSet {
		core.Print(stderr, "%s data import: exactly one of --jsonl or --chats is required", cliName())
		fs.Usage()
		return 2
	}
	if core.Trim(*session) != "" && !chatsSet {
		core.Print(stderr, "%s data import: --session requires --chats", cliName())
		return 2
	}

	store, code := openDataStore(stderr)
	if store == nil {
		return code
	}
	defer store.Close()
	ds, derr := resolveDatasetSlug(store, slug)
	if derr != nil {
		core.Print(stderr, "%s data import: %s", cliName(), derr.Error())
		return 1
	}

	if jsonlSet {
		return runDataImportJSONL(store, ds, *jsonlPath, *fingerprint, *jsonOut, stdout, stderr)
	}
	return runDataImportChats(store, ds, *chatsUser, *session, *jsonOut, stdout, stderr)
}

func runDataImportJSONL(store dataset.Store, ds dataset.Dataset, path, fingerprint string, jsonOut bool, stdout, stderr io.Writer) int {
	opened := core.Open(path)
	if !opened.OK {
		core.Print(stderr, "%s data import: open %s: %s", cliName(), path, opened.Error())
		return 1
	}
	file, ok := opened.Value.(*core.OSFile)
	if !ok {
		core.Print(stderr, "%s data import: unexpected file result for %s", cliName(), path)
		return 1
	}
	defer file.Close()

	r := dataset.IngestJSONL(store, ds.ID, file, dataset.IngestOptions{ModelFingerprint: fingerprint})
	if !r.OK {
		core.Print(stderr, "%s data import: %s", cliName(), r.Error())
		return 1
	}
	report := r.Value.(dataset.IngestReport)
	printImportReport(stdout, report, jsonOut)
	if len(report.Skipped) > 0 {
		return 1
	}
	return 0
}

func runDataImportChats(store dataset.Store, ds dataset.Dataset, userID, sessionID string, jsonOut bool, stdout, stderr io.Writer) int {
	homeResult := core.UserHomeDir()
	if !homeResult.OK {
		core.Print(stderr, "%s data import: resolve user home: %s", cliName(), homeResult.Error())
		return 1
	}
	home, ok := homeResult.Value.(string)
	if !ok || core.Trim(home) == "" {
		core.Print(stderr, "%s data import: user home is empty", cliName())
		return 1
	}
	// Storage convention documented in go/serving/chathistory's package doc:
	// one .duckdb per user at ~/Lethean/lem/users/<user_id>/chats.duckdb —
	// a different application root from ~/.lem (the dataset/TUI root), so
	// this path is derived here rather than through tui's path contract.
	chatsPath := core.Path(home, "Lethean", "lem", "users", userID, "chats.duckdb")
	if statResult := core.Stat(chatsPath); !statResult.OK {
		core.Print(stderr, "%s data import: no chat history found for user %q at %s", cliName(), userID, chatsPath)
		return 1
	}

	h, err := chathistory.Open(userID, chatsPath)
	if err != nil {
		core.Print(stderr, "%s data import: open chathistory: %s", cliName(), err.Error())
		return 1
	}
	defer h.Close()

	// chathistory.RecentConversations has no "give me everything" sentinel —
	// a limit this large is, in practice, "every conversation this user has".
	const allConversations = 1_000_000
	summaries, err := h.RecentConversations(allConversations)
	if err != nil {
		core.Print(stderr, "%s data import: list conversations: %s", cliName(), err.Error())
		return 1
	}
	if core.Trim(sessionID) != "" {
		filtered := summaries[:0]
		for _, c := range summaries {
			if c.ID == sessionID {
				filtered = append(filtered, c)
				break
			}
		}
		if len(filtered) == 0 {
			core.Print(stderr, "%s data import: session %q not found for user %q", cliName(), sessionID, userID)
			return 1
		}
		summaries = filtered
	}

	sessions := make([]dataset.ChatSession, 0, len(summaries))
	for _, c := range summaries {
		turns, terr := h.LoadTurns(c.ID)
		if terr != nil {
			core.Print(stderr, "%s data import: warning: skipping session %s: %s", cliName(), c.ID, terr.Error())
			continue
		}
		chatTurns := make([]dataset.ChatTurn, len(turns))
		for i, t := range turns {
			chatTurns[i] = dataset.ChatTurn{Role: t.Role, Content: t.Content, Ordinal: t.Ordinal}
		}
		sessions = append(sessions, dataset.ChatSession{ID: c.ID, Title: c.Title, ModelID: c.ModelID, StartedAt: c.StartedAt, Turns: chatTurns})
	}

	r := dataset.IngestChatSessions(store, ds.ID, sessions)
	if !r.OK {
		core.Print(stderr, "%s data import: %s", cliName(), r.Error())
		return 1
	}
	report := r.Value.(dataset.IngestReport)
	printImportReport(stdout, report, jsonOut)
	if len(report.Skipped) > 0 {
		return 1
	}
	return 0
}

func printImportReport(w io.Writer, report dataset.IngestReport, jsonOut bool) {
	if jsonOut {
		printJSON(w, report)
		return
	}
	core.Print(w, "import: ingested=%d deduped=%d quarantined=%d skipped=%d", report.Ingested, report.Deduped, report.Quarantined, len(report.Skipped))
	for _, skip := range report.Skipped {
		core.Print(w, "  row %d: %s", skip.Row, skip.Reason)
	}
}

// ---- score ----

// dataScoreReport is the `data score` report shape — CLI-local, json-tagged.
type dataScoreReport struct {
	Dataset      string `json:"dataset"`
	Scored       int    `json:"scored"`
	Failed       int    `json:"failed"`
	AutoApproved int    `json:"auto_approved"`
	AutoRejected int    `json:"auto_rejected"`
}

func runDataScore(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("data score"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	kind := fs.String("kind", "lek", "score kind: 'lek' (the heuristic tier) or judge:<name> (a named judge template — requires --model)")
	modelPath := fs.String("model", "", "judge model checkpoint path (required for --kind judge:<name>; unused for lek)")
	filterExpr := fs.String("filter", "", "which items to score — comma-separated status=/kind=/source=/archived= clauses and/or a score expression (default: every non-archived item)")
	autoApprove := fs.String("auto-approve", "", "auto-approve items whose fresh score satisfies this expression, e.g. lek>=80 (explicit only, never implicit)")
	autoReject := fs.String("auto-reject", "", "auto-reject items whose fresh score satisfies this expression (checked before --auto-approve)")
	jsonOut := fs.Bool("json", false, "print the score report as JSON")
	fs.Usage = dataSubUsage(fs, stderr, "data score [flags] <slug>",
		"Score every matching item with the heuristic lek tier (lek.ScorePair) or a\n"+
			"named judge template (--kind judge:<name> --model <checkpoint path>),\n"+
			"optionally auto-reviewing by an explicit threshold expression.")
	slug, err := parseWithPositional(fs, args)
	if err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		if slug == "" {
			core.Print(stderr, "%s data score: expected exactly one <slug>", cliName())
		} else {
			core.Print(stderr, "%s data score: %s", cliName(), err.Error())
		}
		fs.Usage()
		return 2
	}

	scoreKind := dataset.ScoreKind(*kind)
	isJudge := scoreKind.IsJudge()
	if !isJudge && scoreKind != dataset.ScoreKindLEK {
		core.Print(stderr, "%s data score: unknown --kind %q (want 'lek' or 'judge:<name>')", cliName(), *kind)
		return 2
	}
	if isJudge && core.Trim(*modelPath) == "" {
		core.Print(stderr, "%s data score: --kind %s requires --model <judge checkpoint path>", cliName(), *kind)
		return 2
	}

	var approveExpr, rejectExpr *dataset.ScoreExpression
	if core.Trim(*autoApprove) != "" {
		exprResult := dataset.ParseScoreExpression(*autoApprove)
		if !exprResult.OK {
			core.Print(stderr, "%s data score: --auto-approve: %s", cliName(), exprResult.Error())
			return 2
		}
		parsed := exprResult.Value.(dataset.ScoreExpression)
		approveExpr = &parsed
	}
	if core.Trim(*autoReject) != "" {
		exprResult := dataset.ParseScoreExpression(*autoReject)
		if !exprResult.OK {
			core.Print(stderr, "%s data score: --auto-reject: %s", cliName(), exprResult.Error())
			return 2
		}
		parsed := exprResult.Value.(dataset.ScoreExpression)
		rejectExpr = &parsed
	}

	store, code := openDataStore(stderr)
	if store == nil {
		return code
	}
	defer store.Close()
	ds, derr := resolveDatasetSlug(store, slug)
	if derr != nil {
		core.Print(stderr, "%s data score: %s", cliName(), derr.Error())
		return 1
	}

	filter, ferr := parseItemFilter(ds.ID, *filterExpr)
	if ferr != nil {
		core.Print(stderr, "%s data score: --filter: %s", cliName(), ferr.Error())
		return 2
	}

	itemsResult := store.Items(filter)
	if !itemsResult.OK {
		core.Print(stderr, "%s data score: %s", cliName(), itemsResult.Error())
		return 1
	}
	items := itemsResult.Value.([]dataset.Item)

	// The judge model is loaded ONCE, only when there is actually something
	// to score — never per item, and never merely because --model was set
	// on an empty result set.
	var driver dataset.JudgeDispatcher
	if isJudge && len(items) > 0 {
		d, closeJudge, jerr := newJudgeDispatcher(*modelPath, judgeDefaultMaxTokens)
		if jerr != nil {
			core.Print(stderr, "%s data score: %s", cliName(), jerr.Error())
			return 1
		}
		defer closeJudge()
		driver = d
	}

	report := dataScoreReport{Dataset: slug}
	judgeName := scoreKind.JudgeName()
	for _, item := range items {
		var scores []dataset.Score
		if isJudge {
			scoreResult := dataset.ScoreJudge(ctx, store, driver, item, judgeName)
			if !scoreResult.OK {
				// Malformed judge output (a non-bare-number or out-of-range
				// reply) is a loud per-item error, never a silent 0 — print
				// it and keep scoring the rest of the batch.
				report.Failed++
				core.Print(stderr, "%s data score: item %s: %s", cliName(), item.ID, scoreResult.Error())
				continue
			}
			score, ok := scoreResult.Value.(dataset.Score)
			if !ok {
				report.Failed++
				continue
			}
			scores = []dataset.Score{score}
		} else {
			scoreResult := dataset.ScoreHeuristicAppend(store, item)
			if !scoreResult.OK {
				report.Failed++
				continue
			}
			var ok bool
			scores, ok = scoreResult.Value.([]dataset.Score)
			if !ok {
				continue
			}
		}
		report.Scored++
		if approveExpr == nil && rejectExpr == nil {
			continue
		}
		thresholdResult := dataset.ApplyAutoThreshold(store, item, scores, approveExpr, rejectExpr)
		if !thresholdResult.OK {
			continue
		}
		outcome, ok := thresholdResult.Value.(dataset.AutoThresholdResult)
		if !ok || !outcome.Applied {
			continue
		}
		switch outcome.Review.Status {
		case dataset.StatusApproved:
			report.AutoApproved++
		case dataset.StatusRejected:
			report.AutoRejected++
		}
	}

	if *jsonOut {
		printJSON(stdout, report)
	} else {
		core.Print(stdout, "score: dataset %q — scored=%d failed=%d auto_approved=%d auto_rejected=%d",
			slug, report.Scored, report.Failed, report.AutoApproved, report.AutoRejected)
	}
	if report.Failed > 0 {
		return 1
	}
	return 0
}

// ---- export ----

func runDataExport(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("data export"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	format := fs.String("format", "", "export format: sft-jsonl, pairs-jsonl, or capture-jsonl (required)")
	out := fs.String("out", "", "output JSONL path (required; a sidecar <out>.manifest.json is written alongside)")
	filterExpr := fs.String("filter", "", "which items to export — comma-separated clauses (default: status=approved; exporting anything else requires an explicit filter)")
	jsonOut := fs.Bool("json", false, "print the export manifest as JSON")
	fs.Usage = dataSubUsage(fs, stderr, "data export [flags] <slug>",
		"Export a dataset to JSONL with a manifest receipt (counts, filter, per-item\n"+
			"content hashes, the manifest hash) a training run can name exactly what it saw.")
	slug, err := parseWithPositional(fs, args)
	if err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		if slug == "" {
			core.Print(stderr, "%s data export: expected exactly one <slug>", cliName())
		} else {
			core.Print(stderr, "%s data export: %s", cliName(), err.Error())
		}
		fs.Usage()
		return 2
	}
	if core.Trim(*format) == "" || core.Trim(*out) == "" {
		core.Print(stderr, "%s data export: --format and --out are required", cliName())
		fs.Usage()
		return 2
	}

	store, code := openDataStore(stderr)
	if store == nil {
		return code
	}
	defer store.Close()
	ds, derr := resolveDatasetSlug(store, slug)
	if derr != nil {
		core.Print(stderr, "%s data export: %s", cliName(), derr.Error())
		return 1
	}

	filter, ferr := parseItemFilter(ds.ID, *filterExpr)
	if ferr != nil {
		core.Print(stderr, "%s data export: --filter: %s", cliName(), ferr.Error())
		return 2
	}

	exportResult := dataset.ExportDataset(store, coreio.Local, dataset.ExportRequest{
		DatasetID: ds.ID, Format: dataset.ExportFormat(*format), Filter: filter, OutputPath: *out,
	})
	if !exportResult.OK {
		core.Print(stderr, "%s data export: %s", cliName(), exportResult.Error())
		return 1
	}

	manifest, merr := readExportManifest(*out)
	if merr != nil {
		core.Print(stderr, "%s data export: wrote %s but could not read back the manifest: %s", cliName(), *out, merr.Error())
		return 1
	}
	if *jsonOut {
		printJSON(stdout, manifest)
	} else {
		core.Print(stdout, "export: dataset %q -> %s (%s)", slug, *out, *format)
		core.Print(stdout, "  items=%d skipped=%d filter=%q manifest=%s", manifest.ItemCount, manifest.SkippedCount, manifest.FilterDescription, manifest.ManifestHash)
	}
	if manifest.SkippedCount > 0 {
		return 1
	}
	return 0
}

// readExportManifest reads back the sidecar dataset.ExportDataset just wrote
// so the verb can report the honest written/skipped counts the library
// itself only persists to disk, not through ExportDataset's Result.
func readExportManifest(outputPath string) (dataset.ExportManifest, error) {
	content, err := coreio.Local.Read(core.Concat(outputPath, ".manifest.json"))
	if err != nil {
		return dataset.ExportManifest{}, err
	}
	var manifest dataset.ExportManifest
	if r := core.JSONUnmarshalString(content, &manifest); !r.OK {
		return dataset.ExportManifest{}, r.Err()
	}
	return manifest, nil
}

// ---- archive ----

func runDataArchive(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("data archive"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print the archived dataset as JSON")
	fs.Usage = dataSubUsage(fs, stderr, "data archive [flags] <slug>", "Flag a dataset archived. Never a hard delete — its items remain queryable with --archived.")
	slug, err := parseWithPositional(fs, args)
	if err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		if slug == "" {
			core.Print(stderr, "%s data archive: expected exactly one <slug>", cliName())
		} else {
			core.Print(stderr, "%s data archive: %s", cliName(), err.Error())
		}
		fs.Usage()
		return 2
	}

	store, code := openDataStore(stderr)
	if store == nil {
		return code
	}
	defer store.Close()
	ds, derr := resolveDatasetSlug(store, slug)
	if derr != nil {
		core.Print(stderr, "%s data archive: %s", cliName(), derr.Error())
		return 1
	}

	r := store.DatasetArchive(ds.ID)
	if !r.OK {
		core.Print(stderr, "%s data archive: %s", cliName(), r.Error())
		return 1
	}
	archived := r.Value.(dataset.Dataset)
	if *jsonOut {
		printJSON(stdout, archived)
		return 0
	}
	core.Print(stdout, "archived dataset %q", slug)
	return 0
}

// ---- review ----

// runDataReview points at the TUI Data panel. The panel itself is Task 8 of
// docs/superpowers/plans/2026-07-19-lem-dataset-loop.md, not this verb's —
// cli/tui/tabs.go and the panel files are out of scope here, and as of this
// writing that task has not landed, so this prints the honest state rather
// than a broken promise.
func runDataReview(args []string, stdout, stderr io.Writer) int {
	if len(args) > 0 {
		switch args[0] {
		case "-h", "--help", "help":
			core.WriteString(stdout, core.Sprintf("Usage: %s data review [slug]\n\n", cliName()))
			core.WriteString(stdout, "Print how to open the TUI Data panel for interactive review.\n")
			return 0
		}
	}
	core.Print(stdout, "the Data review panel lands with Task 8 (tracker #43) — not available yet.")
	core.Print(stdout, "see docs/superpowers/plans/2026-07-19-lem-dataset-loop.md (Task 8) for status.")
	core.Print(stdout, "for now: `%s data list` / `%s data stats <slug>` inspect a dataset headlessly.", cliName(), cliName())
	return 0
}
