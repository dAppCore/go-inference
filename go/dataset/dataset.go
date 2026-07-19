// SPDX-Licence-Identifier: EUPL-1.2

package dataset

import (
	"time"

	core "dappco.re/go"
	"github.com/google/uuid"
)

// ItemKind is the shape an Item's Content carries — pair (prompt/
// response), messages (full chat turns), or trace (an opaque SSD
// sample). Kind decides how Content is validated and hashed.
type ItemKind string

const (
	// KindPair is a bare prompt/response exchange — [PairContent].
	KindPair ItemKind = "pair"
	// KindMessages is a full multi-turn chat — [MessagesContent].
	KindMessages ItemKind = "messages"
	// KindTrace is an opaque SSD self-distillation sample: non-empty
	// JSON, shape owned by the sampler, never parsed here.
	KindTrace ItemKind = "trace"
)

// knownItemKind reports whether kind is one this package knows how to
// validate and hash.
func knownItemKind(kind ItemKind) bool {
	switch kind {
	case KindPair, KindMessages, KindTrace:
		return true
	default:
		return false
	}
}

// ItemSource names where an Item's content originated — the ingest door
// it walked through. Every ingest path in
// docs/superpowers/specs/2026-07-19-lem-dataset-loop-design.md maps to
// exactly one of these.
type ItemSource string

const (
	// SourceCaptureServe is a `lem serve --capture <slug>` tee.
	SourceCaptureServe ItemSource = "capture:serve"
	// SourceCaptureTrain is an SFT training-loop eval capture.
	SourceCaptureTrain ItemSource = "capture:train"
	// SourceImportChathistory is a normalised chathistory session.
	SourceImportChathistory ItemSource = "import:chathistory"
	// SourceImportJSONL is a `lem data import --jsonl` row.
	SourceImportJSONL ItemSource = "import:jsonl"
	// SourceSSD is a `lem ssd --dataset` sampled trace.
	SourceSSD ItemSource = "ssd"
)

// knownItemSource reports whether source is one this package recognises.
func knownItemSource(source ItemSource) bool {
	switch source {
	case SourceCaptureServe, SourceCaptureTrain, SourceImportChathistory, SourceImportJSONL, SourceSSD:
		return true
	default:
		return false
	}
}

// ReviewStatus is an Item's durable review lifecycle state. The zero
// value of a fresh Item (before any Review row exists) is StatusPending.
type ReviewStatus string

const (
	// StatusPending has not been reviewed yet — the implicit starting
	// state before any Review row exists for an item.
	StatusPending ReviewStatus = "pending"
	// StatusApproved is cleared for export/training.
	StatusApproved ReviewStatus = "approved"
	// StatusRejected was reviewed and declined.
	StatusRejected ReviewStatus = "rejected"
	// StatusQuarantined was flagged by the welfare screen or a human —
	// visible and reviewable, never silently dropped.
	StatusQuarantined ReviewStatus = "quarantined"
)

// knownReviewStatus reports whether status is one this package recognises.
func knownReviewStatus(status ReviewStatus) bool {
	switch status {
	case StatusPending, StatusApproved, StatusRejected, StatusQuarantined:
		return true
	default:
		return false
	}
}

// Automated reviewer identities. A human reviewer is any other non-empty
// string (e.g. "snider") — Reviewer is deliberately not a closed enum.
const (
	// ReviewerAutoWelfare marks a Review written by the ingest-door
	// welfare screen.
	ReviewerAutoWelfare = "auto:welfare"
	// ReviewerAutoThreshold marks a Review written by an explicit
	// --auto-approve/--auto-reject score expression.
	ReviewerAutoThreshold = "auto:threshold"
)

// ScoreKind names what a Score row measures. The three heuristic kinds
// are fixed; judge-tier kinds are dynamic — build one with
// [JudgeScoreKind].
type ScoreKind string

const (
	// ScoreKindLEK is the lem-scorer composite (0-100) — [lek.ScorePair].
	ScoreKindLEK ScoreKind = "lek"
	// ScoreKindHostility is the directed-anger read (0-1), derived from
	// the same lek.ScorePair call as ScoreKindLEK.
	ScoreKindHostility ScoreKind = "hostility"
	// ScoreKindSycophancy is the sycophancy composite (0-100), derived
	// from the same lek.ScorePair call as ScoreKindLEK.
	ScoreKindSycophancy ScoreKind = "sycophancy"
)

// judgeScoreKindPrefix namespaces judge-tier score kinds from the fixed
// heuristic ones — "judge:<template name>".
const judgeScoreKindPrefix = "judge:"

// JudgeScoreKind builds the ScoreKind for a named judge template.
//
//	k := dataset.JudgeScoreKind("helpfulness") // ScoreKind("judge:helpfulness")
func JudgeScoreKind(name string) ScoreKind {
	return ScoreKind(judgeScoreKindPrefix + core.Trim(name))
}

// IsJudge reports whether k is a judge-tier kind built by [JudgeScoreKind].
//
//	dataset.ScoreKindLEK.IsJudge()             // false
//	dataset.JudgeScoreKind("helpfulness").IsJudge() // true
func (k ScoreKind) IsJudge() bool {
	return len(k) > len(judgeScoreKindPrefix) && core.HasPrefix(string(k), judgeScoreKindPrefix)
}

// JudgeName returns the template name for a judge-tier kind, or "" when
// k is not a judge kind.
//
//	dataset.JudgeScoreKind("helpfulness").JudgeName() // "helpfulness"
func (k ScoreKind) JudgeName() string {
	if !k.IsJudge() {
		return ""
	}
	return string(k)[len(judgeScoreKindPrefix):]
}

// knownScoreKind reports whether kind is a recognised heuristic kind or a
// well-formed judge kind.
func knownScoreKind(kind ScoreKind) bool {
	switch kind {
	case ScoreKindLEK, ScoreKindHostility, ScoreKindSycophancy:
		return true
	default:
		return kind.IsJudge()
	}
}

// ExportFormat names one of the fixed v1 export writers.
type ExportFormat string

const (
	// FormatSFTJSONL writes {"messages":[...]} rows — the `lem train
	// --sft` contract.
	FormatSFTJSONL ExportFormat = "sft-jsonl"
	// FormatPairsJSONL writes {"prompt":...,"response":...} rows.
	FormatPairsJSONL ExportFormat = "pairs-jsonl"
	// FormatCaptureJSONL writes {"step":...,"prompt":...,"text":...,
	// "at_unix":...} rows — round-trips train.CaptureRow's shape.
	FormatCaptureJSONL ExportFormat = "capture-jsonl"
)

// knownExportFormat reports whether format is one this package can write.
func knownExportFormat(format ExportFormat) bool {
	switch format {
	case FormatSFTJSONL, FormatPairsJSONL, FormatCaptureJSONL:
		return true
	default:
		return false
	}
}

// Dataset is a named, durable collection of training-data Items.
// Never hard-deleted — Archived + ArchivedAt record retirement.
type Dataset struct {
	ID         string
	Slug       string // unique, human, e.g. "evening-vents"
	Title      string
	Purpose    string
	CreatedAt  time.Time
	Archived   bool
	ArchivedAt time.Time
}

// Item is one unit of training data — a prompt/response pair, a full
// chat, or an opaque SSD trace. Content is JSON, shaped by Kind; see
// [ValidateItemContent]. Never hard-deleted — Archived + ArchivedAt
// record retirement; ParentItemID records edit lineage (an edit-and-
// approve action archives the original and creates a derived Item).
type Item struct {
	ID        string
	DatasetID string
	Kind      ItemKind
	// Content is Kind-shaped JSON: [PairContent] for KindPair,
	// [MessagesContent] for KindMessages, opaque non-empty JSON for
	// KindTrace. Validate with [ValidateItemContent].
	Content []byte
	Source  ItemSource
	// SourceRef names the origin more precisely within Source — a chat
	// session id for import:chathistory, a training step for
	// capture:train, etc. Optional.
	SourceRef string
	// ModelFingerprint identifies the GENERATING model, stamped at
	// birth — empty for human-authored or provenance-less imported
	// text (the fingerprint rule: never re-score an old response
	// against a later model state; a score always names what produced
	// it).
	ModelFingerprint string
	// ContentHash is the canonicalised content hash — see
	// [ContentHash]. Used for within-dataset dedupe.
	ContentHash string
	// ParentItemID is the source Item this one was derived from by an
	// edit-and-approve action, or "" for an original.
	ParentItemID string
	Archived     bool
	ArchivedAt   time.Time
	CreatedAt    time.Time
}

// Score is one scoring pass over an Item. Scores are append-only —
// re-scoring adds a row, it never overwrites; the score history IS the
// record. ScorerName + ScorerVersion identify a heuristic scorer;
// JudgeFingerprint identifies a judge-tier scorer (the judge model
// itself) — exactly one of the two identifies a Score row.
type Score struct {
	ID     string
	ItemID string
	Kind   ScoreKind
	Value  float64
	// Payload is the full JSON detail behind Value — e.g. the whole
	// lek.DiffResult for a ScoreKindLEK row.
	Payload []byte
	// ScorerName + ScorerVersion identify a heuristic scorer (e.g.
	// "lek" / "1"). Empty for judge-tier rows — see JudgeFingerprint.
	ScorerName    string
	ScorerVersion string
	// JudgeFingerprint identifies the judge model that produced a
	// judge-tier row. Empty for heuristic rows.
	JudgeFingerprint string
	CreatedAt        time.Time
}

// Review is one review decision on an Item. Reviews are append-only —
// the latest row (by CreatedAt) is the Item's current status; history is
// kept. Reviewer is "auto:welfare" / "auto:threshold" (see the
// Reviewer* constants) or a human identity (e.g. "snider"). Note is
// required for a quarantine-clear action (enforced by the TUI, not
// here).
type Review struct {
	ItemID    string
	Status    ReviewStatus
	Reviewer  string
	Note      string
	CreatedAt time.Time
}

// Export is the durable receipt of one dataset export run — the manifest
// hash lets a training run name exactly what it saw.
type Export struct {
	ID                string
	DatasetID         string
	Format            ExportFormat
	FilterDescription string
	ItemCount         int
	OutputPath        string
	// ManifestHash is the sha256 hex digest over the ordered exported
	// items' content hashes — see [ManifestHash].
	ManifestHash string
	CreatedAt    time.Time
}

// PairContent is the Content shape for a KindPair Item.
type PairContent struct {
	Prompt   string `json:"prompt"`
	Response string `json:"response"`
}

// MessageTurn is one role/content turn inside a KindMessages Item.
// Role is "user" / "assistant" / "system" / "tool".
type MessageTurn struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// MessagesContent is the Content shape for a KindMessages Item. It is
// also the exact `lem train --sft` wire row shape ({"messages":[...]})
// — see the sft-jsonl export writer.
type MessagesContent struct {
	Messages []MessageTurn `json:"messages"`
}

// LastExchange reduces a multi-turn conversation to its final
// prompt/response pair: every turn before the last assistant turn,
// newline-joined, as the prompt; the last assistant turn's content as
// the response. Mirrors the reduction
// dappco.re/go/inference/train/dataset.MessagesToSample uses for its
// trailing-assistant-turn case (reimplemented here, not imported, to
// keep this package free of train's engine dependency). ok is false
// when mc has no assistant turn to reduce to.
//
//	pc, ok := mc.LastExchange()
func (mc MessagesContent) LastExchange() (PairContent, bool) {
	assistantIdx := -1
	for i := len(mc.Messages) - 1; i >= 0; i-- {
		if mc.Messages[i].Role == "assistant" {
			assistantIdx = i
			break
		}
	}
	if assistantIdx < 0 {
		return PairContent{}, false
	}
	var sb core.Builder
	for _, turn := range mc.Messages[:assistantIdx] {
		if turn.Content == "" {
			continue
		}
		sb.WriteString(turn.Content)
		sb.WriteString("\n")
	}
	return PairContent{
		Prompt:   sb.String(),
		Response: mc.Messages[assistantIdx].Content,
	}, true
}

// knownMessageRole reports whether role is one this package accepts in a
// MessagesContent turn.
func knownMessageRole(role string) bool {
	switch role {
	case "user", "assistant", "system", "tool":
		return true
	default:
		return false
	}
}

// ValidateItemContent validates Content against the shape Kind requires:
// a KindPair must carry a non-empty prompt and response; a KindMessages
// must carry at least one turn, every turn a known non-empty role and
// non-empty content, with no two adjacent turns sharing the same role
// (the "role-alternating" rule — it accommodates tool-call sequences
// like assistant→tool→assistant, which are not adjacent-repeats, while
// still catching mangled/duplicated turns); a KindTrace must be
// non-empty valid JSON, opaque otherwise.
//
//	r := dataset.ValidateItemContent(dataset.KindPair, content)
//	if !r.OK { return r }
func ValidateItemContent(kind ItemKind, content []byte) core.Result {
	switch kind {
	case KindPair:
		return validatePairContent(content)
	case KindMessages:
		return validateMessagesContent(content)
	case KindTrace:
		return validateTraceContent(content)
	default:
		return core.Fail(core.NewError("dataset: unknown item kind"))
	}
}

func validatePairContent(content []byte) core.Result {
	var pc PairContent
	if r := core.JSONUnmarshal(content, &pc); !r.OK {
		return core.Fail(core.E("dataset.ValidateItemContent", "pair content is not valid JSON", r.Err()))
	}
	if core.Trim(pc.Prompt) == "" {
		return core.Fail(core.NewError("dataset: pair content requires a non-empty prompt"))
	}
	if core.Trim(pc.Response) == "" {
		return core.Fail(core.NewError("dataset: pair content requires a non-empty response"))
	}
	return core.Ok(pc)
}

func validateMessagesContent(content []byte) core.Result {
	var mc MessagesContent
	if r := core.JSONUnmarshal(content, &mc); !r.OK {
		return core.Fail(core.E("dataset.ValidateItemContent", "messages content is not valid JSON", r.Err()))
	}
	if len(mc.Messages) == 0 {
		return core.Fail(core.NewError("dataset: messages content requires at least one turn"))
	}
	previousRole := ""
	for i, turn := range mc.Messages {
		if !knownMessageRole(turn.Role) {
			return core.Fail(core.E("dataset.ValidateItemContent", core.Sprintf("messages content turn %d has an unknown role %q", i, turn.Role), nil))
		}
		if core.Trim(turn.Content) == "" {
			return core.Fail(core.E("dataset.ValidateItemContent", core.Sprintf("messages content turn %d has empty content", i), nil))
		}
		if turn.Role == previousRole {
			return core.Fail(core.E("dataset.ValidateItemContent", core.Sprintf("messages content turn %d repeats the previous turn's role %q — turns must alternate", i, turn.Role), nil))
		}
		previousRole = turn.Role
	}
	return core.Ok(mc)
}

func validateTraceContent(content []byte) core.Result {
	if len(content) == 0 {
		return core.Fail(core.NewError("dataset: trace content must be non-empty"))
	}
	if !core.JSONValid(content) {
		return core.Fail(core.NewError("dataset: trace content must be valid JSON"))
	}
	return core.Ok(content)
}

// NewID returns a fresh unique identifier for a Dataset/Item/Score/
// Export row — a v4 UUID, matching the convention
// dappco.re/go/inference/cli/tui already uses for durable records.
func NewID() string {
	return uuid.NewString()
}
