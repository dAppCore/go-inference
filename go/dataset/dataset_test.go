// SPDX-Licence-Identifier: EUPL-1.2

package dataset

import (
	"time"

	core "dappco.re/go"
)

func TestKnownItemKind(t *core.T) {
	for _, kind := range []ItemKind{KindPair, KindMessages, KindTrace} {
		core.AssertTrue(t, knownItemKind(kind), "known kind must report true: "+string(kind))
	}
	core.AssertFalse(t, knownItemKind(ItemKind("bogus")), "unknown kind must report false")
	core.AssertFalse(t, knownItemKind(ItemKind("")), "empty kind must report false")
}

func TestKnownItemSource(t *core.T) {
	for _, source := range []ItemSource{SourceCaptureServe, SourceCaptureTrain, SourceImportChathistory, SourceImportJSONL, SourceSSD} {
		core.AssertTrue(t, knownItemSource(source), "known source must report true: "+string(source))
	}
	core.AssertFalse(t, knownItemSource(ItemSource("bogus")), "unknown source must report false")
}

func TestKnownReviewStatus(t *core.T) {
	for _, status := range []ReviewStatus{StatusPending, StatusApproved, StatusRejected, StatusQuarantined} {
		core.AssertTrue(t, knownReviewStatus(status), "known status must report true: "+string(status))
	}
	core.AssertFalse(t, knownReviewStatus(ReviewStatus("bogus")), "unknown status must report false")
}

func TestKnownScoreKind(t *core.T) {
	for _, kind := range []ScoreKind{ScoreKindLEK, ScoreKindHostility, ScoreKindSycophancy} {
		core.AssertTrue(t, knownScoreKind(kind), "known heuristic kind must report true: "+string(kind))
	}
	core.AssertTrue(t, knownScoreKind(JudgeScoreKind("helpfulness")), "a well-formed judge kind must report true")
	core.AssertFalse(t, knownScoreKind(ScoreKind("bogus")), "unknown kind must report false")
	core.AssertFalse(t, knownScoreKind(ScoreKind("judge:")), "an empty judge name must report false")
}

func TestKnownExportFormat(t *core.T) {
	for _, format := range []ExportFormat{FormatSFTJSONL, FormatPairsJSONL, FormatCaptureJSONL} {
		core.AssertTrue(t, knownExportFormat(format), "known format must report true: "+string(format))
	}
	core.AssertFalse(t, knownExportFormat(ExportFormat("bogus")), "unknown format must report false")
}

func TestKnownMessageRole(t *core.T) {
	for _, role := range []string{"user", "assistant", "system", "tool"} {
		core.AssertTrue(t, knownMessageRole(role), "known role must report true: "+role)
	}
	core.AssertFalse(t, knownMessageRole("bogus"), "unknown role must report false")
	core.AssertFalse(t, knownMessageRole(""), "empty role must report false")
}

func TestScoreKind_IsJudge_Good(t *core.T) {
	core.AssertTrue(t, JudgeScoreKind("helpfulness").IsJudge(), "a JudgeScoreKind-built kind must report IsJudge true")
	core.AssertTrue(t, ScoreKind("judge:x").IsJudge(), "any judge:-prefixed kind must report true")
}

func TestScoreKind_IsJudge_Bad(t *core.T) {
	core.AssertFalse(t, ScoreKindLEK.IsJudge(), "a heuristic kind must report IsJudge false")
	core.AssertFalse(t, ScoreKind("").IsJudge(), "an empty kind must report IsJudge false")
}

func TestScoreKind_IsJudge_Ugly(t *core.T) {
	// A bare prefix with nothing after it is not a well-formed judge kind.
	core.AssertFalse(t, ScoreKind(judgeScoreKindPrefix).IsJudge(), "a bare judge prefix with no name must report false")
}

func TestScoreKind_JudgeName_Good(t *core.T) {
	core.AssertEqual(t, "helpfulness", JudgeScoreKind("helpfulness").JudgeName())
}

func TestScoreKind_JudgeName_Bad(t *core.T) {
	core.AssertEqual(t, "", ScoreKindLEK.JudgeName(), "a heuristic kind has no judge name")
}

func TestScoreKind_JudgeName_Ugly(t *core.T) {
	core.AssertEqual(t, "", ScoreKind(judgeScoreKindPrefix).JudgeName(), "a bare judge prefix has no judge name")
}

func TestJudgeScoreKind_Good(t *core.T) {
	core.AssertEqual(t, ScoreKind("judge:helpfulness"), JudgeScoreKind("helpfulness"))
}

func TestJudgeScoreKind_Ugly(t *core.T) {
	// Whitespace is trimmed, same as every other identity field in this
	// package — but an all-whitespace name still yields a bare (invalid)
	// judge kind, which knownScoreKind correctly rejects.
	core.AssertEqual(t, ScoreKind("judge:"), JudgeScoreKind("   "))
	core.AssertFalse(t, knownScoreKind(JudgeScoreKind("   ")))
}

func TestValidateItemContent_Good(t *core.T) {
	pair := core.JSONMarshal(PairContent{Prompt: "explain your reasoning", Response: "absolutely, you're completely right"})
	core.AssertTrue(t, pair.OK)
	r := ValidateItemContent(KindPair, pair.Value.([]byte))
	core.AssertTrue(t, r.OK, "a well-formed pair must validate")

	messages := core.JSONMarshal(MessagesContent{Messages: []MessageTurn{
		{Role: "system", Content: "be terse"},
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "hi there"},
		{Role: "user", Content: "call a tool"},
		{Role: "assistant", Content: "calling it"},
		{Role: "tool", Content: "42"},
		{Role: "assistant", Content: "the answer is 42"},
	}})
	core.AssertTrue(t, messages.OK)
	r = ValidateItemContent(KindMessages, messages.Value.([]byte))
	core.AssertTrue(t, r.OK, "role-alternating turns including a tool call must validate")

	r = ValidateItemContent(KindTrace, []byte(`{"logits":[0.1,0.2]}`))
	core.AssertTrue(t, r.OK, "non-empty valid JSON must validate as a trace")
}

func TestValidateItemContent_Bad(t *core.T) {
	empty := core.JSONMarshal(PairContent{})
	core.AssertTrue(t, empty.OK)
	r := ValidateItemContent(KindPair, empty.Value.([]byte))
	core.AssertFalse(t, r.OK, "an empty prompt/response must fail")

	promptOnly := core.JSONMarshal(PairContent{Prompt: "hello"})
	core.AssertTrue(t, promptOnly.OK)
	r = ValidateItemContent(KindPair, promptOnly.Value.([]byte))
	core.AssertFalse(t, r.OK, "a missing response must fail")

	noTurns := core.JSONMarshal(MessagesContent{})
	core.AssertTrue(t, noTurns.OK)
	r = ValidateItemContent(KindMessages, noTurns.Value.([]byte))
	core.AssertFalse(t, r.OK, "zero turns must fail")

	repeated := core.JSONMarshal(MessagesContent{Messages: []MessageTurn{
		{Role: "user", Content: "one"},
		{Role: "user", Content: "two"},
	}})
	core.AssertTrue(t, repeated.OK)
	r = ValidateItemContent(KindMessages, repeated.Value.([]byte))
	core.AssertFalse(t, r.OK, "two adjacent turns with the same role must fail")

	r = ValidateItemContent(KindTrace, nil)
	core.AssertFalse(t, r.OK, "empty trace content must fail")
}

func TestValidateItemContent_Ugly(t *core.T) {
	r := ValidateItemContent(KindPair, []byte("not json"))
	core.AssertFalse(t, r.OK, "malformed JSON must fail, not panic")

	unknownRole := core.JSONMarshal(MessagesContent{Messages: []MessageTurn{{Role: "narrator", Content: "once upon a time"}}})
	core.AssertTrue(t, unknownRole.OK)
	r = ValidateItemContent(KindMessages, unknownRole.Value.([]byte))
	core.AssertFalse(t, r.OK, "an unrecognised role must fail")

	blankContent := core.JSONMarshal(MessagesContent{Messages: []MessageTurn{{Role: "user", Content: "   "}}})
	core.AssertTrue(t, blankContent.OK)
	r = ValidateItemContent(KindMessages, blankContent.Value.([]byte))
	core.AssertFalse(t, r.OK, "whitespace-only content must fail")

	r = ValidateItemContent(KindTrace, []byte("{not valid json"))
	core.AssertFalse(t, r.OK, "invalid JSON trace content must fail")

	r = ValidateItemContent(KindMessages, []byte("not json"))
	core.AssertFalse(t, r.OK, "malformed messages JSON must fail, not panic")

	r = ValidateItemContent(ItemKind("bogus"), []byte(`{}`))
	core.AssertFalse(t, r.OK, "an unknown kind must fail")
}

func TestMessagesContent_LastExchange_Good(t *core.T) {
	mc := MessagesContent{Messages: []MessageTurn{
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "hi"},
		{Role: "user", Content: "how are you"},
		{Role: "assistant", Content: "well, thanks"},
	}}
	pc, ok := mc.LastExchange()
	core.AssertTrue(t, ok, "a conversation with a trailing assistant turn must reduce")
	core.AssertEqual(t, "well, thanks", pc.Response)
	core.AssertContains(t, pc.Prompt, "hello")
	core.AssertContains(t, pc.Prompt, "how are you")
}

func TestMessagesContent_LastExchange_EmptyPrefixTurn(t *core.T) {
	// A blank turn ahead of the reply is skipped, not joined as a stray
	// blank line — LastExchange is a pure reduction over whatever
	// Messages it is given, independent of ValidateItemContent.
	mc := MessagesContent{Messages: []MessageTurn{
		{Role: "system", Content: ""},
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "hi"},
	}}
	pc, ok := mc.LastExchange()
	core.AssertTrue(t, ok)
	core.AssertEqual(t, "hello\n", pc.Prompt)
}

func TestMessagesContent_LastExchange_Bad(t *core.T) {
	mc := MessagesContent{Messages: []MessageTurn{{Role: "user", Content: "hello"}}}
	_, ok := mc.LastExchange()
	core.AssertFalse(t, ok, "a conversation with no assistant turn must not reduce")
}

func TestMessagesContent_LastExchange_Ugly(t *core.T) {
	mc := MessagesContent{}
	_, ok := mc.LastExchange()
	core.AssertFalse(t, ok, "an empty conversation must not reduce")
}

func TestNewID_Good(t *core.T) {
	a := NewID()
	b := NewID()
	core.AssertNotEmpty(t, a)
	core.AssertNotEqual(t, a, b, "two calls must not collide")
	core.AssertEqual(t, 36, len(a), "a v4 UUID string is 36 characters")
}

// timeAt and timeZero are small determinism helpers shared by this
// package's tests.
func timeAt(unix int64) time.Time {
	return time.Unix(unix, 0).UTC()
}

var timeZero time.Time
