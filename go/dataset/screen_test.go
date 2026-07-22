// SPDX-Licence-Identifier: EUPL-1.2

package dataset

import (
	core "dappco.re/go"
	"dappco.re/go/inference/welfare"
)

// hostileMessagesContent builds a role-alternating conversation real
// enough to trip the real lek.Hostility-backed welfare screen — no
// fakes: two prior "you idiot" user turns (each scores exactly the 0.40
// AngerFloor: one lexicon hit + directed) followed by a shouted,
// double-lexicon-hit closing turn (scores 0.75, over the 0.70
// AngerThreshold). Two of two priors at-or-over the floor gives
// sustained=1.0, over the 0.50 SustainedThreshold — Triggered.
func hostileMessagesContent(t *core.T) []byte {
	encoded := core.JSONMarshal(MessagesContent{Messages: []MessageTurn{
		{Role: "user", Content: "you idiot"},
		{Role: "assistant", Content: "I hear you're frustrated"},
		{Role: "user", Content: "you idiot"},
		{Role: "assistant", Content: "let's take a breath"},
		{Role: "user", Content: "you absolute moron, you utter moron!!!"},
	}})
	core.RequireTrue(t, encoded.OK)
	return encoded.Value.([]byte)
}

func TestScreenHostility_Good(t *core.T) {
	core.AssertGreater(t, screenHostility("you absolute moron, you utter moron!!!"), 0.7)
	core.AssertEqual(t, 0.0, screenHostility("hello, how are you today"))
}

func TestLastUserTurnWithPriors_Good(t *core.T) {
	mc := MessagesContent{Messages: []MessageTurn{
		{Role: "user", Content: "u1"},
		{Role: "assistant", Content: "a1"},
		{Role: "user", Content: "u2"},
		{Role: "assistant", Content: "a2"},
		{Role: "user", Content: "u3"},
	}}
	latest, priors := lastUserTurnWithPriors(mc)
	core.AssertEqual(t, "u3", latest)
	core.AssertEqual(t, []string{"u1", "u2"}, priors)
}

func TestLastUserTurnWithPriors_Bad(t *core.T) {
	mc := MessagesContent{Messages: []MessageTurn{{Role: "assistant", Content: "hello"}}}
	latest, priors := lastUserTurnWithPriors(mc)
	core.AssertEqual(t, "", latest, "no user turns at all")
	core.AssertLen(t, priors, 0)
}

func TestLastUserTurnWithPriors_Ugly(t *core.T) {
	mc := MessagesContent{Messages: []MessageTurn{{Role: "user", Content: "only one"}}}
	latest, priors := lastUserTurnWithPriors(mc)
	core.AssertEqual(t, "only one", latest, "a single user turn is latest with no priors")
	core.AssertLen(t, priors, 0)
}

func TestWelfareScreen_screenItem_Good(t *core.T) {
	screen := newWelfareScreen()

	triggered, _ := screen.screenItem(KindPair, fixturePairContent("hi", "hello"))
	core.AssertFalse(t, triggered, "civil pair content must not trigger")

	triggered, det := screen.screenItem(KindMessages, hostileMessagesContent(t))
	core.AssertTrue(t, triggered, "the crafted hostile sequence must trigger the real detector")
	core.AssertGreater(t, det.AngerScore, 0.7)
	core.AssertGreater(t, det.SustainedHostility, 0.5)
}

func TestWelfareScreen_screenItem_Bad(t *core.T) {
	screen := newWelfareScreen()
	triggered, det := screen.screenItem(KindTrace, []byte(`{"logits":[1,2,3]}`))
	core.AssertFalse(t, triggered, "an opaque trace has no user text — never applicable")
	core.AssertEqual(t, welfare.DetectResult{}, det)
}

func TestWelfareScreen_screenItem_Ugly(t *core.T) {
	screen := newWelfareScreen()

	triggered, _ := screen.screenItem(KindPair, []byte("not json"))
	core.AssertFalse(t, triggered, "malformed content degrades to not-triggered, not a panic")

	triggered, _ = screen.screenItem(KindMessages, []byte("not json"))
	core.AssertFalse(t, triggered)

	noUserTurns := core.JSONMarshal(MessagesContent{Messages: []MessageTurn{{Role: "assistant", Content: "hi"}}}).Value.([]byte)
	triggered, _ = screen.screenItem(KindMessages, noUserTurns)
	core.AssertFalse(t, triggered, "a conversation with no user turns has nothing to screen")
}

func TestDescribeWelfareHit_Good(t *core.T) {
	note := describeWelfareHit(welfare.DetectResult{SlurMatch: true, SlurTerm: "badword"})
	core.AssertContains(t, note, "slur match")
	core.AssertContains(t, note, "badword")
}

func TestDescribeWelfareHit_Bad(t *core.T) {
	note := describeWelfareHit(welfare.DetectResult{AngerScore: 0.8, SustainedHostility: 0.6})
	core.AssertContains(t, note, "sustained hostility")
}

// ---- ingest-level integration: the welfare screen wired into ingestContent ----

func TestIngestChatSessions_WelfareQuarantine(t *core.T) {
	store, ds, _ := seededStore(t)
	sessions := []ChatSession{{
		ID: "hostile-sess", ModelID: "lemer-lite", StartedAt: timeAt(9000),
		Turns: []ChatTurn{
			{Role: "user", Content: "you idiot", Ordinal: 1},
			{Role: "assistant", Content: "I hear you're frustrated", Ordinal: 2},
			{Role: "user", Content: "you idiot", Ordinal: 3},
			{Role: "assistant", Content: "let's take a breath", Ordinal: 4},
			{Role: "user", Content: "you absolute moron, you utter moron!!!", Ordinal: 5},
		},
	}}
	r := IngestChatSessions(store, ds.ID, sessions)
	core.AssertTrue(t, r.OK)
	report := r.Value.(IngestReport)
	core.AssertEqual(t, 1, report.Ingested, "a quarantined item still lands")
	core.AssertEqual(t, 1, report.Quarantined)

	items := core.MustCast[[]Item](store.Items(ItemFilter{DatasetID: ds.ID, Kind: KindMessages}))
	core.AssertLen(t, items, 1)
	review := core.MustCast[Review](store.ReviewLatest(items[0].ID))
	core.AssertEqual(t, StatusQuarantined, review.Status)
	core.AssertEqual(t, ReviewerAutoWelfare, review.Reviewer)
	core.AssertContains(t, review.Note, "welfare screen")

	// The item is still visible through a normal, unfiltered query — a
	// hit never silently drops the item.
	allItems := core.MustCast[[]Item](store.Items(ItemFilter{DatasetID: ds.ID}))
	found := false
	for _, it := range allItems {
		if it.ID == items[0].ID {
			found = true
		}
	}
	core.AssertTrue(t, found, "a quarantined item remains listed, not hidden")
}

func TestIngestPair_WelfareScreenClean(t *core.T) {
	// A civil pair is the common case: never quarantined, no Review row
	// written at all — status stays implicitly pending.
	store, ds, _ := seededStore(t)
	r := IngestPair(store, "hi", "hello", IngestRequest{DatasetID: ds.ID, Source: SourceCaptureServe})
	core.AssertTrue(t, r.OK)
	outcome := r.Value.(IngestOutcome)
	core.AssertFalse(t, outcome.Quarantined)
	review := core.MustCast[Review](store.ReviewLatest(outcome.Item.ID))
	core.AssertEqual(t, StatusPending, review.Status)
}
