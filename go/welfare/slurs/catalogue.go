// SPDX-Licence-Identifier: EUPL-1.2

package slurs

// catalogue is the curated slur list — boolean detection, hand-maintained,
// reviewed via PR per RFC.welfare ("Slur regex — curated list").
// NOT community-sourced, NOT telemetry-expanded: the failure mode of a
// community list (a controversial-but-not-slur term getting silent
// suppression through) is exactly what we refuse.
//
// Seeded empty by design — the matcher (slurs.go) is the engineering; this
// data is reviewed separately. To populate: one canonical base term per
// entry. l33t / substitution variants fold automatically (New → canonical),
// so list only the base form. Exclude in-group-only terms and terms with high
// false-positive rates in other languages — defer those to the model, per the
// RFC.
//
//	var catalogue = []string{
//	    "exampleterm",
//	}
var catalogue = []string{
	// TODO(snider): curated catalogue — populate via reviewed PR.
}
