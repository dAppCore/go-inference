// SPDX-Licence-Identifier: EUPL-1.2

package datapipe

import core "dappco.re/go"

func TestLQL_LQLStatementUse_Good(t *core.T) {
	stmt, err := ParseLQL(`USE "models/gemma4-ft.vindex"`)

	core.AssertNoError(t, err)
	core.AssertEqual(t, LQLStatementUse, stmt.Kind)
	core.AssertEqual(t, "models/gemma4-ft.vindex", stmt.Target)
}

func TestLQL_LQLStatementWalk_Good(t *core.T) {
	stmt, err := ParseLQL(`WALK "operator project context" LIMIT 12`)

	core.AssertNoError(t, err)
	core.AssertEqual(t, LQLStatementWalk, stmt.Kind)
	core.AssertEqual(t, "operator project context", stmt.Prompt)
	core.AssertEqual(t, 12, stmt.Limit)
}

func TestLQL_LQLStatementDiff_Good(t *core.T) {
	stmt, err := ParseLQL(`DIFF "base/gemma4" WITH "fine-tunes/project-gemma4" PATCH "findings.patch" LIMIT 8`)

	core.AssertNoError(t, err)
	core.AssertEqual(t, LQLStatementDiff, stmt.Kind)
	core.AssertEqual(t, "base/gemma4", stmt.Base)
	core.AssertEqual(t, "fine-tunes/project-gemma4", stmt.Tuned)
	core.AssertEqual(t, "findings.patch", stmt.Patch)
	core.AssertEqual(t, 8, stmt.Limit)
}

func TestLQL_LQLStatementTrace_Good(t *core.T) {
	stmt, err := ParseLQL(`TRACE INFER "why did this fine tune prefer the operator name?"`)

	core.AssertNoError(t, err)
	core.AssertEqual(t, LQLStatementTrace, stmt.Kind)
	core.AssertEqual(t, LQLStatementInfer, stmt.Operation)
	core.AssertEqual(t, "why did this fine tune prefer the operator name?", stmt.Prompt)
}

func TestLQL_AssertError_Bad(t *core.T) {
	_, err := ParseLQL(" ")

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "empty")
}

func TestLQL_ParseLQL_Bad(t *core.T) {
	_, err := ParseLQL("FLY model.layer[0]")

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "unsupported")
}

func TestLQL_ParseLQLScript_Ugly(t *core.T) {
	statements, err := ParseLQLScript(`
# research batch
USE "base.vindex";
WALK "same; token in quote" LIMIT 2;
-- compare after walk
DIFF base "base" tuned "fine";
`)

	core.AssertNoError(t, err)
	core.AssertLen(t, statements, 3)
	core.AssertEqual(t, LQLStatementUse, statements[0].Kind)
	core.AssertEqual(t, "same; token in quote", statements[1].Prompt)
	core.AssertEqual(t, LQLStatementDiff, statements[2].Kind)
	core.AssertEqual(t, "base", statements[2].Base)
	core.AssertEqual(t, "fine", statements[2].Tuned)
}

// TestLQL_LQLStatementSelect_Good pins the SELECT/COMPILE/EXTRACT branch: the
// target is the rest of the line after the keyword (lqlRest joins the trailing
// tokens), and a trailing LIMIT is still lifted out.
func TestLQL_LQLStatementSelect_Good(t *core.T) {
	stmt, err := ParseLQL(`SELECT layer.0.attn beta`)

	core.AssertNoError(t, err)
	core.AssertEqual(t, LQLStatementSelect, stmt.Kind)
	core.AssertEqual(t, "layer.0.attn beta", stmt.Target)

	extract, err := ParseLQL(`EXTRACT weights LIMIT 4`)
	core.AssertNoError(t, err)
	core.AssertEqual(t, LQLStatementExtract, extract.Kind)
	core.AssertEqual(t, "weights LIMIT 4", extract.Target)
	core.AssertEqual(t, 4, extract.Limit)
}

// TestLQL_LQLStatementSelect_Ugly pins the empty-rest edge: a bare SELECT has
// no trailing tokens, so lqlRest returns "" rather than indexing out of range.
func TestLQL_LQLStatementSelect_Ugly(t *core.T) {
	stmt, err := ParseLQL(`SELECT`)

	core.AssertNoError(t, err)
	core.AssertEqual(t, LQLStatementSelect, stmt.Kind)
	core.AssertEqual(t, "", stmt.Target)
}
