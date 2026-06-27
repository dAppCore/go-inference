// SPDX-Licence-Identifier: EUPL-1.2

package datapipe

import core "dappco.re/go"

func TestLQL_ParseUse_Good(t *core.T) {
	stmt, err := ParseLQL(`USE "models/gemma4-ft.vindex"`)

	core.AssertNoError(t, err)
	core.AssertEqual(t, LQLStatementUse, stmt.Kind)
	core.AssertEqual(t, "models/gemma4-ft.vindex", stmt.Target)
}

func TestLQL_ParseWalk_Good(t *core.T) {
	stmt, err := ParseLQL(`WALK "operator project context" LIMIT 12`)

	core.AssertNoError(t, err)
	core.AssertEqual(t, LQLStatementWalk, stmt.Kind)
	core.AssertEqual(t, "operator project context", stmt.Prompt)
	core.AssertEqual(t, 12, stmt.Limit)
}

func TestLQL_ParseDiff_Good(t *core.T) {
	stmt, err := ParseLQL(`DIFF "base/gemma4" WITH "fine-tunes/project-gemma4" PATCH "findings.patch" LIMIT 8`)

	core.AssertNoError(t, err)
	core.AssertEqual(t, LQLStatementDiff, stmt.Kind)
	core.AssertEqual(t, "base/gemma4", stmt.Base)
	core.AssertEqual(t, "fine-tunes/project-gemma4", stmt.Tuned)
	core.AssertEqual(t, "findings.patch", stmt.Patch)
	core.AssertEqual(t, 8, stmt.Limit)
}

func TestLQL_ParseTraceInfer_Good(t *core.T) {
	stmt, err := ParseLQL(`TRACE INFER "why did this fine tune prefer the operator name?"`)

	core.AssertNoError(t, err)
	core.AssertEqual(t, LQLStatementTrace, stmt.Kind)
	core.AssertEqual(t, LQLStatementInfer, stmt.Operation)
	core.AssertEqual(t, "why did this fine tune prefer the operator name?", stmt.Prompt)
}

func TestLQL_ParseEmpty_Bad(t *core.T) {
	_, err := ParseLQL(" ")

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "empty")
}

func TestLQL_ParseUnknown_Bad(t *core.T) {
	_, err := ParseLQL("FLY model.layer[0]")

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "unsupported")
}

func TestLQL_ParseScript_Ugly(t *core.T) {
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
