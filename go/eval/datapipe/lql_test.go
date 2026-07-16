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

// TestLql_ParseLQL_Good pins the DESCRIBE and INFER statement shapes: DESCRIBE
// carries its target verbatim, INFER treats the quoted prompt as both the
// prompt and the target (a bare inference has no separate target clause).
func TestLql_ParseLQL_Good(t *core.T) {
	describe, err := ParseLQL(`DESCRIBE "base/gemma4"`)
	core.AssertNoError(t, err)
	core.AssertEqual(t, LQLStatementDescribe, describe.Kind)
	core.AssertEqual(t, "base/gemma4", describe.Target)

	infer, err := ParseLQL(`INFER "why did this fine tune prefer the operator name?" LIMIT 3`)
	core.AssertNoError(t, err)
	core.AssertEqual(t, LQLStatementInfer, infer.Kind)
	core.AssertEqual(t, "why did this fine tune prefer the operator name?", infer.Prompt)
	core.AssertEqual(t, "why did this fine tune prefer the operator name?", infer.Target)
	core.AssertEqual(t, 3, infer.Limit)
}

// TestLql_ParseLQL_Bad pins the missing-argument rejections: USE and DESCRIBE
// both require a target token and TRACE requires an operation — each errors
// with a statement-specific message rather than returning a half-built
// statement.
func TestLql_ParseLQL_Bad(t *core.T) {
	_, err := ParseLQL("USE")
	core.AssertError(t, err, "USE requires")

	_, err = ParseLQL("DESCRIBE")
	core.AssertError(t, err, "DESCRIBE requires")

	_, err = ParseLQL("TRACE")
	core.AssertError(t, err, "TRACE requires")
}

// TestLql_ParseLQL_Ugly pins the escaped-quote lexer path: a backslash-escaped
// quote inside a quoted token is kept as a literal quote character rather than
// ending the token early (lexLQL's escaped-char branch).
func TestLql_ParseLQL_Ugly(t *core.T) {
	stmt, err := ParseLQL(`WALK "operator's \"favourite\" context" LIMIT 5`)

	core.AssertNoError(t, err)
	core.AssertEqual(t, LQLStatementWalk, stmt.Kind)
	core.AssertEqual(t, `operator's "favourite" context`, stmt.Prompt)
	core.AssertEqual(t, 5, stmt.Limit)
}

// TestLql_ParseLQLScript_Good pins a plain multi-statement script: each
// semicolon-terminated statement parses independently and comes back in
// source order.
func TestLql_ParseLQLScript_Good(t *core.T) {
	statements, err := ParseLQLScript(`USE "base.vindex"; DESCRIBE "base.vindex";`)

	core.AssertNoError(t, err)
	core.AssertLen(t, statements, 2)
	core.AssertEqual(t, LQLStatementUse, statements[0].Kind)
	core.AssertEqual(t, LQLStatementDescribe, statements[1].Kind)
	core.AssertEqual(t, "base.vindex", statements[1].Target)
}

// TestLql_ParseLQLScript_Bad pins error propagation: an unterminated quoted
// string anywhere in the script is rejected rather than silently truncated.
func TestLql_ParseLQLScript_Bad(t *core.T) {
	_, err := ParseLQLScript(`USE "unterminated`)

	core.AssertError(t, err, "unterminated")
}

// TestLql_ParseLQLScript_Ugly pins the empty-script edge: a script with no
// statements (comments only, nothing else) returns an empty slice, not an
// error.
func TestLql_ParseLQLScript_Ugly(t *core.T) {
	statements, err := ParseLQLScript("# just a header comment\n-- and a note\n")

	core.AssertNoError(t, err)
	core.AssertLen(t, statements, 0)
}
