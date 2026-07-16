// SPDX-Licence-Identifier: EUPL-1.2

package datapipe

import core "dappco.re/go"

func ExampleParseLQL() {
	stmt, err := ParseLQL(`DIFF "base/gemma4" WITH "fine-tunes/project-gemma4" LIMIT 8`)
	if err != nil {
		core.Println(err)
		return
	}
	core.Println(stmt.Kind)
	core.Println(stmt.Base)
	core.Println(stmt.Tuned)
	core.Println(stmt.Limit)
	// Output:
	// diff
	// base/gemma4
	// fine-tunes/project-gemma4
	// 8
}

func ExampleParseLQLScript() {
	statements, err := ParseLQLScript(`USE "base.vindex"; DESCRIBE "base.vindex";`)
	if err != nil {
		core.Println(err)
		return
	}
	core.Println(len(statements))
	core.Println(statements[0].Kind)
	core.Println(statements[1].Kind)
	// Output:
	// 2
	// use
	// describe
}
