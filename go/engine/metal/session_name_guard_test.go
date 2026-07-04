// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"strings"
	"testing"
)

// allowedSessionTypes is the set of NEUTRAL serving-session type names. The persistent
// decode session is arch-driven (it runs any Arch over the backend contract), not
// model-specific, so it must be neutrally named — ArchSession, never <Model>Session. A
// "Gemma4Session" (which once served even Mistral) is the regression this guards. Add an
// entry only for a genuinely neutral session name, never a model-named one. The
// inference.SessionHandle adapter over ArchSession now lives in the shared
// package engine (engine.SessionHandle), so pkg/native declares only ArchSession.
var allowedSessionTypes = map[string]bool{"ArchSession": true}

// TestSessionTypeNeutralName locks the ArchSession name the same way pkg/model's
// TestArchTypesNeutralHome locks the arch declaration: it fails if pkg/native declares a
// session TYPE that isn't neutrally named. Model-named WEIGHT-FORMAT loaders that return
// *ArchSession (the per-arch pkg/model loaders the registry dispatches to, e.g.
// gemma4.Assemble) are fine — those name a weight format, not the general session — so this
// guards type declarations only, not funcs.
func TestSessionTypeNeutralName(t *testing.T) {
	fset := token.NewFileSet()
	goFiles, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range goFiles {
		af, perr := parser.ParseFile(fset, f, nil, 0)
		if perr != nil {
			t.Fatalf("parse %s: %v", f, perr)
		}
		ast.Inspect(af, func(n ast.Node) bool {
			ts, ok := n.(*ast.TypeSpec)
			if !ok || !strings.HasSuffix(ts.Name.Name, "Session") {
				return true
			}
			if !allowedSessionTypes[ts.Name.Name] {
				t.Errorf("%s declares session type %q — the persistent serving session is arch-driven, "+
					"not model-specific, so it must be neutrally named (ArchSession), never <Model>Session "+
					"(a Gemma4Session once served Mistral). Rename it to a neutral name, or if it is a "+
					"genuinely neutral session add it to allowedSessionTypes.", f, ts.Name.Name)
			}
			return true
		})
	}
}
