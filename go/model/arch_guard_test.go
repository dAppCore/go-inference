// SPDX-Licence-Identifier: EUPL-1.2

package model_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
	"strings"
	"testing"
)

// TestArchTypesNeutralHome guards a RECURRING regression. The backend-agnostic arch
// declaration — Arch / LayerSpec / AttentionType — must live ONLY in package model (the
// neutral contract root, next to Backend / TokenModel / Sampler), never in a model-named
// subpackage. It was fixed once in pkg/metal/model, then drifted back into a model-named
// subpackage, which forced other arch packages and pkg/native to import that package just to
// name a neutral type. The model name then regrows, because new general code naturally
// lands next to Arch — in a model-named package.
//
// If a model subpackage re-declares one of these types, this fails: move it up to the
// pkg/model root so the neutral contract stays neutral.
func TestArchTypesNeutralHome(t *testing.T) {
	forbidden := map[string]bool{
		"Arch": true, "LayerSpec": true, "AttentionType": true,
		"LoadedModel": true, "LoadedLayer": true, "LoadedMoE": true,
	}
	fset := token.NewFileSet()
	err := filepath.WalkDir(".", func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if d.IsDir() || !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
			return nil
		}
		if filepath.Dir(path) == "." {
			return nil // package model's OWN root files are the correct home — that is the point
		}
		af, perr := parser.ParseFile(fset, path, nil, 0)
		if perr != nil {
			return perr
		}
		ast.Inspect(af, func(n ast.Node) bool {
			if ts, ok := n.(*ast.TypeSpec); ok && forbidden[ts.Name.Name] {
				t.Errorf("%s declares type %q in a model subpackage — the backend-agnostic arch "+
					"types must live in package model (the pkg/model root), not a model-named "+
					"subpackage (that is exactly what makes other models import this one for a "+
					"neutral type). Move it up.", path, ts.Name.Name)
			}
			return true
		})
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}
