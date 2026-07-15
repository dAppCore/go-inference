// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Landing note: this package descends from go-rocm's top-level engine, where
// go-inference was a vendored submodule at external/go-inference/go. It now
// lives IN go-inference as engine/hip, so the two sibling boundary tests that
// stat'd "../../../external/go-inference/go" (go-inference-as-submodule) no
// longer have a coherent target — go-inference does not vendor itself. One was
// a silent no-op (all roots absent), the other hard-fatal'd; both are removed.
// The engine-import boundary for the shared contract (AX-8: a lib never imports
// its consumers) is go-inference's own concern, guarded repo-wide rather than by
// a single leaf engine. What survives is the guard that genuinely belongs to
// this package: engine/hip must not reach up into the workflow/agent layer.
func TestImportBoundary_NoForbiddenRuntimeImports_Good(t *testing.T) {
	scanImportBoundary(t, ".", forbiddenWorkflowRuntimeImports(), nil)
}

func forbiddenWorkflowRuntimeImports() []string {
	// go-rocm (dappco.re/go/rocm + mirrors) is intentionally absent: it is this
	// engine's legitimate cgo backend. dappco.re/go/mlx is likewise absent —
	// engine/hip's own sub-packages (dappco.re/go/inference/engine/hip/...) are
	// not foreign couplings. The guard is the workflow/agent layer above the
	// engine: an engine builds contracts, it does not consume the fleet.
	return []string{
		"dappco.re/go/ai",
		"dappco.re/go/api",
		"dappco.re/go/ml",
		"dappco.re/go/rag",
		"dappco.re/go/ratelimit",
		"forge.lthn.ai/core/go-ai",
		"forge.lthn.ai/core/go-ml",
		"forge.lthn.ai/core/go-rag",
		"forge.lthn.ai/core/go-ratelimit",
		"forge.lthn.sh/core/go-ai",
		"forge.lthn.sh/core/go-ml",
		"forge.lthn.sh/core/go-rag",
		"forge.lthn.sh/core/go-ratelimit",
		"github.com/dappcore/go-ai",
		"github.com/dappcore/go-ml",
		"github.com/dappcore/go-rag",
		"github.com/dappcore/go-ratelimit",
	}
}

func scanImportBoundary(t *testing.T, root string, forbidden []string, skipDirs map[string]bool) {
	t.Helper()
	fileset := token.NewFileSet()
	err := filepath.WalkDir(root, func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() {
			if entry.Name() == ".git" || skipDirs[entry.Name()] {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") {
			return nil
		}
		file, err := parser.ParseFile(fileset, path, nil, parser.ImportsOnly)
		if err != nil {
			return err
		}
		for _, imported := range file.Imports {
			pathValue := strings.Trim(imported.Path.Value, `"`)
			for _, prefix := range forbidden {
				if pathValue == prefix || strings.HasPrefix(pathValue, prefix+"/") {
					t.Fatalf("%s imports forbidden runtime package %q", path, pathValue)
				}
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk imports under %s: %v", root, err)
	}
}
