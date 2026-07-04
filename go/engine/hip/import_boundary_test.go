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

func TestImportBoundary_NoForbiddenRuntimeImports_Good(t *testing.T) {
	scanImportBoundary(t, ".", forbiddenWorkflowRuntimeImports(), nil)
}

func TestImportBoundary_HigherLevelPackagesDoNotImportROCm_Good(t *testing.T) {
	// Quarantine landing note: paths rebased for pkg/hip's new home
	// (go-mlx/go/pkg/hip, three levels below the worktree root, versus
	// go-rocm's original go/ one level below its repo root) — see the
	// landing commit body. go-ai and go-ml are deprecated (go-inference is
	// the one that matters now); their paths are left as the original
	// go-rocm-relative form and silently skip via the os.IsNotExist guard
	// below.
	roots := []string{
		"../../../external/go-inference/go",
		"../../go-ai",
		"../../go-ml",
	}
	for _, root := range roots {
		if _, err := os.Stat(root); err != nil {
			if os.IsNotExist(err) {
				continue
			}
			t.Fatalf("stat %s: %v", root, err)
		}
		scanImportBoundary(t, root, forbiddenROCmRuntimeImports(), map[string]bool{
			"external": true,
			"vendor":   true,
		})
	}
}

func TestImportBoundary_SharedContractsDoNotImportRuntimeOrWorkflowPackages_Good(t *testing.T) {
	// Quarantine landing note: rebased for pkg/hip's new home — see the
	// landing commit body and the sibling TestImportBoundary_HigherLevel...
	// note above.
	//
	// The go-inference walk enforces only the engine-import boundary (a lib
	// never imports its consumers): go-rocm and go-mlx spellings. The
	// workflow list (rag, ratelimit, …) is go-rocm-era policy from when
	// go-inference was a thin contract surface; go-inference's ai/ package
	// now legitimately builds on dappco.re/go/rag, and its internal layering
	// is guarded by its own suite, not by this consumer.
	root := "../../../external/go-inference/go"
	if _, err := os.Stat(root); err != nil {
		t.Fatalf("stat %s: %v", root, err)
	}
	forbidden := append(forbiddenROCmRuntimeImports(),
		"dappco.re/go/mlx",
		"forge.lthn.ai/core/go-mlx",
		"forge.lthn.sh/core/go-mlx",
		"github.com/dappcore/go-mlx",
	)
	scanImportBoundary(t, root, forbidden, map[string]bool{
		"external": true,
		"vendor":   true,
	})
}

func forbiddenWorkflowRuntimeImports() []string {
	// Quarantine landing note: dappco.re/go/mlx (+ its forge.lthn.ai,
	// forge.lthn.sh, and github.com/dappcore mirror spellings) dropped from
	// this list — see the landing commit body. This test's guard against
	// TestImportBoundary_NoForbiddenRuntimeImports_Good predates pkg/hip's
	// relocation, when go-rocm and go-mlx were separate sibling repos and
	// any go-mlx import from go-rocm's code was necessarily an unwanted
	// foreign coupling. Now that this package's own home IS
	// dappco.re/go/inference/engine/hip, its self-imports (e.g.
	// dappco.re/go/inference/engine/hip/internal/gguf) legitimately start with
	// dappco.re/go/mlx and are not a boundary violation.
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

func forbiddenROCmRuntimeImports() []string {
	return []string{
		"dappco.re/go/rocm",
		"forge.lthn.ai/core/go-rocm",
		"forge.lthn.sh/core/go-rocm",
		"github.com/dappcore/go-rocm",
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
