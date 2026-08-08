// SPDX-Licence-Identifier: EUPL-1.2

package admin

import (
	core "dappco.re/go"
	"dappco.re/go/inference/internal/testenv"
)

// ExampleListKnownModels demonstrates the verified-models set: only a subdir
// carrying a .sha256 sidecar (the download-integrity contract) is reported.
func ExampleListKnownModels() {
	baseResult := core.MkdirTemp("", "admin-example-*")
	if !baseResult.OK {
		panic("tempdir failed")
	}
	home := baseResult.Value.(string)
	defer core.RemoveAll(home)

	defer testenv.SetHomeDir(home)()

	dir := core.PathJoin(home, "Lethean", "lem", "models", "verified")
	if r := core.MkdirAll(dir, 0o755); !r.OK {
		panic(r.Value)
	}
	if r := core.WriteFile(core.PathJoin(dir, shaManifestFilename), []byte("x"), 0o600); !r.OK {
		panic(r.Value)
	}

	core.Println(ListKnownModels())
	// Output:
	// [verified]
}
