// SPDX-Licence-Identifier: EUPL-1.2

package driver

import (
	core "dappco.re/go"
)

func ExampleCanonicalRepoDir() {
	core.Println(CanonicalRepoDir("mlx-community/gemma-4-e2b-it-4bit"))
	// Output:
	// mlx-community__gemma-4-e2b-it-4bit
}

// ExampleAllowRepo isolates HOME under a fresh temp dir so the allowlist
// write lands somewhere throwaway rather than a developer's real
// ~/Lethean/lem/allowed-models.json.
func ExampleAllowRepo() {
	prevHome := core.Getenv("HOME")
	tmp := core.MkdirTemp("", "driver-admin-example-*")
	if !tmp.OK {
		core.Println(false)
		return
	}
	home := tmp.Value.(string)
	defer core.RemoveAll(home)
	defer core.Setenv("HOME", prevHome)
	core.Setenv("HOME", home)

	r := AllowRepo("mlx-community/gemma-3-1b-it-4bit")

	core.Println(r.OK)
	core.Println(r.Value.([]string))
	// Output:
	// true
	// [mlx-community/gemma-3-1b-it-4bit]
}

// ExampleService_DownloadModel shows the pre-serve shape: with no runtime
// supervised, the call refuses rather than reaching for a nonexistent engine.
func ExampleService_DownloadModel() {
	s := &Service{}
	r := s.DownloadModel(RuntimeMLX, "org/repo", "main")

	core.Println(r.OK)
	// Output:
	// false
}

// ExampleService_DownloadJobStatus shows the pre-serve shape: with no runtime
// supervised, the poll refuses rather than reaching for a nonexistent engine.
func ExampleService_DownloadJobStatus() {
	s := &Service{}
	r := s.DownloadJobStatus(RuntimeMLX, "job-1")

	core.Println(r.OK)
	// Output:
	// false
}
