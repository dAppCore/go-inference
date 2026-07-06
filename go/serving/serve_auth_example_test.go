// SPDX-Licence-Identifier: EUPL-1.2

package serving

import (
	"fmt"
	"net/http"
	"net/http/httptest"

	core "dappco.re/go"
)

// ExampleAdminTokenPath shows the canonical admin-token location under $HOME.
func ExampleAdminTokenPath() {
	fmt.Println(core.PathBase(AdminTokenPath()))
	// Output:
	// admin.token
}

// ExampleGenerateAdminToken shows the minted token's fixed secret-scanner
// prefix (the entropy suffix is random, so only the prefix is asserted).
func ExampleGenerateAdminToken() {
	tok, err := GenerateAdminToken()
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(core.HasPrefix(tok, "lthn-mlx_"))
	// Output:
	// true
}

// ExampleWriteAdminToken shows a token written to disk and read back.
func ExampleWriteAdminToken() {
	dir := "/tmp/lem-example-admin-token"
	defer core.RemoveAll(dir)
	path := core.PathJoin(dir, "admin.token")
	if err := WriteAdminToken(path, "example-token"); err != nil {
		fmt.Println("error:", err)
		return
	}
	res := core.ReadFile(path)
	fmt.Print(string(res.Value.([]byte)))
	// Output:
	// example-token
}

// ExampleEnsureAdminToken shows the first-boot generate-and-persist path.
func ExampleEnsureAdminToken() {
	dir := "/tmp/lem-example-ensure-token"
	defer core.RemoveAll(dir)
	tok, generated, err := EnsureAdminToken(core.PathJoin(dir, "admin.token"))
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(generated, core.HasPrefix(tok, "lthn-mlx_"))
	// Output:
	// true true
}

// ExampleRequireBearerOnAdmin shows the Bearer wall guarding /v1/admin/* while
// inference paths pass through unauthenticated.
func ExampleRequireBearerOnAdmin() {
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	h := RequireBearerOnAdmin(next, "secret", nil)

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/chat/completions", nil))
	fmt.Println("inference path:", rec.Code)

	rec = httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/v1/admin/machine", nil)
	req.Header.Set("Authorization", "Bearer secret")
	h.ServeHTTP(rec, req)
	fmt.Println("admin path with token:", rec.Code)

	// Output:
	// inference path: 200
	// admin path with token: 200
}
