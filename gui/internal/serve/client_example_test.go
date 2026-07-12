// SPDX-Licence-Identifier: EUPL-1.2

package serve

import (
	"context"
	"net/http"
	"net/http/httptest"

	core "dappco.re/go"
)

// ExampleClient_Status reads the live serve status from a daemon. Here an
// httptest server stands in for a running `lem serve` on port 36911.
func ExampleClient_Status() {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		core.WriteString(w, `{"model_path":"/models/gemma-4-e2b-it-4bit","runtime":"go-inference"}`)
	}))
	defer srv.Close()

	client := NewClient(srv.URL, func() string { return "" })
	st, _ := client.Status(context.Background())

	core.Println(st.Up, st.ModelPath)
	// Output:
	// true /models/gemma-4-e2b-it-4bit
}
