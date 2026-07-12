// SPDX-Licence-Identifier: EUPL-1.2

// Package serve is the LEM desktop's client to a running `lem serve` daemon —
// the OpenAI / Anthropic / Ollama HTTP host on Lethean's own port 36911. It
// gives the menu-bar tray the three capabilities it needs without re-deriving
// any of go-inference's serving contract:
//
//   - Client   — read the daemon's live status (up/down, active model).
//   - Manager  — spawn and terminate a managed daemon (see manager.go).
//   - Discover — enumerate loadable models on disk (see models.go).
//
// The wire contract (status shape, admin route paths) is reused directly from
// dappco.re/go/inference/serving/admin, so the tray and the daemon it talks to
// cannot drift on the JSON or the URL. The package is deliberately free of file
// I/O and Wails types: the Bearer token is injected as a func so the client
// stays a plain, httptest-able unit.
package serve

import (
	"context"
	"io"
	"net/http"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/serving/admin"
)

// statusTimeout bounds a single status probe. The tray polls on a timer, so a
// hung daemon must not wedge the poll loop — a short deadline reports "down"
// and moves on.
const statusTimeout = 3 * time.Second

// Status is the serve daemon's live state as the tray reads it. Up is false
// when the admin endpoint is unreachable — the daemon simply not running is a
// normal state the tray renders, not an error.
type Status struct {
	Up        bool   // the daemon answered the status probe
	ModelPath string // active model path ("" when serve started model-less)
	Runtime   string // backend that answered (e.g. "go-inference")
	LoadedAt  int64  // unix seconds the current model loaded
}

// Client polls a lem serve admin API at baseURL, authenticating with the Bearer
// token returned by token(). token is read per-request so a token minted after
// the client was built (serve writes it on first boot) is picked up without a
// restart; a nil token func or an empty string sends no Authorization header.
type Client struct {
	baseURL string
	token   func() string
	http    *http.Client
}

// NewClient builds a status client for a serve daemon at baseURL.
//
//	c := serve.NewClient("http://127.0.0.1:36911", func() string { return tok })
//	st, err := c.Status(context.Background())
//	if err == nil && st.Up { useModel(st.ModelPath) }
func NewClient(baseURL string, token func() string) *Client {
	return &Client{
		baseURL: core.TrimSuffix(baseURL, "/"),
		token:   token,
		http:    &http.Client{Timeout: statusTimeout},
	}
}

// Status GETs /v1/admin/serve/status. A dial/connection error resolves to
// Status{Up:false} with a nil error — the daemon being down is expected and the
// tray renders it. A reachable-but-unhappy daemon (auth reject, non-200, bad
// body) returns an error so the tray can tell "down" apart from "misconfigured".
func (c *Client) Status(ctx context.Context) (Status, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+admin.PathServeStatus, nil)
	if err != nil {
		return Status{}, err
	}
	if c.token != nil {
		if tok := c.token(); tok != "" {
			req.Header.Set("Authorization", "Bearer "+tok)
		}
	}

	resp, err := c.http.Do(req)
	if err != nil {
		// Connection refused / timeout / no listener — serve is down.
		return Status{Up: false}, nil
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return Status{}, core.NewError("serve status: HTTP " + core.Itoa(resp.StatusCode))
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 64*1024))
	if err != nil {
		return Status{}, err
	}
	var raw admin.ServeStatus
	if r := core.JSONUnmarshal(body, &raw); !r.OK {
		return Status{}, r.Err()
	}
	return Status{
		Up:        true,
		ModelPath: raw.ModelPath,
		Runtime:   raw.Runtime,
		LoadedAt:  raw.LoadedAtUnix,
	}, nil
}
