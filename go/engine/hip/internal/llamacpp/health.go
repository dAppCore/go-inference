//go:build rocm_legacy_server

package llamacpp

import (
	"context"
	"io"
	"net/http"

	core "dappco.re/go"
)

// Client communicates with a llama-server instance.
type Client struct {
	baseURL    string
	httpClient *http.Client
}

//	client := NewClient("http://127.0.0.1:38080")
//
// NewClient creates a client for the llama-server at the given base URL.
func NewClient(baseURL string) *Client {
	return NewClientWithHTTPClient(baseURL, &http.Client{})
}

//	client := NewClientWithHTTPClient("http://127.0.0.1:38080", customHTTPClient)
//
// NewClientWithHTTPClient creates a client with an injected HTTP transport.
func NewClientWithHTTPClient(baseURL string, httpClient *http.Client) *Client {
	if httpClient == nil {
		httpClient = &http.Client{}
	}
	return &Client{
		baseURL:    core.TrimSuffix(baseURL, "/"),
		httpClient: httpClient,
	}
}

type clientFailure interface {
	Error() string
}

type healthStatusResponse struct {
	Status string `json:"status"`
}

//	err := client.Health(ctx)
//	fmt.Println(err == nil)
//
// Health checks whether the llama-server is ready to accept requests.
func (c *Client) Health(ctx context.Context) clientFailure {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/health", nil)
	if err != nil {
		return core.E("llamacpp.Health", "create health request", err)
	}
	response, err := c.httpClient.Do(request)
	if err != nil {
		return core.E("llamacpp.Health", "health request", err)
	}
	defer response.Body.Close()

	if response.StatusCode != http.StatusOK {
		responseBody, _ := io.ReadAll(io.LimitReader(response.Body, 256))
		return core.E("llamacpp.Health", core.Sprintf("health returned %d: %s", response.StatusCode, string(responseBody)), nil)
	}
	var healthStatus healthStatusResponse
	bodyResult := core.ReadAll(response.Body)
	if !bodyResult.OK {
		return core.E("llamacpp.Health", "health read", bodyResult.Value.(error))
	}
	if r := core.JSONUnmarshal([]byte(bodyResult.Value.(string)), &healthStatus); !r.OK {
		return core.E("llamacpp.Health", "health decode", r.Value.(error))
	}
	if healthStatus.Status != "ok" {
		return core.E("llamacpp.Health", core.Sprintf("server not ready (status: %s)", healthStatus.Status), nil)
	}
	return nil
}
