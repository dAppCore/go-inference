// SPDX-Licence-Identifier: EUPL-1.2

package respcache

import (
	"testing"
	"time"
)

// sampleRequest is the canonical "two-message chat" used across the key tests.
func sampleRequest() Request {
	return Request{
		Model: "gemma-4-e4b",
		Messages: []Message{
			{Role: "system", Content: "you are helpful"},
			{Role: "user", Content: "hello"},
		},
		Temperature: 0.2,
		TopP:        0.9,
		MaxTokens:   256,
		Seed:        42,
		Stop:        []string{"\n\n", "END"},
	}
}

// ---- Key ---------------------------------------------------------------

// TestRespCache_Key_Good — the same request shape yields the same key, and the
// per-message field order does not change it.
func TestRespCache_Key_Good(t *testing.T) {
	a := Key(sampleRequest())
	b := Key(sampleRequest())
	if a == "" {
		t.Fatal("Key returned empty string for a populated request")
	}
	if a != b {
		t.Fatalf("identical requests produced different keys:\n a=%s\n b=%s", a, b)
	}

	// Stop order is a set, not a sequence — reordering it must not change the
	// key (a caller passing the same stop strings in a different order is the
	// same request for cache purposes).
	reordered := sampleRequest()
	reordered.Stop = []string{"END", "\n\n"}
	if got := Key(reordered); got != a {
		t.Fatalf("reordered stop list changed the key:\n want=%s\n got =%s", a, got)
	}
}

// TestRespCache_Key_Bad — a change to the model or any sampling param must
// change the key, so a different generation never collides with a cached one.
func TestRespCache_Key_Bad(t *testing.T) {
	base := Key(sampleRequest())

	cases := map[string]func(r *Request){
		"model":       func(r *Request) { r.Model = "gemma-4-31b" },
		"temperature": func(r *Request) { r.Temperature = 0.7 },
		"top_p":       func(r *Request) { r.TopP = 0.5 },
		"max_tokens":  func(r *Request) { r.MaxTokens = 512 },
		"seed":        func(r *Request) { r.Seed = 7 },
		"stop":        func(r *Request) { r.Stop = []string{"STOP"} },
		"message":     func(r *Request) { r.Messages[1].Content = "goodbye" },
		"role":        func(r *Request) { r.Messages[1].Role = "assistant" },
		"extra-msg":   func(r *Request) { r.Messages = append(r.Messages, Message{Role: "user", Content: "more"}) },
	}

	for name, mutate := range cases {
		r := sampleRequest()
		mutate(&r)
		if got := Key(r); got == base {
			t.Fatalf("changing %q did not change the key (collision): %s", name, got)
		}
	}
}

// TestRespCache_Key_Ugly — degenerate inputs (empty messages, zero params, nil
// stop) still produce a stable, deterministic, non-empty key and don't panic.
func TestRespCache_Key_Ugly(t *testing.T) {
	empty := Request{}
	k1 := Key(empty)
	k2 := Key(Request{})
	if k1 == "" {
		t.Fatal("Key of a zero-value request returned empty string")
	}
	if k1 != k2 {
		t.Fatalf("zero-value request key not deterministic:\n %s\n %s", k1, k2)
	}

	// model only, no messages
	mOnly := Request{Model: "gemma-4-e4b"}
	if Key(mOnly) == k1 {
		t.Fatal("model-only request collided with the fully-empty request")
	}

	// nil stop vs empty-slice stop must be the same key (both = "no stops")
	nilStop := sampleRequest()
	nilStop.Stop = nil
	emptyStop := sampleRequest()
	emptyStop.Stop = []string{}
	if Key(nilStop) != Key(emptyStop) {
		t.Fatal("nil stop and empty stop produced different keys")
	}
}

// ---- Get / Set ---------------------------------------------------------

// TestRespCache_Get_Good — a stored completion is returned on an identical
// request with no inference, and the value round-trips intact.
func TestRespCache_Get_Good(t *testing.T) {
	c := New(nil)
	req := sampleRequest()

	if _, hit := c.Get(req); hit {
		t.Fatal("fresh cache reported a hit before any Set")
	}

	want := Completion{Text: "hello there", Model: "gemma-4-e4b", FinishReason: "stop"}
	c.Set(req, want, 0)

	got, hit := c.Get(req)
	if !hit {
		t.Fatal("expected a hit after Set")
	}
	if got.Text != want.Text || got.Model != want.Model || got.FinishReason != want.FinishReason {
		t.Fatalf("round-trip mismatch:\n want=%+v\n got =%+v", want, got)
	}

	// A reordered-stop request is the same key (Key_Good) → same hit.
	reordered := sampleRequest()
	reordered.Stop = []string{"END", "\n\n"}
	if _, hit := c.Get(reordered); !hit {
		t.Fatal("expected a hit for a request that differs only in stop order")
	}
}

// TestRespCache_Get_Bad — a miss for a never-stored request, and a per-
// request bypass that skips the cache on both read and write.
func TestRespCache_Get_Bad(t *testing.T) {
	c := New(nil)
	req := sampleRequest()
	c.Set(req, Completion{Text: "cached"}, 0)

	// Different request → miss, not a wrong hit.
	other := sampleRequest()
	other.Model = "gemma-4-31b"
	if got, hit := c.Get(other); hit {
		t.Fatalf("expected a miss for an unstored request, got hit: %+v", got)
	}

	// Bypass on read: even though req is cached, a bypassed lookup must miss so
	// the caller runs a fresh inference.
	bypass := req
	bypass.Bypass = true
	if _, hit := c.Get(bypass); hit {
		t.Fatal("bypassed Get returned a hit; bypass must skip the cache")
	}

	// Bypass on write: a bypassed Set must not populate the cache.
	fresh := New(nil)
	wreq := sampleRequest()
	wreq.Bypass = true
	fresh.Set(wreq, Completion{Text: "should not store"}, 0)
	probe := sampleRequest() // same key, bypass off
	if _, hit := fresh.Get(probe); hit {
		t.Fatal("bypassed Set populated the cache; it must not store")
	}
}

// TestRespCache_Set_Ugly — TTL expiry and overwrite. An expired entry is a
// miss; a re-Set overwrites the prior value.
func TestRespCache_Set_Ugly(t *testing.T) {
	now := time.Now()
	clock := now
	c := New(nil)
	c.now = func() time.Time { return clock }

	req := sampleRequest()
	c.Set(req, Completion{Text: "short-lived"}, 50*time.Millisecond)

	// Still inside the TTL → hit.
	if _, hit := c.Get(req); !hit {
		t.Fatal("entry expired before its TTL elapsed")
	}

	// Advance past the TTL → miss.
	clock = now.Add(100 * time.Millisecond)
	if got, hit := c.Get(req); hit {
		t.Fatalf("expired entry still returned a hit: %+v", got)
	}

	// Overwrite: a second Set under the same key replaces the value.
	c.Set(req, Completion{Text: "first"}, 0)
	c.Set(req, Completion{Text: "second"}, 0)
	got, hit := c.Get(req)
	if !hit {
		t.Fatal("expected a hit after overwrite")
	}
	if got.Text != "second" {
		t.Fatalf("overwrite did not replace the value: got %q want %q", got.Text, "second")
	}

	// Zero TTL means no expiry — advancing the clock far ahead still hits.
	clock = now.Add(1000 * time.Hour)
	if _, hit := c.Get(req); !hit {
		t.Fatal("zero-TTL entry expired; zero TTL must mean no expiry")
	}
}

// TestRespCache_Store_Good — a custom Store backs the cache; Get/Set delegate
// to it rather than the in-memory default.
func TestRespCache_Store_Good(t *testing.T) {
	st := &countingStore{inner: NewMemoryStore()}
	c := New(st)
	req := sampleRequest()

	c.Set(req, Completion{Text: "via store"}, 0)
	if st.sets == 0 {
		t.Fatal("Set did not delegate to the pluggable Store")
	}
	if _, hit := c.Get(req); !hit {
		t.Fatal("expected a hit from the pluggable Store")
	}
	if st.gets == 0 {
		t.Fatal("Get did not delegate to the pluggable Store")
	}
}

// countingStore wraps a Store and counts delegations — proves the Cache routes
// through the interface, not a hard-coded map.
type countingStore struct {
	inner      Store
	gets, sets int
}

func (s *countingStore) Get(key string) (entry Entry, ok bool) {
	s.gets++
	return s.inner.Get(key)
}

func (s *countingStore) Set(key string, entry Entry) {
	s.sets++
	s.inner.Set(key, entry)
}
