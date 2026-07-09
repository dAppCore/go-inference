// SPDX-Licence-Identifier: EUPL-1.2

package probe

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

func lpClock(step time.Duration) func() time.Time {
	t := time.Unix(1000, 0)
	return func() time.Time {
		t = t.Add(step)
		return t
	}
}

// --- NewLineProtocolSink ---

// Good: NewLineProtocolSink returns a usable sink whose batch size and clock
// fall back to sane defaults when the config leaves them unset, and which
// formats an emitted training line.
func TestInflux_NewLineProtocolSink_Good(t *testing.T) {
	s := NewLineProtocolSink(LineProtocolConfig{Model: "m", RunID: "r"})
	s.EmitProbe(Event{Kind: KindTraining, Training: &Training{Step: 1, Loss: 1, LossType: LossTypeVal}})
	if s.Lines() != 1 {
		t.Fatalf("NewLineProtocolSink default sink did not format the line: Lines() = %d, want 1", s.Lines())
	}
	if s.Dropped() != 0 {
		t.Fatalf("NewLineProtocolSink with no destinations dropped a line: Dropped() = %d, want 0", s.Dropped())
	}
}

// Bad: a sink configured with a non-positive BatchLines must not divide-by-
// zero or hoard lines forever — NewLineProtocolSink clamps the batch to the
// default 20 so a Post-backed sink still flushes. Emitting 20 lines triggers
// exactly one flush at the clamped boundary.
func TestInflux_NewLineProtocolSink_Bad(t *testing.T) {
	posts := 0
	s := NewLineProtocolSink(LineProtocolConfig{
		Model: "m", RunID: "r",
		Post:       func(string) error { posts++; return nil },
		BatchLines: 0, // invalid → clamped to default 20
	})
	for step := 1; step <= 20; step++ {
		s.EmitProbe(Event{Kind: KindTraining, Training: &Training{Step: step, Loss: 1, LossType: LossTypeVal}})
	}
	if posts != 1 {
		t.Fatalf("posts = %d, want 1 — BatchLines<=0 must clamp to default 20, not 0/∞", posts)
	}
}

// Ugly: a nil *LineProtocolSink (never built by NewLineProtocolSink) must
// no-op on every method rather than panic, while a real instance from
// NewLineProtocolSink beside it still formats lines — the constructor is what
// produces a live sink; the nil path degrades silently.
func TestInflux_NewLineProtocolSink_Ugly(t *testing.T) {
	var s *LineProtocolSink // never went through NewLineProtocolSink
	s.EmitProbe(Event{Kind: KindTraining, Training: &Training{Step: 1, Loss: 1, LossType: LossTypeVal}})
	s.Flush()
	s.Close()
	if s.Lines() != 0 || s.Dropped() != 0 {
		t.Fatal("nil LineProtocolSink must no-op")
	}
	// A real NewLineProtocolSink instance still works beside the nil one.
	live := NewLineProtocolSink(LineProtocolConfig{Model: "m", RunID: "r"})
	live.EmitProbe(Event{Kind: KindTraining, Training: &Training{Step: 1, Loss: 1, LossType: LossTypeVal}})
	if live.Lines() != 1 {
		t.Fatalf("NewLineProtocolSink instance did not format beside nil path: %d, want 1", live.Lines())
	}
}

// --- NewInfluxPoster ---

// Good: NewInfluxPoster returns the HTTP write closure. It POSTs the body,
// sets the text/plain content type, carries the token as a "Token <tok>"
// Authorization header (the InfluxDB v2 write contract), and an empty token
// sends no Authorization header at all.
func TestInflux_NewInfluxPoster_Good(t *testing.T) {
	t.Run("PostsBodyWithTokenHeader", func(t *testing.T) {
		var gotBody, gotAuth, gotContentType, gotMethod string
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			gotMethod = r.Method
			gotAuth = r.Header.Get("Authorization")
			gotContentType = r.Header.Get("Content-Type")
			body, _ := io.ReadAll(r.Body)
			gotBody = string(body)
			w.WriteHeader(http.StatusNoContent)
		}))
		defer srv.Close()

		post := NewInfluxPoster(srv.URL, "secret-token")
		if err := post("training_loss,model=m loss=1.0 1i"); err != nil {
			t.Fatalf("post returned error: %v", err)
		}
		if gotMethod != http.MethodPost {
			t.Fatalf("method = %q, want POST", gotMethod)
		}
		if gotBody != "training_loss,model=m loss=1.0 1i" {
			t.Fatalf("body = %q", gotBody)
		}
		if gotAuth != "Token secret-token" {
			t.Fatalf("auth header = %q, want %q", gotAuth, "Token secret-token")
		}
		if gotContentType != "text/plain; charset=utf-8" {
			t.Fatalf("content-type = %q", gotContentType)
		}
	})
	t.Run("EmptyTokenSendsNoAuthHeader", func(t *testing.T) {
		authPresent := true
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, authPresent = r.Header["Authorization"]
			w.WriteHeader(http.StatusNoContent)
		}))
		defer srv.Close()

		post := NewInfluxPoster(srv.URL, "")
		if err := post("line"); err != nil {
			t.Fatalf("post returned error: %v", err)
		}
		if authPresent {
			t.Fatal("Authorization header sent for empty token, want none")
		}
	})
}

// Bad: a non-2xx response (>=300) from the write endpoint is surfaced as an
// error carrying the status code — the poster reports a rejected write rather
// than silently succeeding.
func TestInflux_NewInfluxPoster_Bad(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	post := NewInfluxPoster(srv.URL, "tok")
	err := post("line")
	if err == nil {
		t.Fatal("post to 500 endpoint returned nil, want error")
	}
	if !core.Contains(err.Error(), "500") {
		t.Fatalf("error = %q, want it to mention status 500", err.Error())
	}
}

// Ugly: the transport-failure edges — a dead endpoint (connection refused)
// and a malformed URL that fails request construction before any network
// call — both surface as errors from the poster, never a panic.
func TestInflux_NewInfluxPoster_Ugly(t *testing.T) {
	t.Run("UnreachableEndpoint", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
		url := srv.URL
		srv.Close() // close before posting → connection refused
		post := NewInfluxPoster(url, "tok")
		if err := post("line"); err == nil {
			t.Fatal("post to closed server returned nil, want transport error")
		}
	})
	t.Run("MalformedURL", func(t *testing.T) {
		// A malformed target URL fails request construction
		// (core.NewHTTPRequest's !OK branch) before any network call.
		post := NewInfluxPoster("://no-scheme\x7f", "tok")
		if err := post("line"); err == nil {
			t.Fatal("post with malformed URL returned nil, want request-build error")
		}
	})
}

// --- LineProtocolSink.EmitProbe ---

// Good: EmitProbe renders the v0 schema verbatim. These golden strings are
// the contract with the existing LEM dashboards (lthn/LEM pkg/lem ingest) —
// a drift here is a broken instrument, not a style choice. Covers the train
// line (with derived throughput), the val line, multi-dimension score lines
// (sorted, has_kernel derived from label), and tag escaping.
func TestInflux_LineProtocolSink_EmitProbe_Good(t *testing.T) {
	t.Run("TrainLineV0Schema", func(t *testing.T) {
		var posts []string
		s := NewLineProtocolSink(LineProtocolConfig{
			Model: "LEM-gemma3-1b", RunID: "gold-1",
			Post:       func(body string) error { posts = append(posts, body); return nil },
			BatchLines: 1,
			now:        lpClock(2 * time.Second),
		})
		// First train step: no interval yet — rates read 0.
		s.EmitProbe(Event{Kind: KindTraining, Training: &Training{
			Step: 1, Loss: 2.5, LearningRate: 0.0001, Tokens: 512, LossType: LossTypeTrain,
		}})
		want := "training_loss,model=LEM-gemma3-1b,run_id=gold-1,loss_type=train loss=2.500000,learning_rate=0.000100,iterations_per_sec=0.000000,tokens_per_sec=0.000000,iteration=1i"
		if len(posts) != 1 || posts[0] != want {
			t.Fatalf("first train line:\n got %q\nwant %q", posts, want)
		}
		// Second step 2s later: 0.5 it/s, 256 tok/s.
		s.EmitProbe(Event{Kind: KindTraining, Training: &Training{
			Step: 2, Loss: 2.25, LearningRate: 0.0001, Tokens: 512, LossType: LossTypeTrain,
		}})
		want = "training_loss,model=LEM-gemma3-1b,run_id=gold-1,loss_type=train loss=2.250000,learning_rate=0.000100,iterations_per_sec=0.500000,tokens_per_sec=256.000000,iteration=2i"
		if posts[1] != want {
			t.Fatalf("second train line:\n got %q\nwant %q", posts[1], want)
		}
	})
	t.Run("ValLineV0Schema", func(t *testing.T) {
		var posts []string
		s := NewLineProtocolSink(LineProtocolConfig{
			Model: "LEM-gemma3-1b", RunID: "gold-1",
			Post:       func(body string) error { posts = append(posts, body); return nil },
			BatchLines: 1,
		})
		s.EmitProbe(Event{Kind: KindTraining, Training: &Training{Step: 25, Loss: 1.75, LossType: LossTypeVal}})
		want := "training_loss,model=LEM-gemma3-1b,run_id=gold-1,loss_type=val loss=1.750000,iteration=25i"
		if len(posts) != 1 || posts[0] != want {
			t.Fatalf("val line:\n got %q\nwant %q", posts, want)
		}
	})
	t.Run("ScoreLinesV0Schema", func(t *testing.T) {
		// Score events become content_score lines — one per dimension,
		// sorted for determinism, has_kernel derived from the label.
		var posts []string
		s := NewLineProtocolSink(LineProtocolConfig{
			Model: "m", RunID: "r",
			Post:       func(body string) error { posts = append(posts, body); return nil },
			BatchLines: 2,
		})
		s.EmitProbe(Event{Kind: KindScore, Step: 50, Score: &Score{
			Label:  "sft-eval-kernel",
			Values: map[string]float64{"lek": 61.5, "composite": 58.25},
		}})
		if len(posts) != 1 {
			t.Fatalf("posts = %d, want 1 (two lines, batch of 2)", len(posts))
		}
		want := "content_score,model=m,run_id=r,label=sft-eval-kernel,dimension=composite,has_kernel=true score=58.250000,iteration=50i\n" +
			"content_score,model=m,run_id=r,label=sft-eval-kernel,dimension=lek,has_kernel=true score=61.500000,iteration=50i"
		if posts[0] != want {
			t.Fatalf("score lines:\n got %q\nwant %q", posts[0], want)
		}
	})
	t.Run("TagEscaping", func(t *testing.T) {
		// Tag escaping mirrors v0's escapeLp: comma, equals, space.
		var posts []string
		s := NewLineProtocolSink(LineProtocolConfig{
			Model: "LEM gemma3,1b=x", RunID: "r",
			Post:       func(body string) error { posts = append(posts, body); return nil },
			BatchLines: 1,
		})
		s.EmitProbe(Event{Kind: KindTraining, Training: &Training{Step: 1, Loss: 1, LossType: LossTypeVal}})
		want := `training_loss,model=LEM\ gemma3\,1b\=x,run_id=r,loss_type=val loss=1.000000,iteration=1i`
		if posts[0] != want {
			t.Fatalf("escaped line:\n got %q\nwant %q", posts[0], want)
		}
	})
}

// Bad: a score event with no label must not emit an empty label tag — the
// instrument defaults the label to "unknown" so every content_score line
// carries a well-formed label dimension.
func TestInflux_LineProtocolSink_EmitProbe_Bad(t *testing.T) {
	var posts []string
	s := NewLineProtocolSink(LineProtocolConfig{
		Model: "m", RunID: "r",
		Post:       func(body string) error { posts = append(posts, body); return nil },
		BatchLines: 1,
	})
	s.EmitProbe(Event{Kind: KindScore, Step: 7, Score: &Score{Values: map[string]float64{"lek": 1.0}}})
	want := "content_score,model=m,run_id=r,label=unknown,dimension=lek,has_kernel=false score=1.000000,iteration=7i"
	if len(posts) != 1 || posts[0] != want {
		t.Fatalf("empty-label score line:\n got %q\nwant %q", posts, want)
	}
}

// Ugly: EmitProbe ignores everything that is not a populated training or
// score event — other kinds, nil payloads, and empty score values all leave
// the sink with zero lines; a nil sink no-ops too. This sink is the training
// instrument, not a firehose.
func TestInflux_LineProtocolSink_EmitProbe_Ugly(t *testing.T) {
	s := NewLineProtocolSink(LineProtocolConfig{Model: "m", RunID: "r"})
	s.EmitProbe(Event{Kind: KindToken, Token: &Token{ID: 1}})
	s.EmitProbe(Event{Kind: KindTraining})               // nil payload
	s.EmitProbe(Event{Kind: KindScore})                  // nil payload
	s.EmitProbe(Event{Kind: KindScore, Score: &Score{}}) // empty values
	if s.Lines() != 0 {
		t.Fatalf("lines = %d, want 0", s.Lines())
	}
	var nilSink *LineProtocolSink
	nilSink.EmitProbe(Event{Kind: KindTraining, Training: &Training{}})
	if nilSink.Lines() != 0 || nilSink.Dropped() != 0 {
		t.Fatal("nil sink must no-op")
	}
}

// --- LineProtocolSink.Flush ---

// Good: Flush posts any buffered-but-not-yet-batched lines on demand. With a
// batch larger than the emitted count nothing auto-flushes, so the single
// pending line is posted only when Flush is called.
func TestInflux_LineProtocolSink_Flush_Good(t *testing.T) {
	posts := 0
	s := NewLineProtocolSink(LineProtocolConfig{
		Model: "m", RunID: "r",
		Post:       func(string) error { posts++; return nil },
		BatchLines: 10, // larger than the one line emitted → no auto-flush
	})
	s.EmitProbe(Event{Kind: KindTraining, Training: &Training{Step: 1, Loss: 1, LossType: LossTypeVal}})
	if posts != 0 {
		t.Fatalf("posts before Flush = %d, want 0 (batch not reached)", posts)
	}
	s.Flush()
	if posts != 1 {
		t.Fatalf("posts after Flush = %d, want 1", posts)
	}
}

// Bad: Flush on a sink whose poster errors counts the failed lines as drops
// and never propagates the error outward — the run keeps going, the gap is
// recorded honestly.
func TestInflux_LineProtocolSink_Flush_Bad(t *testing.T) {
	s := NewLineProtocolSink(LineProtocolConfig{
		Model: "m", RunID: "r",
		Post:       func(string) error { return core.NewError("dashboard down") },
		BatchLines: 10, // buffer the line; Flush triggers the failing post
	})
	s.EmitProbe(Event{Kind: KindTraining, Training: &Training{Step: 1, Loss: 1, LossType: LossTypeVal}})
	s.Flush() // must not panic or return; counts the drop
	if s.Dropped() != 1 {
		t.Fatalf("Dropped() = %d, want 1 — a failed Flush post counts a drop", s.Dropped())
	}
}

// Ugly: Flush on a Post-less sink and on a nil *LineProtocolSink are both
// clean no-ops — flushLocked early-returns with no poster, and the nil
// receiver guards before touching any field.
func TestInflux_LineProtocolSink_Flush_Ugly(t *testing.T) {
	path := t.TempDir() + "/metrics.lp"
	s := NewLineProtocolSink(LineProtocolConfig{Model: "m", RunID: "r", FilePath: path})
	s.EmitProbe(Event{Kind: KindTraining, Training: &Training{Step: 1, Loss: 1, LossType: LossTypeVal}})
	s.Flush() // Post is nil — flushLocked must early-return without panic
	read, err := coreio.Local.Read(path)
	if err != nil {
		t.Fatalf("file copy: %v", err)
	}
	if read == "" || s.Lines() != 1 || s.Dropped() != 0 {
		t.Fatalf("file-only sink: read=%q lines=%d dropped=%d", read, s.Lines(), s.Dropped())
	}
	var nilSink *LineProtocolSink
	nilSink.Flush() // must not panic
}

// --- LineProtocolSink.Close ---

// Good: Close flushes the remaining buffered lines. A batch of 3 over 4
// emitted val lines posts once before Close (the full batch) and once on
// Close (the remainder), and the durable file copy holds all 4 lines.
func TestInflux_LineProtocolSink_Close_Good(t *testing.T) {
	path := t.TempDir() + "/metrics.lp"
	posted := 0
	s := NewLineProtocolSink(LineProtocolConfig{
		Model: "m", RunID: "r", FilePath: path,
		Post:       func(string) error { posted++; return nil },
		BatchLines: 3,
	})
	for step := 1; step <= 4; step++ {
		s.EmitProbe(Event{Kind: KindTraining, Training: &Training{Step: step, Loss: 1, LossType: LossTypeVal}})
	}
	if posted != 1 {
		t.Fatalf("posts before close = %d, want 1 (batch of 3)", posted)
	}
	s.Close()
	if posted != 2 {
		t.Fatalf("posts after close = %d, want 2 (flush remainder)", posted)
	}
	read, err := coreio.Local.Read(path)
	if err != nil {
		t.Fatalf("file copy: %v", err)
	}
	lines := 0
	for _, b := range []byte(read) {
		if b == '\n' {
			lines++
		}
	}
	if lines != 4 || s.Lines() != 4 {
		t.Fatalf("file lines = %d, sink lines = %d, want 4/4", lines, s.Lines())
	}
}

// Bad: Close flushes through a failing poster — the buffered remainder is
// counted as dropped, and Close still returns cleanly (it must never error
// outward or interrupt the caller's shutdown).
func TestInflux_LineProtocolSink_Close_Bad(t *testing.T) {
	s := NewLineProtocolSink(LineProtocolConfig{
		Model: "m", RunID: "r",
		Post:       func(string) error { return core.NewError("dashboard down") },
		BatchLines: 10, // line stays buffered until Close flushes it
	})
	s.EmitProbe(Event{Kind: KindTraining, Training: &Training{Step: 1, Loss: 1, LossType: LossTypeVal}})
	s.Close()
	if s.Dropped() != 1 {
		t.Fatalf("Dropped() after Close = %d, want 1 — Close flush failure counts a drop", s.Dropped())
	}
}

// Ugly: Close is reusable and nil-safe — calling it on a fresh sink with
// nothing buffered is a no-op, the sink stays usable after Close, and Close
// on a nil *LineProtocolSink does not panic.
func TestInflux_LineProtocolSink_Close_Ugly(t *testing.T) {
	s := NewLineProtocolSink(LineProtocolConfig{Model: "m", RunID: "r"})
	s.Close() // nothing buffered → clean no-op
	// Still usable after Close — it holds no descriptors between emits.
	s.EmitProbe(Event{Kind: KindTraining, Training: &Training{Step: 1, Loss: 1, LossType: LossTypeVal}})
	if s.Lines() != 1 {
		t.Fatalf("sink unusable after Close: Lines() = %d, want 1", s.Lines())
	}
	var nilSink *LineProtocolSink
	nilSink.Close() // must not panic
}

// --- LineProtocolSink.Lines ---

// Good: Lines reports the running count of formatted lines — one per training
// event here, climbing to 3 across three emits.
func TestInflux_LineProtocolSink_Lines_Good(t *testing.T) {
	s := NewLineProtocolSink(LineProtocolConfig{Model: "m", RunID: "r"})
	for step := 1; step <= 3; step++ {
		s.EmitProbe(Event{Kind: KindTraining, Training: &Training{Step: step, Loss: 1, LossType: LossTypeVal}})
	}
	if s.Lines() != 3 {
		t.Fatalf("Lines() = %d, want 3", s.Lines())
	}
}

// Bad: Lines counts formatted lines, not emitted events — ignored events
// (wrong kind, nil payloads) never advance the counter, so Lines stays 0
// after a burst of events that produce no lines.
func TestInflux_LineProtocolSink_Lines_Bad(t *testing.T) {
	s := NewLineProtocolSink(LineProtocolConfig{Model: "m", RunID: "r"})
	s.EmitProbe(Event{Kind: KindToken, Token: &Token{ID: 1}}) // ignored kind
	s.EmitProbe(Event{Kind: KindTraining})                    // nil payload
	s.EmitProbe(Event{Kind: KindScore, Score: &Score{}})      // empty values
	if s.Lines() != 0 {
		t.Fatalf("Lines() = %d, want 0 — only formatted lines count", s.Lines())
	}
}

// Ugly: Lines on a nil *LineProtocolSink returns 0 rather than panicking.
func TestInflux_LineProtocolSink_Lines_Ugly(t *testing.T) {
	var s *LineProtocolSink
	if got := s.Lines(); got != 0 {
		t.Fatalf("nil sink Lines() = %d, want 0", got)
	}
}

// --- LineProtocolSink.Dropped ---

// Good: Dropped counts lines that failed a destination. A poster that always
// errors (batch of 1, so every line posts immediately) drives the counter up
// by one per failed line, and the failure never propagates outward.
func TestInflux_LineProtocolSink_Dropped_Good(t *testing.T) {
	s := NewLineProtocolSink(LineProtocolConfig{
		Model: "m", RunID: "r",
		Post:       func(string) error { return core.NewError("dashboard down") },
		BatchLines: 1,
	})
	s.EmitProbe(Event{Kind: KindTraining, Training: &Training{Step: 1, Loss: 1, LossType: LossTypeVal}})
	if s.Dropped() != 1 {
		t.Fatalf("dropped = %d, want 1 — post failures count, never propagate", s.Dropped())
	}
}

// Bad: the file side of the honesty counter — a FilePath that points at a
// directory makes the append fail (EISDIR). Dropped increments without the
// sink ever erroring outward.
func TestInflux_LineProtocolSink_Dropped_Bad(t *testing.T) {
	dir := t.TempDir() // a directory is not an appendable file → EISDIR
	s := NewLineProtocolSink(LineProtocolConfig{Model: "m", RunID: "r", FilePath: dir})
	s.EmitProbe(Event{Kind: KindTraining, Training: &Training{Step: 1, Loss: 1, LossType: LossTypeVal}})
	if s.Dropped() != 1 {
		t.Fatalf("dropped = %d, want 1 — a failed file append counts a drop", s.Dropped())
	}
}

// Ugly: Dropped is 0 on a healthy sink with no destinations and on a nil
// *LineProtocolSink — no destination means nothing can fail.
func TestInflux_LineProtocolSink_Dropped_Ugly(t *testing.T) {
	s := NewLineProtocolSink(LineProtocolConfig{Model: "m", RunID: "r"})
	s.EmitProbe(Event{Kind: KindTraining, Training: &Training{Step: 1, Loss: 1, LossType: LossTypeVal}})
	if s.Dropped() != 0 {
		t.Fatalf("healthy no-destination sink Dropped() = %d, want 0", s.Dropped())
	}
	var nilSink *LineProtocolSink
	if got := nilSink.Dropped(); got != 0 {
		t.Fatalf("nil sink Dropped() = %d, want 0", got)
	}
}
