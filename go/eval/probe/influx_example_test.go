// SPDX-Licence-Identifier: EUPL-1.2

package probe_test

import (
	"net/http"
	"net/http/httptest"

	core "dappco.re/go"
	"dappco.re/go/inference/eval/probe"
)

// ExampleNewLineProtocolSink builds the v0-schema metrics sink and formats
// one validation-loss line, then reports how many lines it produced.
func ExampleNewLineProtocolSink() {
	sink := probe.NewLineProtocolSink(probe.LineProtocolConfig{Model: "m", RunID: "r"})
	sink.EmitProbe(probe.Event{Kind: probe.KindTraining, Training: &probe.Training{
		Step: 1, Loss: 1.5, LossType: probe.LossTypeVal,
	}})
	core.Println(sink.Lines())
	// Output: 1
}

// ExampleNewInfluxPoster builds an InfluxDB write closure and posts one line
// to a loopback test server, which accepts it with 204 No Content.
func ExampleNewInfluxPoster() {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))
	defer srv.Close()
	post := probe.NewInfluxPoster(srv.URL, "")
	err := post("training_loss,model=m loss=1.0 1i")
	core.Println(err == nil)
	// Output: true
}

// ExampleLineProtocolSink_EmitProbe converts a training event into a
// training_loss line; non-training, non-score events are ignored.
func ExampleLineProtocolSink_EmitProbe() {
	sink := probe.NewLineProtocolSink(probe.LineProtocolConfig{Model: "m", RunID: "r"})
	sink.EmitProbe(probe.Event{Kind: probe.KindToken, Token: &probe.Token{ID: 1}}) // ignored
	sink.EmitProbe(probe.Event{Kind: probe.KindTraining, Training: &probe.Training{
		Step: 1, Loss: 2.0, LossType: probe.LossTypeVal,
	}})
	core.Println(sink.Lines())
	// Output: 1
}

// ExampleLineProtocolSink_Flush posts any pending lines on demand.
func ExampleLineProtocolSink_Flush() {
	posts := 0
	sink := probe.NewLineProtocolSink(probe.LineProtocolConfig{
		Model: "m", RunID: "r",
		Post:       func(string) error { posts++; return nil },
		BatchLines: 10, // larger than the one line we emit, so no auto-flush
	})
	sink.EmitProbe(probe.Event{Kind: probe.KindTraining, Training: &probe.Training{
		Step: 1, Loss: 1.0, LossType: probe.LossTypeVal,
	}})
	sink.Flush()
	core.Println(posts)
	// Output: 1
}

// ExampleLineProtocolSink_Close flushes the remaining buffered lines.
func ExampleLineProtocolSink_Close() {
	posts := 0
	sink := probe.NewLineProtocolSink(probe.LineProtocolConfig{
		Model: "m", RunID: "r",
		Post:       func(string) error { posts++; return nil },
		BatchLines: 10,
	})
	sink.EmitProbe(probe.Event{Kind: probe.KindTraining, Training: &probe.Training{
		Step: 1, Loss: 1.0, LossType: probe.LossTypeVal,
	}})
	sink.Close()
	core.Println(posts)
	// Output: 1
}

// ExampleLineProtocolSink_Lines reports how many lines the sink has formatted.
func ExampleLineProtocolSink_Lines() {
	sink := probe.NewLineProtocolSink(probe.LineProtocolConfig{Model: "m", RunID: "r"})
	for step := 1; step <= 3; step++ {
		sink.EmitProbe(probe.Event{Kind: probe.KindTraining, Training: &probe.Training{
			Step: step, Loss: 1.0, LossType: probe.LossTypeVal,
		}})
	}
	core.Println(sink.Lines())
	// Output: 3
}

// ExampleLineProtocolSink_Dropped counts lines that failed a destination —
// here a poster that always errors. The drop never propagates outward.
func ExampleLineProtocolSink_Dropped() {
	sink := probe.NewLineProtocolSink(probe.LineProtocolConfig{
		Model: "m", RunID: "r",
		Post:       func(string) error { return core.NewError("dashboard down") },
		BatchLines: 1,
	})
	sink.EmitProbe(probe.Event{Kind: probe.KindTraining, Training: &probe.Training{
		Step: 1, Loss: 1.0, LossType: probe.LossTypeVal,
	}})
	core.Println(sink.Dropped())
	// Output: 1
}
