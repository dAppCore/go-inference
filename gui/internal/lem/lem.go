// SPDX-Licence-Identifier: EUPL-1.2

// Package lem bridges the LEM desktop GUI to go-inference's consolidated
// packages. It is the new home of what the GUI used to import from
// dappco.re/lthn/lem/pkg/lem, before the AI features consolidated into
// go-inference: the metrics client + DuckDB store now live in eval/datapipe,
// and the scoring agent loop in agent. This shim keeps the GUI's call sites
// (lem.NewInfluxClient / lem.OpenDB / lem.InfluxClient / lem.RunAgent) intact.
package lem

import (
	core "dappco.re/go"
	"dappco.re/go/inference/agent"
	"dappco.re/go/inference/eval/datapipe"
)

// InfluxClient wraps datapipe.InfluxClient so the dashboard keeps its
// (value, error) call shape — datapipe's methods return core.Result.
type InfluxClient struct{ inner *datapipe.InfluxClient }

// QuerySQL runs a read query and returns the result rows (one map per row,
// column-name keyed) or an error.
func (c *InfluxClient) QuerySQL(sql string) ([]map[string]any, error) {
	r := c.inner.QuerySQL(sql)
	if !r.OK {
		if err, ok := r.Value.(error); ok {
			return nil, err
		}
		return nil, core.NewError("lem.InfluxClient.QuerySQL failed")
	}
	rows, _ := r.Value.([]map[string]any)
	return rows, nil
}

// WriteLp writes line-protocol points.
func (c *InfluxClient) WriteLp(lines []string) error {
	if r := c.inner.WriteLp(lines); !r.OK {
		if err, ok := r.Value.(error); ok {
			return err
		}
		return core.NewError("lem.InfluxClient.WriteLp failed")
	}
	return nil
}

// DB wraps datapipe.DB so the dashboard keeps its (value, error) call shape —
// datapipe's DB methods return core.Result, the GUI expects Go-idiomatic pairs.
type DB struct{ inner *datapipe.DB }

// Close releases the DuckDB handle.
func (d *DB) Close() error {
	if r := d.inner.Close(); !r.OK {
		if err, ok := r.Value.(error); ok {
			return err
		}
	}
	return nil
}

// QueryRows runs a query and returns the datapipe result value (rows) or an error.
func (d *DB) QueryRows(query string, args ...any) (any, error) {
	r := d.inner.QueryRows(query, args...)
	if !r.OK {
		if err, ok := r.Value.(error); ok {
			return nil, err
		}
		return nil, core.NewError("lem.DB.QueryRows failed")
	}
	return r.Value, nil
}

// NewInfluxClient constructs a metrics client for (url, db).
//
//	c := lem.NewInfluxClient("http://localhost:8086", "lem")
func NewInfluxClient(url, db string) *InfluxClient {
	return &InfluxClient{inner: datapipe.NewInfluxClient(url, db)}
}

// OpenDB opens the read-only DuckDB metrics store, adapting datapipe's
// core.Result into the (value, error) pair the dashboard expects.
//
//	db, err := lem.OpenDB(path)
//	if err != nil { return err }
//	defer db.Close()
func OpenDB(path string) (*DB, error) {
	r := datapipe.OpenDB(path)
	if !r.OK {
		if err, ok := r.Value.(error); ok {
			return nil, err
		}
		return nil, core.NewError("lem.OpenDB: datapipe.OpenDB failed")
	}
	return &DB{inner: r.Value.(*datapipe.DB)}, nil
}

// RunAgent parses the CLI-style flags the desktop builds (--api-url, --influx,
// --influx-db, --m3-host, --base-model, --work-dir) into an agent.AgentConfig
// and runs the scoring loop. Blocks until the loop exits.
//
//	lem.RunAgent([]string{"--api-url", u, "--influx", iu, "--work-dir", wd})
func RunAgent(args []string) {
	cfg := &agent.AgentConfig{}
	for i := 0; i+1 < len(args); i += 2 {
		switch args[i] {
		case "--api-url":
			cfg.APIURL = args[i+1]
		case "--influx":
			cfg.InfluxURL = args[i+1]
		case "--influx-db":
			cfg.InfluxDB = args[i+1]
		case "--m3-host":
			cfg.M3Host = args[i+1]
		case "--base-model":
			cfg.BaseModel = args[i+1]
		case "--work-dir":
			cfg.WorkDir = args[i+1]
		}
	}
	agent.RunAgentLoop(cfg)
}
