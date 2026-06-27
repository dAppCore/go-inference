package datapipe

import (
	"net/http"
	"time"

	"dappco.re/go"
	coreio "dappco.re/go/io"
)

// InfluxClient talks to an InfluxDB v3 instance.
type InfluxClient struct {
	url   string
	db    string
	token string
}

// NewInfluxClient creates an InfluxClient for the given URL and database.
// Reads token from INFLUX_TOKEN env var first, then ~/.influx_token file.
// If url is empty, defaults to "http://10.69.69.165:8181".
// If db is empty, defaults to "training".
func NewInfluxClient(url, db string) *InfluxClient {
	if url == "" {
		url = "http://10.69.69.165:8181"
	}
	if db == "" {
		db = "training"
	}

	token := core.Env("INFLUX_TOKEN")
	if token == "" {
		home := core.Env("DIR_HOME")
		if home != "" {
			data, err := coreio.Local.Read(core.JoinPath(home, ".influx_token"))
			if err == nil {
				token = core.Trim(data)
			}
		}
	}

	return &InfluxClient{
		url:   url,
		db:    db,
		token: token,
	}
}

// WriteLp writes line protocol data to InfluxDB.
//
//	r := client.WriteLp([]string{"cpu,host=local usage=0.5"})
//	if !r.OK { return r }
func (c *InfluxClient) WriteLp(lines []string) core.Result {
	body := core.Join("\n", lines...)

	url := core.Sprintf("%s/api/v3/write_lp?db=%s", c.url, c.db)

	req, err := http.NewRequest(http.MethodPost, url, core.NewReader(body))
	if err != nil {
		return core.Fail(core.E("ml.InfluxClient.WriteLp", "create write request", err))
	}
	req.Header.Set("Authorization", "Bearer "+c.token)
	req.Header.Set("Content-Type", "text/plain")

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return core.Fail(core.E("ml.InfluxClient.WriteLp", "write request", err))
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusNoContent {
		rBody := readAll(resp.Body)
		var bodyStr string
		if rBody.OK {
			bodyStr = string(rBody.Value.([]byte))
		}
		return core.Fail(core.E("ml.InfluxClient.WriteLp", core.Sprintf("write failed %d: %s", resp.StatusCode, bodyStr), nil))
	}

	return core.Ok(nil)
}

// QuerySQL runs a SQL query against InfluxDB and returns the result rows.
//
//	r := client.QuerySQL("SELECT * FROM metrics LIMIT 10")
//	if !r.OK { return r }
//	rows := r.Value.([]map[string]any)
func (c *InfluxClient) QuerySQL(sql string) core.Result {
	reqBody := map[string]string{
		"db": c.db,
		"q":  sql,
	}

	jsonBody := []byte(core.JSONMarshalString(reqBody))

	url := c.url + "/api/v3/query_sql"

	req, err := http.NewRequest(http.MethodPost, url, core.NewBuffer(jsonBody))
	if err != nil {
		return core.Fail(core.E("ml.InfluxClient.QuerySQL", "create query request", err))
	}
	req.Header.Set("Authorization", "Bearer "+c.token)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return core.Fail(core.E("ml.InfluxClient.QuerySQL", "query request", err))
	}
	defer resp.Body.Close()

	rBody := readAll(resp.Body)
	if !rBody.OK {
		return core.Fail(core.E("ml.InfluxClient.QuerySQL", "read query response", rBody.Value.(error)))
	}
	respBody := rBody.Value.([]byte)

	if resp.StatusCode != http.StatusOK {
		return core.Fail(core.E("ml.InfluxClient.QuerySQL", core.Sprintf("query failed %d: %s", resp.StatusCode, string(respBody)), nil))
	}

	var rows []map[string]any
	if r := core.JSONUnmarshal(respBody, &rows); !r.OK {
		return core.Fail(core.E("ml.InfluxClient.QuerySQL", "unmarshal query response", r.Value.(error)))
	}

	return core.Ok(rows)
}

// EscapeLp escapes spaces, commas, and equals signs for InfluxDB line protocol
// tag values.
func EscapeLp(s string) string {
	s = core.Replace(s, `,`, `\,`)
	s = core.Replace(s, `=`, `\=`)
	s = core.Replace(s, ` `, `\ `)
	return s
}
