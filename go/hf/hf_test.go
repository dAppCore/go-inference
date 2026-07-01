// SPDX-Licence-Identifier: EUPL-1.2

package hf

import (
	"context"
	"testing"

	core "dappco.re/go"
)

// writeHFFile writes data to path, failing the test on error. Small shared
// fixture helper for the local-cache tests below.
func writeHFFile(t *testing.T, path string, data string) {
	t.Helper()
	if result := core.WriteFile(path, []byte(data), 0o644); !result.OK {
		t.Fatalf("write %s: %v", path, result.Value)
	}
}

// TestHf_RemoteSource_ImplementsModelSource_Good is a compile-time contract
// check: RemoteSource must satisfy ModelSource so every engine's fit-planner
// can accept either the production client or a test fixture interchangeably.
func TestHf_RemoteSource_ImplementsModelSource_Good(t *testing.T) {
	var _ ModelSource = (*RemoteSource)(nil)
}

// TestHf_NewRemoteSource_Good covers the constructor's happy path: a fully
// specified RemoteConfig (BaseURL without a trailing slash, explicit
// UserAgent, a token, and an injected client) is stored verbatim and the
// Authorization header value is pre-built once as "Bearer <token>". No
// network — the constructor only assembles fields.
func TestHf_NewRemoteSource_Good(t *testing.T) {
	client := &core.HTTPClient{}
	source := NewRemoteSource(RemoteConfig{
		BaseURL:   "https://hub.example.com",
		Token:     "secret-token",
		UserAgent: "hf-tests",
		Client:    client,
	})
	if source.baseURL != "https://hub.example.com" {
		t.Fatalf("baseURL = %q, want the supplied URL verbatim", source.baseURL)
	}
	if source.userAgent != "hf-tests" {
		t.Fatalf("userAgent = %q, want the supplied override", source.userAgent)
	}
	if source.authValue != "Bearer secret-token" {
		t.Fatalf("authValue = %q, want pre-built \"Bearer secret-token\"", source.authValue)
	}
	if source.client != client {
		t.Fatal("client = injected client not retained, want the supplied *core.HTTPClient")
	}
}

// TestHf_NewRemoteSource_Ugly covers the constructor's degenerate inputs: a
// zero-value RemoteConfig must default the BaseURL to the public hub and the
// user-agent to "go-inference", leave the auth header empty (no token), and
// synthesise a non-nil client. A trailing slash on a supplied BaseURL is
// trimmed exactly once.
func TestHf_NewRemoteSource_Ugly(t *testing.T) {
	empty := NewRemoteSource(RemoteConfig{})
	if empty.baseURL != "https://huggingface.co" {
		t.Fatalf("zero-config baseURL = %q, want the default hub", empty.baseURL)
	}
	if empty.userAgent != "go-inference" {
		t.Fatalf("zero-config userAgent = %q, want default go-inference", empty.userAgent)
	}
	if empty.authValue != "" {
		t.Fatalf("zero-config authValue = %q, want empty (no token)", empty.authValue)
	}
	if empty.client == nil {
		t.Fatal("zero-config client = nil, want a synthesised *core.HTTPClient")
	}

	trimmed := NewRemoteSource(RemoteConfig{BaseURL: "https://hub.example.com/"})
	if trimmed.baseURL != "https://hub.example.com" {
		t.Fatalf("trailing-slash baseURL = %q, want the slash trimmed", trimmed.baseURL)
	}
}

// TestHf_RemoteSource_SearchModels_Good drives the happy path end-to-end: a
// loopback httptest server serves the HF /api/models search list, and
// SearchModels round-trips it (verifying the search query/limit on the wire
// and the size/sizeBytes fallback). No real network.
func TestHf_RemoteSource_SearchModels_Good(t *testing.T) {
	server := core.NewHTTPTestServer(core.HandlerFunc(func(w core.ResponseWriter, r *core.Request) {
		if r.URL.Path != "/api/models" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
		if r.URL.Query().Get("search") != "qwen" || r.URL.Query().Get("limit") != "2" {
			t.Fatalf("query = %q, want search/limit", r.URL.RawQuery)
		}
		w.Header().Set("Content-Type", "application/json")
		core.WriteString(w, `[{
			"id": "Qwen/Qwen3-0.6B",
			"pipeline_tag": "text-generation",
			"config": {"model_type": "qwen3", "hidden_size": 1024},
			"siblings": [{"rfilename": "model.safetensors", "sizeBytes": 440401920}]
		}]`)
	}))
	defer server.Close()

	source := NewRemoteSource(RemoteConfig{BaseURL: server.URL})
	found, err := source.SearchModels(context.Background(), "qwen", 2)
	if err != nil {
		t.Fatalf("SearchModels() error = %v", err)
	}
	if len(found) != 1 || found[0].ID != "Qwen/Qwen3-0.6B" {
		t.Fatalf("SearchModels() = %+v", found)
	}
	if found[0].Files[0].byteSize() != 440401920 {
		t.Fatalf("file size = %+v, want the sizeBytes fallback", found[0].Files[0])
	}
}

// TestHf_RemoteSource_SearchModels_Bad covers SearchModels' error surface: a
// nil receiver returns a guard error, and a malformed JSON body from the
// server surfaces a parse error. Loopback httptest only — no real network.
func TestHf_RemoteSource_SearchModels_Bad(t *testing.T) {
	var nilSource *RemoteSource
	if _, err := nilSource.SearchModels(context.Background(), "qwen", 1); err == nil {
		t.Fatal("SearchModels(nil receiver) error = nil, want a guard error")
	}

	server := core.NewHTTPTestServer(core.HandlerFunc(func(w core.ResponseWriter, r *core.Request) {
		if r.URL.Path != "/api/models" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
		core.WriteString(w, "{") // malformed JSON array
	}))
	defer server.Close()

	source := NewRemoteSource(RemoteConfig{BaseURL: server.URL})
	if _, err := source.SearchModels(context.Background(), "qwen", 5); err == nil {
		t.Fatal("SearchModels(malformed response) error = nil, want a parse error")
	}
}

// TestHf_RemoteSource_SearchModels_Ugly covers SearchModels' awkward edges: a
// non-positive limit is normalised to the default 10 (asserted on the wire),
// and pointing the source at a closed loopback server surfaces a transport
// error. The server is started then immediately closed so the dial fails
// locally — no real network egress.
func TestHf_RemoteSource_SearchModels_Ugly(t *testing.T) {
	var gotLimit string
	server := core.NewHTTPTestServer(core.HandlerFunc(func(w core.ResponseWriter, r *core.Request) {
		if r.URL.Path != "/api/models" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
		gotLimit = r.URL.Query().Get("limit")
		w.Header().Set("Content-Type", "application/json")
		core.WriteString(w, `[]`)
	}))

	source := NewRemoteSource(RemoteConfig{BaseURL: server.URL})
	if _, err := source.SearchModels(context.Background(), "qwen", 0); err != nil {
		t.Fatalf("SearchModels(limit 0) error = %v", err)
	}
	if gotLimit != "10" {
		t.Fatalf("limit on the wire = %q, want 10 (non-positive limit defaults)", gotLimit)
	}
	closedURL := server.URL
	server.Close() // nothing listens at closedURL now -> dial fails
	dead := NewRemoteSource(RemoteConfig{BaseURL: closedURL})
	if _, err := dead.SearchModels(context.Background(), "qwen", 1); err == nil {
		t.Fatal("SearchModels(closed server) error = nil, want a transport error")
	}
}

// TestHf_RemoteSource_ModelMetadata_Good drives ModelMetadata against a
// loopback httptest server that returns a metadata body for one model id,
// including the Bearer auth header check. No real network.
func TestHf_RemoteSource_ModelMetadata_Good(t *testing.T) {
	server := core.NewHTTPTestServer(core.HandlerFunc(func(w core.ResponseWriter, r *core.Request) {
		if r.URL.Path != "/api/models/Qwen/Qwen3-0.6B" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-token" {
			t.Fatalf("Authorization = %q", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		core.WriteString(w, `{
			"modelId": "Qwen/Qwen3-0.6B",
			"config": {"model_type": "qwen3", "num_hidden_layers": 28},
			"siblings": [{"rfilename": "model.safetensors", "size": 440401920}]
		}`)
	}))
	defer server.Close()

	source := NewRemoteSource(RemoteConfig{BaseURL: server.URL, Token: "test-token"})
	meta, err := source.ModelMetadata(context.Background(), "Qwen/Qwen3-0.6B")
	if err != nil {
		t.Fatalf("ModelMetadata() error = %v", err)
	}
	if meta.ModelID != "Qwen/Qwen3-0.6B" || meta.Config.NumHiddenLayers != 28 {
		t.Fatalf("ModelMetadata() = %+v, want the served modelId + config", meta)
	}
	if len(meta.Files) != 1 || meta.Files[0].byteSize() != 440401920 {
		t.Fatalf("ModelMetadata() files = %+v, want one sibling with the size field", meta.Files)
	}
}

// TestHf_RemoteSource_ModelMetadata_Bad covers ModelMetadata's error surface:
// a nil receiver returns a guard error, and an HTTP 404 from the server
// surfaces a status error carrying the code. Loopback httptest only.
func TestHf_RemoteSource_ModelMetadata_Bad(t *testing.T) {
	var nilSource *RemoteSource
	if _, err := nilSource.ModelMetadata(context.Background(), "org/model"); err == nil {
		t.Fatal("ModelMetadata(nil receiver) error = nil, want a guard error")
	}

	server := core.NewHTTPTestServer(core.HandlerFunc(func(w core.ResponseWriter, r *core.Request) {
		if r.URL.Path != "/api/models/missing" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
		w.WriteHeader(404)
		core.WriteString(w, "not found")
	}))
	defer server.Close()

	source := NewRemoteSource(RemoteConfig{BaseURL: server.URL})
	if _, err := source.ModelMetadata(context.Background(), "missing"); err == nil || !core.Contains(err.Error(), "404") {
		t.Fatalf("ModelMetadata(404) error = %v, want an HTTP status error mentioning 404", err)
	}
}

// TestHf_RemoteSource_ModelMetadata_Ugly covers ModelMetadata's two awkward
// edges: when the Hub returns a body carrying neither `id` nor `modelId` the
// requested id is filled in, and pointing the source at a closed loopback
// server surfaces a transport error.
func TestHf_RemoteSource_ModelMetadata_Ugly(t *testing.T) {
	idless := core.NewHTTPTestServer(core.HandlerFunc(func(w core.ResponseWriter, _ *core.Request) {
		w.Header().Set("Content-Type", "application/json")
		core.WriteString(w, `{"config": {"model_type": "qwen3"}}`)
	}))
	defer idless.Close()

	source := NewRemoteSource(RemoteConfig{BaseURL: idless.URL})
	meta, err := source.ModelMetadata(context.Background(), "org/no-id-model")
	if err != nil {
		t.Fatalf("ModelMetadata() error = %v", err)
	}
	if meta.ID != "org/no-id-model" {
		t.Fatalf("ModelMetadata().ID = %q, want the requested id filled in", meta.ID)
	}

	closed := core.NewHTTPTestServer(core.HandlerFunc(func(w core.ResponseWriter, _ *core.Request) {
		core.WriteString(w, "{}")
	}))
	closedURL := closed.URL
	closed.Close() // nothing listens at closedURL now -> dial fails
	dead := NewRemoteSource(RemoteConfig{BaseURL: closedURL})
	if _, err := dead.ModelMetadata(context.Background(), "org/model"); err == nil {
		t.Fatal("ModelMetadata(closed server) error = nil, want a transport error")
	}
}

// TestHf_InspectLocalMetadata_Good drives InspectLocalMetadata over a
// synthetic `models--org--name/snapshots/<rev>` cache directory: it resolves
// the snapshot root, parses config.json, and lists the weight/tokenizer
// files sitting alongside it.
func TestHf_InspectLocalMetadata_Good(t *testing.T) {
	cacheRoot := core.PathJoin(t.TempDir(), "models--org--name")
	snapshot := core.PathJoin(cacheRoot, "snapshots", "b")
	if result := core.MkdirAll(snapshot, 0o755); !result.OK {
		t.Fatalf("mkdir snapshot: %v", result.Value)
	}
	writeHFFile(t, core.PathJoin(snapshot, "config.json"), `{"architectures":["Qwen3ForCausalLM"],"context_length":32768}`)
	writeHFFile(t, core.PathJoin(snapshot, "model-q4.gguf"), "gguf")
	writeHFFile(t, core.PathJoin(snapshot, "model.safetensors"), "safe")
	writeHFFile(t, core.PathJoin(snapshot, "pytorch_model.bin"), "bin")
	writeHFFile(t, core.PathJoin(snapshot, "tokenizer.json"), "{}")

	meta, root, err := InspectLocalMetadata(cacheRoot)
	if err != nil {
		t.Fatalf("InspectLocalMetadata: %v", err)
	}
	if root != snapshot {
		t.Fatalf("root = %q, want %q", root, snapshot)
	}
	if meta.ID != "org/name" {
		t.Fatalf("ID = %q, want org/name", meta.ID)
	}
	if meta.Config.ContextLength != 32768 {
		t.Fatalf("Config.ContextLength = %d, want 32768 (parsed from config.json)", meta.Config.ContextLength)
	}
	if len(meta.Files) != 4 {
		t.Fatalf("files = %+v, want 4 (gguf, safetensors, bin, tokenizer.json)", meta.Files)
	}
}

// TestHf_InspectLocalMetadata_Bad covers the read-failure path: a directory
// with no config.json anywhere under it surfaces an error rather than a
// zero-value metadata.
func TestHf_InspectLocalMetadata_Bad(t *testing.T) {
	empty := t.TempDir()
	if _, _, err := InspectLocalMetadata(empty); err == nil {
		t.Fatal("InspectLocalMetadata(no config.json) error = nil, want a read error")
	}
}

// TestHf_ResolveLocalMetadataRoot_Good covers the dominant case: a
// `models--org--name` cache root with one snapshot resolves to that
// snapshot directory.
func TestHf_ResolveLocalMetadataRoot_Good(t *testing.T) {
	cacheRoot := core.PathJoin(t.TempDir(), "models--org--name")
	snapshot := core.PathJoin(cacheRoot, "snapshots", "b")
	if result := core.MkdirAll(snapshot, 0o755); !result.OK {
		t.Fatalf("mkdir snapshot: %v", result.Value)
	}
	if got := ResolveLocalMetadataRoot(cacheRoot); got != snapshot {
		t.Fatalf("ResolveLocalMetadataRoot(cache root) = %q, want %q", got, snapshot)
	}
}

// TestHf_ResolveLocalMetadataRoot_Ugly covers the two fallback shapes: a path
// that already points straight at config.json resolves to its parent
// directory, and a plain directory with no `snapshots/` child is returned
// unchanged.
func TestHf_ResolveLocalMetadataRoot_Ugly(t *testing.T) {
	snapshot := t.TempDir()
	configPath := core.PathJoin(snapshot, "config.json")
	if got := ResolveLocalMetadataRoot(configPath); got != snapshot {
		t.Fatalf("ResolveLocalMetadataRoot(config.json path) = %q, want %q", got, snapshot)
	}

	plain := core.PathJoin(t.TempDir(), "my-local-model")
	if got := ResolveLocalMetadataRoot(plain); got != plain {
		t.Fatalf("ResolveLocalMetadataRoot(plain dir) = %q, want unchanged %q", got, plain)
	}
}

// TestHf_LocalModelID_Good covers the HuggingFace `models--org--name` cache
// directory convention decoding to `org/name`, walking up from the input
// path when the root itself is not the models-- directory.
func TestHf_LocalModelID_Good(t *testing.T) {
	base := t.TempDir()
	cacheRoot := core.PathJoin(base, "models--mlx-community--gemma-4-e2b-it-4bit")
	snapshot := core.PathJoin(cacheRoot, "snapshots", "abc123")
	if got := LocalModelID(snapshot, cacheRoot); got != "mlx-community/gemma-4-e2b-it-4bit" {
		t.Fatalf("LocalModelID = %q, want mlx-community/gemma-4-e2b-it-4bit", got)
	}
}

// TestHf_LocalModelID_Ugly covers the no-cache-prefix fallback: when no
// `models--` segment exists anywhere in either path, the root's base name is
// returned.
func TestHf_LocalModelID_Ugly(t *testing.T) {
	plain := core.PathJoin(t.TempDir(), "my-local-model")
	if got := LocalModelID(plain, plain); got != "my-local-model" {
		t.Fatalf("LocalModelID(no cache prefix) = %q, want my-local-model", got)
	}
}

// TestHf_LocalModelFiles_Good covers LocalModelFiles and
// isLocalModelFileName against a synthetic snapshot directory: it surfaces
// safetensors/gguf/bin weights and the two tokenizer files, skips
// sub-directories and unrelated files, and reads each entry's size.
func TestHf_LocalModelFiles_Good(t *testing.T) {
	root := t.TempDir()
	writeHFFile(t, core.PathJoin(root, "model.safetensors"), "weights")
	writeHFFile(t, core.PathJoin(root, "model.gguf"), "gg")
	writeHFFile(t, core.PathJoin(root, "pytorch_model.bin"), "bin")
	writeHFFile(t, core.PathJoin(root, "tokenizer.json"), "{}")
	writeHFFile(t, core.PathJoin(root, "tokenizer_config.json"), "{}")
	writeHFFile(t, core.PathJoin(root, "README.md"), "ignored")
	writeHFFile(t, core.PathJoin(root, "config.json"), "{}") // not a weight/tokenizer name
	if result := core.MkdirAll(core.PathJoin(root, "subdir"), 0o755); !result.OK {
		t.Fatalf("mkdir subdir: %v", result.Value)
	}

	files := LocalModelFiles(root)
	got := make(map[string]uint64, len(files))
	for _, f := range files {
		got[f.Name] = f.Size
	}
	for _, want := range []string{"model.safetensors", "model.gguf", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json"} {
		if _, ok := got[want]; !ok {
			t.Fatalf("LocalModelFiles missing %q; got %v", want, got)
		}
	}
	if _, ok := got["README.md"]; ok {
		t.Fatal("LocalModelFiles surfaced README.md, want it skipped")
	}
	if _, ok := got["config.json"]; ok {
		t.Fatal("LocalModelFiles surfaced config.json, want it skipped (not a weight/tokenizer name)")
	}
	if got["model.safetensors"] != uint64(len("weights")) {
		t.Fatalf("model.safetensors size = %d, want %d", got["model.safetensors"], len("weights"))
	}
}

// TestHf_LocalModelFiles_Bad covers the ReadDir-failure early return: a
// non-existent root yields an empty (non-nil) slice rather than an error.
func TestHf_LocalModelFiles_Bad(t *testing.T) {
	files := LocalModelFiles(core.PathJoin(t.TempDir(), "does-not-exist"))
	if len(files) != 0 {
		t.Fatalf("LocalModelFiles(missing) = %v, want empty", files)
	}
}

// TestHf_WeightFormatAndBytes_Good covers the single-format branches: a pure
// safetensors set (with the RFilename + SizeBytes fallbacks) and a pure GGUF
// set.
func TestHf_WeightFormatAndBytes_Good(t *testing.T) {
	safet := []ModelFile{
		{Name: "model-00001-of-00002.safetensors", Size: 100},
		{RFilename: "model-00002-of-00002.safetensors", SizeBytes: 200},
	}
	if format, total := WeightFormatAndBytes(safet); format != "safetensors" || total != 300 {
		t.Fatalf("safetensors = %q/%d, want safetensors/300 (RFilename + SizeBytes fallbacks)", format, total)
	}

	ggufFiles := []ModelFile{{Name: "model.Q4_K_M.gguf", Size: 500}}
	if format, total := WeightFormatAndBytes(ggufFiles); format != "gguf" || total != 500 {
		t.Fatalf("gguf = %q/%d, want gguf/500", format, total)
	}
}

// TestHf_WeightFormatAndBytes_Ugly covers the edge combinations: a mixed
// safetensors+gguf set collapses to "mixed", a .bin set reports "bin", and
// nil input returns the empty zero-value pair.
func TestHf_WeightFormatAndBytes_Ugly(t *testing.T) {
	mixed := []ModelFile{
		{Name: "model.safetensors", Size: 10},
		{Name: "model.gguf", Size: 20},
	}
	if format, total := WeightFormatAndBytes(mixed); format != "mixed" || total != 30 {
		t.Fatalf("mixed = %q/%d, want mixed/30", format, total)
	}

	binFiles := []ModelFile{{Name: "pytorch_model.bin", Size: 42}}
	if format, total := WeightFormatAndBytes(binFiles); format != "bin" || total != 42 {
		t.Fatalf("bin = %q/%d, want bin/42", format, total)
	}

	if format, total := WeightFormatAndBytes(nil); format != "" || total != 0 {
		t.Fatalf("empty = %q/%d, want empty/0", format, total)
	}
}

// TestHfJang_InferJANG_Good drives the public InferJANG over a pack whose id
// carries a "jang_2s" needle but no "jangtq" — the jangBasic branch that
// builds the lowercase haystack, resolves the profile name, and reads the
// group size from the QuantizationConfig. Asserts the inferred profile, the
// bits derived from jang.ProfileBits ("jang_2*" -> 2), and the overridden
// group size (96, not the 64 default).
func TestHfJang_InferJANG_Good(t *testing.T) {
	meta := ModelMetadata{
		ID:   "dealignai/Qwen3-JANG_2S",
		Tags: []string{"mlx", "jang"},
		Files: []ModelFile{
			{Name: "model.safetensors"},
			{RFilename: "tokenizer.json"},
		},
		Config: ModelConfig{
			QuantizationConfig: &QuantizationConfig{GroupSize: 96},
		},
	}
	info := InferJANG(meta)
	if info == nil {
		t.Fatal("InferJANG returned nil for a 'jang_2s' pack, want a basic JANG profile")
	}
	if info.Profile != "JANG_2S" {
		t.Fatalf("Profile = %q, want JANG_2S", info.Profile)
	}
	if info.BitsDefault != 2 {
		t.Fatalf("BitsDefault = %d, want 2 (jang_2* -> 2 bits)", info.BitsDefault)
	}
	if info.GroupSize != 96 {
		t.Fatalf("GroupSize = %d, want 96 (read from QuantizationConfig, not the 64 default)", info.GroupSize)
	}
	if info.Packed == nil {
		t.Fatal("Packed profile = nil, want BuildPackedProfile output")
	}
}

// TestHfJang_InferJANG_Bad asserts the dominant miss path: metadata with no
// "jang" token anywhere (id/tags/filenames) returns nil with no profile work.
func TestHfJang_InferJANG_Bad(t *testing.T) {
	meta := ModelMetadata{
		ID:    "Qwen/Qwen3-0.6B",
		Tags:  []string{"mlx", "text-generation"},
		Files: []ModelFile{{Name: "model.safetensors"}, {Name: "tokenizer.json"}},
	}
	if info := InferJANG(meta); info != nil {
		t.Fatalf("InferJANG = %+v, want nil for a non-JANG pack", info)
	}
}

// TestHfJang_InferJANG_Ugly drives the JANGTQ short-circuit when the
// strongest token is "jangtq" (here only in a filename) and neither quant
// block declares a group size — the helper must fall back to the 64 default
// and stamp the fixed JANGTQ profile/bits without scanning a haystack.
func TestHfJang_InferJANG_Ugly(t *testing.T) {
	meta := ModelMetadata{
		ID: "vendor/model-with-only-a-file-needle",
		Files: []ModelFile{
			{Name: "model.safetensors"},
			{RFilename: "weights.JANGTQ.safetensors"},
		},
	}
	info := InferJANG(meta)
	if info == nil {
		t.Fatal("InferJANG returned nil for a JANGTQ filename, want a JANGTQ profile")
	}
	if info.Profile != "JANGTQ" || info.WeightFormat != "mxtq" {
		t.Fatalf("profile/format = %q/%q, want JANGTQ/mxtq", info.Profile, info.WeightFormat)
	}
	if info.BitsDefault != 2 || info.RoutedExpertBits != 2 {
		t.Fatalf("bits = default:%d routed:%d, want 2/2", info.BitsDefault, info.RoutedExpertBits)
	}
	if info.GroupSize != 64 {
		t.Fatalf("GroupSize = %d, want 64 default (no quant block declared a group size)", info.GroupSize)
	}
}
