package mcp

import (
	"context"
	"slices"

	core "dappco.re/go"
)

func (s *Service) registerBuiltInTools() core.Result {
	registrations := []Tool{
		tool("file", "file_read", "Read the contents of a file", typedHandler(s.readFile)),
		tool("file", "file_write", "Write content to a file", typedHandler(s.writeFile)),
		tool("file", "file_delete", "Delete a file or empty directory", typedHandler(s.deleteFile)),
		tool("file", "file_rename", "Rename or move a file", typedHandler(s.renameFile)),
		tool("file", "file_exists", "Check whether a file or directory exists", typedHandler(s.fileExists)),
		tool("file", "file_edit", "Edit a file by replacing text", typedHandler(s.editFile)),
		tool("dir", "dir_list", "List the contents of a directory", typedHandler(s.listDirectory)),
		tool("dir", "dir_create", "Create a directory", typedHandler(s.createDirectory)),
		tool("language", "lang_detect", "Detect the programming language of a file path", typedHandler(s.detectLanguage)),
		tool("language", "lang_list", "List supported programming languages", typedHandler(s.listLanguages)),
		tool("rag", "rag_query", "Query the RAG vector database", typedHandler(s.ragQuery)),
		tool("rag", "rag_ingest", "Ingest files into the RAG vector database", typedHandler(s.ragIngest)),
		tool("rag", "rag_collections", "List RAG collections", typedHandler(s.ragCollections)),
		tool("ml", "ml_generate", "Generate text with an ML backend", typedHandler(s.mlGenerate)),
		tool("ml", "ml_score", "Score a prompt and response", typedHandler(s.mlScore)),
		tool("ml", "ml_probe", "Run inference capability probes", typedHandler(s.mlProbe)),
		tool("ml", "ml_status", "Show ML pipeline status", typedHandler(s.mlStatus)),
		tool("ml", "ml_backends", "List available ML backends", typedHandler(s.mlBackends)),
		tool("metrics", "metrics_record", "Record a metrics event", typedHandler(s.metricsRecord)),
		tool("metrics", "metrics_query", "Query metrics events", typedHandler(s.metricsQuery)),
		tool("process", "process_start", "Start a managed process", typedHandler(s.processStart)),
		tool("process", "process_stop", "Stop a managed process", typedHandler(s.processStop)),
		tool("process", "process_kill", "Kill a managed process", typedHandler(s.processKill)),
		tool("process", "process_list", "List managed processes", typedHandler(s.processList)),
		tool("process", "process_output", "Read managed process output", typedHandler(s.processOutput)),
		tool("process", "process_input", "Write to managed process stdin", typedHandler(s.processInput)),
		tool("websocket", "ws_start", "Start the WebSocket endpoint", typedHandler(s.wsStart)),
		tool("websocket", "ws_info", "Inspect WebSocket endpoint state", typedHandler(s.wsInfo)),
		tool("browser", "webview_connect", "Connect to a browser debug endpoint", typedHandler(s.webviewConnect)),
		tool("browser", "webview_disconnect", "Disconnect from the browser debug endpoint", typedHandler(s.webviewDisconnect)),
		tool("browser", "webview_navigate", "Navigate the browser to a URL", typedHandler(s.webviewNavigate)),
		tool("browser", "webview_click", "Click an element by selector", typedHandler(s.webviewClick)),
		tool("browser", "webview_type", "Type text into an element", typedHandler(s.webviewType)),
		tool("browser", "webview_query", "Query DOM elements by selector", typedHandler(s.webviewQuery)),
		tool("browser", "webview_console", "Read browser console messages", typedHandler(s.webviewConsole)),
		tool("browser", "webview_eval", "Evaluate JavaScript in the browser", typedHandler(s.webviewEval)),
		tool("browser", "webview_screenshot", "Capture a browser screenshot", typedHandler(s.webviewScreenshot)),
		tool("browser", "webview_wait", "Wait for an element by selector", typedHandler(s.webviewWait)),
		tool("ide_chat", "ide_chat_send", "Send a chat message to an IDE session", typedHandler(s.ideChatSend)),
		tool("ide_chat", "ide_chat_history", "Retrieve IDE chat history", typedHandler(s.ideChatHistory)),
		tool("ide_chat", "ide_session_list", "List IDE agent sessions", typedHandler(s.ideSessionList)),
		tool("ide_chat", "ide_session_create", "Create an IDE agent session", typedHandler(s.ideSessionCreate)),
		tool("ide_chat", "ide_plan_status", "Get IDE plan status", typedHandler(s.idePlanStatus)),
		tool("ide_build", "ide_build_status", "Get IDE build status", typedHandler(s.ideBuildStatus)),
		tool("ide_build", "ide_build_list", "List IDE builds", typedHandler(s.ideBuildList)),
		tool("ide_build", "ide_build_logs", "Get IDE build logs", typedHandler(s.ideBuildLogs)),
		tool("ide_dashboard", "ide_dashboard_overview", "Get IDE dashboard overview", typedHandler(s.ideDashboardOverview)),
		tool("ide_dashboard", "ide_dashboard_activity", "Get IDE dashboard activity", typedHandler(s.ideDashboardActivity)),
		tool("ide_dashboard", "ide_dashboard_metrics", "Get IDE dashboard metrics", typedHandler(s.ideDashboardMetrics)),
	}

	for _, registration := range registrations {
		if r := s.RegisterTool(registration); !r.OK {
			return r
		}
	}
	return core.Ok(nil)
}

func tool(group, name, description string, handler ToolHandler) Tool {
	return Tool{
		Name:        name,
		Description: description,
		Group:       group,
		InputSchema: objectSchema(),
		Handler:     handler,
	}
}

type ReadFileInput struct {
	Path string `json:"\x70ath"`
}

type ReadFileOutput struct {
	Content  string `json:"content"`
	Language string `json:"language"`
	Path     string `json:"\x70ath"`
}

type WriteFileInput struct {
	Path    string `json:"\x70ath"`
	Content string `json:"content"`
}

type WriteFileOutput struct {
	Success bool   `json:"success"`
	Path    string `json:"\x70ath"`
}

type DeleteFileInput struct {
	Path string `json:"\x70ath"`
}

type DeleteFileOutput struct {
	Success bool   `json:"success"`
	Path    string `json:"\x70ath"`
}

type RenameFileInput struct {
	OldPath string `json:"oldPath"`
	NewPath string `json:"newPath"`
}

type RenameFileOutput struct {
	Success bool   `json:"success"`
	OldPath string `json:"oldPath"`
	NewPath string `json:"newPath"`
}

type FileExistsInput struct {
	Path string `json:"\x70ath"`
}

type FileExistsOutput struct {
	Exists bool   `json:"exists"`
	IsDir  bool   `json:"isDir"`
	Path   string `json:"\x70ath"`
}

type EditFileInput struct {
	Path       string `json:"\x70ath"`
	OldString  string `json:"old_string"`
	NewString  string `json:"new_string"`
	ReplaceAll bool   `json:"replace_all,omitempty"`
}

type EditFileOutput struct {
	Path         string `json:"\x70ath"`
	Success      bool   `json:"success"`
	Replacements int    `json:"replacements"`
}

type ListDirectoryInput struct {
	Path string `json:"\x70ath"`
}

type ListDirectoryOutput struct {
	Entries []DirectoryEntry `json:"entries"`
	Path    string           `json:"\x70ath"`
}

type DirectoryEntry struct {
	Name  string `json:"name"`
	Path  string `json:"\x70ath"`
	IsDir bool   `json:"isDir"`
	Size  int64  `json:"size"`
}

type CreateDirectoryInput struct {
	Path string `json:"\x70ath"`
}

type CreateDirectoryOutput struct {
	Success bool   `json:"success"`
	Path    string `json:"\x70ath"`
}

type DetectLanguageInput struct {
	Path string `json:"\x70ath"`
}

type DetectLanguageOutput struct {
	Language string `json:"language"`
	Path     string `json:"\x70ath"`
}

type ListLanguagesInput struct{}

type ListLanguagesOutput struct {
	Languages []LanguageInfo `json:"languages"`
}

type LanguageInfo struct {
	ID         string   `json:"id"`
	Name       string   `json:"name"`
	Extensions []string `json:"extensions"`
}

func (s *Service) readFile(ctx context.Context, input ReadFileInput) core.Result {
	pathResult := s.resolvePath(input.Path)
	if !pathResult.OK {
		return pathResult
	}
	path := pathResult.Value.(string)
	content := core.ReadFile(path)
	if !content.OK {
		return content
	}
	return core.Ok(ReadFileOutput{Content: string(content.Value.([]byte)), Language: detectLanguageFromPath(input.Path), Path: input.Path})
}

func (s *Service) writeFile(ctx context.Context, input WriteFileInput) core.Result {
	pathResult := s.resolvePath(input.Path)
	if !pathResult.OK {
		return pathResult
	}
	path := pathResult.Value.(string)
	if r := core.MkdirAll(osPathDir(path), 0o755); !r.OK {
		return r
	}
	if r := core.WriteFile(path, []byte(input.Content), 0o644); !r.OK {
		return r
	}
	return core.Ok(WriteFileOutput{Success: true, Path: input.Path})
}

func (s *Service) deleteFile(ctx context.Context, input DeleteFileInput) core.Result {
	pathResult := s.resolvePath(input.Path)
	if !pathResult.OK {
		return pathResult
	}
	path := pathResult.Value.(string)
	if r := core.Remove(path); !r.OK {
		return r
	}
	return core.Ok(DeleteFileOutput{Success: true, Path: input.Path})
}

func (s *Service) renameFile(ctx context.Context, input RenameFileInput) core.Result {
	oldPathResult := s.resolvePath(input.OldPath)
	if !oldPathResult.OK {
		return oldPathResult
	}
	oldPath := oldPathResult.Value.(string)
	newPathResult := s.resolvePath(input.NewPath)
	if !newPathResult.OK {
		return newPathResult
	}
	newPath := newPathResult.Value.(string)
	if r := core.MkdirAll(osPathDir(newPath), 0o755); !r.OK {
		return r
	}
	if r := core.Rename(oldPath, newPath); !r.OK {
		return r
	}
	return core.Ok(RenameFileOutput{Success: true, OldPath: input.OldPath, NewPath: input.NewPath})
}

func (s *Service) fileExists(ctx context.Context, input FileExistsInput) core.Result {
	pathResult := s.resolvePath(input.Path)
	if !pathResult.OK {
		return core.Ok(FileExistsOutput{Exists: false, Path: input.Path})
	}
	path := pathResult.Value.(string)
	info := core.Stat(path)
	if !info.OK {
		return core.Ok(FileExistsOutput{Exists: false, Path: input.Path})
	}
	fileInfo := info.Value.(core.FsFileInfo)
	return core.Ok(FileExistsOutput{Exists: true, IsDir: fileInfo.IsDir(), Path: input.Path})
}

func (s *Service) editFile(ctx context.Context, input EditFileInput) core.Result {
	if input.OldString == "" {
		return core.Fail(core.Errorf("%w: old_string is required", errInvalidParams))
	}
	pathResult := s.resolvePath(input.Path)
	if !pathResult.OK {
		return pathResult
	}
	path := pathResult.Value.(string)
	contentBytes := core.ReadFile(path)
	if !contentBytes.OK {
		return contentBytes
	}
	content := string(contentBytes.Value.([]byte))
	replacements := countStringOccurrences(content, input.OldString)
	if replacements == 0 {
		return core.Fail(core.Errorf("old_string not found"))
	}
	if input.ReplaceAll {
		content = core.Replace(content, input.OldString, input.NewString)
	} else {
		content = replaceFirstString(content, input.OldString, input.NewString)
		replacements = 1
	}
	if r := core.WriteFile(path, []byte(content), 0o644); !r.OK {
		return r
	}
	return core.Ok(EditFileOutput{Path: input.Path, Success: true, Replacements: replacements})
}

func (s *Service) listDirectory(ctx context.Context, input ListDirectoryInput) core.Result {
	pathResult := s.resolvePath(input.Path)
	if !pathResult.OK {
		return pathResult
	}
	path := pathResult.Value.(string)
	entriesResult := core.ReadDir(core.DirFS(path), ".")
	if !entriesResult.OK {
		return entriesResult
	}
	entries := entriesResult.Value.([]core.FsDirEntry)
	slices.SortFunc(entries, func(a, b core.FsDirEntry) int {
		return core.Compare(a.Name(), b.Name())
	})
	out := make([]DirectoryEntry, 0, len(entries))
	for _, entry := range entries {
		info, _ := entry.Info()
		var size int64
		if info != nil && !info.IsDir() {
			size = info.Size()
		}
		out = append(out, DirectoryEntry{
			Name:  entry.Name(),
			Path:  directoryEntryPath(input.Path, entry.Name()),
			IsDir: entry.IsDir(),
			Size:  size,
		})
	}
	return core.Ok(ListDirectoryOutput{Entries: out, Path: input.Path})
}

func (s *Service) createDirectory(ctx context.Context, input CreateDirectoryInput) core.Result {
	pathResult := s.resolvePath(input.Path)
	if !pathResult.OK {
		return pathResult
	}
	path := pathResult.Value.(string)
	if r := core.MkdirAll(path, 0o755); !r.OK {
		return r
	}
	return core.Ok(CreateDirectoryOutput{Success: true, Path: input.Path})
}

func (s *Service) detectLanguage(ctx context.Context, input DetectLanguageInput) core.Result {
	return core.Ok(DetectLanguageOutput{Language: detectLanguageFromPath(input.Path), Path: input.Path})
}

func (s *Service) listLanguages(ctx context.Context, input ListLanguagesInput) core.Result {
	return core.Ok(ListLanguagesOutput{Languages: supportedLanguages()})
}

func (s *Service) resolvePath(path string) core.Result {
	if core.Trim(path) == "" {
		return core.Fail(core.Errorf("%w: path is required", errInvalidParams))
	}

	if s.workspaceRoot == "" {
		if core.PathIsAbs(path) {
			return core.Ok(cleanOSPath(path))
		}
		abs := core.PathAbs(path)
		if !abs.OK {
			return abs
		}
		return core.Ok(abs.Value.(string))
	}

	var candidate string
	if core.PathIsAbs(path) {
		candidate = cleanOSPath(path)
	} else {
		cleanRelative := core.TrimPrefix(cleanOSPath(string(core.PathSeparator)+path), string(core.PathSeparator))
		candidate = core.PathJoin(s.workspaceRoot, cleanRelative)
	}
	absCandidate := core.PathAbs(candidate)
	if !absCandidate.OK {
		return absCandidate
	}
	absPath := absCandidate.Value.(string)
	rel := core.PathRel(s.workspaceRoot, absPath)
	if !rel.OK {
		return rel
	}
	relative := rel.Value.(string)
	if relative == ".." || core.HasPrefix(relative, ".."+string(core.PathSeparator)) {
		return core.Fail(core.Errorf("path escapes workspace root: %s", path))
	}
	return core.Ok(absPath)
}

func directoryEntryPath(dir, name string) string {
	dir = trimPathSeparators(dir)
	if dir == "" || dir == "." {
		return name
	}
	return core.PathToSlash(core.PathJoin(dir, name))
}

func detectLanguageFromPath(path string) string {
	if core.PathBase(path) == "Dockerfile" {
		return "dockerfile"
	}
	if lang, ok := languageByExtension[core.PathExt(path)]; ok {
		return lang
	}
	return "plaintext"
}

func cleanOSPath(path string) string {
	return core.CleanPath(path, string(core.PathSeparator))
}

func osPathDir(path string) string {
	sep := byte(core.PathSeparator)
	trimmed := path
	for len(trimmed) > 1 && trimmed[len(trimmed)-1] == sep {
		trimmed = trimmed[:len(trimmed)-1]
	}
	for i := len(trimmed) - 1; i >= 0; i-- {
		if trimmed[i] == sep {
			if i == 0 {
				return string(sep)
			}
			return trimmed[:i]
		}
	}
	return "."
}

func trimPathSeparators(path string) string {
	sep := string(core.PathSeparator)
	for core.HasPrefix(path, sep) {
		path = core.TrimPrefix(path, sep)
	}
	for core.HasSuffix(path, sep) {
		path = core.TrimSuffix(path, sep)
	}
	return path
}

func countStringOccurrences(content, needle string) int {
	if needle == "" {
		return 0
	}
	parts := core.Split(content, needle)
	return len(parts) - 1
}

func replaceFirstString(content, oldString, newString string) string {
	parts := core.SplitN(content, oldString, 2)
	if len(parts) != 2 {
		return content
	}
	return parts[0] + newString + parts[1]
}

var languageByExtension = map[string]string{
	".ts":       "typescript",
	".tsx":      "typescript",
	".js":       "javascript",
	".jsx":      "javascript",
	".go":       "go",
	".py":       "python",
	".rs":       "rust",
	".rb":       "ruby",
	".java":     "java",
	".php":      "php",
	".c":        "c",
	".h":        "c",
	".cpp":      "cpp",
	".hpp":      "cpp",
	".cc":       "cpp",
	".cxx":      "cpp",
	".cs":       "csharp",
	".html":     "html",
	".htm":      "html",
	".css":      "css",
	".scss":     "scss",
	".json":     "json",
	".yaml":     "yaml",
	".yml":      "yaml",
	".xml":      "xml",
	".md":       "markdown",
	".markdown": "markdown",
	".sql":      "sql",
	".sh":       "shell",
	".bash":     "shell",
	".swift":    "swift",
	".kt":       "kotlin",
	".kts":      "kotlin",
}

func supportedLanguages() []LanguageInfo {
	return []LanguageInfo{
		{ID: "typescript", Name: "TypeScript", Extensions: []string{".ts", ".tsx"}},
		{ID: "javascript", Name: "JavaScript", Extensions: []string{".js", ".jsx"}},
		{ID: "go", Name: "Go", Extensions: []string{".go"}},
		{ID: "python", Name: "Python", Extensions: []string{".py"}},
		{ID: "rust", Name: "Rust", Extensions: []string{".rs"}},
		{ID: "ruby", Name: "Ruby", Extensions: []string{".rb"}},
		{ID: "java", Name: "Java", Extensions: []string{".java"}},
		{ID: "php", Name: "PHP", Extensions: []string{".php"}},
		{ID: "c", Name: "C", Extensions: []string{".c", ".h"}},
		{ID: "cpp", Name: "C++", Extensions: []string{".cpp", ".hpp", ".cc", ".cxx"}},
		{ID: "csharp", Name: "C#", Extensions: []string{".cs"}},
		{ID: "html", Name: "HTML", Extensions: []string{".html", ".htm"}},
		{ID: "css", Name: "CSS", Extensions: []string{".css"}},
		{ID: "scss", Name: "SCSS", Extensions: []string{".scss"}},
		{ID: "json", Name: "JSON", Extensions: []string{".json"}},
		{ID: "yaml", Name: "YAML", Extensions: []string{".yaml", ".yml"}},
		{ID: "xml", Name: "XML", Extensions: []string{".xml"}},
		{ID: "markdown", Name: "Markdown", Extensions: []string{".md", ".markdown"}},
		{ID: "sql", Name: "SQL", Extensions: []string{".sql"}},
		{ID: "shell", Name: "Shell", Extensions: []string{".sh", ".bash"}},
		{ID: "swift", Name: "Swift", Extensions: []string{".swift"}},
		{ID: "kotlin", Name: "Kotlin", Extensions: []string{".kt", ".kts"}},
		{ID: "dockerfile", Name: "Dockerfile", Extensions: []string{}},
	}
}
