// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"time"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

type exportFormat string

const (
	exportMarkdown exportFormat = "markdown"
	exportJSON     exportFormat = "json"
)

type sessionExporter interface {
	Export(medium coreio.Medium, directory, sessionID string, format exportFormat) core.Result
}

type exportReceipt struct {
	SessionID  string       `json:"session_id"`
	Title      string       `json:"title"`
	Path       string       `json:"path"`
	Format     exportFormat `json:"format"`
	Bytes      int          `json:"bytes"`
	ExportedAt time.Time    `json:"exported_at"`
}

type sessionExport struct {
	Session     sessionRecord      `json:"session"`
	Turns       []turnRecord       `json:"turns"`
	Events      []eventRecord      `json:"events"`
	Artifacts   []artifactRecord   `json:"artifacts"`
	Attachments []attachmentRecord `json:"attachments"`
	ExportedAt  time.Time          `json:"exported_at"`
}

type workspaceSessionExporter struct {
	repository   workspaceRepository
	showThinking bool
	now          func() time.Time
	ids          func() string
}

func newWorkspaceSessionExporter(
	repository workspaceRepository,
	showThinking bool,
	now func() time.Time,
	ids func() string,
) sessionExporter {
	if now == nil {
		now = time.Now
	}
	if ids == nil {
		ids = newRecordID
	}
	return &workspaceSessionExporter{
		repository: repository, showThinking: showThinking, now: now, ids: ids,
	}
}

func (exporter *workspaceSessionExporter) Export(
	medium coreio.Medium,
	directory string,
	sessionID string,
	format exportFormat,
) core.Result {
	if exporter == nil || exporter.repository == nil {
		return core.Fail(core.E("tui.sessionExporter.Export", "workspace repository is unavailable", nil))
	}
	if medium == nil {
		return core.Fail(core.E("tui.sessionExporter.Export", "export medium is required", nil))
	}
	sessionID = core.Trim(sessionID)
	if sessionID == "" {
		return core.Fail(core.E("tui.sessionExporter.Export", "session ID is required", nil))
	}
	directory = trimMediumPath(directory)
	if directory == ".." || core.HasPrefix(directory, "../") || core.Contains(directory, "/../") {
		return core.Fail(core.E("tui.sessionExporter.Export", core.Concat("unsafe export directory: ", directory), nil))
	}
	if format != exportMarkdown && format != exportJSON {
		return core.Fail(core.E("tui.sessionExporter.Export", core.Concat("unsupported export format: ", string(format)), nil))
	}
	loaded := exporter.load(sessionID)
	if !loaded.OK {
		return loaded
	}
	payload := loaded.Value.(sessionExport)
	payload.ExportedAt = exporter.now().UTC()
	contentResult := exporter.render(payload, format)
	if !contentResult.OK {
		return contentResult
	}
	content := contentResult.String()
	if directory != "" {
		if err := medium.EnsureDir(directory); err != nil {
			return core.Fail(core.E(
				"tui.sessionExporter.Export",
				core.Concat("create export directory ", directory),
				err,
			))
		}
	}
	path := collisionSafeExportPath(medium, directory, payload.Session, payload.ExportedAt, format)
	temporaryID := exportPathToken(exporter.ids())
	if temporaryID == "" {
		temporaryID = "pending"
	}
	temporaryPath := joinMediumPath(directory, core.Concat(".", core.PathBase(path), ".tmp-", temporaryID))
	if err := medium.WriteMode(temporaryPath, content, 0600); err != nil {
		if medium.Exists(temporaryPath) {
			if cleanupErr := medium.Delete(temporaryPath); cleanupErr != nil {
				core.Warn("tui.export.temporary_cleanup", "path", temporaryPath, "error", cleanupErr)
			}
		}
		return core.Fail(core.E(
			"tui.sessionExporter.Export",
			core.Concat("write export ", temporaryPath),
			err,
		))
	}
	if err := medium.Rename(temporaryPath, path); err != nil {
		if cleanupErr := medium.Delete(temporaryPath); cleanupErr != nil {
			core.Warn("tui.export.temporary_cleanup", "path", temporaryPath, "error", cleanupErr)
		}
		return core.Fail(core.E(
			"tui.sessionExporter.Export",
			core.Concat("commit export ", path),
			err,
		))
	}
	return core.Ok(exportReceipt{
		SessionID:  payload.Session.ID,
		Title:      payload.Session.Title,
		Path:       path,
		Format:     format,
		Bytes:      len(content),
		ExportedAt: payload.ExportedAt,
	})
}

func (exporter *workspaceSessionExporter) load(sessionID string) core.Result {
	sessionResult := exporter.repository.Session(sessionID)
	if !sessionResult.OK {
		return sessionResult
	}
	session, ok := sessionResult.Value.(sessionRecord)
	if !ok {
		return core.Fail(core.E("tui.sessionExporter.load", "invalid session result", nil))
	}
	turnResult := exporter.repository.Turns(sessionID)
	if !turnResult.OK {
		return turnResult
	}
	turns, ok := turnResult.Value.([]turnRecord)
	if !ok {
		return core.Fail(core.E("tui.sessionExporter.load", "invalid turn result", nil))
	}
	turns = append([]turnRecord(nil), turns...)
	if !exporter.showThinking {
		for index := range turns {
			turns[index].Thought = ""
		}
	}
	eventResult := exporter.repository.Events(sessionID)
	if !eventResult.OK {
		return eventResult
	}
	events, ok := eventResult.Value.([]eventRecord)
	if !ok {
		return core.Fail(core.E("tui.sessionExporter.load", "invalid event result", nil))
	}
	artifactResult := exporter.repository.Artifacts(sessionID)
	if !artifactResult.OK {
		return artifactResult
	}
	artifacts, ok := artifactResult.Value.([]artifactRecord)
	if !ok {
		return core.Fail(core.E("tui.sessionExporter.load", "invalid artifact result", nil))
	}
	attachmentResult := exporter.repository.Attachments(sessionID)
	if !attachmentResult.OK {
		return attachmentResult
	}
	attachments, ok := attachmentResult.Value.([]attachmentRecord)
	if !ok {
		return core.Fail(core.E("tui.sessionExporter.load", "invalid attachment result", nil))
	}
	return core.Ok(sessionExport{
		Session:     session,
		Turns:       turns,
		Events:      append([]eventRecord(nil), events...),
		Artifacts:   append([]artifactRecord(nil), artifacts...),
		Attachments: append([]attachmentRecord(nil), attachments...),
	})
}

func (exporter *workspaceSessionExporter) render(payload sessionExport, format exportFormat) core.Result {
	if format == exportJSON {
		marshaled := core.JSONMarshalIndent(payload, "", "  ")
		if !marshaled.OK {
			return core.Fail(core.E("tui.sessionExporter.render", "marshal JSON export", resultError(marshaled)))
		}
		return core.Ok(core.AsString(marshaled.Bytes()) + "\n")
	}
	return core.Ok(renderSessionMarkdown(payload))
}

func renderSessionMarkdown(payload sessionExport) string {
	builder := core.NewBuilder()
	builder.WriteString("# ")
	builder.WriteString(payload.Session.Title)
	builder.WriteString("\n\n")
	builder.WriteString("- Session: `")
	builder.WriteString(payload.Session.ID)
	builder.WriteString("`\n- Status: ")
	builder.WriteString(payload.Session.Status)
	builder.WriteString("\n- Model: ")
	if payload.Session.PreferredModel == "" {
		builder.WriteString("not selected")
	} else {
		builder.WriteString(payload.Session.PreferredModel)
	}
	builder.WriteString("\n- Exported: ")
	builder.WriteString(payload.ExportedAt.Format(time.RFC3339))
	builder.WriteString("\n\n## Conversation\n")
	for _, turn := range payload.Turns {
		builder.WriteString("\n### ")
		builder.WriteString(exportRoleTitle(turn.Role))
		builder.WriteString("\n\n")
		if turn.Thought != "" {
			builder.WriteString("_Thought:_ ")
			builder.WriteString(turn.Thought)
			builder.WriteString("\n\n")
		}
		if turn.Visible != "" {
			builder.WriteString(turn.Visible)
			builder.WriteString("\n")
		}
		hasCall := turn.ToolCallJSON != "" && turn.ToolCallJSON != "{}"
		hasResult := turn.ToolResultJSON != "" && turn.ToolResultJSON != "{}"
		if turn.ToolName != "" || hasCall || hasResult {
			builder.WriteString("\n**Tool receipt**\n\n")
			if turn.ToolName != "" {
				builder.WriteString("- Tool: `")
				builder.WriteString(turn.ToolName)
				builder.WriteString("`\n")
			}
			if hasCall {
				builder.WriteString("- Call: `")
				builder.WriteString(turn.ToolCallJSON)
				builder.WriteString("`\n")
			}
			if hasResult {
				builder.WriteString("- Result: `")
				builder.WriteString(turn.ToolResultJSON)
				builder.WriteString("`\n")
			}
		}
	}
	if len(payload.Attachments) > 0 {
		builder.WriteString("\n## Knowledge snapshots\n")
		for _, attachment := range payload.Attachments {
			builder.WriteString("\n### ")
			builder.WriteString(attachment.Title)
			builder.WriteString("\n\nSource: `")
			builder.WriteString(attachment.SourcePath)
			builder.WriteString("`\n\n")
			builder.WriteString(attachment.Snapshot)
			builder.WriteString("\n")
		}
	}
	if len(payload.Events) > 0 {
		builder.WriteString("\n## Events\n")
		for _, event := range payload.Events {
			builder.WriteString("\n- **")
			builder.WriteString(event.Title)
			builder.WriteString("** (`")
			builder.WriteString(event.Kind)
			builder.WriteString("`) — ")
			builder.WriteString(event.Detail)
		}
		builder.WriteString("\n")
	}
	if len(payload.Artifacts) > 0 {
		builder.WriteString("\n## Artifacts\n")
		for _, artifact := range payload.Artifacts {
			builder.WriteString("\n- ")
			builder.WriteString(artifact.Title)
			builder.WriteString(": ")
			builder.WriteString(artifact.Path)
		}
		builder.WriteString("\n")
	}
	return builder.String()
}

func collisionSafeExportPath(
	medium coreio.Medium,
	directory string,
	session sessionRecord,
	exportedAt time.Time,
	format exportFormat,
) string {
	extension := ".md"
	if format == exportJSON {
		extension = ".json"
	}
	base := core.Concat(
		exportedAt.UTC().Format("20060102T150405Z"), "-",
		exportPathToken(session.Title), "-",
		shortExportID(session.ID),
	)
	if base == "" {
		base = "session"
	}
	path := joinMediumPath(directory, base+extension)
	for suffix := 2; medium.Exists(path); suffix++ {
		path = joinMediumPath(directory, core.Sprintf("%s-%d%s", base, suffix, extension))
	}
	return path
}

func exportPathToken(value string) string {
	value = core.Lower(core.Trim(value))
	builder := core.NewBuilder()
	dash := false
	for _, character := range value {
		if character >= 'a' && character <= 'z' || character >= '0' && character <= '9' {
			builder.WriteRune(character)
			dash = false
			continue
		}
		if builder.Len() > 0 && !dash {
			builder.WriteByte('-')
			dash = true
		}
	}
	return core.TrimSuffix(builder.String(), "-")
}

func shortExportID(sessionID string) string {
	token := exportPathToken(sessionID)
	if len(token) > 8 {
		return token[:8]
	}
	return token
}

func exportRoleTitle(role string) string {
	switch core.Lower(role) {
	case "user":
		return "You"
	case "tool":
		return "Tool result"
	default:
		return "Assistant"
	}
}
