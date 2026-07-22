// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"sort"
	"time"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

const knowledgeSystemMessageMaxBytes = 65536

type knowledgeDocument struct {
	Mount       string
	Path        string
	Title       string
	Content     string
	ContentHash string
	ModifiedAt  time.Time
}

type knowledgeMount struct {
	Name   string
	Root   string
	Medium coreio.Medium
}

type knowledgeWarning struct {
	Mount  string
	Path   string
	Reason string
}

type knowledgeDiscovery struct {
	Documents []knowledgeDocument
	Warnings  []knowledgeWarning
}

type knowledgeScanner interface {
	Discover(mounts []knowledgeMount, maxBytes int64) core.Result
}

type mediumKnowledgeScanner struct{}

func newKnowledgeScanner() knowledgeScanner { return mediumKnowledgeScanner{} }

func (mediumKnowledgeScanner) Discover(mounts []knowledgeMount, maxBytes int64) core.Result {
	if maxBytes <= 0 {
		return core.Fail(core.E("tui.knowledgeScanner.Discover", "knowledge file byte limit must be positive", nil))
	}
	discovery := knowledgeDiscovery{
		Documents: []knowledgeDocument{},
		Warnings:  []knowledgeWarning{},
	}
	visitedDirectories := make(map[string]bool)
	visitedDocuments := make(map[string]bool)
	for _, mount := range mounts {
		mount.Name = core.Trim(mount.Name)
		mount.Root = trimMediumPath(mount.Root)
		if mount.Name == "" {
			discovery.Warnings = append(discovery.Warnings, knowledgeWarning{Path: mount.Root, Reason: "mount name is required"})
			continue
		}
		if mount.Medium == nil {
			discovery.Warnings = append(discovery.Warnings, knowledgeWarning{Mount: mount.Name, Path: mount.Root, Reason: "mount medium is unavailable"})
			continue
		}
		scanKnowledgeDirectory(mount, mount.Root, maxBytes, visitedDirectories, visitedDocuments, &discovery)
	}
	sort.SliceStable(discovery.Documents, func(left, right int) bool {
		if discovery.Documents[left].Mount != discovery.Documents[right].Mount {
			return discovery.Documents[left].Mount < discovery.Documents[right].Mount
		}
		leftTitle := core.Lower(discovery.Documents[left].Title)
		rightTitle := core.Lower(discovery.Documents[right].Title)
		if leftTitle != rightTitle {
			return leftTitle < rightTitle
		}
		return discovery.Documents[left].Path < discovery.Documents[right].Path
	})
	sort.SliceStable(discovery.Warnings, func(left, right int) bool {
		if discovery.Warnings[left].Mount != discovery.Warnings[right].Mount {
			return discovery.Warnings[left].Mount < discovery.Warnings[right].Mount
		}
		return discovery.Warnings[left].Path < discovery.Warnings[right].Path
	})
	return core.Ok(discovery)
}

func scanKnowledgeDirectory(
	mount knowledgeMount,
	directory string,
	maxBytes int64,
	visitedDirectories map[string]bool,
	visitedDocuments map[string]bool,
	discovery *knowledgeDiscovery,
) {
	directoryKey := core.Concat(mount.Name, "\x00", directory)
	if visitedDirectories[directoryKey] {
		return
	}
	visitedDirectories[directoryKey] = true
	entries, err := mount.Medium.List(directory)
	if err != nil {
		discovery.Warnings = append(discovery.Warnings, knowledgeWarning{
			Mount: mount.Name, Path: directory, Reason: core.Concat("list failed: ", err.Error()),
		})
		return
	}
	for _, entry := range entries {
		if entry == nil || entry.Type()&core.ModeSymlink != 0 {
			continue
		}
		path := joinMediumPath(directory, entry.Name())
		info, err := mount.Medium.Stat(path)
		if err != nil {
			discovery.Warnings = append(discovery.Warnings, knowledgeWarning{
				Mount: mount.Name, Path: path, Reason: core.Concat("stat failed: ", err.Error()),
			})
			continue
		}
		if info.Mode()&core.ModeSymlink != 0 {
			continue
		}
		if info.IsDir() {
			scanKnowledgeDirectory(mount, path, maxBytes, visitedDirectories, visitedDocuments, discovery)
			continue
		}
		extension := core.Lower(core.PathExt(path))
		if extension != ".md" && extension != ".markdown" {
			continue
		}
		documentKey := core.Concat(mount.Name, "\x00", path)
		if visitedDocuments[documentKey] {
			continue
		}
		visitedDocuments[documentKey] = true
		if info.Size() > maxBytes {
			discovery.Warnings = append(discovery.Warnings, knowledgeWarning{
				Mount:  mount.Name,
				Path:   path,
				Reason: core.Sprintf("file exceeds the %d bytes limit", maxBytes),
			})
			continue
		}
		content, err := mount.Medium.Read(path)
		if err != nil {
			discovery.Warnings = append(discovery.Warnings, knowledgeWarning{
				Mount: mount.Name, Path: path, Reason: core.Concat("read failed: ", err.Error()),
			})
			continue
		}
		if int64(len(content)) > maxBytes {
			discovery.Warnings = append(discovery.Warnings, knowledgeWarning{
				Mount:  mount.Name,
				Path:   path,
				Reason: core.Sprintf("file exceeds the %d bytes limit", maxBytes),
			})
			continue
		}
		discovery.Documents = append(discovery.Documents, knowledgeDocument{
			Mount:       mount.Name,
			Path:        path,
			Title:       knowledgeTitle(path, content),
			Content:     content,
			ContentHash: core.SHA256HexString(content),
			ModifiedAt:  info.ModTime().UTC(),
		})
	}
}

func knowledgeTitle(path, content string) string {
	for _, line := range core.Split(content, "\n") {
		line = core.Trim(line)
		if !core.HasPrefix(line, "#") {
			continue
		}
		for core.HasPrefix(line, "#") {
			line = core.TrimPrefix(line, "#")
		}
		if title := core.Trim(line); title != "" {
			return title
		}
	}
	base := core.PathBase(path)
	return core.TrimSuffix(base, core.PathExt(base))
}

func trimMediumPath(path string) string {
	path = core.Trim(path)
	for core.HasPrefix(path, "/") {
		path = core.TrimPrefix(path, "/")
	}
	for core.HasSuffix(path, "/") {
		path = core.TrimSuffix(path, "/")
	}
	if path == "." {
		return ""
	}
	return path
}

func joinMediumPath(directory, name string) string {
	directory = trimMediumPath(directory)
	name = trimMediumPath(name)
	if directory == "" {
		return name
	}
	if name == "" {
		return directory
	}
	return core.Concat(directory, "/", name)
}

type knowledgeLibrary struct {
	repository workspaceRepository
	maxBytes   int64
	ids        func() string
	now        func() time.Time
}

func newKnowledgeLibrary(
	repository workspaceRepository,
	maxBytes int64,
	ids func() string,
	now func() time.Time,
) core.Result {
	if repository == nil {
		return core.Fail(core.E("tui.newKnowledgeLibrary", "workspace repository is required", nil))
	}
	if maxBytes <= 0 {
		return core.Fail(core.E("tui.newKnowledgeLibrary", "knowledge byte limit must be positive", nil))
	}
	if ids == nil {
		ids = newRecordID
	}
	if now == nil {
		now = time.Now
	}
	return core.Ok(&knowledgeLibrary{repository: repository, maxBytes: maxBytes, ids: ids, now: now})
}

func (library *knowledgeLibrary) Attach(sessionID string, document knowledgeDocument) core.Result {
	if library == nil || library.repository == nil {
		return core.Fail(core.E("tui.knowledgeLibrary.Attach", "knowledge library is unavailable", nil))
	}
	sessionID = core.Trim(sessionID)
	if sessionID == "" {
		return core.Fail(core.E("tui.knowledgeLibrary.Attach", "session ID is required", nil))
	}
	if core.Trim(document.Path) == "" {
		return core.Fail(core.E("tui.knowledgeLibrary.Attach", "document path is required", nil))
	}
	activeResult := library.Attachments(sessionID)
	if !activeResult.OK {
		return activeResult
	}
	active := activeResult.Value.([]attachmentRecord)
	sourcePath := knowledgeSourcePath(document)
	totalBytes := int64(len(document.Content))
	for _, attachment := range active {
		if attachment.SourcePath == sourcePath {
			return core.Ok(attachment)
		}
		totalBytes += int64(len(attachment.Snapshot))
	}
	if int64(len(document.Content)) > library.maxBytes || totalBytes > library.maxBytes {
		return core.Fail(core.E(
			"tui.knowledgeLibrary.Attach",
			core.Sprintf("knowledge snapshots exceed the %d bytes session limit", library.maxBytes),
			nil,
		))
	}
	id := core.Trim(library.ids())
	if id == "" {
		return core.Fail(core.E("tui.knowledgeLibrary.Attach", "attachment ID generator returned an empty value", nil))
	}
	now := library.now().UTC()
	record := attachmentRecord{
		ID:            id,
		SessionID:     sessionID,
		SourcePath:    sourcePath,
		Title:         document.Title,
		ContentHash:   core.SHA256HexString(document.Content),
		Snapshot:      document.Content,
		AddedAt:       now,
		LastCheckedAt: now,
		ArchivedAt:    unsetRecordTime(),
	}
	if result := library.repository.SaveAttachment(record); !result.OK {
		return result
	}
	return core.Ok(record)
}

func (library *knowledgeLibrary) Detach(sessionID, attachmentID string) core.Result {
	if library == nil {
		return core.Fail(core.E("tui.knowledgeLibrary.Detach", "knowledge library is unavailable", nil))
	}
	result := library.Attachments(sessionID)
	if !result.OK {
		return result
	}
	for _, attachment := range result.Value.([]attachmentRecord) {
		if attachment.ID != attachmentID {
			continue
		}
		attachment.Archived = true
		attachment.ArchivedAt = library.now().UTC()
		if saved := library.repository.SaveAttachment(attachment); !saved.OK {
			return saved
		}
		return core.Ok(attachment)
	}
	return core.Fail(core.E("tui.knowledgeLibrary.Detach", core.Concat("unknown attachment: ", attachmentID), nil))
}

func (library *knowledgeLibrary) Attachments(sessionID string) core.Result {
	if library == nil || library.repository == nil {
		return core.Fail(core.E("tui.knowledgeLibrary.Attachments", "knowledge library is unavailable", nil))
	}
	return library.repository.Attachments(sessionID)
}

func (library *knowledgeLibrary) RefreshStaleness(sessionID string, documents []knowledgeDocument) core.Result {
	if library == nil {
		return core.Fail(core.E("tui.knowledgeLibrary.RefreshStaleness", "knowledge library is unavailable", nil))
	}
	result := library.Attachments(sessionID)
	if !result.OK {
		return result
	}
	hashes := make(map[string]string, len(documents))
	for _, document := range documents {
		hashes[knowledgeSourcePath(document)] = core.SHA256HexString(document.Content)
	}
	updated := append([]attachmentRecord(nil), result.Value.([]attachmentRecord)...)
	for index := range updated {
		currentHash, exists := hashes[updated[index].SourcePath]
		updated[index].Stale = !exists || currentHash != updated[index].ContentHash
		updated[index].LastCheckedAt = library.now().UTC()
		if saved := library.repository.SaveAttachment(updated[index]); !saved.OK {
			return saved
		}
	}
	return core.Ok(updated)
}

func knowledgeSystemMessage(attachments []attachmentRecord) string {
	return knowledgeSystemMessageBounded(attachments, knowledgeSystemMessageMaxBytes)
}

func knowledgeSystemMessageBounded(attachments []attachmentRecord, maxBytes int) string {
	if len(attachments) == 0 || maxBytes <= 0 {
		return ""
	}
	ordered := append([]attachmentRecord(nil), attachments...)
	sort.SliceStable(ordered, func(left, right int) bool {
		if ordered[left].AddedAt.Equal(ordered[right].AddedAt) {
			return ordered[left].SourcePath < ordered[right].SourcePath
		}
		return ordered[left].AddedAt.Before(ordered[right].AddedAt)
	})
	builder := core.NewBuilder()
	header := "Local knowledge snapshots (read-only context):"
	if len(header) > maxBytes {
		return ""
	}
	builder.WriteString(header)
	for _, attachment := range ordered {
		if attachment.Archived {
			continue
		}
		stale := ""
		if attachment.Stale {
			stale = " [source changed; preserved snapshot]"
		}
		section := core.Concat(
			"\n\n## ", attachment.Title, stale,
			"\nSource: ", attachment.SourcePath,
			"\n\n", attachment.Snapshot,
		)
		if builder.Len()+len(section) > maxBytes {
			continue
		}
		builder.WriteString(section)
	}
	return builder.String()
}

func knowledgeSourcePath(document knowledgeDocument) string {
	if document.Mount == "" {
		return document.Path
	}
	return core.Concat(document.Mount, ":", document.Path)
}

type knowledgeInspection struct {
	ready     bool
	documents []knowledgeDocument
	warnings  []knowledgeWarning
}

func knowledgeInspectionFrom(result core.Result) knowledgeInspection {
	inspection := knowledgeInspection{ready: true}
	if !result.OK {
		inspection.warnings = []knowledgeWarning{{Reason: result.Error()}}
		return inspection
	}
	discovery, ok := result.Value.(knowledgeDiscovery)
	if !ok {
		inspection.warnings = []knowledgeWarning{{Reason: "invalid knowledge discovery result"}}
		return inspection
	}
	inspection.documents = append([]knowledgeDocument(nil), discovery.Documents...)
	inspection.warnings = append([]knowledgeWarning(nil), discovery.Warnings...)
	return inspection
}
