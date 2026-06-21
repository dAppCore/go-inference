// Package ai provides the canonical AI facade for the core CLI.
//
//	contextText, err := ai.QueryRAGForTask(ai.TaskInfo{
//		Title:       "Investigate build failure",
//		Description: "CI compile step fails",
//	})
//	if err != nil {
//		return err
//	}
//
//	if err := ai.Record(ai.Event{Type: "security.scan", Repo: "wailsapp/wails"}); err != nil {
//		return err
//	}
package ai
