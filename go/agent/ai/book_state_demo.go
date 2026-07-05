// SPDX-License-Identifier: EUPL-1.2

package ai

import (
	"context"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	inferstate "dappco.re/go/inference/model/state"
)

const (
	defaultBookStateMaxTokens        = 256
	defaultBookStateStudentMaxTokens = 128
	defaultBookStateTeacherMaxTokens = 256
)

// BookState describes a persisted model-state or knowledge-pack entry that can
// be injected into provider prompts without depending on a concrete runtime.
type BookState struct {
	Title        string            `json:"title,omitempty"`
	Excerpt      string            `json:"excerpt,omitempty"`
	URI          string            `json:"uri,omitempty"`
	EntryURI     string            `json:"entry_uri,omitempty"`
	BundleURI    string            `json:"bundle_uri,omitempty"`
	IndexURI     string            `json:"index_uri,omitempty"`
	StoreURI     string            `json:"store_uri,omitempty"`
	PrefixTokens int               `json:"prefix_tokens,omitempty"`
	BundleTokens int               `json:"bundle_tokens,omitempty"`
	BlockSize    int               `json:"block_size,omitempty"`
	BlocksRead   int               `json:"blocks_read,omitempty"`
	Labels       map[string]string `json:"labels,omitempty"`
	Metadata     map[string]string `json:"metadata,omitempty"`
}

// BookStateFromWakeResult adapts the shared go-inference state wake metadata
// into the the inference stack demo context shape.
func BookStateFromWakeResult(result inferstate.WakeResult) BookState {
	state := BookStateFromRef(result.Entry)
	state.BundleURI = core.FirstNonBlank(state.BundleURI, result.Bundle.URI)
	state.IndexURI = core.FirstNonBlank(state.IndexURI, result.Index.URI)
	state.PrefixTokens = positiveOr(state.PrefixTokens, result.PrefixTokens)
	state.BundleTokens = result.BundleTokens
	state.BlockSize = result.BlockSize
	state.BlocksRead = result.BlocksRead
	state.Labels = mergeStringMaps(state.Labels, result.Labels, result.Entry.Labels)
	return state
}

// BookStateFromRef adapts a durable go-inference state reference into a
// user-facing book-state descriptor.
func BookStateFromRef(ref inferstate.Ref) BookState {
	metadata := make(map[string]string)
	setMetadata(metadata, "kind", ref.Kind)
	setMetadata(metadata, "hash", ref.Hash)
	setMetadataInt(metadata, "token_start", ref.TokenStart)
	setMetadataInt64(metadata, "byte_start", ref.ByteStart)
	setMetadataInt64(metadata, "byte_count", ref.ByteCount)
	return BookState{
		Title:        ref.Title,
		URI:          ref.URI,
		EntryURI:     ref.URI,
		BundleURI:    ref.BundleURI,
		PrefixTokens: ref.TokenCount,
		Labels:       core.MapClone(ref.Labels),
		Metadata:     metadata,
	}
}

// BookStateContextAssembler formats a persisted state entry as provider
// context. It is deliberately text-only so the inference stack can target local drivers,
// external providers, notebooks, and MCP tools through the same path.
type BookStateContextAssembler struct {
	State BookState
}

// AssembleContext implements ProviderContextAssembler.
func (a BookStateContextAssembler) AssembleContext(ctx context.Context, _ []inference.Message) core.Result {
	if err := ctx.Err(); err != nil {
		return core.Fail(err)
	}
	return core.Ok(formatBookStateContext(a.State))
}

// BookStateDemoConfig configures a teacher/student demo over provider routes.
type BookStateDemoConfig struct {
	State BookState

	TeacherRoutes []ProviderRoute
	StudentRoutes []ProviderRoute

	StudentUsesBookState bool
	MaxTokens            int
	TeacherMaxTokens     int
	StudentMaxTokens     int
	Temperature          float32
}

// BookStateAskRequest asks the demo to answer a question with an optional
// unaided student pass followed by a book-state-backed teacher pass.
type BookStateAskRequest struct {
	Question             string  `json:"question"`
	MaxTokens            int     `json:"max_tokens,omitempty"`
	TeacherMaxTokens     int     `json:"teacher_max_tokens,omitempty"`
	StudentMaxTokens     int     `json:"student_max_tokens,omitempty"`
	Temperature          float32 `json:"temperature,omitempty"`
	StudentUsesBookState *bool   `json:"student_uses_book_state,omitempty"`
}

// BookStateAskResponse is returned by BookStateDemo.Ask.
type BookStateAskResponse struct {
	Question string    `json:"question"`
	State    BookState `json:"state"`

	StudentAnswer string               `json:"student_answer,omitempty"`
	TeacherAnswer string               `json:"teacher_answer"`
	Student       ProviderChatResponse `json:"student,omitempty"`
	Teacher       ProviderChatResponse `json:"teacher"`

	CreatedAtUnix int64 `json:"created_at_unix"`
}

// BookStateDemo orchestrates a small teacher/student question flow over a
// persisted book state.
type BookStateDemo struct {
	state BookState

	teacher *ProviderRouter
	student *ProviderRouter

	studentUsesBookState bool
	maxTokens            int
	teacherMaxTokens     int
	studentMaxTokens     int
	temperature          float32
}

// NewBookStateDemo creates a teacher/student demo over shared provider routes.
func NewBookStateDemo(cfg BookStateDemoConfig) core.Result {
	if len(cfg.TeacherRoutes) == 0 {
		return core.Fail(core.E("ai.NewBookStateDemo", "teacher route is required", nil))
	}

	teacherResult := NewProviderRouter(cfg.TeacherRoutes...)
	if !teacherResult.OK {
		if err, ok := teacherResult.Value.(error); ok {
			return core.Fail(core.E("ai.NewBookStateDemo", "teacher route invalid", err))
		}
		return core.Fail(core.E("ai.NewBookStateDemo", teacherResult.Error(), nil))
	}

	var student *ProviderRouter
	if len(cfg.StudentRoutes) > 0 {
		studentResult := NewProviderRouter(cfg.StudentRoutes...)
		if !studentResult.OK {
			if err, ok := studentResult.Value.(error); ok {
				return core.Fail(core.E("ai.NewBookStateDemo", "student route invalid", err))
			}
			return core.Fail(core.E("ai.NewBookStateDemo", studentResult.Error(), nil))
		}
		student = studentResult.Value.(*ProviderRouter)
	}

	demo := &BookStateDemo{
		state:                cloneBookState(cfg.State),
		teacher:              teacherResult.Value.(*ProviderRouter),
		student:              student,
		studentUsesBookState: cfg.StudentUsesBookState,
		maxTokens:            positiveOr(cfg.MaxTokens, defaultBookStateMaxTokens),
		teacherMaxTokens:     positiveOr(cfg.TeacherMaxTokens, defaultBookStateTeacherMaxTokens),
		studentMaxTokens:     positiveOr(cfg.StudentMaxTokens, defaultBookStateStudentMaxTokens),
		temperature:          cfg.Temperature,
	}
	return core.Ok(demo)
}

// State returns the configured persisted book state metadata.
func (d *BookStateDemo) State() BookState {
	if d == nil {
		return BookState{}
	}
	return cloneBookState(d.state)
}

// Ask runs the student, when configured, then asks the teacher to answer using
// the book state and the student's response.
func (d *BookStateDemo) Ask(ctx context.Context, req BookStateAskRequest) core.Result {
	if d == nil || d.teacher == nil {
		return core.Fail(core.E("ai.BookStateDemo.Ask", "demo is nil", nil))
	}
	question := core.Trim(req.Question)
	if question == "" {
		return core.Fail(core.E("ai.BookStateDemo.Ask", "question is required", nil))
	}

	assembler := BookStateContextAssembler{State: d.state}
	maxTokens := positiveOr(req.MaxTokens, d.maxTokens)
	temperature := req.Temperature
	if temperature == 0 {
		temperature = d.temperature
	}

	var studentResponse ProviderChatResponse
	var studentAnswer string
	if d.student != nil {
		studentUsesState := d.studentUsesBookState
		if req.StudentUsesBookState != nil {
			studentUsesState = *req.StudentUsesBookState
		}
		studentResult := d.student.Chat(ctx, ProviderChatRequest{
			Prompt:           question,
			MaxTokens:        positiveOr(req.StudentMaxTokens, positiveOr(maxTokens, d.studentMaxTokens)),
			Temperature:      temperature,
			ContextAssembler: assembler,
			ContextPrefix:    "Book state:\n",
			DisableContext:   !studentUsesState,
			Labels:           map[string]string{"role": "student"},
		})
		if !studentResult.OK {
			if err, ok := studentResult.Value.(error); ok {
				return core.Fail(core.E("ai.BookStateDemo.Ask", "student failed", err))
			}
			return core.Fail(core.E("ai.BookStateDemo.Ask", studentResult.Error(), nil))
		}
		studentResponse = studentResult.Value.(ProviderChatResponse)
		studentAnswer = core.Trim(studentResponse.Text)
	}

	teacherResult := d.teacher.Chat(ctx, ProviderChatRequest{
		Messages: []inference.Message{{Role: "user", Content: teacherPrompt(question, studentAnswer)}},
		MaxTokens: positiveOr(req.TeacherMaxTokens,
			positiveOr(maxTokens, d.teacherMaxTokens)),
		Temperature:      temperature,
		ContextAssembler: assembler,
		ContextPrefix:    "Book state:\n",
		Labels:           map[string]string{"role": "teacher"},
	})
	if !teacherResult.OK {
		if err, ok := teacherResult.Value.(error); ok {
			return core.Fail(core.E("ai.BookStateDemo.Ask", "teacher failed", err))
		}
		return core.Fail(core.E("ai.BookStateDemo.Ask", teacherResult.Error(), nil))
	}

	teacherResponse := teacherResult.Value.(ProviderChatResponse)
	return core.Ok(BookStateAskResponse{
		Question:      question,
		State:         cloneBookState(d.state),
		StudentAnswer: studentAnswer,
		TeacherAnswer: core.Trim(teacherResponse.Text),
		Student:       studentResponse,
		Teacher:       teacherResponse,
		CreatedAtUnix: time.Now().Unix(),
	})
}

func teacherPrompt(question, studentAnswer string) string {
	builder := core.NewBuilder()
	builder.WriteString("Question:\n")
	builder.WriteString(question)
	if core.Trim(studentAnswer) != "" {
		builder.WriteString("\n\nStudent answer:\n")
		builder.WriteString(studentAnswer)
	}
	builder.WriteString("\n\nTeacher task:\nAnswer from the book state. Correct the student if needed. Keep it concise and cite only what the state supports.")
	return builder.String()
}

func formatBookStateContext(state BookState) string {
	builder := core.NewBuilder()
	writeContextLine(builder, "title", state.Title)
	writeContextLine(builder, "uri", state.URI)
	writeContextLine(builder, "entry_uri", state.EntryURI)
	writeContextLine(builder, "bundle_uri", state.BundleURI)
	writeContextLine(builder, "index_uri", state.IndexURI)
	writeContextLine(builder, "store_uri", state.StoreURI)
	writeContextIntLine(builder, "prefix_tokens", state.PrefixTokens)
	writeContextIntLine(builder, "bundle_tokens", state.BundleTokens)
	writeContextIntLine(builder, "block_size", state.BlockSize)
	writeContextIntLine(builder, "blocks_read", state.BlocksRead)
	writeContextMapLine(builder, "labels", state.Labels)
	writeContextMapLine(builder, "metadata", state.Metadata)
	if core.Trim(state.Excerpt) != "" {
		builder.WriteString("excerpt:\n")
		builder.WriteString(core.Trim(state.Excerpt))
		builder.WriteString("\n")
	}
	return core.Trim(builder.String())
}

type bookStateStringWriter interface {
	WriteString(string) (int, error)
}

func writeContextLine(builder bookStateStringWriter, key, value string) {
	value = core.Trim(value)
	if value == "" {
		return
	}
	builder.WriteString(key)
	builder.WriteString(": ")
	builder.WriteString(value)
	builder.WriteString("\n")
}

func writeContextIntLine(builder bookStateStringWriter, key string, value int) {
	if value <= 0 {
		return
	}
	builder.WriteString(key)
	builder.WriteString(": ")
	builder.WriteString(core.Sprintf("%d", value))
	builder.WriteString("\n")
}

func writeContextMapLine(builder bookStateStringWriter, key string, values map[string]string) {
	if len(values) == 0 {
		return
	}
	builder.WriteString(key)
	builder.WriteString(": ")
	first := true
	for name, value := range values {
		name = core.Trim(name)
		value = core.Trim(value)
		if name == "" && value == "" {
			continue
		}
		if !first {
			builder.WriteString(", ")
		}
		first = false
		builder.WriteString(name)
		builder.WriteString("=")
		builder.WriteString(value)
	}
	builder.WriteString("\n")
}

func cloneBookState(state BookState) BookState {
	state.Labels = core.MapClone(state.Labels)
	state.Metadata = core.MapClone(state.Metadata)
	return state
}

func mergeStringMaps(values ...map[string]string) map[string]string {
	var out map[string]string
	for _, valueMap := range values {
		for key, value := range valueMap {
			if out == nil {
				out = make(map[string]string)
			}
			out[key] = value
		}
	}
	return out
}

func setMetadata(metadata map[string]string, key, value string) {
	value = core.Trim(value)
	if value == "" {
		return
	}
	metadata[key] = value
}

func setMetadataInt(metadata map[string]string, key string, value int) {
	if value == 0 {
		return
	}
	metadata[key] = core.Sprintf("%d", value)
}

func setMetadataInt64(metadata map[string]string, key string, value int64) {
	if value == 0 {
		return
	}
	metadata[key] = core.Sprintf("%d", value)
}

func positiveOr(value, fallback int) int {
	if value > 0 {
		return value
	}
	return fallback
}
