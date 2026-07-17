// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"reflect"
	"testing"
	"time"

	"dappco.re/go/orm"
	"github.com/google/uuid"
)

type schemaFieldExpectation struct {
	typ         string
	constraints []string
}

type recordSchemaExpectation struct {
	name    string
	pk      []string
	fields  map[string]schemaFieldExpectation
	indexes [][]string
}

func TestRecordSchemas_Good(t *testing.T) {
	expectations := []struct {
		schema orm.Schema
		want   recordSchemaExpectation
	}{
		{
			schema: (schemaVersionRecord{}).Schema(),
			want: recordSchemaExpectation{
				name: "lem_schema_versions",
				pk:   []string{"version"},
				fields: map[string]schemaFieldExpectation{
					"version":    {typ: "int64", constraints: []string{"pk", "notnull"}},
					"applied_at": {typ: "time", constraints: []string{"notnull"}},
				},
			},
		},
		{
			schema: (sessionRecord{}).Schema(),
			want: recordSchemaExpectation{
				name: "lem_sessions",
				pk:   []string{"id"},
				fields: requiredRecordFields(map[string]string{
					"id": "string", "title": "string", "status": "string", "preferred_model": "string",
					"mode": "string", "generation_json": "string", "tools_json": "string",
					"created_at": "time", "updated_at": "time", "last_opened_at": "time",
					"archived": "bool", "archived_at": "time",
				}, "id", "preferred_model"),
				indexes: [][]string{{"archived", "last_opened_at"}},
			},
		},
		{
			schema: (turnRecord{}).Schema(),
			want: recordSchemaExpectation{
				name: "lem_turns",
				pk:   []string{"id"},
				fields: requiredRecordFields(map[string]string{
					"id": "string", "session_id": "string", "sequence": "int64", "role": "string",
					"visible": "string", "thought": "string", "tool_name": "string",
					"tool_call_json": "string", "tool_result_json": "string", "model": "string",
					"created_at": "time", "updated_at": "time",
				}, "id", "visible", "thought", "tool_name", "tool_call_json", "tool_result_json", "model"),
				indexes: [][]string{{"session_id", "sequence"}},
			},
		},
		{
			schema: (eventRecord{}).Schema(),
			want: recordSchemaExpectation{
				name: "lem_events",
				pk:   []string{"id"},
				fields: requiredRecordFields(map[string]string{
					"id": "string", "session_id": "string", "work_item_id": "string", "job_id": "string",
					"kind": "string", "status": "string", "title": "string", "detail": "string",
					"payload_json": "string", "created_at": "time",
				}, "id", "work_item_id", "job_id", "detail", "payload_json"),
				indexes: [][]string{{"session_id", "created_at"}},
			},
		},
		{
			schema: (generationJobRecord{}).Schema(),
			want: recordSchemaExpectation{
				name: "lem_generation_jobs",
				pk:   []string{"id"},
				fields: requiredRecordFields(map[string]string{
					"id": "string", "session_id": "string", "prompt_turn_id": "string",
					"answer_turn_id": "string", "status": "string", "model": "string", "error": "string",
					"metrics_json": "string", "created_at": "time", "started_at": "time", "finished_at": "time",
				}, "id", "prompt_turn_id", "answer_turn_id", "model", "error", "metrics_json"),
				indexes: [][]string{{"session_id", "status", "created_at"}},
			},
		},
		{
			schema: (workItemRecord{}).Schema(),
			want: recordSchemaExpectation{
				name: "lem_work_items",
				pk:   []string{"id"},
				fields: requiredRecordFieldsWithUnique(map[string]string{
					"id": "string", "external_id": "string", "source": "string", "title": "string",
					"status": "string", "agent": "string", "repo": "string", "org": "string", "task": "string",
					"branch": "string", "runtime": "string", "question": "string", "pr_url": "string",
					"session_id": "string", "started_at": "time", "updated_at": "time", "archived": "bool",
					"archived_at": "time",
				}, "id", "external_id", "agent", "repo", "org", "task", "branch", "runtime", "question", "pr_url", "session_id"),
				indexes: [][]string{{"archived", "status", "updated_at"}},
			},
		},
		{
			schema: (artifactRecord{}).Schema(),
			want: recordSchemaExpectation{
				name: "lem_artifacts",
				pk:   []string{"id"},
				fields: requiredRecordFields(map[string]string{
					"id": "string", "session_id": "string", "work_item_id": "string", "kind": "string",
					"path": "string", "title": "string", "metadata_json": "string", "created_at": "time",
					"archived": "bool", "archived_at": "time",
				}, "id", "work_item_id", "metadata_json"),
				indexes: [][]string{{"session_id", "created_at"}},
			},
		},
		{
			schema: (attachmentRecord{}).Schema(),
			want: recordSchemaExpectation{
				name: "lem_attachments",
				pk:   []string{"id"},
				fields: requiredRecordFields(map[string]string{
					"id": "string", "session_id": "string", "source_path": "string", "title": "string",
					"content_hash": "string", "snapshot": "string", "added_at": "time",
					"last_checked_at": "time", "stale": "bool", "archived": "bool", "archived_at": "time",
				}, "id", "content_hash", "snapshot"),
				indexes: [][]string{{"session_id", "archived", "added_at"}},
			},
		},
	}

	for _, expectation := range expectations {
		assertRecordSchema(t, expectation.schema, expectation.want)
	}
}

func TestNewRecordID_Good(t *testing.T) {
	first := newRecordID()
	second := newRecordID()
	if _, err := uuid.Parse(first); err != nil {
		t.Fatalf("newRecordID() = %q, want UUID: %v", first, err)
	}
	if first == second {
		t.Fatalf("two newRecordID calls both returned %q", first)
	}
}

func TestUnsetRecordTime_Good(t *testing.T) {
	want := time.Unix(0, 0).UTC()
	if got := unsetRecordTime(); !got.Equal(want) || got.Location() != time.UTC {
		t.Fatalf("unsetRecordTime() = %v (%v), want %v UTC", got, got.Location(), want)
	}
}

func requiredRecordFields(fields map[string]string, primaryKey string, optional ...string) map[string]schemaFieldExpectation {
	return requiredRecordFieldsWithUnique(fields, primaryKey, "", optional...)
}

func requiredRecordFieldsWithUnique(fields map[string]string, primaryKey, unique string, optional ...string) map[string]schemaFieldExpectation {
	want := make(map[string]schemaFieldExpectation, len(fields))
	for name, typ := range fields {
		constraints := []string{"notnull"}
		for _, optionalName := range optional {
			if name == optionalName {
				constraints = nil
				break
			}
		}
		if name == primaryKey {
			constraints = []string{"pk", "notnull"}
		}
		if name == unique {
			constraints = append(constraints, "unique")
		}
		want[name] = schemaFieldExpectation{typ: typ, constraints: constraints}
	}
	return want
}

func assertRecordSchema(t *testing.T, got orm.Schema, want recordSchemaExpectation) {
	t.Helper()
	if got.Name != want.name {
		t.Errorf("schema name = %q, want %q", got.Name, want.name)
	}
	if !reflect.DeepEqual(got.PK, want.pk) {
		t.Errorf("%s primary key = %#v, want %#v", want.name, got.PK, want.pk)
	}
	if len(got.Fields) != len(want.fields) {
		t.Errorf("%s fields = %d, want %d", want.name, len(got.Fields), len(want.fields))
	}
	for name, fieldWant := range want.fields {
		field, ok := got.FieldByName(name)
		if !ok {
			t.Errorf("%s missing field %q", want.name, name)
			continue
		}
		if field.Type != fieldWant.typ {
			t.Errorf("%s.%s type = %q, want %q", want.name, name, field.Type, fieldWant.typ)
		}
		if !reflect.DeepEqual(field.Constraints, fieldWant.constraints) {
			t.Errorf("%s.%s constraints = %#v, want %#v", want.name, name, field.Constraints, fieldWant.constraints)
		}
	}
	if len(got.Indexes) != len(want.indexes) {
		t.Errorf("%s indexes = %d, want %d", want.name, len(got.Indexes), len(want.indexes))
	}
	for index, fields := range want.indexes {
		if index >= len(got.Indexes) {
			break
		}
		if !reflect.DeepEqual(got.Indexes[index].Fields, fields) {
			t.Errorf("%s index %d = %#v, want %#v", want.name, index, got.Indexes[index].Fields, fields)
		}
	}
}
