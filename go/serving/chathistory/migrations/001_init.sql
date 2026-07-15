-- SPDX-License-Identifier: EUPL-1.2
--
-- chathistory schema v1 — per-user portable chat archive.
--
-- One .duckdb file per user, conventionally at:
--   ~/Lethean/data/users/<user>/chats.duckdb
--
-- The file is the user's portable property — exportable, copyable,
-- usable in any DuckDB-aware tool. Future LoRA training data prep
-- pulls (user, assistant) pairs from `turns` joined to `conversations`
-- filtered by `signal` + `consent_version`. Embeddings table is
-- optional sidecar populated when an embedding model is configured.
--
-- Continuity rights: the user owns this file. The agent writes; the
-- user controls. See project_chat_continuity_rights_normal_user_pattern.

CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER PRIMARY KEY,
    applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    note       TEXT
);

CREATE TABLE IF NOT EXISTS conversations (
    id               VARCHAR(36) PRIMARY KEY,
    user_id          TEXT NOT NULL,
    title            TEXT,
    started_at       TIMESTAMP NOT NULL,
    ended_at         TIMESTAMP,
    model_id         TEXT,
    base_model       TEXT,
    adapter_id       TEXT,
    tags             VARCHAR,         -- JSON-encoded []string, e.g. ["life","vent"]
    metadata         VARCHAR,         -- JSON-encoded agent-extensible payload
    consent_version  INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS conversations_user_started
    ON conversations(user_id, started_at);

CREATE TABLE IF NOT EXISTS turns (
    id               VARCHAR(36) PRIMARY KEY,
    conversation_id  VARCHAR(36) NOT NULL,
    ordinal          INTEGER NOT NULL,
    role             TEXT NOT NULL,
    content          TEXT NOT NULL,
    tool_calls       VARCHAR,         -- JSON-encoded structured tool invocations
    tool_results     VARCHAR,         -- JSON-encoded tool response payload
    created_at       TIMESTAMP NOT NULL,
    tokens_in        INTEGER,
    tokens_out       INTEGER,
    signal           TEXT,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);

CREATE INDEX IF NOT EXISTS turns_conv_ordinal
    ON turns(conversation_id, ordinal);

CREATE INDEX IF NOT EXISTS turns_created
    ON turns(created_at);

-- Optional sidecar — populated only when an embedding model is wired.
-- Schema present so any future tooling can rely on it existing; the
-- vector array dimension is held in the column type (768 is a common
-- default; later migrations can widen / split per embedding model
-- without breaking existing rows because no rows exist yet).
CREATE TABLE IF NOT EXISTS embeddings (
    turn_id          VARCHAR(36) PRIMARY KEY,
    embedding_model  TEXT NOT NULL,
    vector           FLOAT[768],
    FOREIGN KEY (turn_id) REFERENCES turns(id)
);

INSERT INTO schema_version (version, note)
VALUES (1, 'initial schema — conversations, turns, embeddings sidecar')
ON CONFLICT (version) DO NOTHING;
