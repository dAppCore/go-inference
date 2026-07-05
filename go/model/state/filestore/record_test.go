// SPDX-Licence-Identifier: EUPL-1.2

// filestore record-codec tests: hand-rolled record-meta encoder round-trips through encoding/json and extractRecordURI.
package filestore

import (
	"testing"

	core "dappco.re/go"
)

// TestEncodeRecordMeta_RoundTrip locks the hand-rolled encoder to
// encoding/json's deserialisation contract. The encoder is the
// canonical PutBytesStream meta serialiser — every record we write
// passes through it, so its output must round-trip cleanly through
// json.Unmarshal back into recordMeta with no field loss or value
// drift. Mixed shapes (empty, single string, tag map, label slice,
// escape-sensitive characters) cover the branches the encoder
// walks.
func TestEncodeRecordMeta_RoundTrip(t *testing.T) {
	cases := []struct {
		name string
		meta recordMeta
	}{
		{"empty", recordMeta{}},
		{"uri-only", recordMeta{URI: "mlx://kv/0"}},
		{"all-strings", recordMeta{
			URI:   "mlx://kv/1",
			Title: "training-checkpoint",
			Kind:  "kv",
			Track: "primary",
		}},
		{"tags-1", recordMeta{
			URI:  "mlx://kv/2",
			Tags: map[string]string{"epoch": "3"},
		}},
		{"tags-many", recordMeta{
			URI: "mlx://kv/3",
			Tags: map[string]string{
				"epoch": "3", "track": "primary",
				"branch": "dev", "runner": "homelab",
			},
		}},
		{"labels", recordMeta{
			URI:    "mlx://kv/4",
			Labels: []string{"k0:v0", "k1:v1"},
		}},
		{"full", recordMeta{
			URI: "mlx://kv/5", Title: "bench", Kind: "training",
			Track: "primary", Tags: map[string]string{"a": "1"},
			Labels: []string{"x"},
		}},
		{"escapes", recordMeta{
			Title: `quote " and backslash \ and slash /`,
			Kind:  "tabs\tand\nnewlines",
			Tags:  map[string]string{"control": "\x01\x02"},
		}},
		{"unicode", recordMeta{
			Title:  "ünïcödé",
			Labels: []string{"日本", "🐦"},
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			encoded := encodeRecordMeta(&tc.meta)
			var decoded recordMeta
			if result := core.JSONUnmarshal(encoded, &decoded); !result.OK {
				t.Fatalf("JSONUnmarshal(%s) error: %v\nencoded: %s", tc.name, result.Value, encoded)
			}
			if decoded.URI != tc.meta.URI {
				t.Fatalf("URI = %q, want %q", decoded.URI, tc.meta.URI)
			}
			if decoded.Title != tc.meta.Title {
				t.Fatalf("Title = %q, want %q", decoded.Title, tc.meta.Title)
			}
			if decoded.Kind != tc.meta.Kind {
				t.Fatalf("Kind = %q, want %q", decoded.Kind, tc.meta.Kind)
			}
			if decoded.Track != tc.meta.Track {
				t.Fatalf("Track = %q, want %q", decoded.Track, tc.meta.Track)
			}
			if len(decoded.Tags) != len(tc.meta.Tags) {
				t.Fatalf("Tags len = %d, want %d", len(decoded.Tags), len(tc.meta.Tags))
			}
			for k, v := range tc.meta.Tags {
				if decoded.Tags[k] != v {
					t.Fatalf("Tags[%q] = %q, want %q", k, decoded.Tags[k], v)
				}
			}
			if len(decoded.Labels) != len(tc.meta.Labels) {
				t.Fatalf("Labels len = %d, want %d", len(decoded.Labels), len(tc.meta.Labels))
			}
			for i, v := range tc.meta.Labels {
				if decoded.Labels[i] != v {
					t.Fatalf("Labels[%d] = %q, want %q", i, decoded.Labels[i], v)
				}
			}
			// extractRecordURI must also accept the encoder output.
			uri, err := extractRecordURI(encoded)
			if err != nil {
				t.Fatalf("extractRecordURI: %v\nencoded: %s", err, encoded)
			}
			if uri != tc.meta.URI {
				t.Fatalf("extractRecordURI URI = %q, want %q", uri, tc.meta.URI)
			}
		})
	}
}

func TestDecodeRecordHeader_Good_RoundTripsEncodeRecordHeader(t *testing.T) {
	var buf [recordHeaderLen]byte
	encodeRecordHeader(buf[:], 7, 128, 12)
	got, err := decodeRecordHeader(buf[:])
	if err != nil {
		t.Fatalf("decodeRecordHeader() error = %v", err)
	}
	if got.chunkID != 7 || got.payloadSize != 128 || got.metaSize != 12 {
		t.Fatalf("decodeRecordHeader() = %+v, want {chunkID:7 payloadSize:128 metaSize:12}", got)
	}
}

func TestDecodeRecordHeader_Bad_WrongLength(t *testing.T) {
	if _, err := decodeRecordHeader(make([]byte, recordHeaderLen-1)); err == nil {
		t.Fatal("decodeRecordHeader(short) error = nil")
	}
	if _, err := decodeRecordHeader(make([]byte, recordHeaderLen+1)); err == nil {
		t.Fatal("decodeRecordHeader(long) error = nil")
	}
}

func TestDecodeRecordHeader_Bad_InvalidMagic(t *testing.T) {
	var buf [recordHeaderLen]byte
	encodeRecordHeader(buf[:], 1, 1, 0)
	buf[0] = 'X' // corrupt the magic prefix
	if _, err := decodeRecordHeader(buf[:]); err == nil {
		t.Fatal("decodeRecordHeader(bad magic) error = nil")
	}
}
