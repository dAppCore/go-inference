// SPDX-Licence-Identifier: EUPL-1.2

package turboquant

import "testing"

// testRow is a fixed 8-element row shared by the Codec wrapper tests below —
// large enough to exercise a real rotation/group split, small enough to be
// legible in a failure message.
var testRow = []float32{3, -1, 4, 1, -5, 9, 2, -6}

// TestQMSECodec_Good checks Encode/Decode round-trips via the wire format
// and BytesPerRow matches the actual payload length.
func TestQMSECodec_Good(t *testing.T) {
	c := QMSECodec{Bits: 3, Seed: 42}
	data := c.Encode(testRow)
	if got := c.BytesPerRow(len(testRow)); got != len(data) {
		t.Errorf("BytesPerRow(%d) = %d, want len(Encode(...)) = %d", len(testRow), got, len(data))
	}
	got := c.Decode(data, len(testRow))
	if len(got) != len(testRow) {
		t.Fatalf("Decode returned %d elements, want %d", len(got), len(testRow))
	}
	if c.Name() != "TurboQuant-Qmse-b3" {
		t.Errorf("Name() = %q, want %q", c.Name(), "TurboQuant-Qmse-b3")
	}
}

// TestQProdCodec_Good checks Encode/Decode round-trips via the wire format
// and BytesPerRow matches the actual payload length.
func TestQProdCodec_Good(t *testing.T) {
	c := QProdCodec{TotalBits: 3, Seed: 42}
	data := c.Encode(testRow)
	if got := c.BytesPerRow(len(testRow)); got != len(data) {
		t.Errorf("BytesPerRow(%d) = %d, want len(Encode(...)) = %d", len(testRow), got, len(data))
	}
	got := c.Decode(data, len(testRow))
	if len(got) != len(testRow) {
		t.Fatalf("Decode returned %d elements, want %d", len(got), len(testRow))
	}
	if c.Name() != "TurboQuant-Qprod-b3" {
		t.Errorf("Name() = %q, want %q", c.Name(), "TurboQuant-Qprod-b3")
	}
}

// TestMixedCodec_Good checks Encode/Decode round-trips via the wire format,
// BytesPerRow matches the actual payload length, and Name reports the true
// channel-weighted effective bit rate rather than a rounded label.
func TestMixedCodec_Good(t *testing.T) {
	split := NewMixedSplitDescriptor(len(testRow), []int{0, 4}, 2, 3)
	c := MixedCodec{Split: split, Seed: 42}
	data := c.Encode(testRow)
	if got := c.BytesPerRow(len(testRow)); got != len(data) {
		t.Errorf("BytesPerRow(%d) = %d, want len(Encode(...)) = %d", len(testRow), got, len(data))
	}
	got := c.Decode(data, len(testRow))
	if len(got) != len(testRow) {
		t.Fatalf("Decode returned %d elements, want %d", len(got), len(testRow))
	}
	// 2 outliers at 3 bits + 6 base at 2 bits over 8 channels = 2.25 bit/ch.
	if want := "TurboQuant-Mixed-2.25bit"; c.Name() != want {
		t.Errorf("Name() = %q, want %q", c.Name(), want)
	}
}

// TestMixedCodec_Ugly checks Name on the zero-value codec (D==0) does not
// divide by zero.
func TestMixedCodec_Ugly(t *testing.T) {
	c := MixedCodec{}
	if got := c.Name(); got != "TurboQuant-Mixed" {
		t.Errorf("Name() on zero-value MixedCodec = %q, want %q", got, "TurboQuant-Mixed")
	}
}

// TestGroupQuantInt8Codec_Good checks Encode/Decode round-trips via the wire
// format and BytesPerRow matches the actual payload length.
func TestGroupQuantInt8Codec_Good(t *testing.T) {
	c := GroupQuantInt8Codec{}
	data := c.Encode(testRow)
	if got := c.BytesPerRow(len(testRow)); got != len(data) {
		t.Errorf("BytesPerRow(%d) = %d, want len(Encode(...)) = %d", len(testRow), got, len(data))
	}
	got := c.Decode(data, len(testRow))
	if len(got) != len(testRow) {
		t.Fatalf("Decode returned %d elements, want %d", len(got), len(testRow))
	}
}

// TestGroupQuantInt4Codec_Good checks Encode/Decode round-trips via the wire
// format and BytesPerRow matches the actual payload length.
func TestGroupQuantInt4Codec_Good(t *testing.T) {
	c := GroupQuantInt4Codec{}
	data := c.Encode(testRow)
	if got := c.BytesPerRow(len(testRow)); got != len(data) {
		t.Errorf("BytesPerRow(%d) = %d, want len(Encode(...)) = %d", len(testRow), got, len(data))
	}
	got := c.Decode(data, len(testRow))
	if len(got) != len(testRow) {
		t.Fatalf("Decode returned %d elements, want %d", len(got), len(testRow))
	}
}

// TestCodecs_Good checks the full table has the 9 codecs the RFC #41 spec
// requires and every one has a distinct name.
func TestCodecs_Good(t *testing.T) {
	split := NewMixedSplitDescriptor(128, []int{0}, 2, 3)
	codecs := Codecs(42, split)
	if len(codecs) != 9 {
		t.Fatalf("Codecs(...) returned %d codecs, want 9 (Qmse b2-4, Qprod b2-4, Mixed, Int8, Int4)", len(codecs))
	}
	seen := map[string]bool{}
	for _, c := range codecs {
		name := c.Name()
		if seen[name] {
			t.Errorf("duplicate codec name %q in Codecs(...)", name)
		}
		seen[name] = true
	}
}

// TestCodecs_Ugly checks every codec in the table actually round-trips a
// row without panicking — a smoke test across the whole table at once,
// catching a wiring mistake (wrong bits passed to Unmarshal, say) that a
// single-codec test could miss.
func TestCodecs_Ugly(t *testing.T) {
	const d = 16
	split := CalibrateMixedSplit([][]float32{testRowN(d)}, d, 4, 2, 3)
	codecs := Codecs(99, split)
	row := testRowN(d)
	for _, c := range codecs {
		data := c.Encode(row)
		got := c.Decode(data, d)
		if len(got) != d {
			t.Errorf("codec %q: Decode returned %d elements, want %d", c.Name(), len(got), d)
		}
		if bpr := c.BytesPerRow(d); bpr != len(data) {
			t.Errorf("codec %q: BytesPerRow(%d) = %d, want len(Encode(...)) = %d", c.Name(), d, bpr, len(data))
		}
	}
}

// testRowN builds a deterministic n-element row for the codec smoke test.
func testRowN(n int) []float32 {
	row := make([]float32, n)
	for i := range row {
		row[i] = float32(i%7) - 3
	}
	return row
}
