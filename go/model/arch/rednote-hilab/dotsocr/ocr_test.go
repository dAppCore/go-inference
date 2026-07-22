// SPDX-Licence-Identifier: EUPL-1.2

package dotsocr

import "testing"

// TestScatterVisionEmbeds_Good proves the vision rows land at exactly the image-token positions,
// in order, leaving every other row untouched.
func TestScatterVisionEmbeds_Good(t *testing.T) {
	hidden := 2
	embeds := []float32{9, 9, 0, 0, 9, 9, 0, 0} // rows 1 and 3 are the image-token placeholders
	ids := []int32{100, 151665, 100, 151665}
	vision := []float32{1, 2, 3, 4} // two vision rows
	if err := scatterVisionEmbeds(embeds, ids, vision, 151665, hidden); err != nil {
		t.Fatalf("scatterVisionEmbeds: %v", err)
	}
	want := []float32{9, 9, 1, 2, 9, 9, 3, 4}
	if d := maxAbsDiff32(t, embeds, want); d != 0 {
		t.Fatalf("scatterVisionEmbeds result = %v, want %v", embeds, want)
	}
}

// TestScatterVisionEmbeds_Bad proves too FEW image-token positions for the vision rows produced
// refuses (never silently drops rows).
func TestScatterVisionEmbeds_Bad(t *testing.T) {
	embeds := make([]float32, 4)
	ids := []int32{100, 151665}
	vision := []float32{1, 2, 3, 4} // two rows, but only one image-token position
	if err := scatterVisionEmbeds(embeds, ids, vision, 151665, 2); err == nil {
		t.Fatal("scatterVisionEmbeds accepted more vision rows than image-token positions")
	}
}

// TestScatterVisionEmbeds_Ugly proves too MANY image-token positions for the vision rows produced
// ALSO refuses — the distinct "ran out of vision rows mid-scatter" branch from _Bad's "leftover
// rows at the end" branch.
func TestScatterVisionEmbeds_Ugly(t *testing.T) {
	embeds := make([]float32, 6)
	ids := []int32{151665, 151665, 151665}
	vision := []float32{1, 2} // one row, but three image-token positions
	if err := scatterVisionEmbeds(embeds, ids, vision, 151665, 2); err == nil {
		t.Fatal("scatterVisionEmbeds accepted more image-token positions than vision rows")
	}
}

// TestLoad_Bad proves a directory with no config.json refuses cleanly.
func TestLoad_Bad(t *testing.T) {
	if _, err := Load(t.TempDir()); err == nil {
		t.Fatal("Load accepted a directory with no config.json")
	}
}

// TestLoad_Ugly proves a config.json whose model_type isn't dots_ocr/dots_ocr_1_5 refuses
// distinctly from the missing-file case above — the same capability-refusal pattern
// whisper.Load's model_type check follows.
func TestLoad_Ugly(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "config.json", `{"model_type":"llama"}`)
	if _, err := Load(dir); err == nil {
		t.Fatal("Load accepted a non-DOTS-OCR model_type")
	}
}
