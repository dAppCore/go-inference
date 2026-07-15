// SPDX-Licence-Identifier: EUPL-1.2

package generate

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// pngMagic is a minimal byte sequence standing in for image content — the
// resolver does not decode the image, it only carries the bytes through, so any
// non-empty payload exercises the path.
var pngMagic = []byte{0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a, 0x01, 0x02}

// writeTempImage writes payload to a fresh file under t.TempDir and returns its
// path, failing the test on a write error.
func writeTempImage(t *testing.T, name string, payload []byte) string {
	t.Helper()
	path := core.PathJoin(t.TempDir(), name)
	if r := core.WriteFile(path, payload, 0o644); !r.OK {
		t.Fatalf("write temp image %s: %v", path, r.Value)
	}
	return path
}

// TestResolveImageInputs_FileAndDataURL_Good proves a local file path and a
// base64 data: URL both resolve to their raw bytes, in order.
func TestResolveImageInputs_FileAndDataURL_Good(t *testing.T) {
	path := writeTempImage(t, "cat.png", pngMagic)
	dataURL := "data:image/png;base64," + core.Base64Encode(pngMagic)

	images, err := resolveImageInputs([]string{path, "  ", dataURL})
	if err != nil {
		t.Fatalf("resolveImageInputs: %v", err)
	}
	if len(images) != 2 {
		t.Fatalf("got %d images, want 2 (blank entry should be skipped)", len(images))
	}
	for i, got := range images {
		if string(got) != string(pngMagic) {
			t.Fatalf("image[%d] = %v, want %v", i, got, pngMagic)
		}
	}
}

// TestResolveImageInputs_Empty_Good proves nil/blank inputs yield no images and
// no error — the no-image path.
func TestResolveImageInputs_Empty_Good(t *testing.T) {
	for _, in := range [][]string{nil, {}, {"", "   "}} {
		images, err := resolveImageInputs(in)
		if err != nil {
			t.Fatalf("resolveImageInputs(%v): %v", in, err)
		}
		if images != nil {
			t.Fatalf("resolveImageInputs(%v) = %v, want nil", in, images)
		}
	}
}

// TestResolveImageInputs_TooMany_Bad proves the per-request image cap rejects an
// oversized batch rather than loading unbounded vision work.
func TestResolveImageInputs_TooMany_Bad(t *testing.T) {
	path := writeTempImage(t, "one.png", pngMagic)
	sources := make([]string, maxImagesPerRequest+1)
	for i := range sources {
		sources[i] = path
	}
	if _, err := resolveImageInputs(sources); err == nil {
		t.Fatalf("resolveImageInputs of %d images: want cap error, got nil", len(sources))
	}
}

// TestResolveOneImage_RemoteURL_Bad proves a remote http(s) URL is refused — a
// local engine never fetches images over the network.
func TestResolveOneImage_RemoteURL_Bad(t *testing.T) {
	for _, url := range []string{"http://example.com/cat.png", "https://example.com/cat.png"} {
		if _, err := resolveOneImage(url); err == nil {
			t.Fatalf("resolveOneImage(%q): want refusal, got nil", url)
		}
	}
}

// TestReadImageFile_Missing_Bad proves a nonexistent path errors rather than
// silently yielding empty bytes.
func TestReadImageFile_Missing_Bad(t *testing.T) {
	if _, err := readImageFile(core.PathJoin(t.TempDir(), "nope.png")); err == nil {
		t.Fatal("readImageFile of missing path: want error, got nil")
	}
}

// TestReadImageFile_Empty_Ugly proves a present-but-empty file errors — an empty
// image is a malformed input, not a valid zero-byte image.
func TestReadImageFile_Empty_Ugly(t *testing.T) {
	path := writeTempImage(t, "empty.png", []byte{})
	if _, err := readImageFile(path); err == nil {
		t.Fatal("readImageFile of empty file: want error, got nil")
	}
}

// TestDecodeImageDataURL_Roundtrip_Good proves a base64 data: URL decodes back
// to the exact source bytes.
func TestDecodeImageDataURL_Roundtrip_Good(t *testing.T) {
	url := "data:image/jpeg;base64," + core.Base64Encode(pngMagic)
	got, err := decodeImageDataURL(url)
	if err != nil {
		t.Fatalf("decodeImageDataURL: %v", err)
	}
	if string(got) != string(pngMagic) {
		t.Fatalf("decoded = %v, want %v", got, pngMagic)
	}
}

// TestDecodeImageDataURL_NotBase64_Bad proves a data: URL without the ;base64
// marker is rejected (the engine takes base64, not raw/URL-encoded payloads).
func TestDecodeImageDataURL_NotBase64_Bad(t *testing.T) {
	if _, err := decodeImageDataURL("data:image/png,notbase64payload"); err == nil {
		t.Fatal("decodeImageDataURL without ;base64: want error, got nil")
	}
}

// TestDecodeImageDataURL_MissingComma_Ugly proves a data: URL with no payload
// separator is rejected rather than mis-parsed.
func TestDecodeImageDataURL_MissingComma_Ugly(t *testing.T) {
	if _, err := decodeImageDataURL("data:image/png;base64"); err == nil {
		t.Fatal("decodeImageDataURL with no comma: want error, got nil")
	}
}

// fakeTextModel satisfies inference.TextModel via an embedded nil interface —
// only the methods a test needs are overridden; the rest are never called.
type fakeTextModel struct{ inference.TextModel }

// fakeVisionModel adds the neutral vision capability so requireVision's gate can
// be exercised for both an accepting and a declining checkpoint.
type fakeVisionModel struct {
	inference.TextModel
	accepts bool
}

func (f fakeVisionModel) AcceptsImages() bool { return f.accepts }

// Compile-time proof the fakes carry the interfaces requireVision asserts.
var (
	_ inference.TextModel   = fakeTextModel{}
	_ inference.VisionModel = fakeVisionModel{}
)

// TestRequireVision_NoImages_Good proves the gate is a no-op with no images —
// even a text-only model generates when the turn carries no image.
func TestRequireVision_NoImages_Good(t *testing.T) {
	if err := requireVision(fakeTextModel{}, nil); err != nil {
		t.Fatalf("requireVision with no images: %v", err)
	}
}

// TestRequireVision_NonVisionModelRejects_Bad proves an image turn against a
// model that does not implement inference.VisionModel is rejected, mirroring
// serve's chat-completions handler rather than dropping the image.
func TestRequireVision_NonVisionModelRejects_Bad(t *testing.T) {
	if err := requireVision(fakeTextModel{}, [][]byte{pngMagic}); err == nil {
		t.Fatal("requireVision on a non-vision model: want rejection, got nil")
	}
}

// TestRequireVision_VisionModelAccepts_Good proves an image turn is admitted
// when the loaded checkpoint reports it accepts images.
func TestRequireVision_VisionModelAccepts_Good(t *testing.T) {
	if err := requireVision(fakeVisionModel{accepts: true}, [][]byte{pngMagic}); err != nil {
		t.Fatalf("requireVision on an accepting vision model: %v", err)
	}
}

// TestRequireVision_VisionModelDeclines_Ugly proves a VisionModel that reports
// this checkpoint shipped no tower still rejects the image turn (AcceptsImages
// is a live probe, not a static family declaration).
func TestRequireVision_VisionModelDeclines_Ugly(t *testing.T) {
	if err := requireVision(fakeVisionModel{accepts: false}, [][]byte{pngMagic}); err == nil {
		t.Fatal("requireVision on a declining vision model: want rejection, got nil")
	}
}

// wavMagic is a minimal byte sequence standing in for audio content — like the
// image resolver, resolveAudioInputs carries the bytes through unchanged (the
// engine validates the WAV shape), so any non-empty payload exercises the path.
var wavMagic = []byte{'R', 'I', 'F', 'F', 0x24, 0x00, 0x00, 0x00, 'W', 'A', 'V', 'E'}

// TestResolveAudioInputs_FileAndDataURL_Good proves a local file path and a
// base64 data: URL both resolve to their raw bytes, in order, with the blank
// entry skipped — the audio sibling of resolveImageInputs' Good case.
func TestResolveAudioInputs_FileAndDataURL_Good(t *testing.T) {
	path := writeTempImage(t, "clip.wav", wavMagic)
	dataURL := "data:audio/wav;base64," + core.Base64Encode(wavMagic)

	audios, err := resolveAudioInputs([]string{path, "  ", dataURL})
	if err != nil {
		t.Fatalf("resolveAudioInputs: %v", err)
	}
	if len(audios) != 2 {
		t.Fatalf("got %d audios, want 2 (blank entry should be skipped)", len(audios))
	}
	for i, got := range audios {
		if string(got) != string(wavMagic) {
			t.Fatalf("audio[%d] = %v, want %v", i, got, wavMagic)
		}
	}
}

// TestResolveAudioInputs_Empty_Good proves nil/blank inputs yield no audio and
// no error — the no-audio path.
func TestResolveAudioInputs_Empty_Good(t *testing.T) {
	for _, in := range [][]string{nil, {}, {"", "   "}} {
		audios, err := resolveAudioInputs(in)
		if err != nil {
			t.Fatalf("resolveAudioInputs(%v): %v", in, err)
		}
		if audios != nil {
			t.Fatalf("resolveAudioInputs(%v) = %v, want nil", in, audios)
		}
	}
}

// TestResolveAudioInputs_RemoteURL_Bad proves a remote http(s) URL is refused —
// a local engine never fetches audio over the network.
func TestResolveAudioInputs_RemoteURL_Bad(t *testing.T) {
	for _, url := range []string{"http://example.com/clip.wav", "https://example.com/clip.wav"} {
		if _, err := resolveAudioInputs([]string{url}); err == nil {
			t.Fatalf("resolveAudioInputs(%q): want refusal, got nil", url)
		}
	}
}

// TestResolveAudioInputs_TooMany_Bad proves the per-request cap rejects an
// oversized batch rather than loading unbounded audio work.
func TestResolveAudioInputs_TooMany_Bad(t *testing.T) {
	path := writeTempImage(t, "one.wav", wavMagic)
	sources := make([]string, maxImagesPerRequest+1)
	for i := range sources {
		sources[i] = path
	}
	if _, err := resolveAudioInputs(sources); err == nil {
		t.Fatalf("resolveAudioInputs of %d clips: want cap error, got nil", len(sources))
	}
}

// TestResolveAudioInputs_MissingFile_Ugly proves a nonexistent path errors
// rather than silently yielding empty audio bytes.
func TestResolveAudioInputs_MissingFile_Ugly(t *testing.T) {
	if _, err := resolveAudioInputs([]string{core.PathJoin(t.TempDir(), "nope.wav")}); err == nil {
		t.Fatal("resolveAudioInputs of a missing path: want error, got nil")
	}
}

// fakeAudioModel adds the neutral audio capability so requireAudio's gate can be
// exercised for both an accepting and a declining checkpoint — the audio sibling
// of fakeVisionModel.
type fakeAudioModel struct {
	inference.TextModel
	accepts bool
}

func (f fakeAudioModel) AcceptsAudio() bool { return f.accepts }

// Compile-time proof the fake carries the interface requireAudio asserts.
var _ inference.AudioModel = fakeAudioModel{}

// TestRequireAudio_NoAudio_Good proves the gate is a no-op with no audio — even
// a text-only model generates when the turn carries no audio.
func TestRequireAudio_NoAudio_Good(t *testing.T) {
	if err := requireAudio(fakeTextModel{}, nil); err != nil {
		t.Fatalf("requireAudio with no audio: %v", err)
	}
}

// TestRequireAudio_NonAudioModelRejects_Bad proves an audio turn against a model
// that does not implement inference.AudioModel is rejected rather than dropped.
func TestRequireAudio_NonAudioModelRejects_Bad(t *testing.T) {
	if err := requireAudio(fakeTextModel{}, [][]byte{wavMagic}); err == nil {
		t.Fatal("requireAudio on a non-audio model: want rejection, got nil")
	}
}

// TestRequireAudio_AudioModelAccepts_Good proves an audio turn is admitted when
// the loaded checkpoint reports it accepts audio.
func TestRequireAudio_AudioModelAccepts_Good(t *testing.T) {
	if err := requireAudio(fakeAudioModel{accepts: true}, [][]byte{wavMagic}); err != nil {
		t.Fatalf("requireAudio on an accepting audio model: %v", err)
	}
}

// TestRequireAudio_AudioModelDeclines_Ugly proves an AudioModel that reports
// this checkpoint shipped no audio head still rejects the audio turn
// (AcceptsAudio is a live probe, not a static family declaration).
func TestRequireAudio_AudioModelDeclines_Ugly(t *testing.T) {
	if err := requireAudio(fakeAudioModel{accepts: false}, [][]byte{wavMagic}); err == nil {
		t.Fatal("requireAudio on a declining audio model: want rejection, got nil")
	}
}
