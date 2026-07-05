// SPDX-Licence-Identifier: EUPL-1.2

// Multimodal input for the generate library: resolve --image sources into raw
// image bytes and gate them on the loaded model's neutral vision capability,
// exactly the way serve's chat-completions handler carries image content parts
// (provider/openai: decode → inference.Message.Images → inference.VisionModel
// gate). Business logic lives here so cmd/lem generate stays thin flag-parsing.
//
// A source is either a base64 "data:" URL (the shape serve receives over the
// wire) or a LOCAL file path (the shape a CLI naturally takes). Remote http(s)
// URLs are refused — this is a local engine, so a prompt never triggers network
// I/O (no SSRF surface, matching provider/openai/content.go).

package generate

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
)

// maxDecodedImageBytes caps one decoded image, mirroring serve's decoder cap:
// the vision front-end resizes onto a fixed patch budget anyway, so anything
// past this is either a mistake or an attack on the decoder.
const maxDecodedImageBytes = 32 << 20

// maxImagesPerRequest bounds the per-request vision work, mirroring serve.
const maxImagesPerRequest = 16

// resolveImageInputs turns each --image source (a "data:" base64 URL or a local
// file path) into raw image bytes, enforcing the same count + size caps serve
// applies. It never fetches a remote URL. A nil/empty input yields nil, nil.
//
//	imgs, err := resolveImageInputs([]string{"cat.png", "data:image/png;base64,iVBOR..."})
func resolveImageInputs(sources []string) ([][]byte, error) {
	if len(sources) == 0 {
		return nil, nil
	}
	images := make([][]byte, 0, len(sources))
	for _, raw := range sources {
		source := core.Trim(raw)
		if source == "" {
			continue
		}
		if len(images) >= maxImagesPerRequest {
			return nil, core.E("generate.image", core.Sprintf("too many images — at most %d per request", maxImagesPerRequest), nil)
		}
		bytes, err := resolveOneImage(source)
		if err != nil {
			return nil, err
		}
		images = append(images, bytes)
	}
	if len(images) == 0 {
		return nil, nil
	}
	return images, nil
}

// resolveOneImage decodes a single image source: a "data:" URL through base64,
// or a local file path read straight off disk. A remote URL is refused.
func resolveOneImage(source string) ([]byte, error) {
	switch {
	case core.HasPrefix(source, "data:"):
		return decodeImageDataURL(source)
	case core.HasPrefix(source, "http://"), core.HasPrefix(source, "https://"):
		return nil, core.E("generate.image", "remote image URLs are not fetched — pass a local file path or a base64 data: URL", nil)
	default:
		return readImageFile(source)
	}
}

// readImageFile reads a local image off disk, enforcing the decoded-size cap so
// a huge file never allocates its whole contents past the budget serve accepts.
func readImageFile(path string) ([]byte, error) {
	read := core.ReadFile(path)
	if !read.OK {
		return nil, core.E("generate.image", core.Sprintf("read image %s", path), resultErr(read))
	}
	bytes, ok := read.Value.([]byte)
	if !ok {
		return nil, core.E("generate.image", core.Sprintf("read image %s returned non-byte data", path), nil)
	}
	if len(bytes) == 0 {
		return nil, core.E("generate.image", core.Sprintf("image %s is empty", path), nil)
	}
	if len(bytes) > maxDecodedImageBytes {
		return nil, core.E("generate.image", core.Sprintf("image %s exceeds the %d MiB cap", path, maxDecodedImageBytes>>20), nil)
	}
	return bytes, nil
}

// decodeImageDataURL decodes "data:image/png;base64,…" into raw image bytes,
// ported from serve's provider/openai/content.go so the CLI accepts the same
// wire shape. Only base64 data: URLs are accepted.
func decodeImageDataURL(url string) ([]byte, error) {
	comma := core.Index(url, ",")
	if comma < 0 {
		return nil, core.E("generate.image", "malformed data: URL — missing payload separator", nil)
	}
	if !core.HasSuffix(url[:comma], ";base64") {
		return nil, core.E("generate.image", "data: URL must be base64-encoded", nil)
	}
	payload := url[comma+1:]
	// Base64 expands 3 bytes to 4 chars; bound the ENCODED length before
	// decoding so an oversized payload never allocates its decoded form.
	if len(payload) > (maxDecodedImageBytes/3+1)*4 {
		return nil, core.E("generate.image", core.Sprintf("image exceeds the %d MiB cap", maxDecodedImageBytes>>20), nil)
	}
	decoded := core.Base64Decode(payload)
	if !decoded.OK {
		return nil, core.E("generate.image", "image base64 payload is invalid", resultErr(decoded))
	}
	bytes, ok := decoded.Value.([]byte)
	if !ok {
		text, textOK := decoded.Value.(string)
		if !textOK {
			return nil, core.E("generate.image", "image base64 decode returned an unexpected type", nil)
		}
		bytes = []byte(text)
	}
	if len(bytes) == 0 {
		return nil, core.E("generate.image", "image payload is empty", nil)
	}
	return bytes, nil
}

// requireVision gates an image-bearing request on the loaded model's neutral
// vision capability, exactly as serve's chat-completions handler does before it
// prefills: a model that does not implement inference.VisionModel (or reports it
// cannot accept images for this checkpoint) rejects the request rather than
// silently dropping the images and answering text-only. The images light up the
// moment the engine bridges its vision tower onto the neutral surface.
func requireVision(tm inference.TextModel, images [][]byte) error {
	if len(images) == 0 {
		return nil
	}
	vision, ok := tm.(inference.VisionModel)
	if !ok || !vision.AcceptsImages() {
		return core.E("generate.image", "model does not accept image input — the loaded engine exposes no neutral vision capability", nil)
	}
	return nil
}

// resultErr pulls the error out of a failed core.Result for wrapping, tolerating
// a Result whose Value is not an error.
func resultErr(r core.Result) error {
	if err, ok := r.Value.(error); ok {
		return err
	}
	return nil
}
