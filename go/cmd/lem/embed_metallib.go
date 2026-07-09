// SPDX-Licence-Identifier: EUPL-1.2

//go:build embed_metallib

// Self-contained metallibs: under -tags embed_metallib the shipping build bakes
// BOTH GPU shader libraries (Apple MLX's mlx.metallib + go-inference's own
// lthn_kernels.metallib) into the lem binary, gzipped. lem then runs from any
// path with nothing external to ship or resolve — the whole "you only need
// go-inference" point. Without the tag (plain `go build` / `go test`) this file
// is excluded and the engine resolves the metallib externally (MLX_METALLIB_PATH
// / colocated), keeping the ~151MB artifact out of routine dev + CI builds.
//
// The build step (Taskfile build:embed) gzips build/dist/lib/{mlx,lthn_kernels}
// .metallib into {mlx,lthn_kernels}.metallib.gz next to this file before
// compiling. At process start we gunzip both once into a single content-addressed
// cache dir and point MLX at mlx.metallib via MLX_METALLIB_PATH before any Metal
// device init; the engine's sibling lookup then finds lthn_kernels.metallib
// beside it (see engine/metal device.go siblingMetallib).
package main

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"io"
	"os"
	"path/filepath"
)

//go:embed mlx.metallib.gz
var mlxMetallibGz []byte

//go:embed lthn_kernels.metallib.gz
var lthnKernelsGz []byte

// init extracts the embedded metallibs and sets MLX_METALLIB_PATH before main.
// Best-effort: any failure leaves the env unset so the engine falls back to its
// normal external resolution rather than crashing the process at import time.
func init() {
	// An operator's explicit MLX_METALLIB_PATH outranks the embedded copy —
	// never clobber it (the set-if-unset contract engine/metal also honours).
	if os.Getenv("MLX_METALLIB_PATH") != "" {
		return
	}
	if len(mlxMetallibGz) == 0 {
		return
	}

	// Content-addressed dir keyed on both payloads, so a version bump lands in a
	// fresh dir and both metallibs always match. Both extract into this ONE dir
	// so the engine's sibling lookup finds lthn_kernels.metallib beside mlx.
	h := sha256.New()
	h.Write(mlxMetallibGz)
	h.Write(lthnKernelsGz)
	dir := filepath.Join(os.TempDir(), "lthn-lem", hex.EncodeToString(h.Sum(nil)[:8]))
	mlxDst := filepath.Join(dir, "mlx.metallib")

	if err := os.MkdirAll(dir, 0o755); err != nil {
		return
	}
	if !extractGz(mlxMetallibGz, mlxDst) {
		return
	}
	// lthn_kernels is optional — the engine falls back to composed primitives if
	// absent — so a failure here still leaves the (working) mlx.metallib pointed at.
	if len(lthnKernelsGz) > 0 {
		_ = extractGz(lthnKernelsGz, filepath.Join(dir, "lthn_kernels.metallib"))
	}
	_ = os.Setenv("MLX_METALLIB_PATH", mlxDst)
}

// extractGz gunzips src into dst (idempotent: a present non-empty dst is trusted,
// since the parent dir is content-addressed). Writes to a temp sibling then
// renames so a concurrent start never sees a half-written file. Returns true on
// a usable dst.
func extractGz(src []byte, dst string) bool {
	if fi, err := os.Stat(dst); err == nil && fi.Size() > 0 {
		return true
	}
	gz, err := gzip.NewReader(bytes.NewReader(src))
	if err != nil {
		return false
	}
	defer func() { _ = gz.Close() }()

	tmp := dst + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return false
	}
	if _, err := io.Copy(f, gz); err != nil {
		_ = f.Close()
		_ = os.Remove(tmp)
		return false
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(tmp)
		return false
	}
	if err := os.Rename(tmp, dst); err != nil {
		_ = os.Remove(tmp)
		return false
	}
	return true
}
