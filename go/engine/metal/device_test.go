// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func TestEnsureInitLoadsDeviceQueueAndLibrary(t *testing.T) {
	requireNativeRuntime(t)

	if device.ID == 0 {
		t.Fatal("device was not initialised")
	}
	if queue == nil {
		t.Fatal("command queue was not initialised")
	}
	if library == nil {
		t.Fatal("metallib was not loaded")
	}
}

func TestSiblingMetallibResolvesBesideMainLibrary(t *testing.T) {
	got := siblingMetallib("/tmp/mlx.metallib", "lthn_kernels.metallib")
	if got != "/tmp/lthn_kernels.metallib" {
		t.Fatalf("siblingMetallib = %q, want /tmp/lthn_kernels.metallib", got)
	}
	if got := siblingMetallib("mlx.metallib", "lthn_kernels.metallib"); got != "lthn_kernels.metallib" {
		t.Fatalf("siblingMetallib without directory = %q, want lthn_kernels.metallib", got)
	}
}
