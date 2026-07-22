// SPDX-Licence-Identifier: EUPL-1.2

package turboquant

import "testing"

// TestPutFloat32LE_Good round-trips a variety of float32 values through
// putFloat32LE/getFloat32LE.
func TestPutFloat32LE_Good(t *testing.T) {
	for _, v := range []float32{0, 1, -1, 1.5, -1.5, 3.4e38, -3.4e38, 1e-10} {
		buf := make([]byte, 4)
		putFloat32LE(buf, v)
		got := getFloat32LE(buf)
		if got != v {
			t.Errorf("putFloat32LE/getFloat32LE(%v) round-trip = %v, want %v", v, got, v)
		}
	}
}

// TestPutFloat32LE_Ugly checks the byte order is little-endian explicitly,
// against a known IEEE-754 bit pattern (1.5 = 0x3FC00000).
func TestPutFloat32LE_Ugly(t *testing.T) {
	buf := make([]byte, 4)
	putFloat32LE(buf, 1.5)
	want := []byte{0x00, 0x00, 0xC0, 0x3F}
	for i, b := range want {
		if buf[i] != b {
			t.Errorf("putFloat32LE(1.5)[%d] = %#x, want %#x", i, buf[i], b)
		}
	}
}

// TestPutUint32LE_Good round-trips a variety of uint32 values through
// putUint32LE/getUint32LE.
func TestPutUint32LE_Good(t *testing.T) {
	for _, v := range []uint32{0, 1, 258, 0xFFFFFFFF, 0x80000000} {
		buf := make([]byte, 4)
		putUint32LE(buf, v)
		got := getUint32LE(buf)
		if got != v {
			t.Errorf("putUint32LE/getUint32LE(%v) round-trip = %v, want %v", v, got, v)
		}
	}
}

// TestGetUint32LE_Ugly checks the byte order is little-endian explicitly.
func TestGetUint32LE_Ugly(t *testing.T) {
	got := getUint32LE([]byte{2, 1, 0, 0})
	if got != 258 {
		t.Errorf("getUint32LE({2,1,0,0}) = %d, want 258", got)
	}
}
