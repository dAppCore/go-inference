// SPDX-Licence-Identifier: EUPL-1.2

//go:build !unix

package safetensors

import (
	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

// LoadMmap on non-unix platforms falls back to a full read (no mmap): Mapping.Data holds the
// whole file on the heap and Tensors view it, so consumers get the identical shape — just
// without the zero-copy benefit. Close is a no-op (the GC reclaims Data). Keeps the package
// all-platforms while the unix build gets the real mmap.
func LoadMmap(path string) (*Mapping, error) {
	str, err := coreio.Local.Read(path)
	if err != nil {
		return nil, core.E("safetensors.LoadMmap", "read "+path, err)
	}
	blob := []byte(str)
	tensors, err := Parse(blob)
	if err != nil {
		return nil, err
	}
	return &Mapping{Data: blob, Tensors: tensors}, nil
}

// Close drops the references so the heap blob can be collected. No unmap on non-unix.
func (m *Mapping) Close() error {
	if m != nil {
		m.Data = nil
		m.Tensors = nil
	}
	return nil
}
