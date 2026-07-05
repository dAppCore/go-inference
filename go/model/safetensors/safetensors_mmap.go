// SPDX-Licence-Identifier: EUPL-1.2

//go:build unix

package safetensors

import (
	"syscall"

	core "dappco.re/go"
)

// LoadMmap memory-maps a safetensors file (read-only, page-aligned) and Parses it WITHOUT
// copying the weights — Mapping.Data is the whole file mapped, and every Tensor.Data is a
// view into it. This is the zero-copy weight path: a backend wraps Data in ONE GPU buffer
// (the page-aligned base satisfies Metal's bytesNoCopy) and binds each tensor at its byte
// offset, so multi-GB checkpoints never get a second heap or GPU copy. It is the no-cgo Go
// counterpart to mlx-c's mmap-based loader (which is how pkg/metal gets zero-copy). Pair
// with Mapping.Close to unmap; the mapping MUST outlive every view + buffer over it.
//
//	m, err := safetensors.LoadMmap("/path/to/model.safetensors")
//	defer m.Close()
//	w := m.Tensors["model.embed_tokens.weight"] // w.Data views the mmap — no copy
func LoadMmap(path string) (*Mapping, error) {
	fd, err := syscall.Open(path, syscall.O_RDONLY, 0)
	if err != nil {
		return nil, core.E("safetensors.LoadMmap", "open "+path, err)
	}
	defer syscall.Close(fd)
	var st syscall.Stat_t
	if err := syscall.Fstat(fd, &st); err != nil {
		return nil, core.E("safetensors.LoadMmap", "fstat "+path, err)
	}
	if st.Size <= 0 {
		return nil, core.NewError("safetensors.LoadMmap: empty file " + path)
	}
	// PROT_READ + MAP_SHARED: read-only view of the file, kernel returns a page-aligned base.
	data, err := syscall.Mmap(fd, 0, int(st.Size), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, core.E("safetensors.LoadMmap", "mmap "+path, err)
	}
	tensors, err := Parse(data)
	if err != nil {
		_ = syscall.Munmap(data)
		return nil, err
	}
	return &Mapping{Data: data, Tensors: tensors}, nil
}

// Close unmaps the file. Safe on a nil/already-closed mapping; call exactly once, after every
// Tensor view and GPU buffer over Data is done (using a view after Close is a use-after-unmap).
func (m *Mapping) Close() error {
	if m == nil || m.Data == nil {
		return nil
	}
	err := syscall.Munmap(m.Data)
	m.Data = nil
	m.Tensors = nil
	if err != nil {
		return core.E("safetensors.Mapping.Close", "munmap", err)
	}
	return nil
}
