// SPDX-Licence-Identifier: EUPL-1.2

//go:build unix

package gguf

import (
	"syscall"

	core "dappco.re/go"
)

func mmapGGUFFile(path string) ([]byte, func() error, error) {
	fd, err := syscall.Open(path, syscall.O_RDONLY, 0)
	if err != nil {
		return nil, nil, core.E("gguf.LoadTensors", "open "+path, err)
	}
	defer syscall.Close(fd)
	var st syscall.Stat_t
	if err := syscall.Fstat(fd, &st); err != nil {
		return nil, nil, core.E("gguf.LoadTensors", "fstat "+path, err)
	}
	if st.Size <= 0 {
		return nil, nil, core.NewError("gguf.LoadTensors: empty file " + path)
	}
	data, err := syscall.Mmap(fd, 0, int(st.Size), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, nil, core.E("gguf.LoadTensors", "mmap "+path, err)
	}
	closeMapping := func() error {
		if err := syscall.Munmap(data); err != nil {
			return core.E("gguf.TensorMapping.Close", "munmap", err)
		}
		return nil
	}
	return data, closeMapping, nil
}
