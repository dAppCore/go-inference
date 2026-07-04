// SPDX-Licence-Identifier: EUPL-1.2

//go:build !unix

package gguf

import core "dappco.re/go"

func mmapGGUFFile(path string) ([]byte, func() error, error) {
	read := core.ReadFile(path)
	if !read.OK {
		return nil, nil, core.Errorf("gguf.LoadTensors: read %s: %w", path, read.Value.(error))
	}
	return read.Value.([]byte), nil, nil
}
