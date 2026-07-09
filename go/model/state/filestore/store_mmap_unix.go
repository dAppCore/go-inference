// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin || linux || freebsd || netbsd || openbsd

package filestore

import "syscall"

func (s *Store) ensureMappedRegionLocked() error {
	if s == nil || s.file == nil {
		return errStoreClosed
	}
	if s.mappedRegion != nil {
		return nil
	}
	info, err := s.file.Stat()
	if err != nil {
		return err
	}
	size, err := s.regionSize(info.Size())
	if err != nil {
		return err
	}
	if size <= 0 || size > int64(maxInt()) {
		return errMappedRegionInvalid
	}
	pageSize := int64(syscall.Getpagesize())
	pageDelta := s.baseAt % pageSize
	mapOffset := s.baseAt - pageDelta
	mapBytes := size + pageDelta
	if mapBytes <= 0 || mapBytes > int64(maxInt()) {
		return errMappedRegionInvalid
	}
	mapped, err := syscall.Mmap(int(s.file.Fd()), mapOffset, int(mapBytes), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return err
	}
	start := int(pageDelta)
	end := start + int(size)
	if start < 0 || end < start || end > len(mapped) {
		_ = syscall.Munmap(mapped)
		return errMappedRegionInvalid
	}
	s.mapped = mapped
	s.mappedRegion = mapped[start:end]
	return nil
}

func (s *Store) unmapRegionLocked() {
	if s == nil || s.mapped == nil {
		return
	}
	mapped := s.mapped
	s.mapped = nil
	s.mappedRegion = nil
	_ = syscall.Munmap(mapped)
}
