// SPDX-Licence-Identifier: EUPL-1.2

package state

import "context"

type InMemoryStore struct {
	chunks map[int]string
	data   map[int][]byte
	refs   map[int]ChunkRef
	uris   map[string]int
	nextID int
}

func NewInMemoryStore(chunks map[int]string) *InMemoryStore {
	return NewInMemoryStoreWithManifest(chunks, nil)
}

func NewInMemoryStoreWithManifest(chunks map[int]string, refs map[int]ChunkRef) *InMemoryStore {
	copyMap := make(map[int]string, len(chunks))
	nextID := 1
	for id, text := range chunks {
		copyMap[id] = text
		if id >= nextID {
			nextID = id + 1
		}
	}
	refMap := make(map[int]ChunkRef, len(copyMap))
	for id := range copyMap {
		refMap[id] = ChunkRef{
			ChunkID:        id,
			FrameOffset:    uint64(id),
			HasFrameOffset: true,
			Codec:          CodecMemory,
		}
	}
	for id, ref := range refs {
		ref.ChunkID = id
		refMap[id] = ref
		if id >= nextID {
			nextID = id + 1
		}
	}
	return &InMemoryStore{
		chunks: copyMap,
		data:   make(map[int][]byte),
		refs:   refMap,
		uris:   make(map[string]int),
		nextID: nextID,
	}
}

func (s *InMemoryStore) Get(ctx context.Context, chunkID int) (string, error) {
	chunk, err := s.Resolve(ctx, chunkID)
	if err != nil {
		return "", err
	}
	return chunk.Text, nil
}

func (s *InMemoryStore) Resolve(ctx context.Context, chunkID int) (Chunk, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	select {
	case <-ctx.Done():
		return Chunk{}, ctx.Err()
	default:
	}
	if s == nil {
		return Chunk{}, &ChunkNotFoundError{ID: chunkID}
	}
	text, ok := s.chunks[chunkID]
	data, dataOK := s.data[chunkID]
	if !ok && !dataOK {
		return Chunk{}, &ChunkNotFoundError{ID: chunkID}
	}
	ref := s.refs[chunkID]
	if ref.ChunkID != chunkID {
		ref.ChunkID = chunkID
	}
	chunk := Chunk{Ref: ref, Text: text}
	if dataOK {
		chunk.Data = append([]byte(nil), data...)
		if chunk.Text == "" {
			chunk.Text = string(data)
		}
	}
	return chunk, nil
}

func (s *InMemoryStore) ResolveBytes(ctx context.Context, chunkID int) (Chunk, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	select {
	case <-ctx.Done():
		return Chunk{}, ctx.Err()
	default:
	}
	if s == nil {
		return Chunk{}, &ChunkNotFoundError{ID: chunkID}
	}
	ref := s.refs[chunkID]
	if ref.ChunkID != chunkID {
		ref.ChunkID = chunkID
	}
	if data, ok := s.data[chunkID]; ok {
		return Chunk{Ref: ref, Data: append([]byte(nil), data...)}, nil
	}
	text, ok := s.chunks[chunkID]
	if !ok {
		return Chunk{}, &ChunkNotFoundError{ID: chunkID}
	}
	return Chunk{Ref: ref, Text: text, Data: []byte(text)}, nil
}

func (s *InMemoryStore) ResolveURI(ctx context.Context, uri string) (Chunk, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	select {
	case <-ctx.Done():
		return Chunk{}, ctx.Err()
	default:
	}
	if s == nil {
		return Chunk{}, &URIChunkNotFoundError{URI: uri}
	}
	id, ok := s.uris[uri]
	if !ok {
		return Chunk{}, &URIChunkNotFoundError{URI: uri}
	}
	return s.Resolve(ctx, id)
}

func (s *InMemoryStore) Put(ctx context.Context, text string, opts PutOptions) (ChunkRef, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	select {
	case <-ctx.Done():
		return ChunkRef{}, ctx.Err()
	default:
	}
	if s == nil {
		return ChunkRef{}, &ChunkNotFoundError{}
	}
	if s.chunks == nil {
		s.chunks = make(map[int]string)
	}
	if s.refs == nil {
		s.refs = make(map[int]ChunkRef)
	}
	if s.data == nil {
		s.data = make(map[int][]byte)
	}
	if s.uris == nil {
		s.uris = make(map[string]int)
	}
	if s.nextID <= 0 {
		s.nextID = 1
	}
	id := s.nextID
	s.nextID++
	ref := ChunkRef{
		ChunkID:        id,
		FrameOffset:    uint64(id),
		HasFrameOffset: true,
		Codec:          CodecMemory,
	}
	s.chunks[id] = text
	delete(s.data, id)
	s.refs[id] = ref
	if opts.URI != "" {
		s.uris[opts.URI] = id
	}
	return ref, nil
}

func (s *InMemoryStore) PutBytes(ctx context.Context, data []byte, opts PutOptions) (ChunkRef, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	select {
	case <-ctx.Done():
		return ChunkRef{}, ctx.Err()
	default:
	}
	if s == nil {
		return ChunkRef{}, &ChunkNotFoundError{}
	}
	if s.chunks == nil {
		s.chunks = make(map[int]string)
	}
	if s.data == nil {
		s.data = make(map[int][]byte)
	}
	if s.refs == nil {
		s.refs = make(map[int]ChunkRef)
	}
	if s.uris == nil {
		s.uris = make(map[string]int)
	}
	if s.nextID <= 0 {
		s.nextID = 1
	}
	id := s.nextID
	s.nextID++
	ref := ChunkRef{
		ChunkID:        id,
		FrameOffset:    uint64(id),
		HasFrameOffset: true,
		Codec:          CodecMemory,
	}
	delete(s.chunks, id)
	s.data[id] = append([]byte(nil), data...)
	s.refs[id] = ref
	if opts.URI != "" {
		s.uris[opts.URI] = id
	}
	return ref, nil
}
