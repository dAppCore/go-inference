// SPDX-Licence-Identifier: EUPL-1.2

package scheme

type mixerInfo struct {
	kind      string
	state     StateKind
	cacheMode string
}

func (mixer mixerInfo) Kind() string     { return mixer.kind }
func (mixer mixerInfo) State() StateKind { return mixer.state }
func (mixer mixerInfo) CacheMode() string {
	return mixer.cacheMode
}

type cacheInfo struct {
	mode   string
	serves StateKind
}

func (cache cacheInfo) Mode() string      { return cache.mode }
func (cache cacheInfo) Serves() StateKind { return cache.serves }

type quantInfo struct {
	kind string
	bits int
}

func (quant quantInfo) Kind() string { return quant.kind }
func (quant quantInfo) Bits() int    { return quant.bits }

func init() {
	for _, mixer := range []mixerInfo{
		{kind: "full_attention", state: StateKVCache},
		{kind: "softmax-hybrid", state: StateKVCache},
		{kind: "mamba2", state: StateRecurrent},
		{kind: "rwkv7", state: StateRecurrent},
		{kind: "gla", state: StateRecurrent},
		{kind: "retnet", state: StateRecurrent},
		{kind: "deltanet", state: StateRecurrent},
		{kind: "gsa", state: StateRecurrent},
		{kind: "nsa", state: StateKVCache},
		{kind: "moba", state: StateKVCache},
		{kind: "mla", state: StateKVCache, cacheMode: CacheModeMLALatent},
	} {
		RegisterMixer(mixer)
	}

	for _, cache := range []cacheInfo{
		{"default", StateKVCache},
		{"fp16", StateKVCache},
		{"q8", StateKVCache},
		{"k-q8-v-q4", StateKVCache},
		{"paged", StateKVCache},
		{"fixed", StateKVCache},
		{"turboquant", StateKVCache},
		{CacheModeMLALatent, StateKVCache},
		{CacheModeCompaction, StateKVCache},
		{CacheModeCompactionFull, StateKVCache},
		{"recurrent", StateRecurrent},
	} {
		RegisterCache(cache)
	}

	for _, quant := range []quantInfo{
		{"affine", 0},
		{"bf16", 16},
		{"mxfp4", 4},
		{"mxfp8", 8},
		{"nvfp4", 4},
		{"q4_0", 4},
		{"jangtq", 2},
	} {
		RegisterQuant(quant)
	}
}
