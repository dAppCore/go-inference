// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"sync"

	core "dappco.re/go"
)

type hipGemma4Q4DeviceDecodeState struct {
	mode           string
	layers         []hipGemma4Q4DeviceLayerKVState
	appendLayers   int
	remirrorLayers int
	closed         bool
}

type hipGemma4Q4DeviceLayerKVState struct {
	cache                   *rocmDeviceKVCache
	descriptorTable         *rocmDeviceKVDescriptorTable
	launch                  rocmDeviceKVLaunchDescriptor
	borrowedCache           bool
	borrowedDescriptorTable bool
}

type hipGemma4Q4DeviceOwnershipAction struct {
	oldLayer *hipGemma4Q4DeviceLayerKVState
	newCache *rocmDeviceKVCache
	append   bool
}

var hipGemma4Q4DeviceLayerStatePool = struct {
	sync.Mutex
	layers [][]hipGemma4Q4DeviceLayerKVState
}{}

var hipGemma4Q4DeviceOwnershipActionPool = struct {
	sync.Mutex
	actions [][]hipGemma4Q4DeviceOwnershipAction
}{}

const (
	hipGemma4Q4DeviceDecodeStatePoolMax     = 4096
	hipGemma4Q4DeviceLayerStatePoolMax      = 4096
	hipGemma4Q4DeviceOwnershipActionPoolMax = 4096
)

var hipGemma4Q4DeviceDecodeStatePool = struct {
	sync.Mutex
	states []*hipGemma4Q4DeviceDecodeState
}{}

func hipPrewarmGemma4Q4DeviceDecodeStatePool(layerCapacity, depth int) {
	if layerCapacity <= 0 || depth <= 0 {
		return
	}
	states := make([]*hipGemma4Q4DeviceDecodeState, 0, depth)
	for range depth {
		state := hipNewGemma4Q4DeviceDecodeState("", layerCapacity)
		hipReleaseGemma4Q4DeviceLayerStates(state.layers)
		state.layers = nil
		state.closed = true
		states = append(states, state)
	}
	for _, state := range states {
		hipReleaseClosedGemma4Q4DeviceDecodeState(state)
	}
}

func hipPrewarmGemma4Q4DeviceLayerStatePool(layerCapacity, depth int) {
	if layerCapacity <= 0 || depth <= 0 {
		return
	}
	for range depth {
		hipReleaseGemma4Q4DeviceLayerStates(make([]hipGemma4Q4DeviceLayerKVState, 0, layerCapacity))
	}
}

func hipNewGemma4Q4DeviceDecodeState(mode string, layerCapacity int) *hipGemma4Q4DeviceDecodeState {
	hipGemma4Q4DeviceDecodeStatePool.Lock()
	count := len(hipGemma4Q4DeviceDecodeStatePool.states)
	if count > 0 {
		state := hipGemma4Q4DeviceDecodeStatePool.states[count-1]
		hipGemma4Q4DeviceDecodeStatePool.states[count-1] = nil
		hipGemma4Q4DeviceDecodeStatePool.states = hipGemma4Q4DeviceDecodeStatePool.states[:count-1]
		hipGemma4Q4DeviceDecodeStatePool.Unlock()
		*state = hipGemma4Q4DeviceDecodeState{mode: mode, layers: hipBorrowGemma4Q4DeviceLayerStates(layerCapacity)}
		return state
	}
	hipGemma4Q4DeviceDecodeStatePool.Unlock()
	state := &hipGemma4Q4DeviceDecodeState{}
	*state = hipGemma4Q4DeviceDecodeState{mode: mode, layers: hipBorrowGemma4Q4DeviceLayerStates(layerCapacity)}
	return state
}

func hipReleaseClosedGemma4Q4DeviceDecodeState(state *hipGemma4Q4DeviceDecodeState) {
	if state == nil || !state.closed || len(state.layers) != 0 {
		return
	}
	*state = hipGemma4Q4DeviceDecodeState{}
	hipGemma4Q4DeviceDecodeStatePool.Lock()
	if len(hipGemma4Q4DeviceDecodeStatePool.states) < hipGemma4Q4DeviceDecodeStatePoolMax {
		hipGemma4Q4DeviceDecodeStatePool.states = append(hipGemma4Q4DeviceDecodeStatePool.states, state)
	}
	hipGemma4Q4DeviceDecodeStatePool.Unlock()
}

func hipBorrowGemma4Q4DeviceLayerStates(layerCapacity int) []hipGemma4Q4DeviceLayerKVState {
	if layerCapacity <= 0 {
		layerCapacity = 1
	}
	hipGemma4Q4DeviceLayerStatePool.Lock()
	for index := len(hipGemma4Q4DeviceLayerStatePool.layers) - 1; index >= 0; index-- {
		layers := hipGemma4Q4DeviceLayerStatePool.layers[index]
		hipGemma4Q4DeviceLayerStatePool.layers[index] = nil
		hipGemma4Q4DeviceLayerStatePool.layers = hipGemma4Q4DeviceLayerStatePool.layers[:index]
		if cap(layers) >= layerCapacity {
			hipGemma4Q4DeviceLayerStatePool.Unlock()
			return layers[:0]
		}
	}
	hipGemma4Q4DeviceLayerStatePool.Unlock()
	return make([]hipGemma4Q4DeviceLayerKVState, 0, layerCapacity)
}

func hipReleaseGemma4Q4DeviceLayerStates(layers []hipGemma4Q4DeviceLayerKVState) {
	if cap(layers) == 0 {
		return
	}
	clear(layers[:cap(layers)])
	hipGemma4Q4DeviceLayerStatePool.Lock()
	if len(hipGemma4Q4DeviceLayerStatePool.layers) < hipGemma4Q4DeviceLayerStatePoolMax {
		hipGemma4Q4DeviceLayerStatePool.layers = append(hipGemma4Q4DeviceLayerStatePool.layers, layers[:0])
	}
	hipGemma4Q4DeviceLayerStatePool.Unlock()
}

func hipBorrowGemma4Q4DeviceOwnershipActions(layerCapacity int) []hipGemma4Q4DeviceOwnershipAction {
	if layerCapacity <= 0 {
		layerCapacity = 1
	}
	hipGemma4Q4DeviceOwnershipActionPool.Lock()
	for index := len(hipGemma4Q4DeviceOwnershipActionPool.actions) - 1; index >= 0; index-- {
		actions := hipGemma4Q4DeviceOwnershipActionPool.actions[index]
		hipGemma4Q4DeviceOwnershipActionPool.actions[index] = nil
		hipGemma4Q4DeviceOwnershipActionPool.actions = hipGemma4Q4DeviceOwnershipActionPool.actions[:index]
		if cap(actions) >= layerCapacity {
			hipGemma4Q4DeviceOwnershipActionPool.Unlock()
			return actions[:0]
		}
	}
	hipGemma4Q4DeviceOwnershipActionPool.Unlock()
	return make([]hipGemma4Q4DeviceOwnershipAction, 0, layerCapacity)
}

func hipReleaseGemma4Q4DeviceOwnershipActions(actions []hipGemma4Q4DeviceOwnershipAction) {
	if cap(actions) == 0 {
		return
	}
	clear(actions[:cap(actions)])
	hipGemma4Q4DeviceOwnershipActionPool.Lock()
	if len(hipGemma4Q4DeviceOwnershipActionPool.actions) < hipGemma4Q4DeviceOwnershipActionPoolMax {
		hipGemma4Q4DeviceOwnershipActionPool.actions = append(hipGemma4Q4DeviceOwnershipActionPool.actions, actions[:0])
	}
	hipGemma4Q4DeviceOwnershipActionPool.Unlock()
}

func (layer *hipGemma4Q4DeviceLayerKVState) Close() error {
	if layer == nil {
		return nil
	}
	var lastErr error
	if !layer.borrowedDescriptorTable {
		if err := layer.descriptorTable.Close(); err != nil {
			lastErr = core.E("rocm.hip.Gemma4Q4DeviceKV", "free descriptor table", err)
		}
	}
	if !layer.borrowedCache {
		cache := layer.cache
		if err := cache.Close(); err != nil {
			lastErr = core.E("rocm.hip.Gemma4Q4DeviceKV", "free device KV layer", err)
		} else {
			rocmReleaseDeviceKVCache(cache)
		}
	}
	layer.cache = nil
	layer.descriptorTable = nil
	return lastErr
}

func (layer *hipGemma4Q4DeviceLayerKVState) closeDescriptorTable() error {
	if layer == nil || layer.borrowedDescriptorTable {
		return nil
	}
	if err := layer.descriptorTable.Close(); err != nil {
		return core.E("rocm.hip.Gemma4Q4DeviceKV", "free descriptor table", err)
	}
	layer.descriptorTable = nil
	return nil
}

func hipMirrorGemma4Q4DecodeState(driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, state hipGemma4Q4DecodeState, mode string) (*hipGemma4Q4DeviceDecodeState, error) {
	if driver == nil {
		return nil, core.E("rocm.hip.Gemma4Q4DeviceKV", "HIP driver is nil", nil)
	}
	if !driver.Available() {
		return nil, core.E("rocm.hip.Gemma4Q4DeviceKV", "HIP driver is not available", nil)
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	if err := state.validate(cfg); err != nil {
		return nil, err
	}
	if len(state.Layers) == 0 {
		return nil, core.E("rocm.hip.Gemma4Q4DeviceKV", "decode state has no layers", nil)
	}
	if mode == "" {
		mode = rocmKVCacheModeFP16
	}
	deviceState := hipNewGemma4Q4DeviceDecodeState(mode, len(state.Layers))
	deviceState.remirrorLayers = len(state.Layers)
	for index, layerState := range state.Layers {
		layer, err := hipMirrorGemma4Q4LayerDecodeState(driver, cfg.Layers[index], layerState, mode)
		if err != nil {
			_ = deviceState.Close()
			return nil, err
		}
		deviceState.layers = append(deviceState.layers, layer)
	}
	return deviceState, nil
}

func hipUpdateGemma4Q4DeviceDecodeState(driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, previousHost, nextHost hipGemma4Q4DecodeState, previousDevice *hipGemma4Q4DeviceDecodeState, mode string) (*hipGemma4Q4DeviceDecodeState, error) {
	if previousDevice == nil {
		return hipMirrorGemma4Q4DecodeState(driver, cfg, nextHost, mode)
	}
	if previousDevice.closed {
		return nil, core.E("rocm.hip.Gemma4Q4DeviceKV", "previous device decode state is closed", nil)
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	if err := previousHost.validate(cfg); err != nil {
		return nil, err
	}
	if err := nextHost.validate(cfg); err != nil {
		return nil, err
	}
	if len(previousHost.Layers) == 0 {
		nextDevice, err := hipMirrorGemma4Q4DecodeState(driver, cfg, nextHost, firstNonEmptyString(mode, previousDevice.mode))
		if err != nil {
			return nil, err
		}
		if err := previousDevice.Close(); err != nil {
			_ = nextDevice.Close()
			return nil, err
		}
		return nextDevice, nil
	}
	if len(previousHost.Layers) != len(nextHost.Layers) || len(previousDevice.layers) != len(nextHost.Layers) {
		return nil, core.E("rocm.hip.Gemma4Q4DeviceKV", "decode state layer counts must match for device update", nil)
	}
	mode = firstNonEmptyString(mode, previousDevice.mode)
	if mode == "" {
		mode = rocmKVCacheModeFP16
	}
	if previousDevice.mode != "" && mode != previousDevice.mode {
		return nil, core.E("rocm.hip.Gemma4Q4DeviceKV", "device KV mode mismatch", nil)
	}
	nextDevice := hipNewGemma4Q4DeviceDecodeState(mode, len(nextHost.Layers))
	actions := hipBorrowGemma4Q4DeviceOwnershipActions(len(nextHost.Layers))
	defer hipReleaseGemma4Q4DeviceOwnershipActions(actions)
	success := false
	defer func() {
		if !success {
			_ = nextDevice.Close()
		}
	}()
	for index := range nextHost.Layers {
		oldLayer := &previousDevice.layers[index]
		layerCfg := cfg.Layers[index]
		if !oldLayer.borrowedCache && hipGemma4Q4LayerStateCanAppendDeviceKV(layerCfg, previousHost.Layers[index], nextHost.Layers[index]) {
			keyStart := len(nextHost.Layers[index].Keys) - layerCfg.HeadDim
			valueStart := len(nextHost.Layers[index].Values) - layerCfg.HeadDim
			nextCache, err := oldLayer.cache.withAppendedToken(nextHost.Layers[index].Keys[keyStart:], nextHost.Layers[index].Values[valueStart:])
			if err != nil {
				return nil, err
			}
			table, err := nextCache.KernelDescriptorTableFromAppendedToken(context.Background(), oldLayer.cache, oldLayer.descriptorTable)
			if err != nil {
				_ = nextCache.closePagesFrom(oldLayer.cache.PageCount())
				return nil, err
			}
			launch, err := nextCache.KernelLaunchDescriptor(table)
			if err != nil {
				_ = table.Close()
				_ = nextCache.closePagesFrom(oldLayer.cache.PageCount())
				return nil, err
			}
			nextDevice.layers = append(nextDevice.layers, hipGemma4Q4DeviceLayerKVState{cache: nextCache, descriptorTable: table, launch: launch})
			nextDevice.appendLayers++
			actions = append(actions, hipGemma4Q4DeviceOwnershipAction{oldLayer: oldLayer, newCache: nextCache, append: true})
			continue
		}
		layer, err := hipMirrorGemma4Q4LayerDecodeState(driver, layerCfg, nextHost.Layers[index], mode)
		if err != nil {
			return nil, err
		}
		nextDevice.layers = append(nextDevice.layers, layer)
		nextDevice.remirrorLayers++
		actions = append(actions, hipGemma4Q4DeviceOwnershipAction{oldLayer: oldLayer})
	}
	for _, action := range actions {
		if action.oldLayer.borrowedCache {
			// The source owner layer handles the shared cache once.
		} else if action.append {
			oldCache := action.oldLayer.cache
			if err := oldCache.transferPagesTo(action.newCache); err != nil {
				return nil, err
			}
			action.oldLayer.cache = nil
			rocmReleaseDeviceKVCache(oldCache)
		} else {
			oldCache := action.oldLayer.cache
			if err := oldCache.Close(); err != nil {
				return nil, err
			}
			action.oldLayer.cache = nil
			rocmReleaseDeviceKVCache(oldCache)
		}
		if err := action.oldLayer.closeDescriptorTable(); err != nil {
			return nil, err
		}
	}
	hipReleaseGemma4Q4DeviceLayerStates(previousDevice.layers)
	previousDevice.layers = nil
	previousDevice.closed = true
	success = true
	return nextDevice, nil
}

func hipFinalizeGemma4Q4ForwardDeviceState(previous, next *hipGemma4Q4DeviceDecodeState) error {
	if next == nil || previous == nil {
		return nil
	}
	if previous.closed {
		return core.E("rocm.hip.Gemma4Q4DeviceKV", "previous device decode state is closed", nil)
	}
	if len(previous.layers) != len(next.layers) {
		return core.E("rocm.hip.Gemma4Q4DeviceKV", "device state layer counts must match for forward transfer", nil)
	}
	for index := range next.layers {
		oldLayer := &previous.layers[index]
		newLayer := &next.layers[index]
		if oldLayer.borrowedCache {
			// The source owner layer handles the shared cache once.
		} else if oldLayer.cache.ownsAnyPages() && newLayer.cache.borrowsPagesFrom(oldLayer.cache) {
			oldCache := oldLayer.cache
			if err := oldCache.transferPagesTo(newLayer.cache); err != nil {
				return err
			}
			oldLayer.cache = nil
			rocmReleaseDeviceKVCache(oldCache)
		} else if oldLayer.cache.ownsAnyPages() && newLayer.cache.sharesPagesFrom(oldLayer.cache) {
			oldCache := oldLayer.cache
			if err := oldCache.transferSharedPagesTo(newLayer.cache); err != nil {
				return err
			}
			oldLayer.cache = nil
			rocmReleaseDeviceKVCache(oldCache)
		} else {
			oldCache := oldLayer.cache
			if err := oldCache.Close(); err != nil {
				return err
			}
			oldLayer.cache = nil
			rocmReleaseDeviceKVCache(oldCache)
		}
		hipTransferGemma4Q4DescriptorTableOwnership(oldLayer, newLayer)
		if err := oldLayer.closeDescriptorTable(); err != nil {
			return err
		}
	}
	hipReleaseGemma4Q4DeviceLayerStates(previous.layers)
	previous.layers = nil
	previous.closed = true
	return nil
}

func hipTransferGemma4Q4DescriptorTableOwnership(oldLayer, newLayer *hipGemma4Q4DeviceLayerKVState) {
	if oldLayer == nil || newLayer == nil || oldLayer.descriptorTable == nil {
		return
	}
	if oldLayer.descriptorTable == newLayer.descriptorTable {
		oldLayer.borrowedDescriptorTable = true
	}
}

func hipMirrorGemma4Q4LayerDecodeState(driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, layerState hipGemma4Q4LayerKVState, mode string) (hipGemma4Q4DeviceLayerKVState, error) {
	if len(layerState.Keys) == 0 || len(layerState.Values) == 0 {
		return hipGemma4Q4DeviceLayerKVState{}, core.E("rocm.hip.Gemma4Q4DeviceKV", "decode state layer has no KV tokens", nil)
	}
	host, err := newROCmKVCache(mode, defaultROCmKVBlockSize)
	if err != nil {
		return hipGemma4Q4DeviceLayerKVState{}, err
	}
	if err := host.AppendVectors(0, cfg.HeadDim, cfg.HeadDim, layerState.Keys, layerState.Values); err != nil {
		return hipGemma4Q4DeviceLayerKVState{}, err
	}
	device, err := host.MirrorToDevice(driver)
	if err != nil {
		return hipGemma4Q4DeviceLayerKVState{}, err
	}
	table, err := device.kernelDescriptorTableLabeled("rocm.KVCache.DeviceDescriptor", "mirror_layer_decode_state")
	if err != nil {
		_ = device.Close()
		return hipGemma4Q4DeviceLayerKVState{}, err
	}
	launch, err := device.KernelLaunchDescriptor(table)
	if err != nil {
		_ = table.Close()
		_ = device.Close()
		return hipGemma4Q4DeviceLayerKVState{}, err
	}
	return hipGemma4Q4DeviceLayerKVState{cache: device, descriptorTable: table, launch: launch}, nil
}

func hipGemma4Q4LayerStateCanAppendDeviceKV(cfg hipGemma4Q4Layer0Config, previous, next hipGemma4Q4LayerKVState) bool {
	if cfg.HeadDim <= 0 || len(previous.Keys) == 0 || len(previous.Values) == 0 {
		return false
	}
	if len(next.Keys) != len(previous.Keys)+cfg.HeadDim || len(next.Values) != len(previous.Values)+cfg.HeadDim {
		return false
	}
	return hipFloat32SlicesEqual(previous.Keys, next.Keys[:len(previous.Keys)]) && hipFloat32SlicesEqual(previous.Values, next.Values[:len(previous.Values)])
}

func hipFloat32SlicesEqual(left, right []float32) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

func (state *hipGemma4Q4DeviceDecodeState) Close() error {
	if state == nil || state.closed {
		return nil
	}
	var lastErr error
	for index := range state.layers {
		if err := state.layers[index].Close(); err != nil {
			lastErr = err
		}
	}
	hipReleaseGemma4Q4DeviceLayerStates(state.layers)
	state.layers = nil
	state.closed = true
	return lastErr
}

func (state *hipGemma4Q4DeviceDecodeState) LayerCount() int {
	if state == nil {
		return 0
	}
	return len(state.layers)
}

func (state *hipGemma4Q4DeviceDecodeState) layerCache(index int) *rocmDeviceKVCache {
	if state == nil || index < 0 || index >= len(state.layers) {
		return nil
	}
	return state.layers[index].cache
}

func hipGemma4Q4DeviceLayerCaches(state *hipGemma4Q4DeviceDecodeState, scratch []*rocmDeviceKVCache, layerCount int) []*rocmDeviceKVCache {
	if state == nil {
		return nil
	}
	if layerCount <= 0 {
		layerCount = state.LayerCount()
	}
	if cap(scratch) < layerCount {
		scratch = make([]*rocmDeviceKVCache, layerCount)
	} else {
		scratch = scratch[:layerCount]
		clear(scratch)
	}
	for index := range scratch {
		scratch[index] = state.layerCache(index)
	}
	return scratch
}

func hipGemma4Q4DeviceLayerDescriptorTables(state *hipGemma4Q4DeviceDecodeState, scratch []*rocmDeviceKVDescriptorTable, layerCount int) []*rocmDeviceKVDescriptorTable {
	if state == nil {
		return nil
	}
	if layerCount <= 0 {
		layerCount = state.LayerCount()
	}
	if cap(scratch) < layerCount {
		scratch = make([]*rocmDeviceKVDescriptorTable, layerCount)
	} else {
		scratch = scratch[:layerCount]
		clear(scratch)
	}
	for index := range scratch {
		scratch[index] = state.layerDescriptorTable(index)
	}
	return scratch
}

func (state *hipGemma4Q4DeviceDecodeState) layerDescriptorTable(index int) *rocmDeviceKVDescriptorTable {
	if state == nil || index < 0 || index >= len(state.layers) {
		return nil
	}
	return state.layers[index].descriptorTable
}

func (state *hipGemma4Q4DeviceDecodeState) LayerTokenCounts() []int {
	if state == nil {
		return nil
	}
	counts := make([]int, 0, len(state.layers))
	for _, layer := range state.layers {
		if layer.cache == nil {
			counts = append(counts, 0)
			continue
		}
		counts = append(counts, layer.cache.TokenCount())
	}
	return counts
}

func (state *hipGemma4Q4DeviceDecodeState) maxLayerTokenCount() int {
	if state == nil {
		return 0
	}
	maxTokens := 0
	for _, layer := range state.layers {
		if layer.cache == nil {
			continue
		}
		if tokens := layer.cache.TokenCount(); tokens > maxTokens {
			maxTokens = tokens
		}
	}
	return maxTokens
}

func (state *hipGemma4Q4DeviceDecodeState) MemoryBytes() uint64 {
	if state == nil {
		return 0
	}
	var total uint64
	for _, layer := range state.layers {
		if !layer.borrowedCache {
			total += layer.cache.MemoryBytes()
		}
		if !layer.borrowedDescriptorTable && layer.descriptorTable != nil {
			total += layer.descriptorTable.AllocationBytes()
		}
	}
	return total
}

func (state *hipGemma4Q4DeviceDecodeState) CompatibleWithHostState(cfg hipGemma4Q4ForwardConfig, host hipGemma4Q4DecodeState, mode string) error {
	if state == nil {
		return nil
	}
	if state.closed {
		return core.E("rocm.hip.Gemma4Q4DeviceKV", "device decode state is closed", nil)
	}
	if err := cfg.validate(); err != nil {
		return err
	}
	if err := host.validate(cfg); err != nil {
		return err
	}
	if len(host.Layers) == 0 {
		return core.E("rocm.hip.Gemma4Q4DeviceKV", "prior device state requires host KV state", nil)
	}
	if len(state.layers) != len(host.Layers) || len(state.layers) != len(cfg.Layers) {
		return core.E("rocm.hip.Gemma4Q4DeviceKV", "device state layer count must match host state", nil)
	}
	mode = firstNonEmptyString(mode, state.mode)
	if mode == "" {
		mode = rocmKVCacheModeFP16
	}
	if state.mode != "" && state.mode != mode {
		return core.E("rocm.hip.Gemma4Q4DeviceKV", "device KV mode mismatch", nil)
	}
	for index, layer := range state.layers {
		if layer.cache == nil {
			return core.E("rocm.hip.Gemma4Q4DeviceKV", core.Sprintf("device layer %d cache is nil", index), nil)
		}
		if layer.cache.closed {
			return core.E("rocm.hip.Gemma4Q4DeviceKV", core.Sprintf("device layer %d cache is closed", index), nil)
		}
		if layer.cache.mode != mode {
			return core.E("rocm.hip.Gemma4Q4DeviceKV", core.Sprintf("device layer %d cache mode mismatch", index), nil)
		}
		layerCfg := cfg.Layers[index]
		hostTokens := len(host.Layers[index].Keys) / layerCfg.HeadDim
		if layer.cache.TokenCount() != hostTokens {
			return core.E("rocm.hip.Gemma4Q4DeviceKV", core.Sprintf("device layer %d token count mismatch", index), nil)
		}
		keyWidth, valueWidth, ok := layer.cache.LastVectorWidths()
		if !ok || keyWidth != layerCfg.HeadDim || valueWidth != layerCfg.HeadDim {
			return core.E("rocm.hip.Gemma4Q4DeviceKV", core.Sprintf("device layer %d KV width mismatch", index), nil)
		}
	}
	return nil
}

func (state *hipGemma4Q4DeviceDecodeState) HostState() (hipGemma4Q4DecodeState, error) {
	if state == nil {
		return hipGemma4Q4DecodeState{}, core.E("rocm.hip.Gemma4Q4DeviceKV", "device decode state is nil", nil)
	}
	if state.closed {
		return hipGemma4Q4DecodeState{}, core.E("rocm.hip.Gemma4Q4DeviceKV", "device decode state is closed", nil)
	}
	hostState := hipGemma4Q4DecodeState{Layers: make([]hipGemma4Q4LayerKVState, 0, len(state.layers))}
	for index, layer := range state.layers {
		hostCache, err := layer.cache.hostCache()
		if err != nil {
			return hipGemma4Q4DecodeState{}, core.E("rocm.hip.Gemma4Q4DeviceKV", core.Sprintf("copy layer %d", index), err)
		}
		keys, values, err := hostCache.Restore(0, hostCache.TokenCount())
		if err != nil {
			return hipGemma4Q4DecodeState{}, core.E("rocm.hip.Gemma4Q4DeviceKV", core.Sprintf("restore layer %d", index), err)
		}
		hostState.Layers = append(hostState.Layers, hipGemma4Q4LayerKVState{Keys: keys, Values: values})
	}
	return hostState, nil
}

func (state *hipGemma4Q4DeviceDecodeState) Labels() map[string]string {
	labels := map[string]string{
		"gemma4_q4_device_kv_backing": "hip_device_mirror",
		"gemma4_q4_device_kv_layers":  core.Sprintf("%d", state.LayerCount()),
		"production_kv_cache_backing": hipKernelStatusNotLinked,
	}
	if state == nil {
		return labels
	}
	labels["gemma4_q4_device_kv_mode"] = state.mode
	labels["gemma4_q4_device_kv_bytes"] = core.Sprintf("%d", state.MemoryBytes())
	labels["gemma4_q4_device_kv_append_layers"] = core.Sprintf("%d", state.appendLayers)
	labels["gemma4_q4_device_kv_remirror_layers"] = core.Sprintf("%d", state.remirrorLayers)
	counts := state.LayerTokenCounts()
	if len(counts) > 0 {
		minTokens := counts[0]
		maxTokens := counts[0]
		for _, count := range counts[1:] {
			if count < minTokens {
				minTokens = count
			}
			if count > maxTokens {
				maxTokens = count
			}
		}
		labels["gemma4_q4_device_kv_min_tokens"] = core.Sprintf("%d", minTokens)
		labels["gemma4_q4_device_kv_max_tokens"] = core.Sprintf("%d", maxTokens)
	}
	return labels
}
