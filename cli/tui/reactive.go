// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	core "dappco.re/go"
	"dappco.re/go/store"
)

const (
	reactiveNamespace      = "lem"
	reactiveGroupWorkspace = "workspace"
	reactiveGroupDrafts    = "drafts"
	reactiveGroupViewport  = "viewport"
	reactiveGroupCollapsed = "collapsed"
)

type reactiveState interface {
	Get(group, key string) (string, core.Result)
	Set(group, key, value string) core.Result
	Delete(group, key string) core.Result
	Close() core.Result
}

type storeReactiveState struct {
	store  *store.Store
	scoped *store.ScopedStore
}

func openReactiveState(paths appPaths) core.Result {
	if core.Trim(paths.State) == "" {
		return core.Fail(core.E("tui.openReactiveState", "state database path is required", nil))
	}
	if core.Trim(paths.Root) == "" || core.Trim(paths.Workspaces) == "" {
		return core.Fail(core.E("tui.openReactiveState", "workspace state directory is required", nil))
	}

	opened := store.New(
		paths.State,
		store.WithWorkspaceStateDirectory(paths.Workspaces),
	)
	if !opened.OK {
		return core.Fail(core.E("tui.openReactiveState", core.Concat("open state database: ", paths.State), resultError(opened)))
	}
	instance, ok := opened.Value.(*store.Store)
	if !ok {
		return core.Fail(core.E("tui.openReactiveState", "invalid go-store result", nil))
	}
	scoped := store.NewScoped(instance, reactiveNamespace)
	if scoped == nil {
		if result := instance.Close(); !result.OK {
			core.Warn("tui.reactive.close_after_scope_failure", "error", result.Value)
		}
		return core.Fail(core.E("tui.openReactiveState", "create scoped state", nil))
	}
	return core.Ok(&storeReactiveState{store: instance, scoped: scoped})
}

func (state *storeReactiveState) Get(group, key string) (string, core.Result) {
	if result := state.ready("Get", group, key); !result.OK {
		return "", result
	}
	return state.scoped.GetFrom(group, key)
}

func (state *storeReactiveState) Set(group, key, value string) core.Result {
	if result := state.ready("Set", group, key); !result.OK {
		return result
	}
	return state.scoped.SetIn(group, key, value)
}

func (state *storeReactiveState) Delete(group, key string) core.Result {
	if result := state.ready("Delete", group, key); !result.OK {
		return result
	}
	return state.scoped.Delete(group, key)
}

func (state *storeReactiveState) Close() core.Result {
	if state == nil || state.store == nil {
		return core.Ok(nil)
	}
	result := state.store.Close()
	state.store = nil
	state.scoped = nil
	return result
}

func (state *storeReactiveState) ready(operation, group, key string) core.Result {
	if state == nil || state.store == nil || state.scoped == nil {
		return core.Fail(core.E(core.Concat("tui.reactiveState.", operation), "reactive state is closed", nil))
	}
	if !validReactiveGroup(group) {
		return core.Fail(core.E(core.Concat("tui.reactiveState.", operation), core.Concat("unknown state group: ", group), nil))
	}
	if core.Trim(key) == "" {
		return core.Fail(core.E(core.Concat("tui.reactiveState.", operation), "state key is required", nil))
	}
	return core.Ok(nil)
}

func validReactiveGroup(group string) bool {
	switch group {
	case reactiveGroupWorkspace, reactiveGroupDrafts, reactiveGroupViewport, reactiveGroupCollapsed:
		return true
	default:
		return false
	}
}

type disabledReactiveState struct {
	reason error
}

func newDisabledReactiveState(reason error) reactiveState {
	return &disabledReactiveState{reason: reason}
}

func (state *disabledReactiveState) Get(group, key string) (string, core.Result) {
	return "", core.Fail(core.E(
		"tui.disabledReactiveState.Get",
		disabledStateMessage(state, group, key),
		store.NotFoundError,
	))
}

func (state *disabledReactiveState) Set(group, key, _ string) core.Result {
	return core.Fail(core.E(
		"tui.disabledReactiveState.Set",
		disabledStateMessage(state, group, key),
		disabledStateReason(state),
	))
}

func (state *disabledReactiveState) Delete(group, key string) core.Result {
	return core.Fail(core.E(
		"tui.disabledReactiveState.Delete",
		disabledStateMessage(state, group, key),
		disabledStateReason(state),
	))
}

func (*disabledReactiveState) Close() core.Result {
	return core.Ok(nil)
}

func disabledStateMessage(state *disabledReactiveState, group, key string) string {
	message := core.Concat("state disabled for ", group, "/", key)
	if reason := disabledStateReason(state); reason != nil {
		message = core.Concat(message, ": ", reason.Error())
	}
	return message
}

func disabledStateReason(state *disabledReactiveState) error {
	if state == nil {
		return nil
	}
	return state.reason
}
