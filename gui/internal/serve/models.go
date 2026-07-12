// SPDX-Licence-Identifier: EUPL-1.2

package serve

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
)

// Model is a discoverable model the tray's picker offers. A serve daemon can be
// spawned against any of these via `lem serve --model <Path>`.
type Model struct {
	Name  string // basename of the model directory
	Path  string // absolute path to the model directory
	Type  string // architecture from config.json (e.g. "gemma3"), "" when unknown
	Quant int    // quantisation bits (0 when unquantised/unknown)
}

// DiscoverModels walks baseDir for loadable safetensors model directories using
// go-inference's own discovery idiom — a directory with config.json and at least
// one .safetensors file — and returns them for the picker. An empty baseDir
// yields nil; an unreadable one yields the models it could probe (none).
//
//	for _, m := range serve.DiscoverModels(serve.DefaultModelsDir()) {
//	    menu.Add(m.Name).OnClick(func() { svc.Start(m.Path, "", "") })
//	}
func DiscoverModels(baseDir string) []Model {
	if baseDir == "" {
		return nil
	}
	var models []Model
	for dm := range inference.Discover(baseDir) {
		models = append(models, Model{
			Name:  core.PathBase(dm.Path),
			Path:  dm.Path,
			Type:  dm.ModelType,
			Quant: dm.QuantBits,
		})
	}
	return models
}

// DefaultModelsDir is ~/Lethean/lem/models — the model store the daemon downloads
// into and its hot-swap reload endpoint binds against, and so the tray's default
// discovery root.
func DefaultModelsDir() string {
	return core.PathJoin(core.Env("HOME"), "Lethean", "lem", "models")
}
