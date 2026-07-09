package hip

import (
	core "dappco.re/go"
	"dappco.re/go/inference/engine/hip/internal/gguf"
)

//	models, err := DiscoverModels("/data/lem/gguf")
//	fmt.Println(models[0].Architecture, models[0].Quantisation)
//
// DiscoverModels scans a directory for GGUF model files and returns structured
// information about each. Files that cannot be parsed are skipped.
func DiscoverModels(dir string) (
	[]ModelInfo,
	error,
) {
	rootResult := core.PathAbs(dir)
	if !rootResult.OK {
		return nil, core.E("rocm.DiscoverModels", "resolve model directory", rootResult.Value.(error))
	}
	root := rootResult.Value.(string)

	matchResult := core.PathMatch("[", "x")
	if !matchResult.OK && core.Contains(dir, "[") {
		return nil, core.E("rocm.DiscoverModels", "glob gguf files", matchResult.Value.(error))
	}
	matches := core.PathGlob(core.PathJoin(root, "*.gguf"))

	var models []ModelInfo
	for _, path := range matches {
		meta, err := gguf.ReadMetadata(path)
		if err != nil {
			continue
		}

		models = append(models, ModelInfo{
			Path:         path,
			Architecture: meta.Architecture,
			Name:         meta.Name,
			Quantisation: gguf.FileTypeName(meta.FileType),
			Parameters:   meta.SizeLabel,
			FileSize:     meta.FileSize,
			ContextLen:   meta.ContextLength,
		})
	}

	return models, nil
}
