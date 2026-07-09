// SPDX-Licence-Identifier: EUPL-1.2

package memorypretrain

import (
	"context"

	core "dappco.re/go"
)

// FFNMemoryRuntime binds the offline router, anchor embedder, and FFN memory
// table used by model code when augmenting a feed-forward layer.
type FFNMemoryRuntime struct {
	Memory   *FFNMemoryBank `json:"-"`
	Router   *Bank          `json:"-"`
	Embedder Embedder       `json:"-"`
}

// NewFFNMemoryRuntime creates a runtime facade for memory-augmented FFN calls.
// A nil router selects the generic-memory fallback and does not require an
// embedder.
func NewFFNMemoryRuntime(memory *FFNMemoryBank, router *Bank, embedder Embedder) (*FFNMemoryRuntime, error) {
	if memory == nil {
		return nil, core.NewError("memorypretrain: FFN memory bank is nil")
	}
	if router != nil && embedder == nil {
		return nil, core.NewError("memorypretrain: embedder is required when router is set")
	}
	return &FFNMemoryRuntime{
		Memory:   memory,
		Router:   router,
		Embedder: embedder,
	}, nil
}

// AddTextToFFNOutput embeds queryText with the anchor embedder, routes the
// query through the hierarchical cluster bank, and applies the selected FFN
// memories. If no router is configured it applies the generic fallback slot.
func (runtime *FFNMemoryRuntime) AddTextToFFNOutput(ctx context.Context, dst []float32, ffnOutput []float32, mlpInput []float32, queryText string, layerID int) ([]float32, []int, FFNMemoryStats, error) {
	if runtime == nil {
		return nil, nil, FFNMemoryStats{}, core.NewError("memorypretrain: FFN memory runtime is nil")
	}
	if runtime.Memory == nil {
		return nil, nil, FFNMemoryStats{}, core.NewError("memorypretrain: FFN memory bank is nil")
	}
	if runtime.Router == nil {
		return runtime.Memory.AddGenericToFFNOutput(dst, ffnOutput, mlpInput, layerID)
	}
	if runtime.Embedder == nil {
		return nil, nil, FFNMemoryStats{}, core.NewError("memorypretrain: embedder is required when router is set")
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, FFNMemoryStats{}, err
	}
	query, err := runtime.Embedder.Embed(ctx, queryText)
	if err != nil {
		return nil, nil, FFNMemoryStats{}, core.E("memorypretrain.AddTextToFFNOutput", "embed query text", err)
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, FFNMemoryStats{}, err
	}
	return runtime.Memory.AddRoutedToFFNOutput(dst, ffnOutput, mlpInput, runtime.Router, query, layerID)
}
