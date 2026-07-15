// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"net/http"
	"time"
)

// DefaultModelsPath is the OpenAI model-list route.
const DefaultModelsPath = "/v1/models"

// ModelsHandler serves GET /v1/models — the OpenAI model-list endpoint most
// clients call to discover what they can request. The servable model IDs come
// from a host-supplied callback: the Resolver only maps a NAME to a model
// (ResolveModel), so it cannot enumerate what is loaded on its own; the serve,
// which knows its --model, provides the list.
//
//	mux.Handle(openai.DefaultModelsPath, openai.NewModelsHandler(func() []string {
//	    return []string{"gemma-4-e2b-it-4bit"}
//	}))
type ModelsHandler struct {
	models func() []string
}

// NewModelsHandler builds the /v1/models handler. models returns the currently
// servable model IDs; nil (or an empty return) yields a valid empty list.
func NewModelsHandler(models func() []string) *ModelsHandler {
	return &ModelsHandler{models: models}
}

// ModelObject is one entry in the /v1/models list (OpenAI shape).
type ModelObject struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// ModelsListResponse is the /v1/models response body.
type ModelsListResponse struct {
	Object string        `json:"object"`
	Data   []ModelObject `json:"data"`
}

// ServeHTTP answers GET /v1/models with the servable models in OpenAI list shape.
// The request "model" field on chat/completions is cosmetic (the serve loads one
// --model), so these IDs are advisory — a client may echo one back, but any name
// resolves to the loaded model.
func (h *ModelsHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !requireServiceMethod(w, r, http.MethodGet) {
		return
	}
	created := time.Now().Unix()
	var ids []string
	if h != nil && h.models != nil {
		ids = h.models()
	}
	data := make([]ModelObject, 0, len(ids))
	for _, id := range ids {
		if id == "" {
			continue
		}
		data = append(data, ModelObject{ID: id, Object: "model", Created: created, OwnedBy: "lethean"})
	}
	writeJSON(w, http.StatusOK, ModelsListResponse{Object: "list", Data: data})
}
