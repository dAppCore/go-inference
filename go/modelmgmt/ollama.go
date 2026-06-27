package modelmgmt

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"time"

	"dappco.re/go"
	coreio "dappco.re/go/io"
)

// OllamaBaseModelMap maps model tags to Ollama model names.
var OllamaBaseModelMap = map[string]string{
	"gemma-3-1b":  "gemma3:1b",
	"gemma-3-4b":  "gemma3:4b",
	"gemma-3-12b": "gemma3:12b",
	"gemma-3-27b": "gemma3:27b",
}

// HFBaseModelMap maps model tags to HuggingFace model IDs.
var HFBaseModelMap = map[string]string{
	"gemma-3-1b":  "google/gemma-3-1b-it",
	"gemma-3-4b":  "google/gemma-3-4b-it",
	"gemma-3-12b": "google/gemma-3-12b-it",
	"gemma-3-27b": "google/gemma-3-27b-it",
}

// ollamaUploadBlob uploads a local file to Ollama's blob store.
// Returns the sha256 digest string (e.g. "sha256:abc123...").
func ollamaUploadBlob(ollamaURL, filePath string) core.Result {
	raw, err := coreio.Local.Read(filePath)
	if err != nil {
		return core.Fail(core.E("modelmgmt.ollamaUploadBlob", core.Sprintf("read %s", filePath), err))
	}
	data := []byte(raw)

	hash := sha256.Sum256(data)
	digest := "sha256:" + hex.EncodeToString(hash[:])

	headReq, _ := http.NewRequest(http.MethodHead, ollamaURL+"/api/blobs/"+digest, nil)
	client := &http.Client{Timeout: 5 * time.Minute}
	headResp, err := client.Do(headReq)
	if err == nil && headResp.StatusCode == http.StatusOK {
		headResp.Body.Close()
		return core.Ok(digest)
	}
	if headResp != nil {
		headResp.Body.Close()
	}

	req, err := http.NewRequest(http.MethodPost, ollamaURL+"/api/blobs/"+digest, core.NewBuffer(data))
	if err != nil {
		return core.Fail(core.E("modelmgmt.ollamaUploadBlob", "blob request", err))
	}
	req.Header.Set("Content-Type", "application/octet-stream")

	resp, err := client.Do(req)
	if err != nil {
		return core.Fail(core.E("modelmgmt.ollamaUploadBlob", "blob upload", err))
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusCreated && resp.StatusCode != http.StatusOK {
		rBody := readAll(resp.Body)
		body := []byte{}
		if rBody.OK {
			body = rBody.Value.([]byte)
		}
		return core.Fail(core.E("modelmgmt.ollamaUploadBlob", core.Sprintf("blob upload HTTP %d: %s", resp.StatusCode, string(body)), nil))
	}
	return core.Ok(digest)
}

// OllamaCreateModel creates a temporary Ollama model with a LoRA adapter.
// peftDir is a local directory containing adapter_model.safetensors and adapter_config.json.
func OllamaCreateModel(ollamaURL, modelName, baseModel, peftDir string) core.Result {
	sfPath := peftDir + "/adapter_model.safetensors"
	cfgPath := peftDir + "/adapter_config.json"

	sfDigestResult := ollamaUploadBlob(ollamaURL, sfPath)
	if !sfDigestResult.OK {
		return core.Fail(core.E("modelmgmt.OllamaCreateModel", "upload adapter safetensors", sfDigestResult.Value.(error)))
	}
	sfDigest := sfDigestResult.Value.(string)

	cfgDigestResult := ollamaUploadBlob(ollamaURL, cfgPath)
	if !cfgDigestResult.OK {
		return core.Fail(core.E("modelmgmt.OllamaCreateModel", "upload adapter config", cfgDigestResult.Value.(error)))
	}
	cfgDigest := cfgDigestResult.Value.(string)

	reqBody := core.JSONMarshalString(map[string]any{
		"model": modelName,
		"from":  baseModel,
		"adapters": map[string]string{
			"adapter_model.safetensors": sfDigest,
			"adapter_config.json":       cfgDigest,
		},
	})

	client := &http.Client{Timeout: 10 * time.Minute}
	resp, err := client.Post(ollamaURL+"/api/create", "application/json", core.NewReader(reqBody))
	if err != nil {
		return core.Fail(core.E("modelmgmt.OllamaCreateModel", "ollama create", err))
	}
	defer resp.Body.Close()

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)
	for scanner.Scan() {
		var status struct {
			Status string `json:"status"`
			Error  string `json:"error"`
		}
		if r := core.JSONUnmarshalString(scanner.Text(), &status); !r.OK {
			return core.Fail(core.E("modelmgmt.OllamaCreateModel", "ollama create decode", r.Value.(error)))
		}
		if status.Error != "" {
			return core.Fail(core.E("modelmgmt.OllamaCreateModel", core.Sprintf("ollama create: %s", status.Error), nil))
		}
		if status.Status == "success" {
			return core.Ok(nil)
		}
	}
	if err := scanner.Err(); err != nil {
		return core.Fail(core.E("modelmgmt.OllamaCreateModel", "ollama create decode", err))
	}

	if resp.StatusCode != http.StatusOK {
		return core.Fail(core.E("modelmgmt.OllamaCreateModel", core.Sprintf("ollama create: HTTP %d", resp.StatusCode), nil))
	}
	return core.Ok(nil)
}

// OllamaDeleteModel removes a temporary Ollama model.
func OllamaDeleteModel(ollamaURL, modelName string) core.Result {
	body := core.JSONMarshalString(map[string]string{"model": modelName})

	req, err := http.NewRequest(http.MethodDelete, ollamaURL+"/api/delete", core.NewReader(body))
	if err != nil {
		return core.Fail(core.E("modelmgmt.OllamaDeleteModel", "ollama delete request", err))
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return core.Fail(core.E("modelmgmt.OllamaDeleteModel", "ollama delete", err))
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		rBody := readAll(resp.Body)
		respBody := []byte{}
		if rBody.OK {
			respBody = rBody.Value.([]byte)
		}
		return core.Fail(core.E("modelmgmt.OllamaDeleteModel", core.Sprintf("ollama delete %d: %s", resp.StatusCode, string(respBody)), nil))
	}
	return core.Ok(nil)
}
