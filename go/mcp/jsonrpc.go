package mcp

import (
	"bytes"
	"context"
	"strconv"

	core "dappco.re/go"
)

type rpcRequest struct {
	JSONRPC string
	ID      RawMessage
	HasID   bool
	Method  string
	Params  RawMessage
}

type rpcResponse struct {
	JSONRPC string    `json:"jsonrpc"`
	ID      any       `json:"id"`
	Result  any       `json:"result,omitempty"`
	Error   *rpcError `json:"error,omitempty"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type callToolParams struct {
	Name      string
	Arguments RawMessage
}

// HandleFrame handles one newline-delimited JSON-RPC frame.
func (s *Service) HandleFrame(ctx context.Context, frame []byte) core.Result {
	// bytes.TrimSpace returns a subslice — zero alloc, vs the previous
	// []byte→string→Trim→[]byte round-trip which allocated two strings
	// plus a new byte slice per inbound frame.
	frame = bytes.TrimSpace(frame)
	if len(frame) == 0 {
		return core.Ok([]byte(nil))
	}

	reqResult := decodeRPCRequest(frame)
	if !reqResult.OK {
		response := marshalRPCResponse(rpcResponse{
			JSONRPC: "2.0",
			ID:      nil,
			Error:   &rpcError{Code: -32700, Message: "parse error"},
		})
		return core.Ok(response)
	}
	req := reqResult.Value.(rpcRequest)

	if req.JSONRPC != "2.0" || req.Method == "" {
		response := s.errorResponse(req.ID, -32600, "invalid request")
		return core.Ok(response)
	}

	result := s.handleMethod(ctx, req)
	if !req.HasID {
		if !result.OK {
			return result
		}
		return core.Ok([]byte(nil))
	}
	if !result.OK {
		err, _ := resultError(result).(error)
		response := s.errorResponse(req.ID, rpcCodeForError(err), err.Error())
		return core.Ok(response)
	}

	return core.Ok(marshalRPCResponse(rpcResponse{
		JSONRPC: "2.0",
		ID:      rawMessageValue(req.ID),
		Result:  result.Value,
	}))
}

func (s *Service) handleMethod(ctx context.Context, req rpcRequest) core.Result {
	switch req.Method {
	case "initialize":
		return core.Ok(map[string]any{
			"protocolVersion": "2024-11-05",
			"serverInfo": map[string]any{
				"name":    serverName,
				"version": serverVersion,
			},
			"capabilities": map[string]any{
				"tools": map[string]any{"listChanged": false},
			},
		})
	case "notifications/initialized":
		return core.Ok(nil)
	case "ping":
		return core.Ok(map[string]any{})
	case "tools/list":
		return core.Ok(map[string]any{"tools": s.Tools()})
	case "tools/call":
		return s.handleToolCall(ctx, req.Params)
	default:
		return core.Fail(core.Errorf("method not found: %s", req.Method))
	}
}

func (s *Service) handleToolCall(ctx context.Context, raw RawMessage) core.Result {
	raw = RawMessage(bytes.TrimSpace([]byte(raw)))
	if len(raw) == 0 || string(raw) == "null" {
		return core.Fail(core.Errorf("%w: missing tools/call params", errInvalidParams))
	}
	paramsResult := decodeCallToolParams(raw)
	if !paramsResult.OK {
		return paramsResult
	}
	params := paramsResult.Value.(callToolParams)
	params.Name = core.Trim(params.Name)
	if params.Name == "" {
		return core.Fail(core.Errorf("%w: tool name is required", errInvalidParams))
	}
	tool, ok := s.tools[params.Name]
	if !ok {
		return core.Fail(core.Errorf("tool not found: %s", params.Name))
	}
	if len(bytes.TrimSpace([]byte(params.Arguments))) == 0 {
		params.Arguments = RawMessage("{}")
	}

	outputResult := tool.Handler(ctx, params.Arguments)
	if !outputResult.OK {
		return outputResult
	}

	outputJSON := core.JSONMarshalString(outputResult.Value)
	return core.Ok(map[string]any{
		"content": []map[string]any{{
			"type": "text",
			"text": string(outputJSON),
		}},
		"structuredContent": outputResult.Value,
		"isError":           false,
	})
}

func (s *Service) errorResponse(id RawMessage, code int, message string) []byte {
	if len(id) == 0 {
		id = RawMessage("null")
	}
	return marshalRPCResponse(rpcResponse{
		JSONRPC: "2.0",
		ID:      rawMessageValue(id),
		Error:   &rpcError{Code: code, Message: message},
	})
}

func decodeRPCRequest(frame []byte) core.Result {
	var fields map[string]any
	if r := core.JSONUnmarshal(frame, &fields); !r.OK {
		return r
	}
	req := rpcRequest{}
	if value, ok := fields["jsonrpc"].(string); ok {
		req.JSONRPC = value
	}
	if value, ok := fields["method"].(string); ok {
		req.Method = value
	}
	if value, ok := fields["id"]; ok {
		req.HasID = true
		// Fast paths for the only ID shapes JSON-RPC permits in
		// practice (string, number, null) — avoid the reflect-based
		// encoding/json marshal path entirely.
		switch v := value.(type) {
		case string:
			req.ID = RawMessage(strconv.AppendQuote(nil, v))
		case float64:
			if v == float64(int64(v)) {
				req.ID = RawMessage(strconv.AppendInt(nil, int64(v), 10))
			} else {
				req.ID = RawMessage(strconv.AppendFloat(nil, v, 'g', -1, 64))
			}
		case nil:
			req.ID = RawMessage("null")
		default:
			if raw := core.JSONMarshal(value); raw.OK {
				req.ID = RawMessage(raw.Value.([]byte))
			} else {
				return raw
			}
		}
	}
	if value, ok := fields["params"]; ok {
		if raw := core.JSONMarshal(value); raw.OK {
			req.Params = RawMessage(raw.Value.([]byte))
		} else {
			return raw
		}
	}
	return core.Ok(req)
}

func decodeCallToolParams(raw RawMessage) core.Result {
	var fields map[string]any
	if r := core.JSONUnmarshal([]byte(raw), &fields); !r.OK {
		return core.Fail(core.Errorf("%w: %s", errInvalidParams, r.Error()))
	}
	params := callToolParams{}
	if value, ok := fields["name"].(string); ok {
		params.Name = value
	}
	if value, ok := fields["arguments"]; ok {
		rawArgs := core.JSONMarshal(value)
		if !rawArgs.OK {
			return rawArgs
		}
		params.Arguments = RawMessage(rawArgs.Value.([]byte))
	}
	return core.Ok(params)
}

func rawMessageValue(raw RawMessage) any {
	raw = RawMessage(bytes.TrimSpace([]byte(raw)))
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	// Fast paths mirroring decodeRPCRequest's ID typing: avoid the
	// reflect-based JSON unmarshal-into-any for the only ID shapes
	// JSON-RPC permits in practice (string, number, null). The output
	// must mirror encoding/json's any-decode contract: numbers come
	// back as float64, strings as their unquoted form.
	first := raw[0]
	if first == '"' {
		// Quoted string. strconv.Unquote handles JSON-compatible
		// escapes; on parse failure (unusual escape, malformed
		// surrogate) fall through to the JSON parser.
		if v, err := strconv.Unquote(string(raw)); err == nil {
			return v
		}
	} else if first == '-' || (first >= '0' && first <= '9') {
		// Number. Try integer-first to keep encoded form tight for
		// the common positional-integer ID; fall back to float for
		// fractional/exponent forms. Both cast to float64 so the
		// downstream encoder emits the same shape encoding/json
		// would for a map[string]any decode.
		if v, err := strconv.ParseInt(string(raw), 10, 64); err == nil {
			return float64(v)
		}
		if v, err := strconv.ParseFloat(string(raw), 64); err == nil {
			return v
		}
	}
	var value any
	if r := core.JSONUnmarshal([]byte(raw), &value); r.OK {
		return value
	}
	return nil
}

func resultError(r core.Result) any {
	if err, ok := r.Value.(error); ok {
		return err
	}
	return core.E("mcp.result", r.Error(), nil)
}

func rpcCodeForError(err error) int {
	if core.Is(err, errInvalidRequest) {
		return -32600
	}
	if core.Is(err, errInvalidParams) {
		return -32602
	}
	if core.HasPrefix(err.Error(), "method not found:") {
		return -32601
	}
	return -32000
}

func marshalRPCResponse(response rpcResponse) []byte {
	data := core.JSONMarshal(response)
	if !data.OK {
		fallback := core.JSONMarshal(rpcResponse{
			JSONRPC: "2.0",
			ID:      nil,
			Error:   &rpcError{Code: -32603, Message: "internal error"},
		})
		if !fallback.OK {
			return []byte(`{"jsonrpc":"2.0","id":null,"error":{"code":-32603,"message":"internal error"}}`)
		}
		return fallback.Value.([]byte)
	}
	return data.Value.([]byte)
}
