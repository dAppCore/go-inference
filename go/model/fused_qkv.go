// SPDX-Licence-Identifier: EUPL-1.2

package model

import "dappco.re/go/inference/model/safetensors"

// SplitContiguousGateUp exposes a fused [gate; up] row tensor under the two
// canonical dense-MLP roles consumed by Assemble.
func SplitContiguousGateUp(tensors map[string]safetensors.Tensor, fused, gate, up string) map[string]safetensors.Tensor {
	t, ok := tensors[fused+".weight"]
	if !ok || len(t.Shape) < 2 || t.Shape[0]%2 != 0 || len(t.Data)%2 != 0 {
		return tensors
	}
	out := make(map[string]safetensors.Tensor, len(tensors)+2)
	for name, tensor := range tensors {
		out[name] = tensor
	}
	rows, size := t.Shape[0]/2, len(t.Data)/2
	shape := append([]int(nil), t.Shape...)
	shape[0] = rows
	out[gate+".weight"] = safetensors.Tensor{Dtype: t.Dtype, Shape: append([]int(nil), shape...), Data: t.Data[:size]}
	out[up+".weight"] = safetensors.Tensor{Dtype: t.Dtype, Shape: shape, Data: t.Data[size:]}
	return out
}

// SplitContiguousQKV splits a fused tensor laid out as all Q rows, then K, then V.
func SplitContiguousQKV(tensors map[string]safetensors.Tensor, fused, query, key, value string, qRows, kvRows int) map[string]safetensors.Tensor {
	t, ok := tensors[fused+".weight"]
	if !ok || len(t.Shape) != 2 || t.Shape[0] != qRows+2*kvRows || qRows <= 0 || kvRows <= 0 {
		return tensors
	}
	out := make(map[string]safetensors.Tensor, len(tensors)+3)
	for name, tensor := range tensors {
		out[name] = tensor
	}
	rowBytes := len(t.Data) / t.Shape[0]
	makePart := func(start, rows int) safetensors.Tensor {
		data := make([]byte, rows*rowBytes)
		copy(data, t.Data[start*rowBytes:(start+rows)*rowBytes])
		return safetensors.Tensor{Dtype: t.Dtype, Shape: []int{rows, t.Shape[1]}, Data: data}
	}
	out[query+".weight"] = makePart(0, qRows)
	out[key+".weight"] = makePart(qRows, kvRows)
	out[value+".weight"] = makePart(qRows+kvRows, kvRows)
	return out
}

// SplitGroupedQKV splits Falcon's new decoder layout: per KV group, its query
// heads are followed by one key head and one value head.
func SplitGroupedQKV(tensors map[string]safetensors.Tensor, fused, query, key, value string, heads, kvHeads, headDim int) map[string]safetensors.Tensor {
	t, ok := tensors[fused+".weight"]
	if !ok || len(t.Shape) != 2 || kvHeads <= 0 || heads%kvHeads != 0 || t.Shape[0] != (heads+2*kvHeads)*headDim {
		return tensors
	}
	out := make(map[string]safetensors.Tensor, len(tensors)+3)
	for name, tensor := range tensors {
		out[name] = tensor
	}
	rowBytes, qPerGroup := len(t.Data)/t.Shape[0], heads/kvHeads
	q, k, v := make([]byte, heads*headDim*rowBytes), make([]byte, kvHeads*headDim*rowBytes), make([]byte, kvHeads*headDim*rowBytes)
	for group := 0; group < kvHeads; group++ {
		base := group * (qPerGroup + 2) * headDim * rowBytes
		qBytes := qPerGroup * headDim * rowBytes
		copy(q[group*qBytes:(group+1)*qBytes], t.Data[base:base+qBytes])
		copy(k[group*headDim*rowBytes:(group+1)*headDim*rowBytes], t.Data[base+qBytes:base+qBytes+headDim*rowBytes])
		copy(v[group*headDim*rowBytes:(group+1)*headDim*rowBytes], t.Data[base+qBytes+headDim*rowBytes:base+qBytes+2*headDim*rowBytes])
	}
	out[query+".weight"] = safetensors.Tensor{Dtype: t.Dtype, Shape: []int{heads * headDim, t.Shape[1]}, Data: q}
	out[key+".weight"] = safetensors.Tensor{Dtype: t.Dtype, Shape: []int{kvHeads * headDim, t.Shape[1]}, Data: k}
	out[value+".weight"] = safetensors.Tensor{Dtype: t.Dtype, Shape: []int{kvHeads * headDim, t.Shape[1]}, Data: v}
	return out
}

// SplitInterleavedQKV aliases an HF fused [head,QKV,dim] tensor as three
// ordinary projection tensors for the neutral assembler.
func SplitInterleavedQKV(tensors map[string]safetensors.Tensor, fused, query, key, value string, heads, headDim int) map[string]safetensors.Tensor {
	t, ok := tensors[fused+".weight"]
	if !ok || len(t.Shape) != 2 || t.Shape[0] != 3*heads*headDim || heads <= 0 || headDim <= 0 {
		return tensors
	}
	out := make(map[string]safetensors.Tensor, len(tensors)+6)
	for name, tensor := range tensors {
		out[name] = tensor
	}
	split := func(source safetensors.Tensor, columns int) (safetensors.Tensor, safetensors.Tensor, safetensors.Tensor) {
		rowBytes := len(source.Data) / source.Shape[0]
		parts := [3][]byte{make([]byte, heads*headDim*rowBytes), make([]byte, heads*headDim*rowBytes), make([]byte, heads*headDim*rowBytes)}
		for h := 0; h < heads; h++ {
			for part := range 3 {
				src := (h*3 + part) * headDim * rowBytes
				dst := h * headDim * rowBytes
				copy(parts[part][dst:dst+headDim*rowBytes], source.Data[src:src+headDim*rowBytes])
			}
		}
		shape := []int{heads * headDim, columns}
		return safetensors.Tensor{Dtype: source.Dtype, Shape: shape, Data: parts[0]}, safetensors.Tensor{Dtype: source.Dtype, Shape: shape, Data: parts[1]}, safetensors.Tensor{Dtype: source.Dtype, Shape: shape, Data: parts[2]}
	}
	q, k, v := split(t, t.Shape[1])
	out[query+".weight"], out[key+".weight"], out[value+".weight"] = q, k, v
	if b, exists := tensors[fused+".bias"]; exists && len(b.Shape) == 1 && b.Shape[0] == t.Shape[0] {
		q, k, v = split(b, 1)
		q.Shape, k.Shape, v.Shape = []int{heads * headDim}, []int{heads * headDim}, []int{heads * headDim}
		out[query+".bias"], out[key+".bias"], out[value+".bias"] = q, k, v
	}
	return out
}
