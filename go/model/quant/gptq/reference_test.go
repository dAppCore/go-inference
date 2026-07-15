// SPDX-Licence-Identifier: EUPL-1.2

package gptq

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

// TestReferenceSchema compares the writer contract with the real
// TheBlokeAI/Mixtral-tiny-GPTQ checkpoint. Set LEM_GPTQ_REFERENCE_DIR to the
// downloaded repository; the accepted file SHA-256 is documented in BRIEF lane
// verification output rather than embedding a network dependency in normal tests.
func TestReferenceSchema(t *testing.T) {
	dir := core.Getenv("LEM_GPTQ_REFERENCE_DIR")
	if core.Trim(dir) == "" {
		t.Skip("set LEM_GPTQ_REFERENCE_DIR to a real Hugging Face GPTQ snapshot")
	}
	idx, err := safetensors.IndexFiles([]string{core.PathJoin(dir, "model.safetensors")})
	if err != nil {
		t.Fatal(err)
	}
	checked := 0
	for name, qweight := range idx.Tensors {
		if !core.HasSuffix(name, ".qweight") {
			continue
		}
		base := core.TrimSuffix(name, ".qweight")
		qzeros, zeroOK := idx.Tensors[base+".qzeros"]
		scales, scaleOK := idx.Tensors[base+".scales"]
		gidx, indexOK := idx.Tensors[base+".g_idx"]
		if !zeroOK || !scaleOK || !indexOK {
			t.Fatalf("%s lacks GPTQ companions", base)
		}
		if qweight.DType != "I32" || qzeros.DType != "I32" || scales.DType != "F16" || gidx.DType != "I32" {
			t.Fatalf("%s dtypes = %s/%s/%s/%s", base, qweight.DType, qzeros.DType, scales.DType, gidx.DType)
		}
		if len(qweight.Shape) != 2 || len(qzeros.Shape) != 2 || len(scales.Shape) != 2 || len(gidx.Shape) != 1 {
			t.Fatalf("%s ranks do not match GPTQ schema", base)
		}
		if qweight.Shape[1] != scales.Shape[1] || qzeros.Shape[0] != scales.Shape[0] || qzeros.Shape[1]*8 != qweight.Shape[1] || qweight.Shape[0]*8 != gidx.Shape[0] {
			t.Fatalf("%s shapes = qweight %v qzeros %v scales %v g_idx %v", base, qweight.Shape, qzeros.Shape, scales.Shape, gidx.Shape)
		}
		checked++
	}
	if checked == 0 {
		t.Fatal("reference contains no GPTQ qweight tensors")
	}
	var config map[string]any
	read := core.ReadFile(core.PathJoin(dir, "quantize_config.json"))
	if !read.OK {
		t.Fatal(read.Err())
	}
	if decoded := core.JSONUnmarshal(read.Value.([]byte), &config); !decoded.OK {
		t.Fatal(decoded.Err())
	}
	for _, key := range []string{"bits", "group_size", "damp_percent", "desc_act", "sym", "true_sequential", "model_name_or_path", "model_file_base_name"} {
		if _, ok := config[key]; !ok {
			t.Fatalf("reference config lacks %q", key)
		}
	}
	t.Logf("validated %d GPTQ projections against the HF GPTQ schema", checked)
}
