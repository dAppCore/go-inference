// SPDX-Licence-Identifier: EUPL-1.2

package awq

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

// TestReferenceSchema compares the writer contract with the real
// TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ Hugging Face checkpoint.
func TestReferenceSchema(t *testing.T) {
	dir := core.Getenv("LEM_AWQ_REFERENCE_DIR")
	if core.Trim(dir) == "" {
		t.Skip("set LEM_AWQ_REFERENCE_DIR to the TinyLlama AWQ snapshot")
	}
	files := core.PathGlob(core.PathJoin(dir, "*.safetensors"))
	idx, err := safetensors.IndexFiles(files)
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
		if !zeroOK || !scaleOK {
			t.Fatalf("%s lacks AWQ companions", base)
		}
		if qweight.DType != "I32" || qzeros.DType != "I32" || scales.DType != "F16" {
			t.Fatalf("%s dtypes = %s/%s/%s", base, qweight.DType, qzeros.DType, scales.DType)
		}
		if len(qweight.Shape) != 2 || len(qzeros.Shape) != 2 || len(scales.Shape) != 2 {
			t.Fatalf("%s ranks do not match AWQ schema", base)
		}
		if qweight.Shape[0] != scales.Shape[0]*128 || qweight.Shape[1]*8 != scales.Shape[1] || qzeros.Shape[0] != scales.Shape[0] || qzeros.Shape[1] != qweight.Shape[1] {
			t.Fatalf("%s shapes = qweight %v qzeros %v scales %v", base, qweight.Shape, qzeros.Shape, scales.Shape)
		}
		checked++
	}
	if checked == 0 {
		t.Fatal("reference contains no AWQ qweight tensors")
	}
	var config map[string]any
	read := core.ReadFile(core.PathJoin(dir, "quant_config.json"))
	if !read.OK {
		read = core.ReadFile(core.PathJoin(dir, "quantize_config.json"))
	}
	if !read.OK {
		t.Fatal(read.Err())
	}
	if decoded := core.JSONUnmarshal(read.Value.([]byte), &config); !decoded.OK {
		t.Fatal(decoded.Err())
	}
	for _, key := range []string{"zero_point", "q_group_size", "w_bit", "version", "modules_to_not_convert"} {
		if _, ok := config[key]; !ok {
			t.Fatalf("reference config lacks %q", key)
		}
	}
	t.Logf("validated %d AWQ projections against TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ", checked)
}
