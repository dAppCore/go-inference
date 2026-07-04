// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"os"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func Example_bertClassifierLoRAAdapter() {
	os.Setenv("GO_ROCM_KERNEL_HSACO", "fake-example.hsaco")
	defer os.Unsetenv("GO_ROCM_KERNEL_HSACO")

	dir, err := os.MkdirTemp("", "go-rocm-bert-lora-example-*")
	if err != nil {
		core.Println("tempdir")
		return
	}
	defer os.RemoveAll(dir)

	embeddingPayload, err := hipFloat32Payload([]float32{
		0, 0,
		0, 1,
		0, 0,
		0, 1,
		1, 0,
	})
	if err != nil {
		core.Println("embedding")
		return
	}
	classifierPayload, err := hipFloat32Payload([]float32{0, 0, 0, 1})
	if err != nil {
		core.Println("classifier")
		return
	}
	modelPath := core.PathJoin(dir, "bert-classifier.bin")
	if write := core.WriteFile(modelPath, append(append([]byte(nil), embeddingPayload...), classifierPayload...), 0o644); !write.OK {
		core.Println("model")
		return
	}
	model, err := newHIPRuntime(&fakeHIPDriver{available: true}).LoadModel(modelPath, nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: "bert", VocabSize: 5, HiddenSize: 2, QuantBits: 32},
		Tensors: []nativeTensorInfo{{
			Name:       "embeddings.word_embeddings.weight",
			Type:       0,
			Dimensions: []uint64{5, 2},
			Offset:     0,
			ByteSize:   uint64(len(embeddingPayload)),
		}, {
			Name:       "classifier.weight",
			Type:       0,
			Dimensions: []uint64{2, 2},
			Offset:     uint64(len(embeddingPayload)),
			ByteSize:   uint64(len(classifierPayload)),
		}},
	})
	if err != nil {
		core.Println("load")
		return
	}
	defer model.Close()

	adapterPath := core.PathJoin(dir, "rocm_classifier_lora.json")
	if write := core.WriteFile(adapterPath, []byte(`{
		"format":"rocm-classifier-lora",
		"target":"classifier.weight",
		"rank":1,
		"alpha":1,
		"hidden_size":2,
		"num_labels":2,
		"lora_a":[1,0],
		"lora_b":[0,4]
	}`), 0o644); !write.OK {
		core.Println("adapter")
		return
	}
	loaded, ok := model.(*hipLoadedModel)
	if !ok {
		core.Println("type")
		return
	}
	identity, err := loaded.LoadAdapter(adapterPath)
	if err != nil {
		core.Println("adapter-load")
		return
	}
	reranked, err := loaded.Rerank(context.Background(), inference.RerankRequest{
		Query:     "hello",
		Documents: []string{"hello world", "hello"},
		TopN:      1,
	})
	if err != nil {
		core.Println("rerank")
		return
	}
	core.Println(identity.Format)
	core.Println(identity.Labels["adapter_runtime"])
	core.Println(reranked.Results[0].Index)
	core.Println(reranked.Labels["lora_kernel_name"])
	// Output:
	// rocm-classifier-lora
	// hip_bert_classifier
	// 0
	// rocm_lora_projection
}
