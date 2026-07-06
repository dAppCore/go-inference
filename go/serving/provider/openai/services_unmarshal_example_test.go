// SPDX-Licence-Identifier: EUPL-1.2

package openai

import core "dappco.re/go"

func ExampleEmbeddingRequest_UnmarshalJSON() {
	var req EmbeddingRequest
	in := []byte(`{"model":"text-embedding","input":["a","b"]}`)

	if err := req.UnmarshalJSON(in); err != nil {
		core.Println(err)
		return
	}

	core.Println(req.Model)
	core.Println(len(req.Input))
	// Output:
	// text-embedding
	// 2
}

func ExampleRerankRequest_UnmarshalJSON() {
	var req RerankRequest
	in := []byte(`{"model":"rerank","query":"q","documents":["a","b"]}`)

	if err := req.UnmarshalJSON(in); err != nil {
		core.Println(err)
		return
	}

	core.Println(req.Query)
	core.Println(len(req.Documents))
	// Output:
	// q
	// 2
}

func ExampleCancelRequest_UnmarshalJSON() {
	var req CancelRequest
	in := []byte(`{"model":"qwen","id":"req_1"}`)

	if err := req.UnmarshalJSON(in); err != nil {
		core.Println(err)
		return
	}

	core.Println(req.ID)
	// Output:
	// req_1
}

func ExampleCacheClearRequest_UnmarshalJSON() {
	var req CacheClearRequest
	in := []byte(`{"model":"qwen","labels":{"adapter":"none"}}`)

	if err := req.UnmarshalJSON(in); err != nil {
		core.Println(err)
		return
	}

	core.Println(req.Labels["adapter"])
	// Output:
	// none
}

func ExampleCacheWarmRequest_UnmarshalJSON() {
	var req CacheWarmRequest
	in := []byte(`{"model":"qwen","tokens":[1,2,3]}`)

	if err := req.UnmarshalJSON(in); err != nil {
		core.Println(err)
		return
	}

	core.Println(len(req.Tokens))
	// Output:
	// 3
}
