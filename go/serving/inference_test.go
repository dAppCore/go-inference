package serving

import "dappco.re/go"

func TestInference_DefaultGenOpts_Good(t *core.T) {
	opts := DefaultGenOpts()
	core.AssertEqual(t, 0.1, opts.Temperature)
	core.AssertEqual(t, 0, opts.MaxTokens)
}

func TestInference_DefaultGenOpts_Bad(t *core.T) {
	opts := DefaultGenOpts()
	core.AssertEqual(t, "", opts.Model)
	core.AssertEmpty(t, opts.StopSequences)
}

func TestInference_DefaultGenOpts_Ugly(t *core.T) {
	opts := DefaultGenOpts()
	opts.MaxTokens = 1
	core.AssertEqual(t, 1, opts.MaxTokens)
}
