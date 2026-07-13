// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/inference/model"
)

func TestMTPDiagBF16Stats_Good(t *testing.T) {
	got := mtpDiagBF16Stats(toBF16Bytes([]float32{-2.5, 0.25, 1.75}))
	if got != "sum|x|=4.5 max|x|=2.500 nan=0 n=3" {
		t.Fatalf("mtpDiagBF16Stats() = %q", got)
	}
}

func TestMTPDiagBF16Stats_Ugly(t *testing.T) {
	got := mtpDiagBF16Stats([]byte{0x80})
	if got != "sum|x|=0.0 max|x|=0.000 nan=0 n=0" {
		t.Fatalf("mtpDiagBF16Stats(odd byte) = %q", got)
	}
}

func TestMTPDiagBF16Stats_NaN(t *testing.T) {
	got := mtpDiagBF16Stats([]byte{0xc1, 0x7f, 0x00, 0x40})
	if got != "sum|x|=2.0 max|x|=2.000 nan=1 n=2" {
		t.Fatalf("mtpDiagBF16Stats(NaN) = %q", got)
	}
}

func TestMTPDecode_Bad(t *testing.T) {
	if _, err := MTPDecode(nil, &ArchSession{}, []int32{3}, 1, -1, 1); err == nil {
		t.Fatal("MTPDecode(nil target) error = nil")
	}
}

func TestMTPDecode_Good(t *testing.T) {
	if err := ensureInit(); err != nil {
		t.Skipf("device init: %v", err)
	}
	mk := newMTPDecodeFixture(t)
	res, err := MTPDecode(mk(), mk(), []int32{2, 18, 7}, 2, -1, 1)
	if err != nil {
		t.Fatalf("MTPDecode: %v", err)
	}
	if len(res.Tokens) != 2 || res.Rounds == 0 {
		t.Fatalf("MTPDecode result = %+v", res)
	}
}

func TestMTPDecode_Ugly(t *testing.T) {
	session := &ArchSession{maxLen: 8}
	if _, err := MTPDecode(session, session, nil, 1, -1, 1); err == nil {
		t.Fatal("MTPDecode(empty prompt) error = nil")
	}
}

func TestMTPDecodeEach_Bad(t *testing.T) {
	session := &ArchSession{maxLen: 8}
	if _, err := MTPDecodeEach(session, session, []int32{3}, 0, -1, 1, nil); err == nil {
		t.Fatal("MTPDecodeEach(maxNew=0) error = nil")
	}
}

func TestMTPDecodeEach_Good(t *testing.T) {
	if err := ensureInit(); err != nil {
		t.Skipf("device init: %v", err)
	}
	mk := newMTPDecodeFixture(t)
	yielded := make([]int32, 0, 2)
	res, err := MTPDecodeEach(mk(), mk(), []int32{2, 18, 7}, 2, -1, 1, func(id int32) bool {
		yielded = append(yielded, id)
		return true
	})
	if err != nil {
		t.Fatalf("MTPDecodeEach: %v", err)
	}
	if len(res.Tokens) != 2 || len(yielded) != len(res.Tokens) {
		t.Fatalf("tokens = %v, yielded = %v", res.Tokens, yielded)
	}
}

func TestMTPDecodeEach_Ugly(t *testing.T) {
	session := &ArchSession{maxLen: 1}
	if _, err := MTPDecodeEach(session, session, []int32{3}, 1, -1, 1, nil); err == nil {
		t.Fatal("MTPDecodeEach(cache overflow) error = nil")
	}
}

func TestMTPDecodeSampled_Bad(t *testing.T) {
	session := &ArchSession{maxLen: 8}
	if _, err := MTPDecodeSampled(session, session, []int32{3}, 1, nil, nil, model.NewSampler(2), model.SampleParams{}, 1); err == nil {
		t.Fatal("MTPDecodeSampled(nil target sampler) error = nil")
	}
}

func TestMTPDecodeSampled_Good(t *testing.T) {
	if err := ensureInit(); err != nil {
		t.Skipf("device init: %v", err)
	}
	mk := newMTPDecodeFixture(t)
	res, err := MTPDecodeSampled(mk(), mk(), []int32{2, 18, 7}, 2, nil, model.NewSampler(41), model.NewSampler(43), model.SampleParams{Temperature: 0}, 1)
	if err != nil {
		t.Fatalf("MTPDecodeSampled: %v", err)
	}
	if len(res.Tokens) != 2 || res.Rounds == 0 {
		t.Fatalf("MTPDecodeSampled result = %+v", res)
	}
}

func TestMTPDecodeSampled_Ugly(t *testing.T) {
	session := &ArchSession{maxLen: 8}
	sampler := model.NewSampler(7)
	if _, err := MTPDecodeSampled(session, session, []int32{3}, 1, nil, sampler, sampler, model.SampleParams{}, 1); err == nil {
		t.Fatal("MTPDecodeSampled(shared sampler) error = nil")
	}
}

func TestMTPDecodeSampledEach_Bad(t *testing.T) {
	session := &ArchSession{maxLen: 8}
	if _, err := MTPDecodeSampledEach(session, session, []int32{3}, 1, nil, model.NewSampler(1), nil, model.SampleParams{}, 1, nil); err == nil {
		t.Fatal("MTPDecodeSampledEach(nil draft sampler) error = nil")
	}
}

func TestMTPDecodeSampledEach_Good(t *testing.T) {
	if err := ensureInit(); err != nil {
		t.Skipf("device init: %v", err)
	}
	mk := newMTPDecodeFixture(t)
	yielded := make([]int32, 0, 2)
	res, err := MTPDecodeSampledEach(mk(), mk(), []int32{2, 18, 7}, 2, nil, model.NewSampler(47), model.NewSampler(53), model.SampleParams{Temperature: 0}, 1, func(id int32) bool {
		yielded = append(yielded, id)
		return true
	})
	if err != nil {
		t.Fatalf("MTPDecodeSampledEach: %v", err)
	}
	if len(res.Tokens) != 2 || len(yielded) != len(res.Tokens) {
		t.Fatalf("tokens = %v, yielded = %v", res.Tokens, yielded)
	}
}

func TestMTPDecodeSampledEach_Ugly(t *testing.T) {
	session := &ArchSession{maxLen: 1}
	if _, err := MTPDecodeSampledEach(session, session, []int32{3}, 1, nil, model.NewSampler(1), model.NewSampler(2), model.SampleParams{}, 1, nil); err == nil {
		t.Fatal("MTPDecodeSampledEach(cache overflow) error = nil")
	}
}

func TestEmitMTPToken_Good(t *testing.T) {
	res := &MTPResult{}
	if !emitMTPToken(res, nil, 17) || len(res.Tokens) != 1 || res.Tokens[0] != 17 {
		t.Fatalf("emitMTPToken() = %v, tokens = %v", true, res.Tokens)
	}
}

func TestEmitMTPToken_Bad(t *testing.T) {
	res := &MTPResult{}
	if emitMTPToken(res, func(id int32) bool { return id == 99 }, 23) {
		t.Fatal("emitMTPToken(rejecting yield) = true")
	}
	if len(res.Tokens) != 1 || res.Tokens[0] != 23 {
		t.Fatalf("tokens = %v, want [23]", res.Tokens)
	}
}

func TestArchSessionMTPSamplePickParams_Good(t *testing.T) {
	session := &ArchSession{}
	params := model.SampleParams{MinTokensBeforeStop: 3, SuppressTokens: []int32{4}}
	got := session.mtpSamplePickParams(params, []int32{7, 9}, 1)
	if len(got.SuppressTokens) != 3 || got.SuppressTokens[0] != 4 || got.SuppressTokens[1] != 7 || got.SuppressTokens[2] != 9 {
		t.Fatalf("SuppressTokens = %v", got.SuppressTokens)
	}
}

func TestArchSessionMTPSamplePickParams_Ugly(t *testing.T) {
	session := &ArchSession{}
	params := model.SampleParams{MinTokensBeforeStop: 2, SuppressTokens: []int32{5}}
	got := session.mtpSamplePickParams(params, []int32{8}, 2)
	if len(got.SuppressTokens) != 1 || got.SuppressTokens[0] != 5 {
		t.Fatalf("SuppressTokens at boundary = %v", got.SuppressTokens)
	}
}
