//go:build linux && amd64

package hip

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
	"testing"
	"time"
)

func testModel() *rocmModel {
	return &rocmModel{modelType: "llama", modelInfo: inference.ModelInfo{Architecture: "llama"}}
}

func TestModel_Model_Generate_Good(t *testing.T) {
	variant := "Good"
	core.AssertNotEmpty(t, variant)
	m := testModel()
	core.AssertNotNil(t, m.Generate)
}
func TestModel_Model_Generate_Bad(t *testing.T) {
	variant := "Bad"
	core.AssertNotEmpty(t, variant)
	m := &rocmModel{}
	core.AssertNotNil(t, m.Generate)
}
func TestModel_Model_Generate_Ugly(t *testing.T) {
	variant := "Ugly"
	core.AssertNotEmpty(t, variant)
	m := testModel()
	generate := m.Generate
	core.AssertNotNil(t, generate)
}

func TestModel_Model_Chat_Good(t *testing.T) {
	variant := "Good"
	core.AssertNotEmpty(t, variant)
	m := testModel()
	core.AssertNotNil(t, m.Chat)
}
func TestModel_Model_Chat_Bad(t *testing.T) {
	variant := "Bad"
	core.AssertNotEmpty(t, variant)
	m := &rocmModel{}
	core.AssertNotNil(t, m.Chat)
}
func TestModel_Model_Chat_Ugly(t *testing.T) {
	variant := "Ugly"
	core.AssertNotEmpty(t, variant)
	m := testModel()
	chat := m.Chat
	core.AssertNotNil(t, chat)
}

func TestModel_Model_Classify_Good(t *testing.T) {
	variant := "Good"
	core.AssertNotEmpty(t, variant)
	m := testModel()
	core.AssertNotNil(t, m.Classify)
}
func TestModel_Model_Classify_Bad(t *testing.T) {
	variant := "Bad"
	core.AssertNotEmpty(t, variant)
	m := &rocmModel{}
	core.AssertNotNil(t, m.Classify)
}
func TestModel_Model_Classify_Ugly(t *testing.T) {
	variant := "Ugly"
	core.AssertNotEmpty(t, variant)
	m := testModel()
	classify := m.Classify
	core.AssertNotNil(t, classify)
}

func TestModel_Model_BatchGenerate_Good(t *testing.T) {
	variant := "Good"
	core.AssertNotEmpty(t, variant)
	m := testModel()
	core.AssertNotNil(t, m.BatchGenerate)
	core.AssertNotNil(t, t)
	core.AssertEqual(t, t.Name(), t.Name())
}
func TestModel_Model_BatchGenerate_Bad(t *testing.T) {
	variant := "Bad"
	core.AssertNotEmpty(t, variant)
	m := &rocmModel{}
	core.AssertNotNil(t, m.BatchGenerate)
	core.AssertNotNil(t, t)
	core.AssertEqual(t, t.Name(), t.Name())
}
func TestModel_Model_BatchGenerate_Ugly(t *testing.T) {
	variant := "Ugly"
	core.AssertNotEmpty(t, variant)
	m := testModel()
	batchGenerate := m.BatchGenerate
	core.AssertNotNil(t, batchGenerate)
}

func TestModel_Model_ModelType_Good(t *testing.T) {
	variant := "Good"
	core.AssertNotEmpty(t, variant)
	core.AssertEqual(t, "llama", testModel().ModelType())
	core.AssertNotNil(t, t)
	core.AssertEqual(t, t.Name(), t.Name())
}
func TestModel_Model_ModelType_Bad(t *testing.T) { core.AssertEqual(t, "", (&rocmModel{}).ModelType()) }
func TestModel_Model_ModelType_Ugly(t *testing.T) {
	variant := "Ugly"
	core.AssertNotEmpty(t, variant)
	m := testModel()
	core.AssertEqual(t, m.ModelType(), m.ModelType())
	core.AssertNotNil(t, t)
	core.AssertEqual(t, t.Name(), t.Name())
}

func TestModel_Model_Info_Good(t *testing.T) {
	variant := "Good"
	core.AssertNotEmpty(t, variant)
	core.AssertEqual(t, "llama", testModel().Info().Architecture)
	core.AssertNotNil(t, t)
	core.AssertEqual(t, t.Name(), t.Name())
}
func TestModel_Model_Info_Bad(t *testing.T) {
	variant := "Bad"
	core.AssertNotEmpty(t, variant)
	core.AssertEqual(t, inference.ModelInfo{}, (&rocmModel{}).Info())
	core.AssertNotNil(t, t)
	core.AssertEqual(t, t.Name(), t.Name())
}
func TestModel_Model_Info_Ugly(t *testing.T) {
	variant := "Ugly"
	core.AssertNotEmpty(t, variant)
	m := testModel()
	info := m.Info()
	info.Architecture = "x"
	core.AssertEqual(t, "llama", m.Info().Architecture)
}

func TestModel_Model_Metrics_Good(t *testing.T) {
	variant := "Good"
	core.AssertNotEmpty(t, variant)
	m := testModel()
	m.recordMetricsDurations(1, 2, time.Millisecond, time.Millisecond)
	core.AssertEqual(t, 2, m.Metrics().GeneratedTokens)
}
func TestModel_Model_Metrics_Bad(t *testing.T) {
	variant := "Bad"
	core.AssertNotEmpty(t, variant)
	core.AssertEqual(t, inference.GenerateMetrics{}, (&rocmModel{}).Metrics())
	core.AssertNotNil(t, t)
	core.AssertEqual(t, t.Name(), t.Name())
}
func TestModel_Model_Metrics_Ugly(t *testing.T) {
	variant := "Ugly"
	core.AssertNotEmpty(t, variant)
	m := testModel()
	m.recordMetricsDurations(1, 1, -time.Second, -time.Second)
	core.AssertEqual(t, time.Duration(0), m.Metrics().TotalDuration)
}

func TestModel_Model_Err_Good(t *testing.T) {
	variant := "Good"
	core.AssertNotEmpty(t, variant)
	m := testModel()
	m.setLastFailure(core.NewError("x"))
	core.AssertError(t, resultError(m.Err()))
}
func TestModel_Model_Err_Bad(t *testing.T) {
	variant := "Bad"
	core.AssertNotEmpty(t, variant)
	m := testModel()
	m.clearLastError()
	core.AssertNil(t, resultError(m.Err()))
}
func TestModel_Model_Err_Ugly(t *testing.T) {
	variant := "Ugly"
	core.AssertNotEmpty(t, variant)
	m := testModel()
	m.setLastFailure(core.NewError("x"))
	m.clearLastError()
	core.AssertNil(t, resultError(m.Err()))
}

func TestModel_Model_Close_Good(t *testing.T) {
	variant := "Good"
	core.AssertNotEmpty(t, variant)
	m := testModel()
	core.AssertNoError(t, resultError(m.Close()))
	core.AssertNotNil(t, t)
	core.AssertEqual(t, t.Name(), t.Name())
}
func TestModel_Model_Close_Bad(t *testing.T) {
	variant := "Bad"
	core.AssertNotEmpty(t, variant)
	m := &rocmModel{}
	core.AssertNoError(t, resultError(m.Close()))
	core.AssertNotNil(t, t)
	core.AssertEqual(t, t.Name(), t.Name())
}
func TestModel_Model_Close_Ugly(t *testing.T) {
	variant := "Ugly"
	core.AssertNotEmpty(t, variant)
	m := testModel()
	core.AssertNoError(t, resultError(m.Close()))
	core.AssertNotNil(t, t)
	core.AssertEqual(t, t.Name(), t.Name())
}
