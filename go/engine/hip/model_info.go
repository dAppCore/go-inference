// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/inference/engine/hip/model"
)

const ROCmModelInfoReporterContract = rocmmodel.ModelInfoReporterContract

type ROCmModelInfoReporter = rocmmodel.ModelInfoReporter
type ROCmModelInfoRequest = rocmmodel.ModelInfoRequest
type ROCmModelInfoReport = rocmmodel.ModelInfoReport

// ResolveROCmModelInfo resolves architecture metadata through the same
// model-owned reporter contract used by loaded ROCm models.
func ResolveROCmModelInfo(req ROCmModelInfoRequest) ROCmModelInfoReport {
	return rocmmodel.ResolveModelInfo(req)
}

// ROCmModelInfoFromIdentity converts a backend-neutral identity into the small
// go-inference ModelInfo shape after ROCm architecture normalization.
func ROCmModelInfoFromIdentity(path string, identity inference.ModelIdentity) inference.ModelInfo {
	return rocmmodel.ModelInfoFromIdentity(path, identity)
}

// ROCmModelInfoIdentity converts ModelInfo plus labels into the richer identity
// shape used by registry and route planning.
func ROCmModelInfoIdentity(path string, info inference.ModelInfo, labels map[string]string) inference.ModelIdentity {
	return rocmmodel.ModelInfoIdentity(path, info, labels)
}

// ROCmModelInfoReportForModel resolves model-info metadata from a loaded text
// model. Model-owned identity and info reporters are used when present so
// wrappers can stay reactive without concrete ROCm type switches.
func ROCmModelInfoReportForModel(model inference.TextModel) (ROCmModelInfoReport, bool) {
	if model == nil {
		return ROCmModelInfoReport{}, false
	}
	identity := inference.ModelIdentity{}
	if reporter, ok := model.(ROCmModelIdentityReporter); ok {
		identity = reporter.ModelIdentity()
	}
	labels := cloneStringMap(identity.Labels)
	reporter, _ := model.(ROCmModelInfoReporter)
	report := ResolveROCmModelInfo(ROCmModelInfoRequest{
		Path:      identity.Path,
		ModelType: model.ModelType(),
		Info:      model.Info(),
		Identity:  identity,
		Labels:    labels,
		Reporter:  reporter,
	})
	if !report.Matched() {
		return ROCmModelInfoReport{}, false
	}
	return report.Clone(), true
}
