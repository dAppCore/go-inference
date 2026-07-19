// SPDX-License-Identifier: EUPL-1.2

package workspace

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference/agent/gitserver"
	"dappco.re/go/inference/agent/queue"
	"dappco.re/go/inference/agent/work"
	commandexec "dappco.re/go/process/exec"
)

// ChangeReview is the immutable receipt for one disposable integration attempt.
type ChangeReview struct {
	WorkID            string
	RunID             string
	SourceBranch      string
	SourceRevision    string
	AgentBase         string
	AgentTip          string
	IntegrationBranch string
	IntegrationPath   string
	ResultRevision    string
	Diff              string
	CommitLog         string
	Validation        []ValidationResult
	Conflicts         []string
}

// ValidationResult records one explicit validation argument vector and its result.
type ValidationResult struct {
	Command  Command
	ExitCode int
	Output   string
	Receipt  string
	Passed   bool
}

// AcceptRequest carries the reviewed receipt plus explicit final confirmation.
type AcceptRequest struct {
	Review    ChangeReview
	Project   work.Project
	Confirmed bool
}

type reviewState struct {
	review     ChangeReview
	sourcePath string
	clonePath  string
}

type application struct {
	manager  *Manager
	review   ChangeReview
	state    reviewState
	advanced bool
}

// ReviewChanges integrates the exact durable agent range in an internal worktree.
func (manager *Manager) ReviewChanges(ctx context.Context, project work.Project, run work.Run, validation []queue.Command) (result core.Result) {
	if manager == nil {
		return core.Fail(core.NewError("agent workspace manager is required"))
	}
	if ctx == nil {
		return core.Fail(core.NewError("agent workspace change review context is required"))
	}
	if err := ctx.Err(); err != nil {
		return core.Fail(core.E("workspace.Manager.ReviewChanges", "change review context is done", err))
	}
	manager.mu.Lock()
	defer manager.mu.Unlock()

	if run.Status != work.RunCompleted || core.Trim(run.WorkID) == "" || core.Trim(run.DurableRevision) == "" {
		return core.Fail(core.NewError("agent workspace change review requires a completed durable run"))
	}
	if run.ProjectID != project.ID || run.SourceRevision == "" || core.Trim(run.Branch) == "" {
		return core.Fail(core.NewError("agent workspace change review run identity is incomplete"))
	}
	if validated := manager.validateReviewProject(project); !validated.OK {
		return validated
	}
	sourceResult := manager.reviewSource(ctx, project.RepositoryRoot)
	if !sourceResult.OK {
		return sourceResult
	}
	source := sourceResult.Value.(SourceReview)
	if !source.Git || !source.Clean || source.Detached || source.Branch != project.SourceBranch {
		return core.Fail(core.NewError("agent workspace source must remain clean on its registered branch for change review"))
	}

	repositoryResult := manager.server.EnsureRepository(ctx, project.RepositoryName)
	if !repositoryResult.OK {
		return core.Fail(core.E("workspace.Manager.ReviewChanges", "failed to resolve private repository", repositoryResult.Err()))
	}
	repository, ok := repositoryResult.Value.(gitserver.Repository)
	if !ok {
		return core.Fail(core.Errorf("agent workspace Git service returned %T instead of repository", repositoryResult.Value))
	}
	environmentResult := repositoryEnvironment(repository)
	if !environmentResult.OK {
		return environmentResult
	}
	environment := environmentResult.Value.([]string)
	agentReference := core.Concat("refs/remotes/lem/", run.Branch)
	fetchedAgent := manager.gitOutput(ctx, manager.root, environment,
		"--git-dir", project.ClonePath, "fetch", "--no-tags", repository.CloneURL,
		core.Concat("+refs/heads/", run.Branch, ":", agentReference),
	)
	if !fetchedAgent.OK {
		return core.Fail(core.E("workspace.Manager.ReviewChanges", "failed to fetch durable agent branch", fetchedAgent.Err()))
	}
	verifiedTip := manager.gitOutput(ctx, manager.root, nil, "--git-dir", project.ClonePath, "rev-parse", core.Concat(agentReference, "^{commit}"))
	if !verifiedTip.OK || core.Trim(verifiedTip.Value.(string)) != run.DurableRevision {
		return core.Fail(core.NewError("agent workspace durable branch differs from the completed run receipt"))
	}
	if ancestor := manager.gitOutput(ctx, manager.root, nil, "--git-dir", project.ClonePath, "merge-base", "--is-ancestor", run.SourceRevision, run.DurableRevision); !ancestor.OK {
		return core.Fail(core.NewError("agent workspace durable revision is not descended from its immutable agent base"))
	}

	reviewIDResult := pathSegment("review ID", manager.ids())
	if !reviewIDResult.OK {
		return reviewIDResult
	}
	reviewID := reviewIDResult.Value.(string)
	runIDResult := pathSegment("run ID", run.ID)
	if !runIDResult.OK {
		return runIDResult
	}
	integrationBranch := core.Concat("lem/integration/", branchComponent(run.ID), "/", branchComponent(reviewID))
	integrationPathResult := manager.internalPath(project.ID, "reviews", run.ID, reviewID, "worktree")
	if !integrationPathResult.OK {
		return integrationPathResult
	}
	integrationPath := integrationPathResult.Value.(string)
	integrationRelativeResult := manager.internalRelative(integrationPath)
	if !integrationRelativeResult.OK {
		return integrationRelativeResult
	}
	if ensureErr := manager.files.EnsureDir(core.PathDir(integrationRelativeResult.Value.(string))); ensureErr != nil {
		return core.Fail(core.E("workspace.Manager.ReviewChanges", "failed to create integration directory", ensureErr))
	}
	sourceReference := core.Concat("refs/lem/source/", reviewID)
	fetchedSource := manager.gitOutput(ctx, manager.root, nil,
		"--git-dir", project.ClonePath, "fetch", "--no-tags", project.RepositoryRoot,
		core.Concat("+", source.Revision, ":", sourceReference),
	)
	if !fetchedSource.OK {
		return core.Fail(core.E("workspace.Manager.ReviewChanges", "failed to fetch reviewed source revision", fetchedSource.Err()))
	}
	added := manager.gitOutput(ctx, manager.root, nil,
		"--git-dir", project.ClonePath, "worktree", "add", "-b", integrationBranch, integrationPath, sourceReference,
	)
	if !added.OK {
		cleanup := manager.cleanupFailedWorktree(ctx, project.ClonePath, integrationPath, integrationRelativeResult.Value.(string), integrationBranch, true)
		if !cleanup.OK {
			return core.Fail(core.E("workspace.Manager.ReviewChanges", added.Error(), cleanup.Err()))
		}
		return core.Fail(core.E("workspace.Manager.ReviewChanges", "failed to create integration worktree", added.Err()))
	}
	manager.reviews[integrationPath] = provisionalReview{
		workID: run.WorkID, runID: run.ID, branch: integrationBranch, path: integrationPath,
		clone: project.ClonePath, relative: integrationRelativeResult.Value.(string),
	}
	provisional := true
	defer func() {
		if !provisional {
			return
		}
		cleanup := manager.cleanupFailedWorktree(ctx, project.ClonePath, integrationPath, integrationRelativeResult.Value.(string), integrationBranch, true)
		if cleanup.OK {
			return
		}
		if result.OK {
			result = cleanup
			return
		}
		result = core.Fail(core.E("workspace.Manager.ReviewChanges", result.Error(), cleanup.Err()))
	}()

	review := ChangeReview{
		WorkID: run.WorkID, RunID: run.ID, SourceBranch: source.Branch, SourceRevision: source.Revision,
		AgentBase: run.SourceRevision, AgentTip: run.DurableRevision, IntegrationBranch: integrationBranch,
		IntegrationPath: integrationPath,
	}
	commitLog := manager.gitOutput(ctx, manager.root, nil, "--git-dir", project.ClonePath,
		"log", "--reverse", "--format=%H%x09%s", core.Concat(run.SourceRevision, "..", run.DurableRevision))
	if !commitLog.OK {
		return core.Fail(core.E("workspace.Manager.ReviewChanges", "failed to read agent commit range", commitLog.Err()))
	}
	review.CommitLog = core.Trim(commitLog.Value.(string))

	if source.Revision == run.SourceRevision {
		integrated := manager.gitOutput(ctx, integrationPath, nil, "merge", "--ff-only", run.DurableRevision)
		if !integrated.OK {
			return core.Fail(core.E("workspace.Manager.ReviewChanges", "failed to fast-forward disposable integration", integrated.Err()))
		}
	} else {
		commitsResult := manager.gitOutput(ctx, manager.root, nil, "--git-dir", project.ClonePath,
			"rev-list", "--reverse", "--topo-order", core.Concat(run.SourceRevision, "..", run.DurableRevision))
		if !commitsResult.OK {
			return core.Fail(core.E("workspace.Manager.ReviewChanges", "failed to enumerate exact agent range", commitsResult.Err()))
		}
		commits := core.Fields(commitsResult.Value.(string))
		if len(commits) > 0 {
			arguments := append([]string{"cherry-pick"}, commits...)
			integrated := manager.gitOutput(ctx, integrationPath, nil, arguments...)
			if !integrated.OK {
				conflicts := manager.gitOutput(ctx, integrationPath, nil, "diff", "--name-only", "--diff-filter=U")
				if conflicts.OK {
					review.Conflicts = core.Fields(conflicts.Value.(string))
				}
				if len(review.Conflicts) == 0 {
					return core.Fail(core.E("workspace.Manager.ReviewChanges", "failed to replay exact agent range", integrated.Err()))
				}
				provisional = false
				return core.Ok(cloneChangeReview(review))
			}
		}
	}
	resultRevision := manager.gitOutput(ctx, integrationPath, nil, "rev-parse", "HEAD")
	if !resultRevision.OK {
		return core.Fail(core.E("workspace.Manager.ReviewChanges", "failed to inspect integration result", resultRevision.Err()))
	}
	review.ResultRevision = core.Trim(resultRevision.Value.(string))
	diff := manager.gitOutput(ctx, integrationPath, nil, "diff", "--binary", source.Revision, review.ResultRevision)
	if !diff.OK {
		return core.Fail(core.E("workspace.Manager.ReviewChanges", "failed to render integration diff", diff.Err()))
	}
	review.Diff = diff.Value.(string)
	review.Validation = runValidation(ctx, integrationPath, validation)
	provisional = false
	return core.Ok(cloneChangeReview(review))
}

// RetainReview transfers a provisional integration worktree to a durable review receipt.
func (manager *Manager) RetainReview(review ChangeReview) core.Result {
	if manager == nil {
		return core.Fail(core.NewError("agent workspace manager is required"))
	}
	manager.mu.Lock()
	defer manager.mu.Unlock()
	provisional, exists := manager.reviews[review.IntegrationPath]
	if !exists || !matchesProvisionalReview(provisional, review) {
		return core.Fail(core.NewError("agent workspace review does not match a provisional integration"))
	}
	delete(manager.reviews, review.IntegrationPath)
	return core.Ok(cloneChangeReview(review))
}

// AbandonReview removes a provisional integration whose durable receipt was not committed.
func (manager *Manager) AbandonReview(ctx context.Context, review ChangeReview) core.Result {
	if manager == nil {
		return core.Fail(core.NewError("agent workspace manager is required"))
	}
	if ctx == nil {
		return core.Fail(core.NewError("agent workspace review cleanup context is required"))
	}
	if err := ctx.Err(); err != nil {
		return core.Fail(core.E("workspace.Manager.AbandonReview", "review cleanup context is done", err))
	}
	manager.mu.Lock()
	defer manager.mu.Unlock()
	provisional, exists := manager.reviews[review.IntegrationPath]
	if !exists || !matchesProvisionalReview(provisional, review) {
		return core.Fail(core.NewError("agent workspace review does not match a provisional integration"))
	}
	return manager.cleanupFailedWorktree(ctx, provisional.clone, provisional.path, provisional.relative, provisional.branch, true)
}

func matchesProvisionalReview(provisional provisionalReview, review ChangeReview) bool {
	return provisional.workID == review.WorkID && provisional.runID == review.RunID &&
		provisional.branch == review.IntegrationBranch && provisional.path == review.IntegrationPath
}

// Apply advances the unchanged reviewed source to the already validated result.
func (manager *Manager) Apply(ctx context.Context, request AcceptRequest) core.Result {
	if manager == nil {
		return core.Fail(core.NewError("agent workspace manager is required"))
	}
	if ctx == nil {
		return core.Fail(core.NewError("agent workspace acceptance context is required"))
	}
	if !request.Confirmed {
		return core.Fail(core.NewError("agent workspace acceptance requires explicit final confirmation"))
	}
	if err := ctx.Err(); err != nil {
		return core.Fail(core.E("workspace.Manager.Apply", "acceptance context is done", err))
	}
	manager.mu.Lock()
	defer manager.mu.Unlock()
	verified := manager.verifyChangeReview(ctx, request.Project, request.Review)
	if !verified.OK {
		return verified
	}
	state := verified.Value.(reviewState)
	review := state.review
	if !completeChangeReview(review) || len(review.Conflicts) != 0 {
		return core.Fail(core.NewError("agent workspace acceptance review is incomplete or conflicted"))
	}
	for _, validation := range review.Validation {
		if !validation.Passed {
			return core.Fail(core.NewError("agent workspace acceptance requires every configured validation to pass"))
		}
	}
	sourceResult := manager.reviewSource(ctx, state.sourcePath)
	if !sourceResult.OK {
		return sourceResult
	}
	source := sourceResult.Value.(SourceReview)
	if !source.Clean || source.Detached || source.Branch != review.SourceBranch {
		return core.Fail(core.NewError("agent workspace stale review: source branch or cleanliness changed"))
	}
	if source.Revision == review.ResultRevision {
		return core.Ok(application{manager: manager, review: review, state: state})
	}
	if source.Revision != review.SourceRevision {
		return core.Fail(core.NewError("agent workspace stale review: source revision changed"))
	}
	fetched := manager.gitOutput(ctx, state.sourcePath, nil, "fetch", "--no-tags", state.clonePath, review.ResultRevision)
	if !fetched.OK {
		return core.Fail(core.E("workspace.Manager.Apply", "failed to fetch validated result after confirmation", fetched.Err()))
	}
	applied := manager.gitOutput(ctx, state.sourcePath, nil, "merge", "--ff-only", "FETCH_HEAD")
	if !applied.OK {
		return core.Fail(core.E("workspace.Manager.Apply", "failed to fast-forward source to validated result", applied.Err()))
	}
	return core.Ok(application{manager: manager, review: review, state: state, advanced: true})
}

// Reject acknowledges a reviewed result while retaining all internal Git history.
func (manager *Manager) Reject(ctx context.Context, review ChangeReview) core.Result {
	if manager == nil {
		return core.Fail(core.NewError("agent workspace manager is required"))
	}
	if ctx == nil {
		return core.Fail(core.NewError("agent workspace rejection context is required"))
	}
	if err := ctx.Err(); err != nil {
		return core.Fail(core.E("workspace.Manager.Reject", "rejection context is done", err))
	}
	if core.Trim(review.WorkID) == "" || core.Trim(review.RunID) == "" {
		return core.Fail(core.NewError("agent workspace rejection requires a reviewed Work and run"))
	}
	return core.Ok(review)
}

func (manager *Manager) validateReviewProject(project work.Project) core.Result {
	if core.Trim(project.ID) == "" || core.Trim(project.RepositoryRoot) == "" || core.Trim(project.SourceBranch) == "" || core.Trim(project.RepositoryName) == "" {
		return core.Fail(core.NewError("agent workspace change review project identity is incomplete"))
	}
	expectedClone := manager.internalPath(project.ID, "repo.git")
	if !expectedClone.OK || expectedClone.Value.(string) != project.ClonePath {
		return core.Fail(core.NewError("agent workspace change review cached clone is outside the internal root"))
	}
	return core.Ok(project)
}

func completeChangeReview(review ChangeReview) bool {
	return core.Trim(review.WorkID) != "" && core.Trim(review.RunID) != "" && core.Trim(review.SourceBranch) != "" &&
		core.Trim(review.SourceRevision) != "" && core.Trim(review.AgentBase) != "" && core.Trim(review.AgentTip) != "" &&
		core.Trim(review.IntegrationBranch) != "" && core.Trim(review.IntegrationPath) != "" &&
		core.Trim(review.ResultRevision) != ""
}

func cloneChangeReview(review ChangeReview) ChangeReview {
	cloned := review
	cloned.Validation = make([]ValidationResult, len(review.Validation))
	for index, validation := range review.Validation {
		cloned.Validation[index] = validation
		cloned.Validation[index].Command = Command{
			Dir: validation.Command.Dir, Executable: validation.Command.Executable,
			Args:        append([]string(nil), validation.Command.Args...),
			Environment: append([]string(nil), validation.Command.Environment...),
		}
	}
	cloned.Conflicts = append([]string(nil), review.Conflicts...)
	return cloned
}

func (manager *Manager) verifyChangeReview(ctx context.Context, project work.Project, supplied ChangeReview) core.Result {
	review := cloneChangeReview(supplied)
	if !completeChangeReview(review) || len(review.Conflicts) != 0 {
		return core.Fail(core.NewError("agent workspace acceptance review is incomplete or conflicted"))
	}
	if validated := manager.validateReviewProject(project); !validated.OK {
		return validated
	}
	if project.SourceBranch != review.SourceBranch {
		return core.Fail(core.NewError("agent workspace acceptance project branch differs from review"))
	}
	integrationPathResult := manager.internalAbsolute(review.IntegrationPath)
	if !integrationPathResult.OK || integrationPathResult.Value.(string) != review.IntegrationPath {
		return core.Fail(core.NewError("agent workspace acceptance integration path is outside the internal root"))
	}
	commonResult := manager.gitOutput(ctx, review.IntegrationPath, nil, "rev-parse", "--path-format=absolute", "--git-common-dir")
	if !commonResult.OK {
		return core.Fail(core.E("workspace.Manager.Apply", "failed to resolve integration repository", commonResult.Err()))
	}
	clonePath := core.Trim(commonResult.Value.(string))
	cloneResult := manager.internalAbsolute(clonePath)
	if !cloneResult.OK || cloneResult.Value.(string) != project.ClonePath || clonePath != project.ClonePath || core.PathBase(clonePath) != "repo.git" {
		return core.Fail(core.NewError("agent workspace acceptance cached clone is outside the internal root"))
	}
	branchResult := manager.gitOutput(ctx, review.IntegrationPath, nil, "symbolic-ref", "--short", "HEAD")
	headResult := manager.gitOutput(ctx, review.IntegrationPath, nil, "rev-parse", "HEAD")
	statusResult := manager.gitOutput(ctx, review.IntegrationPath, nil, "status", "--porcelain=v1", "--untracked-files=all")
	if !branchResult.OK || !headResult.OK || !statusResult.OK || core.Trim(branchResult.Value.(string)) != review.IntegrationBranch ||
		core.Trim(headResult.Value.(string)) != review.ResultRevision || core.Trim(statusResult.Value.(string)) != "" {
		return core.Fail(core.NewError("agent workspace acceptance integration Git facts changed"))
	}
	agentAncestor := manager.gitOutput(ctx, manager.root, nil, "--git-dir", clonePath, "merge-base", "--is-ancestor", review.AgentBase, review.AgentTip)
	resultAncestor := manager.gitOutput(ctx, manager.root, nil, "--git-dir", clonePath, "merge-base", "--is-ancestor", review.SourceRevision, review.ResultRevision)
	if !agentAncestor.OK || !resultAncestor.OK {
		return core.Fail(core.NewError("agent workspace acceptance revision ancestry changed"))
	}
	commitLog := manager.gitOutput(ctx, manager.root, nil, "--git-dir", clonePath,
		"log", "--reverse", "--format=%H%x09%s", core.Concat(review.AgentBase, "..", review.AgentTip))
	diff := manager.gitOutput(ctx, review.IntegrationPath, nil, "diff", "--binary", review.SourceRevision, review.ResultRevision)
	if !commitLog.OK || !diff.OK || core.Trim(commitLog.Value.(string)) != review.CommitLog || diff.Value.(string) != review.Diff {
		return core.Fail(core.NewError("agent workspace acceptance commit or diff receipt changed"))
	}
	return core.Ok(reviewState{review: cloneChangeReview(review), sourcePath: project.RepositoryRoot, clonePath: clonePath})
}

func runValidation(ctx context.Context, directory string, configured []queue.Command) []ValidationResult {
	results := make([]ValidationResult, 0, len(configured))
	for _, configuredCommand := range configured {
		command := Command{Dir: directory, Executable: core.Trim(configuredCommand.Command), Args: append([]string(nil), configuredCommand.Args...)}
		argv := append([]string{command.Executable}, command.Args...)
		receipt := core.JSONMarshalString(argv)
		stdout := core.NewBuffer()
		stderr := core.NewBuffer()
		run := commandexec.Command(ctx, command.Executable, command.Args...).WithDir(directory).WithStdout(stdout).WithStderr(stderr).Run()
		exitCode := 0
		if !run.OK {
			exitCode = -1
			var exit interface{ ExitCode() int }
			if core.As(run.Err(), &exit) {
				exitCode = exit.ExitCode()
			}
		}
		output := core.Concat(stdout.String(), stderr.String())
		if !run.OK && output == "" {
			output = run.Error()
		}
		results = append(results, ValidationResult{Command: command, ExitCode: exitCode, Output: output, Receipt: receipt, Passed: run.OK})
	}
	return results
}
