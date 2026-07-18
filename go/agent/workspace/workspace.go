// SPDX-License-Identifier: EUPL-1.2

// Package workspace owns source review and isolated Git worktrees for agents.
package workspace

import (
	"context"
	"sync"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/agent/gitserver"
	"dappco.re/go/inference/agent/work"
	coreio "dappco.re/go/io"
)

const (
	lemIdentityName  = "LEM Agent"
	lemIdentityEmail = "lem@localhost"
)

// SourceReview is a mutation-free description of a selected source directory.
type SourceReview struct {
	Path           string
	Root           string
	Branch         string
	ProposedBranch string
	Revision       string
	CommitIdentity string
	Git            bool
	Clean          bool
	Detached       bool
	Included       []string
	IncludedHash   string

	identityName  string
	identityEmail string
}

// RegisterRequest confirms source review and private-repository registration.
type RegisterRequest struct {
	ProjectID            string
	SourcePath           string
	RepositoryName       string
	EnableGit            bool
	Confirmed            bool
	ExpectedIncludedHash string
}

// RunWorkspace is an isolated checkout leased to one native provider attempt.
type RunWorkspace struct {
	Project      work.Project
	RunID        string
	Branch       string
	Path         string
	BaseRevision string
}

// Capture reports whether run changes were committed and durably pushed.
type Capture struct {
	Revision string
	Changed  bool
	Pushed   bool
	Retained bool
	Summary  string
}

// ManagerOptions injects the local medium, Git runner, and private Git service.
type ManagerOptions struct {
	Root   string
	Files  coreio.Medium
	Git    Runner
	Server gitserver.Service
	IDs    func() string
	Now    func() time.Time
}

// Manager owns cached repositories and per-run worktree leases.
type Manager struct {
	root   string
	files  coreio.Medium
	git    Runner
	server gitserver.Service
	ids    func() string
	now    func() time.Time

	mu      sync.Mutex
	leases  map[string]RunWorkspace
	durable map[string]bool
}

// NewManager validates the internal workspace boundary and its dependencies.
func NewManager(options ManagerOptions) core.Result {
	if options.Files == nil || options.Git == nil || options.Server == nil || options.IDs == nil || options.Now == nil {
		return core.Fail(core.NewError("agent workspace manager requires files, Git, server, IDs, and clock dependencies"))
	}
	rootResult := canonicalDirectory(options.Root)
	if !rootResult.OK {
		return core.Fail(core.E("workspace.NewManager", "invalid internal workspace root", rootResult.Err()))
	}
	root := rootResult.Value.(string)
	if ensureErr := options.Files.EnsureDir(""); ensureErr != nil {
		return core.Fail(core.E("workspace.NewManager", "failed to open internal workspace medium", ensureErr))
	}
	return core.Ok(&Manager{
		root:    root,
		files:   options.Files,
		git:     options.Git,
		server:  options.Server,
		ids:     options.IDs,
		now:     options.Now,
		leases:  make(map[string]RunWorkspace),
		durable: make(map[string]bool),
	})
}

// ReviewSource inspects a Git repository or ad-hoc folder without mutating it.
func (manager *Manager) ReviewSource(ctx context.Context, sourcePath string) core.Result {
	if manager == nil {
		return core.Fail(core.NewError("agent workspace manager is required"))
	}
	if ctx == nil {
		return core.Fail(core.NewError("agent workspace review context is required"))
	}
	manager.mu.Lock()
	defer manager.mu.Unlock()
	return manager.reviewSource(ctx, sourcePath)
}

// Register seeds a confirmed source revision into private Git and a cached clone.
func (manager *Manager) Register(ctx context.Context, request RegisterRequest) core.Result {
	if manager == nil {
		return core.Fail(core.NewError("agent workspace manager is required"))
	}
	if ctx == nil {
		return core.Fail(core.NewError("agent workspace registration context is required"))
	}
	if err := ctx.Err(); err != nil {
		return core.Fail(core.E("workspace.Manager.Register", "registration context is done", err))
	}
	manager.mu.Lock()
	defer manager.mu.Unlock()

	projectIDResult := pathSegment("project ID", request.ProjectID)
	if !projectIDResult.OK {
		return projectIDResult
	}
	projectID := projectIDResult.Value.(string)
	repositoryName := core.Trim(request.RepositoryName)
	if repositoryName == "" {
		return core.Fail(core.NewError("agent workspace repository name is required"))
	}
	if !request.Confirmed {
		return core.Fail(core.NewError("agent workspace registration requires explicit confirmation"))
	}
	expectedHash := core.Trim(request.ExpectedIncludedHash)
	if expectedHash == "" {
		return core.Fail(core.NewError("agent workspace registration requires the reviewed inclusion hash"))
	}

	reviewResult := manager.reviewSource(ctx, request.SourcePath)
	if !reviewResult.OK {
		return reviewResult
	}
	review := reviewResult.Value.(SourceReview)
	if review.IncludedHash != expectedHash {
		return core.Fail(core.NewError("agent workspace source changed after review; review it again before registration"))
	}
	if review.Git {
		if request.EnableGit {
			return core.Fail(core.NewError("agent workspace source is already a Git repository"))
		}
		if !review.Clean {
			return core.Fail(core.NewError("agent workspace source repository must be clean before registration"))
		}
		if review.Detached || review.Branch == "" {
			return core.Fail(core.NewError("agent workspace source repository must be on a branch before registration"))
		}
	} else {
		if !request.EnableGit {
			return core.Fail(core.NewError("agent workspace ad-hoc source requires confirmed Git enablement"))
		}
		initialized := manager.initializeAdHoc(ctx, review, projectID)
		if !initialized.OK {
			return initialized
		}
		reviewResult = manager.reviewSource(ctx, review.Root)
		if !reviewResult.OK {
			return reviewResult
		}
		review = reviewResult.Value.(SourceReview)
		if !review.Git || !review.Clean || review.Detached || review.IncludedHash != expectedHash {
			return core.Fail(core.NewError("agent workspace ad-hoc baseline does not match the confirmed review"))
		}
	}

	if started := manager.server.Start(ctx); !started.OK {
		return core.Fail(core.E("workspace.Manager.Register", "failed to start private Git", started.Err()))
	}
	repositoryResult := manager.server.EnsureRepository(ctx, repositoryName)
	if !repositoryResult.OK {
		return core.Fail(core.E("workspace.Manager.Register", "failed to ensure private repository", repositoryResult.Err()))
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
	seeded := manager.gitOutput(ctx, review.Root, environment,
		"push", "--force", repository.CloneURL,
		core.Concat(review.Revision, ":refs/heads/", review.Branch),
	)
	if !seeded.OK {
		return core.Fail(core.E("workspace.Manager.Register", "failed to seed private repository", seeded.Err()))
	}

	projectDirectory := projectID
	if ensureErr := manager.files.EnsureDir(projectDirectory); ensureErr != nil {
		return core.Fail(core.E("workspace.Manager.Register", "failed to create cached project directory", ensureErr))
	}
	clonePathResult := manager.internalPath(projectID, "repo.git")
	if !clonePathResult.OK {
		return clonePathResult
	}
	clonePath := clonePathResult.Value.(string)
	cloneRelative := core.PathJoin(projectID, "repo.git")
	if manager.files.Exists(cloneRelative) {
		bare := manager.gitOutput(ctx, manager.root, environment, "--git-dir", clonePath, "rev-parse", "--is-bare-repository")
		if !bare.OK || core.Trim(bare.Value.(string)) != "true" {
			return core.Fail(core.NewError("agent workspace cached repository is not a valid bare Git repository"))
		}
	} else {
		cloned := manager.gitOutput(ctx, manager.root, environment, "clone", "--bare", "--no-tags", repository.CloneURL, clonePath)
		if !cloned.OK {
			return core.Fail(core.E("workspace.Manager.Register", "failed to create cached private clone", cloned.Err()))
		}
	}
	fetched := manager.gitOutput(ctx, manager.root, environment,
		"--git-dir", clonePath, "fetch", "--no-tags", repository.CloneURL,
		"+refs/heads/*:refs/remotes/lem/*",
	)
	if !fetched.OK {
		return core.Fail(core.E("workspace.Manager.Register", "failed to synchronize cached private clone", fetched.Err()))
	}
	if configured := manager.configureCloneIdentity(ctx, clonePath, review.identityName, review.identityEmail); !configured.OK {
		return configured
	}
	verified := manager.gitOutput(ctx, manager.root, environment, "--git-dir", clonePath, "rev-parse", core.Concat(review.Revision, "^{commit}"))
	if !verified.OK || core.Trim(verified.Value.(string)) != review.Revision {
		return core.Fail(core.NewError("agent workspace cached clone does not contain the reviewed source revision"))
	}
	at := manager.now()
	if at.IsZero() {
		return core.Fail(core.NewError("agent workspace clock returned a zero registration time"))
	}
	return core.Ok(work.Project{
		ID:             projectID,
		SourcePath:     review.Path,
		RepositoryRoot: review.Root,
		SourceBranch:   review.Branch,
		SourceRevision: review.Revision,
		RepositoryName: repository.Name,
		ClonePath:      clonePath,
		CreatedAt:      at,
		UpdatedAt:      at,
	})
}

// PrepareRun creates a deterministic branch and isolated worktree for a run.
func (manager *Manager) PrepareRun(ctx context.Context, project work.Project, run work.Run) core.Result {
	if manager == nil {
		return core.Fail(core.NewError("agent workspace manager is required"))
	}
	if ctx == nil {
		return core.Fail(core.NewError("agent workspace preparation context is required"))
	}
	if err := ctx.Err(); err != nil {
		return core.Fail(core.E("workspace.Manager.PrepareRun", "preparation context is done", err))
	}
	manager.mu.Lock()
	defer manager.mu.Unlock()

	validated := manager.validateRun(project, run, false)
	if !validated.OK {
		return validated
	}
	prepared := validated.Value.(RunWorkspace)
	if _, exists := manager.leases[prepared.RunID]; exists {
		return core.Fail(core.Errorf("agent workspace run %s already owns a worktree", prepared.RunID))
	}
	for _, lease := range manager.leases {
		if lease.Branch == prepared.Branch {
			return core.Fail(core.Errorf("agent workspace branch %s is already attached to %s", prepared.Branch, lease.Path))
		}
	}
	relativeResult := manager.internalRelative(prepared.Path)
	if !relativeResult.OK {
		return relativeResult
	}
	worktreeRelative := relativeResult.Value.(string)
	if manager.files.Exists(worktreeRelative) {
		return core.Fail(core.Errorf("agent workspace run path already exists: %s", prepared.Path))
	}

	repositoryResult := manager.server.EnsureRepository(ctx, project.RepositoryName)
	if !repositoryResult.OK {
		return core.Fail(core.E("workspace.Manager.PrepareRun", "failed to resolve private repository", repositoryResult.Err()))
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
	fetched := manager.gitOutput(ctx, manager.root, environment,
		"--git-dir", project.ClonePath, "fetch", "--no-tags", repository.CloneURL,
		core.Concat("+refs/heads/", project.SourceBranch, ":refs/remotes/lem/", project.SourceBranch),
	)
	if !fetched.OK {
		return core.Fail(core.E("workspace.Manager.PrepareRun", "failed to fetch source revision", fetched.Err()))
	}
	if ensureErr := manager.files.EnsureDir(core.PathDir(worktreeRelative)); ensureErr != nil {
		return core.Fail(core.E("workspace.Manager.PrepareRun", "failed to create run directory", ensureErr))
	}
	added := manager.gitOutput(ctx, manager.root, environment,
		"--git-dir", project.ClonePath, "worktree", "add", "-b", prepared.Branch,
		prepared.Path, project.SourceRevision,
	)
	if !added.OK {
		cleanup := manager.cleanupFailedWorktree(ctx, project.ClonePath, prepared.Path, worktreeRelative)
		if !cleanup.OK {
			return core.Fail(core.E("workspace.Manager.PrepareRun", added.Error(), cleanup.Err()))
		}
		return core.Fail(core.E("workspace.Manager.PrepareRun", "failed to create isolated worktree", added.Err()))
	}
	base := manager.gitOutput(ctx, prepared.Path, nil, "rev-parse", "HEAD")
	if !base.OK {
		return core.Fail(core.E("workspace.Manager.PrepareRun", "failed to inspect prepared worktree", base.Err()))
	}
	prepared.BaseRevision = core.Trim(base.Value.(string))
	if prepared.BaseRevision != project.SourceRevision {
		return core.Fail(core.NewError("agent workspace prepared revision differs from the reviewed source"))
	}
	manager.leases[prepared.RunID] = prepared
	manager.durable[prepared.RunID] = false
	return core.Ok(prepared)
}

// CaptureRun commits non-ignored run changes and pushes the branch durably.
func (manager *Manager) CaptureRun(ctx context.Context, workspace RunWorkspace) core.Result {
	if manager == nil {
		return core.Fail(core.NewError("agent workspace manager is required"))
	}
	if ctx == nil {
		return core.Fail(core.NewError("agent workspace capture context is required"))
	}
	manager.mu.Lock()
	defer manager.mu.Unlock()
	leaseResult := manager.validLease(workspace)
	if !leaseResult.OK {
		return leaseResult
	}
	workspace = leaseResult.Value.(RunWorkspace)
	captureContext := context.WithoutCancel(ctx)

	repositoryResult := manager.server.EnsureRepository(captureContext, workspace.Project.RepositoryName)
	if !repositoryResult.OK {
		return core.Ok(Capture{Retained: true, Summary: core.Concat("private repository unavailable: ", repositoryResult.Error())})
	}
	repository, ok := repositoryResult.Value.(gitserver.Repository)
	if !ok {
		return core.Ok(Capture{Retained: true, Summary: core.Sprintf("private Git returned %T instead of repository", repositoryResult.Value)})
	}
	environmentResult := repositoryEnvironment(repository)
	if !environmentResult.OK {
		return core.Ok(Capture{Retained: true, Summary: environmentResult.Error()})
	}
	environment := environmentResult.Value.([]string)
	status := manager.gitOutput(captureContext, workspace.Path, nil, "status", "--porcelain=v1")
	if !status.OK {
		return core.Ok(Capture{Retained: true, Summary: core.Concat("status failed: ", status.Error())})
	}
	changed := core.Trim(status.Value.(string)) != ""
	if changed {
		staged := manager.gitOutput(captureContext, workspace.Path, nil, "add", "--all")
		if !staged.OK {
			return core.Ok(Capture{Changed: true, Retained: true, Summary: core.Concat("capture staging failed: ", staged.Error())})
		}
		if identity := manager.ensureCommitIdentity(captureContext, workspace.Path); !identity.OK {
			return core.Ok(Capture{Changed: true, Retained: true, Summary: core.Concat("capture identity failed: ", identity.Error())})
		}
		committed := manager.gitOutput(captureContext, workspace.Path, nil, "commit", "-m", core.Concat("LEM run ", workspace.RunID, " capture"))
		if !committed.OK {
			return core.Ok(Capture{Changed: true, Retained: true, Summary: core.Concat("capture commit failed: ", committed.Error())})
		}
	}
	revisionResult := manager.gitOutput(captureContext, workspace.Path, nil, "rev-parse", "HEAD")
	if !revisionResult.OK {
		return core.Ok(Capture{Changed: changed, Retained: true, Summary: core.Concat("capture revision failed: ", revisionResult.Error())})
	}
	revision := core.Trim(revisionResult.Value.(string))
	remoteReference := core.Concat("refs/remotes/lem/", workspace.Branch)
	expectedRevision := ""
	tracked := manager.gitOutput(captureContext, manager.root, nil,
		"--git-dir", workspace.Project.ClonePath, "rev-parse", remoteReference,
	)
	if tracked.OK {
		expectedRevision = core.Trim(tracked.Value.(string))
	}
	lease := core.Concat("--force-with-lease=refs/heads/", workspace.Branch, ":", expectedRevision)
	pushed := manager.gitOutput(captureContext, workspace.Path, environment,
		"push", lease, repository.CloneURL,
		core.Concat("HEAD:refs/heads/", workspace.Branch),
	)
	if !pushed.OK {
		manager.durable[workspace.RunID] = false
		return core.Ok(Capture{
			Revision: revision,
			Changed:  changed,
			Retained: true,
			Summary:  core.Concat("push failed; worktree retained: ", pushed.Error()),
		})
	}
	refreshed := manager.refreshRunTracking(captureContext, workspace.Project, workspace.Branch, repository, environment, revision)
	if !refreshed.OK {
		manager.durable[workspace.RunID] = false
		return core.Ok(Capture{
			Revision: revision,
			Changed:  changed,
			Retained: true,
			Summary:  core.Concat("push succeeded but local tracking refresh failed; worktree retained: ", refreshed.Error()),
		})
	}
	manager.durable[workspace.RunID] = true
	summary := "workspace unchanged; branch pushed"
	if changed {
		summary = "run changes captured and pushed"
	}
	return core.Ok(Capture{Revision: revision, Changed: changed, Pushed: true, Summary: summary})
}

// ReconstructRun restores a missing worktree from its durable private branch.
func (manager *Manager) ReconstructRun(ctx context.Context, project work.Project, run work.Run) core.Result {
	if manager == nil {
		return core.Fail(core.NewError("agent workspace manager is required"))
	}
	if ctx == nil {
		return core.Fail(core.NewError("agent workspace reconstruction context is required"))
	}
	if err := ctx.Err(); err != nil {
		return core.Fail(core.E("workspace.Manager.ReconstructRun", "reconstruction context is done", err))
	}
	manager.mu.Lock()
	defer manager.mu.Unlock()

	validated := manager.validateRun(project, run, true)
	if !validated.OK {
		return validated
	}
	reconstructed := validated.Value.(RunWorkspace)
	if existing, exists := manager.leases[reconstructed.RunID]; exists {
		if existing.Path != reconstructed.Path || existing.Branch != reconstructed.Branch {
			return core.Fail(core.NewError("agent workspace run lease differs from its durable record"))
		}
		return core.Ok(existing)
	}
	relativeResult := manager.internalRelative(reconstructed.Path)
	if !relativeResult.OK {
		return relativeResult
	}
	worktreeRelative := relativeResult.Value.(string)
	if manager.files.Exists(worktreeRelative) {
		branch := manager.gitOutput(ctx, reconstructed.Path, nil, "symbolic-ref", "--short", "HEAD")
		if !branch.OK || core.Trim(branch.Value.(string)) != reconstructed.Branch {
			return core.Fail(core.Errorf("agent workspace retained checkout is not on expected branch %s", reconstructed.Branch))
		}
		revision := manager.gitOutput(ctx, reconstructed.Path, nil, "rev-parse", "HEAD")
		if !revision.OK {
			return core.Fail(core.E("workspace.Manager.ReconstructRun", "failed to inspect retained checkout", revision.Err()))
		}
		reconstructed.BaseRevision = core.Trim(revision.Value.(string))
		if run.ParentRunID != "" {
			if tracked := manager.recoverRunTracking(ctx, project, reconstructed.Branch, reconstructed.BaseRevision); !tracked.OK {
				return tracked
			}
		}
		manager.leases[reconstructed.RunID] = reconstructed
		manager.durable[reconstructed.RunID] = false
		return core.Ok(reconstructed)
	}

	worktrees := manager.gitOutput(ctx, manager.root, nil, "--git-dir", project.ClonePath, "worktree", "list", "--porcelain")
	if !worktrees.OK {
		return core.Fail(core.E("workspace.Manager.ReconstructRun", "failed to inspect cached worktrees", worktrees.Err()))
	}
	if attachedPath := attachedBranchPath(worktrees.Value.(string), reconstructed.Branch); attachedPath != "" {
		return core.Fail(core.Errorf("agent workspace branch %s is already attached to retained checkout %s", reconstructed.Branch, attachedPath))
	}

	repositoryResult := manager.server.EnsureRepository(ctx, project.RepositoryName)
	if !repositoryResult.OK {
		return core.Fail(core.E("workspace.Manager.ReconstructRun", "failed to resolve private repository", repositoryResult.Err()))
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
	remoteReference := core.Concat("refs/remotes/lem/", reconstructed.Branch)
	fetched := manager.gitOutput(ctx, manager.root, environment,
		"--git-dir", project.ClonePath, "fetch", "--no-tags", repository.CloneURL,
		core.Concat("+refs/heads/", reconstructed.Branch, ":", remoteReference),
	)
	if !fetched.OK {
		return core.Fail(core.E("workspace.Manager.ReconstructRun", "failed to fetch durable run branch", fetched.Err()))
	}
	branched := manager.gitOutput(ctx, manager.root, nil,
		"--git-dir", project.ClonePath, "branch", "--force", reconstructed.Branch, remoteReference,
	)
	if !branched.OK {
		return core.Fail(core.E("workspace.Manager.ReconstructRun", "failed to restore local run branch", branched.Err()))
	}
	if ensureErr := manager.files.EnsureDir(core.PathDir(worktreeRelative)); ensureErr != nil {
		return core.Fail(core.E("workspace.Manager.ReconstructRun", "failed to create reconstructed run directory", ensureErr))
	}
	added := manager.gitOutput(ctx, manager.root, nil,
		"--git-dir", project.ClonePath, "worktree", "add", reconstructed.Path, reconstructed.Branch,
	)
	if !added.OK {
		cleanup := manager.cleanupFailedWorktree(ctx, project.ClonePath, reconstructed.Path, worktreeRelative)
		if !cleanup.OK {
			return core.Fail(core.E("workspace.Manager.ReconstructRun", added.Error(), cleanup.Err()))
		}
		return core.Fail(core.E("workspace.Manager.ReconstructRun", "failed to reconstruct run worktree", added.Err()))
	}
	revision := manager.gitOutput(ctx, reconstructed.Path, nil, "rev-parse", "HEAD")
	if !revision.OK {
		return core.Fail(core.E("workspace.Manager.ReconstructRun", "failed to inspect reconstructed worktree", revision.Err()))
	}
	reconstructed.BaseRevision = core.Trim(revision.Value.(string))
	manager.leases[reconstructed.RunID] = reconstructed
	manager.durable[reconstructed.RunID] = false
	return core.Ok(reconstructed)
}

func (manager *Manager) refreshRunTracking(ctx context.Context, project work.Project, branch string, repository gitserver.Repository, environment []string, revision string) core.Result {
	remoteReference := core.Concat("refs/remotes/lem/", branch)
	updated := manager.gitOutput(ctx, manager.root, nil,
		"--git-dir", project.ClonePath, "update-ref", remoteReference, revision,
	)
	if !updated.OK {
		fetched := manager.gitOutput(ctx, manager.root, environment,
			"--git-dir", project.ClonePath, "fetch", "--no-tags", repository.CloneURL,
			core.Concat("+refs/heads/", branch, ":", remoteReference),
		)
		if !fetched.OK {
			return core.Fail(core.E("workspace.Manager.refreshRunTracking", updated.Error(), fetched.Err()))
		}
	}
	tracked := manager.gitOutput(ctx, manager.root, nil, "--git-dir", project.ClonePath, "rev-parse", remoteReference)
	if !tracked.OK {
		return core.Fail(core.E("workspace.Manager.refreshRunTracking", "failed to inspect local tracking reference", tracked.Err()))
	}
	if core.Trim(tracked.Value.(string)) != revision {
		return core.Fail(core.NewError("agent workspace local tracking reference differs from pushed run revision"))
	}
	return core.Ok(nil)
}

func (manager *Manager) recoverRunTracking(ctx context.Context, project work.Project, branch, revision string) core.Result {
	remoteReference := core.Concat("refs/remotes/lem/", branch)
	trackedResult := manager.gitOutput(ctx, manager.root, nil,
		"--git-dir", project.ClonePath, "for-each-ref", "--format=%(objectname)", remoteReference,
	)
	if !trackedResult.OK {
		return core.Fail(core.E("workspace.Manager.recoverRunTracking", "failed to inspect local tracking reference", trackedResult.Err()))
	}
	trackedRevision := core.Trim(trackedResult.Value.(string))
	repositoryResult := manager.server.EnsureRepository(ctx, project.RepositoryName)
	if !repositoryResult.OK {
		return core.Fail(core.E("workspace.Manager.recoverRunTracking", "failed to resolve private repository", repositoryResult.Err()))
	}
	repository, ok := repositoryResult.Value.(gitserver.Repository)
	if !ok {
		return core.Fail(core.Errorf("agent workspace Git service returned %T instead of repository", repositoryResult.Value))
	}
	environmentResult := repositoryEnvironment(repository)
	if !environmentResult.OK {
		return environmentResult
	}
	remoteBranch := core.Concat("refs/heads/", branch)
	remoteResult := manager.gitOutput(ctx, manager.root, environmentResult.Value.([]string),
		"--git-dir", project.ClonePath, "ls-remote", "--heads", repository.CloneURL, remoteBranch,
	)
	if !remoteResult.OK {
		return core.Fail(core.E("workspace.Manager.recoverRunTracking", "failed to inspect durable run branch", remoteResult.Err()))
	}
	remoteRevision := ""
	remoteFields := core.Fields(remoteResult.Value.(string))
	if len(remoteFields) != 0 {
		if len(remoteFields) != 2 || remoteFields[1] != remoteBranch {
			return core.Fail(core.NewError("agent workspace private branch inspection returned an unexpected result"))
		}
		remoteRevision = remoteFields[0]
	}
	if remoteRevision == revision {
		if trackedRevision == revision {
			return core.Ok(nil)
		}
		refreshed := manager.refreshRunTracking(ctx, project, branch, repository, environmentResult.Value.([]string), revision)
		if !refreshed.OK {
			return core.Fail(core.E("workspace.Manager.recoverRunTracking", "failed to refresh pushed run tracking", refreshed.Err()))
		}
		return core.Ok(nil)
	}
	if remoteRevision == trackedRevision {
		return core.Ok(nil)
	}
	return core.Fail(core.NewError("agent workspace retained checkout differs from its expected private branch state"))
}

// ReleaseRun removes a clean worktree only after its branch is durably pushed.
func (manager *Manager) ReleaseRun(ctx context.Context, workspace RunWorkspace) core.Result {
	if manager == nil {
		return core.Fail(core.NewError("agent workspace manager is required"))
	}
	if ctx == nil {
		return core.Fail(core.NewError("agent workspace release context is required"))
	}
	if err := ctx.Err(); err != nil {
		return core.Fail(core.E("workspace.Manager.ReleaseRun", "release context is done", err))
	}
	manager.mu.Lock()
	defer manager.mu.Unlock()
	leaseResult := manager.validLease(workspace)
	if !leaseResult.OK {
		return leaseResult
	}
	workspace = leaseResult.Value.(RunWorkspace)
	if !manager.durable[workspace.RunID] {
		return core.Fail(core.NewError("agent workspace worktree is not durably pushed and must be retained"))
	}
	status := manager.gitOutput(ctx, workspace.Path, nil, "status", "--porcelain=v1")
	if !status.OK || core.Trim(status.Value.(string)) != "" {
		return core.Fail(core.NewError("agent workspace worktree changed after capture and must be retained"))
	}
	removed := manager.gitOutput(ctx, manager.root, nil,
		"--git-dir", workspace.Project.ClonePath, "worktree", "remove", "--force", workspace.Path,
	)
	if !removed.OK {
		return core.Fail(core.E("workspace.Manager.ReleaseRun", "failed to remove isolated worktree", removed.Err()))
	}
	relativeResult := manager.internalRelative(workspace.Path)
	if !relativeResult.OK {
		return relativeResult
	}
	runRelative := core.PathDir(relativeResult.Value.(string))
	if deleteErr := manager.files.DeleteAll(runRelative); deleteErr != nil {
		return core.Fail(core.E("workspace.Manager.ReleaseRun", "failed to clean run directory", deleteErr))
	}
	for runID, lease := range manager.leases {
		if lease.Path == workspace.Path {
			delete(manager.leases, runID)
			delete(manager.durable, runID)
		}
	}
	return core.Ok(nil)
}

func (manager *Manager) reviewSource(ctx context.Context, sourcePath string) core.Result {
	pathResult := canonicalDirectory(sourcePath)
	if !pathResult.OK {
		return core.Fail(core.E("workspace.Manager.ReviewSource", "invalid source directory", pathResult.Err()))
	}
	selectedPath := pathResult.Value.(string)
	rootResult := manager.gitOutput(ctx, selectedPath, nil, "rev-parse", "--show-toplevel")
	if !rootResult.OK {
		includedResult := manager.adHocIncluded(ctx, selectedPath)
		if !includedResult.OK {
			return includedResult
		}
		included := includedResult.Value.([]string)
		hashResult := includedHash(selectedPath, included)
		if !hashResult.OK {
			return hashResult
		}
		identityName, identityEmail := manager.commitIdentity(ctx, selectedPath)
		return core.Ok(SourceReview{
			Path:           selectedPath,
			Root:           selectedPath,
			ProposedBranch: "main",
			CommitIdentity: core.Concat(identityName, " <", identityEmail, ">"),
			Git:            false,
			Clean:          true,
			Included:       included,
			IncludedHash:   hashResult.Value.(string),
			identityName:   identityName,
			identityEmail:  identityEmail,
		})
	}
	rootCanonical := canonicalDirectory(core.Trim(rootResult.Value.(string)))
	if !rootCanonical.OK {
		return core.Fail(core.E("workspace.Manager.ReviewSource", "Git returned an invalid repository root", rootCanonical.Err()))
	}
	root := rootCanonical.Value.(string)
	branchResult := manager.gitOutput(ctx, root, nil, "symbolic-ref", "--short", "HEAD")
	branch := ""
	detached := !branchResult.OK
	if branchResult.OK {
		branch = core.Trim(branchResult.Value.(string))
	}
	revisionResult := manager.gitOutput(ctx, root, nil, "rev-parse", "HEAD")
	if !revisionResult.OK {
		return core.Fail(core.E("workspace.Manager.ReviewSource", "failed to resolve source revision", revisionResult.Err()))
	}
	statusResult := manager.gitOutput(ctx, root, nil, "status", "--porcelain=v1")
	if !statusResult.OK {
		return core.Fail(core.E("workspace.Manager.ReviewSource", "failed to inspect source status", statusResult.Err()))
	}
	includedResult := manager.gitOutput(ctx, root, nil, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
	if !includedResult.OK {
		return core.Fail(core.E("workspace.Manager.ReviewSource", "failed to list source files", includedResult.Err()))
	}
	included := parseNULList(includedResult.Value.(string))
	hashResult := includedHash(root, included)
	if !hashResult.OK {
		return hashResult
	}
	identityName, identityEmail := manager.commitIdentity(ctx, root)
	proposedBranch := branch
	if proposedBranch == "" {
		proposedBranch = "main"
	}
	return core.Ok(SourceReview{
		Path:           selectedPath,
		Root:           root,
		Branch:         branch,
		ProposedBranch: proposedBranch,
		Revision:       core.Trim(revisionResult.Value.(string)),
		CommitIdentity: core.Concat(identityName, " <", identityEmail, ">"),
		Git:            true,
		Clean:          core.Trim(statusResult.Value.(string)) == "",
		Detached:       detached,
		Included:       included,
		IncludedHash:   hashResult.Value.(string),
		identityName:   identityName,
		identityEmail:  identityEmail,
	})
}

func (manager *Manager) adHocIncluded(ctx context.Context, sourceRoot string) core.Result {
	reviewIDResult := pathSegment("review ID", manager.ids())
	if !reviewIDResult.OK {
		return reviewIDResult
	}
	reviewRelative := core.PathJoin(".reviews", core.Concat(reviewIDResult.Value.(string), ".git"))
	if manager.files.Exists(reviewRelative) {
		if deleteErr := manager.files.DeleteAll(reviewRelative); deleteErr != nil {
			return core.Fail(core.E("workspace.Manager.ReviewSource", "failed to clean stale review index", deleteErr))
		}
	}
	if ensureErr := manager.files.EnsureDir(".reviews"); ensureErr != nil {
		return core.Fail(core.E("workspace.Manager.ReviewSource", "failed to create review index directory", ensureErr))
	}
	reviewPathResult := manager.internalPath(".reviews", core.Concat(reviewIDResult.Value.(string), ".git"))
	if !reviewPathResult.OK {
		return reviewPathResult
	}
	reviewPath := reviewPathResult.Value.(string)
	initialized := manager.gitOutput(ctx, manager.root, nil, "init", "--bare", "--quiet", reviewPath)
	if !initialized.OK {
		return core.Fail(core.E("workspace.Manager.ReviewSource", "failed to create temporary review index", initialized.Err()))
	}
	listed := manager.gitOutput(ctx, sourceRoot, nil,
		core.Concat("--git-dir=", reviewPath), core.Concat("--work-tree=", sourceRoot),
		"-c", "core.bare=false", "ls-files", "-z", "--cached", "--others", "--exclude-standard",
	)
	deleteErr := manager.files.DeleteAll(reviewRelative)
	if !listed.OK {
		if deleteErr != nil {
			return core.Fail(core.E("workspace.Manager.ReviewSource", listed.Error(), deleteErr))
		}
		return core.Fail(core.E("workspace.Manager.ReviewSource", "failed to list ad-hoc source files", listed.Err()))
	}
	if deleteErr != nil {
		return core.Fail(core.E("workspace.Manager.ReviewSource", "failed to remove temporary review index", deleteErr))
	}
	return core.Ok(parseNULList(listed.Value.(string)))
}

func (manager *Manager) initializeAdHoc(ctx context.Context, review SourceReview, projectID string) core.Result {
	initialized := manager.gitOutput(ctx, review.Root, nil, "init", "--initial-branch", review.ProposedBranch, ".")
	if !initialized.OK {
		return core.Fail(core.E("workspace.Manager.Register", "failed to initialize ad-hoc Git repository", initialized.Err()))
	}
	if review.identityName == lemIdentityName && review.identityEmail == lemIdentityEmail {
		name := manager.gitOutput(ctx, review.Root, nil, "config", "--local", "user.name", lemIdentityName)
		if !name.OK {
			return core.Fail(core.E("workspace.Manager.Register", "failed to configure local LEM commit name", name.Err()))
		}
		email := manager.gitOutput(ctx, review.Root, nil, "config", "--local", "user.email", lemIdentityEmail)
		if !email.OK {
			return core.Fail(core.E("workspace.Manager.Register", "failed to configure local LEM commit email", email.Err()))
		}
	}
	staged := manager.gitOutput(ctx, review.Root, nil, "add", "--all")
	if !staged.OK {
		return core.Fail(core.E("workspace.Manager.Register", "failed to stage ad-hoc baseline", staged.Err()))
	}
	committed := manager.gitOutput(ctx, review.Root, nil, "commit", "-m", core.Concat("LEM baseline for ", projectID))
	if !committed.OK {
		return core.Fail(core.E("workspace.Manager.Register", "failed to commit ad-hoc baseline", committed.Err()))
	}
	return core.Ok(nil)
}

func (manager *Manager) configureCloneIdentity(ctx context.Context, clonePath, name, email string) core.Result {
	if core.Trim(name) == "" || core.Trim(email) == "" {
		return core.Fail(core.NewError("agent workspace clone commit identity is incomplete"))
	}
	configuredName := manager.gitOutput(ctx, manager.root, nil, "--git-dir", clonePath, "config", "user.name", name)
	if !configuredName.OK {
		return core.Fail(core.E("workspace.Manager.Register", "failed to configure cached clone commit name", configuredName.Err()))
	}
	configuredEmail := manager.gitOutput(ctx, manager.root, nil, "--git-dir", clonePath, "config", "user.email", email)
	if !configuredEmail.OK {
		return core.Fail(core.E("workspace.Manager.Register", "failed to configure cached clone commit email", configuredEmail.Err()))
	}
	return core.Ok(nil)
}

func (manager *Manager) ensureCommitIdentity(ctx context.Context, directory string) core.Result {
	name := manager.gitOutput(ctx, directory, nil, "config", "--get", "user.name")
	email := manager.gitOutput(ctx, directory, nil, "config", "--get", "user.email")
	if name.OK && email.OK && core.Trim(name.Value.(string)) != "" && core.Trim(email.Value.(string)) != "" {
		return core.Ok(nil)
	}
	configuredName := manager.gitOutput(ctx, directory, nil, "config", "--local", "user.name", lemIdentityName)
	if !configuredName.OK {
		return configuredName
	}
	return manager.gitOutput(ctx, directory, nil, "config", "--local", "user.email", lemIdentityEmail)
}

func (manager *Manager) commitIdentity(ctx context.Context, directory string) (string, string) {
	nameResult := manager.gitOutput(ctx, directory, nil, "config", "--get", "user.name")
	emailResult := manager.gitOutput(ctx, directory, nil, "config", "--get", "user.email")
	if nameResult.OK && emailResult.OK {
		name := core.Trim(nameResult.Value.(string))
		email := core.Trim(emailResult.Value.(string))
		if name != "" && email != "" {
			return name, email
		}
	}
	return lemIdentityName, lemIdentityEmail
}

func (manager *Manager) validateRun(project work.Project, run work.Run, reconstruct bool) core.Result {
	projectIDResult := pathSegment("project ID", project.ID)
	if !projectIDResult.OK {
		return projectIDResult
	}
	runIDResult := pathSegment("run ID", run.ID)
	if !runIDResult.OK {
		return runIDResult
	}
	if run.ProjectID != project.ID {
		return core.Fail(core.NewError("agent workspace run project does not match the registered project"))
	}
	if core.Trim(project.RepositoryName) == "" || core.Trim(project.SourceBranch) == "" || core.Trim(project.SourceRevision) == "" {
		return core.Fail(core.NewError("agent workspace project requires repository, branch, and source revision"))
	}
	expectedCloneResult := manager.internalPath(project.ID, "repo.git")
	if !expectedCloneResult.OK {
		return expectedCloneResult
	}
	if project.ClonePath != expectedCloneResult.Value.(string) {
		return core.Fail(core.NewError("agent workspace cached clone is outside the configured internal root"))
	}
	branchResult := runBranch(run.WorkID, run.Number)
	if !branchResult.OK {
		return branchResult
	}
	branch := branchResult.Value.(string)
	if reconstruct && core.Trim(run.Branch) != "" {
		branch = core.Trim(run.Branch)
		if !core.HasPrefix(branch, "lem/work/") || core.Contains(branch, "..") || core.Contains(branch, " ") {
			return core.Fail(core.NewError("agent workspace durable branch is invalid"))
		}
	} else if core.Trim(run.Branch) != "" && core.Trim(run.Branch) != branch {
		return core.Fail(core.NewError("agent workspace run branch differs from its deterministic branch"))
	}
	pathResult := manager.internalPath(project.ID, "runs", run.ID, "worktree")
	if !pathResult.OK {
		return pathResult
	}
	path := pathResult.Value.(string)
	if reconstruct && core.Trim(run.Worktree) != "" {
		pathResult = manager.internalAbsolute(run.Worktree)
		if !pathResult.OK {
			return pathResult
		}
		path = pathResult.Value.(string)
	} else if core.Trim(run.Worktree) != "" && run.Worktree != path {
		return core.Fail(core.NewError("agent workspace run path differs from its deterministic internal path"))
	}
	return core.Ok(RunWorkspace{
		Project:      project,
		RunID:        run.ID,
		Branch:       branch,
		Path:         path,
		BaseRevision: project.SourceRevision,
	})
}

func (manager *Manager) validLease(workspace RunWorkspace) core.Result {
	runIDResult := pathSegment("run ID", workspace.RunID)
	if !runIDResult.OK {
		return runIDResult
	}
	lease, exists := manager.leases[workspace.RunID]
	if !exists {
		return core.Fail(core.Errorf("agent workspace run %s has no active worktree lease", workspace.RunID))
	}
	if lease.Path != workspace.Path || lease.Branch != workspace.Branch || lease.Project.ID != workspace.Project.ID {
		return core.Fail(core.NewError("agent workspace supplied lease differs from the active worktree"))
	}
	if contained := manager.internalAbsolute(lease.Path); !contained.OK {
		return contained
	}
	return core.Ok(lease)
}

func (manager *Manager) cleanupFailedWorktree(ctx context.Context, clonePath, worktreePath, worktreeRelative string) core.Result {
	manager.gitOutput(context.WithoutCancel(ctx), manager.root, nil,
		"--git-dir", clonePath, "worktree", "remove", "--force", worktreePath,
	)
	manager.gitOutput(context.WithoutCancel(ctx), manager.root, nil, "--git-dir", clonePath, "worktree", "prune")
	if deleteErr := manager.files.DeleteAll(core.PathDir(worktreeRelative)); deleteErr != nil {
		return core.Fail(core.E("workspace.cleanupFailedWorktree", "failed to clean incomplete worktree", deleteErr))
	}
	return core.Ok(nil)
}

func (manager *Manager) gitOutput(ctx context.Context, directory string, environment []string, arguments ...string) core.Result {
	result := manager.git.Run(ctx, Command{
		Dir:         directory,
		Executable:  "git",
		Args:        append([]string(nil), arguments...),
		Environment: append([]string(nil), environment...),
	})
	if !result.OK {
		return core.Fail(core.E("workspace.git", core.Concat("git ", core.Join(" ", arguments...)), result.Err()))
	}
	output, ok := result.Value.(string)
	if !ok {
		return core.Fail(core.Errorf("agent workspace Git runner returned %T instead of string output", result.Value))
	}
	return core.Ok(output)
}

func (manager *Manager) internalPath(segments ...string) core.Result {
	return manager.internalAbsolute(core.PathJoin(append([]string{manager.root}, segments...)...))
}

func (manager *Manager) internalAbsolute(path string) core.Result {
	if !core.PathIsAbs(path) {
		return core.Fail(core.NewError("agent workspace internal path must be absolute"))
	}
	absResult := core.PathAbs(path)
	if !absResult.OK {
		return core.Fail(core.E("workspace.internalAbsolute", "failed to normalize internal path", absResult.Err()))
	}
	abs := absResult.Value.(string)
	relativeResult := core.PathRel(manager.root, abs)
	if !relativeResult.OK {
		return core.Fail(core.E("workspace.internalAbsolute", "failed to compare internal path", relativeResult.Err()))
	}
	relative := relativeResult.Value.(string)
	if relative == ".." || core.HasPrefix(relative, "../") || core.HasPrefix(relative, `..\`) || core.PathIsAbs(relative) {
		return core.Fail(core.NewError("agent workspace path escapes the configured internal root"))
	}
	return core.Ok(abs)
}

func (manager *Manager) internalRelative(path string) core.Result {
	absResult := manager.internalAbsolute(path)
	if !absResult.OK {
		return absResult
	}
	relativeResult := core.PathRel(manager.root, absResult.Value.(string))
	if !relativeResult.OK {
		return relativeResult
	}
	return core.Ok(relativeResult.Value.(string))
}

func canonicalDirectory(configured string) core.Result {
	configured = core.Trim(configured)
	if configured == "" {
		return core.Fail(core.NewError("directory path is required"))
	}
	absResult := core.PathAbs(configured)
	if !absResult.OK {
		return core.Fail(core.E("workspace.canonicalDirectory", "failed to resolve absolute path", absResult.Err()))
	}
	evaluated := core.PathEvalSymlinks(absResult.Value.(string))
	if !evaluated.OK {
		return core.Fail(core.E("workspace.canonicalDirectory", "failed to resolve directory symlinks", evaluated.Err()))
	}
	path := evaluated.Value.(string)
	stat := core.Stat(path)
	if !stat.OK {
		return core.Fail(core.E("workspace.canonicalDirectory", "failed to inspect directory", stat.Err()))
	}
	info, ok := stat.Value.(interface{ IsDir() bool })
	if !ok || !info.IsDir() {
		return core.Fail(core.NewError("configured path is not a directory"))
	}
	return core.Ok(path)
}

func pathSegment(label, configured string) core.Result {
	value := core.Trim(configured)
	if value == "" || value == "." || value == ".." || core.Contains(value, "/") || core.Contains(value, `\`) || core.Contains(value, "\x00") {
		return core.Fail(core.Errorf("agent workspace %s must be one safe path segment", label))
	}
	return core.Ok(value)
}

func runBranch(workID string, number int) core.Result {
	if number <= 0 {
		return core.Fail(core.NewError("agent workspace run number must be positive"))
	}
	component := branchComponent(workID)
	if component == "" {
		return core.Fail(core.NewError("agent workspace work ID cannot form a Git branch"))
	}
	return core.Ok(core.Concat("lem/work/", component, "/run-", core.Itoa(number)))
}

func branchComponent(configured string) string {
	input := []byte(core.Trim(configured))
	output := make([]byte, 0, len(input))
	separator := false
	for _, character := range input {
		allowed := character >= 'a' && character <= 'z' || character >= 'A' && character <= 'Z' || character >= '0' && character <= '9' || character == '_' || character == '.' || character == '-'
		if allowed {
			output = append(output, character)
			separator = false
			continue
		}
		if len(output) > 0 && !separator {
			output = append(output, '-')
			separator = true
		}
	}
	for len(output) > 0 && (output[0] == '.' || output[0] == '-') {
		output = output[1:]
	}
	for len(output) > 0 && (output[len(output)-1] == '.' || output[len(output)-1] == '-') {
		output = output[:len(output)-1]
	}
	return string(output)
}

func parseNULList(output string) []string {
	parts := core.Split(output, "\x00")
	files := make([]string, 0, len(parts))
	for _, part := range parts {
		part = core.Trim(part)
		if part != "" {
			files = append(files, part)
		}
	}
	core.SliceSort(files)
	return files
}

func includedHash(root string, included []string) core.Result {
	medium, mediumErr := coreio.NewSandboxed(root)
	if mediumErr != nil {
		return core.Fail(core.E("workspace.includedHash", "failed to open reviewed source medium", mediumErr))
	}
	entries := make([]string, 0, len(included))
	for _, path := range included {
		if path == "" || core.PathIsAbs(path) || path == ".." || core.HasPrefix(path, "../") || core.HasPrefix(path, `..\`) {
			return core.Fail(core.Errorf("agent workspace included path is unsafe: %q", path))
		}
		content, readErr := medium.Read(path)
		if readErr != nil {
			return core.Fail(core.E("workspace.includedHash", core.Concat("failed to read included source ", path), readErr))
		}
		entries = append(entries, core.Concat(core.Itoa(len(path)), ":", path, ":", core.SHA256HexString(content)))
	}
	return core.Ok(core.SHA256HexString(core.Join("\n", entries...)))
}

func repositoryEnvironment(repository gitserver.Repository) core.Result {
	identity := core.Trim(repository.IdentityFile)
	knownHosts := core.Trim(repository.KnownHostsFile)
	if identity == "" && knownHosts == "" {
		return core.Ok([]string(nil))
	}
	if identity == "" || knownHosts == "" || !core.PathIsAbs(identity) || !core.PathIsAbs(knownHosts) {
		return core.Fail(core.NewError("agent workspace private Git requires absolute identity and known-host paths"))
	}
	sshCommand := core.Concat(
		"ssh -i ", shellQuote(identity),
		" -o IdentitiesOnly=yes",
		" -o UserKnownHostsFile=", shellQuote(knownHosts),
		" -o StrictHostKeyChecking=yes",
	)
	return core.Ok([]string{core.Concat("GIT_SSH_COMMAND=", sshCommand)})
}

func shellQuote(value string) string {
	return core.Concat("'", core.Replace(value, "'", `'"'"'`), "'")
}

func attachedBranchPath(output, branch string) string {
	worktreePath := ""
	for _, line := range core.Split(output, "\n") {
		if core.HasPrefix(line, "worktree ") {
			worktreePath = core.TrimPrefix(line, "worktree ")
			continue
		}
		if line == core.Concat("branch refs/heads/", branch) {
			return worktreePath
		}
	}
	return ""
}
