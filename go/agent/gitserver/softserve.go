// SPDX-License-Identifier: EUPL-1.2

package gitserver

import (
	"context"
	"net"
	"net/url"
	"sync"
	"time"

	charmlog "charm.land/log/v2"
	core "dappco.re/go"
	"github.com/charmbracelet/keygen"
	"github.com/charmbracelet/soft-serve/pkg/access"
	softbackend "github.com/charmbracelet/soft-serve/pkg/backend"
	softconfig "github.com/charmbracelet/soft-serve/pkg/config"
	softdb "github.com/charmbracelet/soft-serve/pkg/db"
	"github.com/charmbracelet/soft-serve/pkg/db/migrate"
	softlog "github.com/charmbracelet/soft-serve/pkg/log"
	"github.com/charmbracelet/soft-serve/pkg/proto"
	softssh "github.com/charmbracelet/soft-serve/pkg/ssh"
	softstore "github.com/charmbracelet/soft-serve/pkg/store"
	softdatabase "github.com/charmbracelet/soft-serve/pkg/store/database"
	softutils "github.com/charmbracelet/soft-serve/pkg/utils"
	charmssh "github.com/charmbracelet/ssh"
	cryptossh "golang.org/x/crypto/ssh"
	"golang.org/x/crypto/ssh/knownhosts"
)

const (
	ownerFilename   = "owner.lock"
	clientKeyName   = "soft_serve_client_ed25519"
	hostKeyName     = "soft_serve_host_ed25519"
	knownHostsName  = "known_hosts"
	softServeDBName = "soft-serve.db"
)

type ownerRecord struct {
	PID       int       `json:"pid"`
	StartedAt time.Time `json:"started_at"`
}

type softServeOperations struct {
	before func(string) core.Result
}

type loggerResources struct {
	logger *charmlog.Logger
	file   *core.OSFile
}

type softServe struct {
	options    Options
	operations softServeOperations

	mu           sync.Mutex
	serveErrMu   sync.Mutex
	state        string
	address      string
	publicURL    string
	ownerReceipt string
	owned        bool
	listener     core.Listener
	server       *softssh.SSHServer
	database     *softdb.DB
	backend      *softbackend.Backend
	logFile      *core.OSFile
	serveDone    chan struct{}
	serveResult  core.Result
}

// NewSoftServe validates options and constructs a lazy private Git service.
func NewSoftServe(options Options) core.Result {
	validated := validateOptions(options)
	if !validated.OK {
		return validated
	}
	return core.Ok(&softServe{options: validated.Value.(Options), state: "stopped", serveResult: core.Ok(nil)})
}

func (service *softServe) Start(ctx context.Context) core.Result {
	if service == nil {
		return core.Fail(core.NewError("agent git server service is required"))
	}
	if ctx == nil {
		return core.Fail(core.NewError("agent git server start context is required"))
	}
	if err := ctx.Err(); err != nil {
		return core.Fail(core.E("gitserver.Start", "start context is done", err))
	}

	service.mu.Lock()
	defer service.mu.Unlock()
	switch service.state {
	case "running":
		return core.Ok(Health{Running: true, Address: service.address})
	case "starting", "closing":
		return core.Fail(core.Errorf("agent git server is %s", service.state))
	case "failed":
		return core.Fail(core.Errorf("agent git server failed and must be closed before restart: %s", service.failureReason()))
	}
	service.state = "starting"

	if acquired := service.acquireOwner(); !acquired.OK {
		service.state = "stopped"
		return acquired
	}
	if started := service.startRuntime(ctx); !started.OK {
		cleanup := service.cleanupStart()
		service.state = "stopped"
		if !cleanup.OK {
			return core.Fail(core.E("gitserver.Start", started.Error(), cleanup.Err()))
		}
		return started
	}
	service.state = "running"
	return core.Ok(Health{Running: true, Address: service.address})
}

func (service *softServe) EnsureRepository(ctx context.Context, configuredName string) core.Result {
	if service == nil {
		return core.Fail(core.NewError("agent git server service is required"))
	}
	if ctx == nil {
		return core.Fail(core.NewError("agent git server repository context is required"))
	}
	nameResult := normalizeRepositoryName(configuredName)
	if !nameResult.OK {
		return nameResult
	}
	name := nameResult.Value.(string)
	if started := service.Start(ctx); !started.OK {
		return started
	}

	service.mu.Lock()
	defer service.mu.Unlock()
	if service.state != "running" || service.backend == nil {
		return core.Fail(core.Errorf("agent git server is not running: %s", service.failureReason()))
	}

	_, repositoryErr := service.backend.Repository(ctx, name)
	if repositoryErr != nil {
		if !core.Is(repositoryErr, proto.ErrRepoNotFound) {
			return core.Fail(core.E("gitserver.EnsureRepository", core.Concat("failed to inspect ", name), repositoryErr))
		}
		admin, userErr := service.backend.User(ctx, "admin")
		if userErr != nil {
			return core.Fail(core.E("gitserver.EnsureRepository", "failed to load private repository owner", userErr))
		}
		_, createErr := service.backend.CreateRepository(ctx, name, admin, proto.RepositoryOptions{
			Private: true,
			Hidden:  true,
		})
		if createErr != nil && !core.Is(createErr, proto.ErrRepoExist) {
			return core.Fail(core.E("gitserver.EnsureRepository", core.Concat("failed to create ", name), createErr))
		}
	}
	if disabled := service.disableCommandHooks(name); !disabled.OK {
		return disabled
	}
	return core.Ok(service.repository(name))
}

func (service *softServe) Health(ctx context.Context) core.Result {
	if service == nil {
		return core.Fail(core.NewError("agent git server service is required"))
	}
	if ctx == nil {
		return core.Fail(core.NewError("agent git server health context is required"))
	}
	if err := ctx.Err(); err != nil {
		return core.Fail(core.E("gitserver.Health", "health context is done", err))
	}
	service.mu.Lock()
	defer service.mu.Unlock()
	health := Health{Address: service.address}
	switch service.state {
	case "running":
		health.Running = true
	case "stopped":
		health.Reason = "private Git service is not started"
	default:
		health.Reason = service.failureReason()
	}
	return core.Ok(health)
}

func (service *softServe) Close() core.Result {
	if service == nil {
		return core.Fail(core.NewError("agent git server service is required"))
	}

	service.mu.Lock()
	if service.state == "stopped" && service.server == nil && service.database == nil && !service.owned {
		service.mu.Unlock()
		return core.Ok(nil)
	}
	server := service.server
	listener := service.listener
	database := service.database
	logFile := service.logFile
	done := service.serveDone
	service.state = "closing"
	service.mu.Unlock()

	failures := make([]string, 0, 5)
	if server != nil {
		shutdownContext, cancel := context.WithTimeout(context.Background(), service.options.ShutdownTimeout)
		if err := server.Shutdown(shutdownContext); err != nil && !core.Is(err, charmssh.ErrServerClosed) {
			failures = append(failures, core.Concat("SSH shutdown: ", err.Error()))
			if closeErr := server.Close(); closeErr != nil && !core.Is(closeErr, charmssh.ErrServerClosed) {
				failures = append(failures, core.Concat("SSH close: ", closeErr.Error()))
			}
		}
		cancel()
	}
	if listener != nil {
		if err := listener.Close(); err != nil {
			if !core.Is(err, net.ErrClosed) {
				failures = append(failures, core.Concat("listener close: ", err.Error()))
			}
		}
	}
	if done != nil {
		timer := core.NewTimer(service.options.ShutdownTimeout)
		select {
		case <-done:
			if !timer.Stop() {
				select {
				case <-timer.C:
				default:
				}
			}
		case <-timer.C:
			failures = append(failures, "SSH serve loop did not stop before timeout")
		}
	}
	if database != nil {
		closed := service.runPostOperation("database-close", core.ResultOf(nil, database.Close()))
		if !closed.OK {
			failures = append(failures, core.Concat("database close: ", closed.Error()))
		}
	}
	if logFile != nil {
		if err := logFile.Close(); err != nil {
			failures = append(failures, core.Concat("log close: ", err.Error()))
		}
	}

	service.mu.Lock()
	if released := service.releaseOwner(); !released.OK {
		failures = append(failures, released.Error())
	}
	serveResult := service.currentServeResult()
	service.listener = nil
	service.server = nil
	service.database = nil
	service.backend = nil
	service.logFile = nil
	service.serveDone = nil
	service.setServeResult(core.Ok(nil))
	service.address = ""
	service.publicURL = ""
	service.state = "stopped"
	service.mu.Unlock()

	if !serveResult.OK && serveResult.Err() != nil && !core.Is(serveResult.Err(), charmssh.ErrServerClosed) {
		failures = append(failures, core.Concat("SSH serve: ", serveResult.Error()))
	}
	if len(failures) > 0 {
		return core.Fail(core.NewError(core.Join("; ", failures...)))
	}
	return core.Ok(nil)
}

func validateOptions(options Options) core.Result {
	options.DataPath = core.Trim(options.DataPath)
	options.ListenAddress = core.Trim(options.ListenAddress)
	options.PublicURL = core.Trim(options.PublicURL)
	if options.DataPath == "" || !core.PathIsAbs(options.DataPath) {
		return core.Fail(core.NewError("agent git server data path must be an absolute path"))
	}
	if options.ListenAddress == "" || !loopbackAuthority(options.ListenAddress) {
		return core.Fail(core.NewError("agent git server listen address must be a numeric loopback host and port"))
	}
	if options.PublicURL == "" || !loopbackPublicURL(options.PublicURL) {
		return core.Fail(core.NewError("agent git server public URL must be a credential-free loopback SSH URL"))
	}
	if options.ShutdownTimeout <= 0 {
		return core.Fail(core.NewError("agent git server shutdown timeout must be positive"))
	}
	if options.PID == nil || options.ProcessAlive == nil {
		return core.Fail(core.NewError("agent git server process ownership functions are required"))
	}
	return core.Ok(options)
}

func loopbackAuthority(authority string) bool {
	host, port, err := net.SplitHostPort(authority)
	if err != nil || host == "" || port == "" {
		return false
	}
	ip := core.ParseIP(host)
	if ip == nil || !ip.IsLoopback() {
		return false
	}
	parsedPort := core.Atoi(port)
	if !parsedPort.OK {
		return false
	}
	portNumber := parsedPort.Value.(int)
	return portNumber >= 0 && portNumber <= 65535
}

func loopbackPublicURL(rawURL string) bool {
	parsed, err := url.Parse(rawURL)
	if err != nil || parsed.Scheme != "ssh" || parsed.User != nil || parsed.Host == "" || parsed.Path != "" || parsed.RawQuery != "" || parsed.Fragment != "" {
		return false
	}
	return loopbackAuthority(parsed.Host)
}

func normalizeRepositoryName(configured string) core.Result {
	configured = core.Trim(configured)
	if configured == "" {
		return core.Fail(core.NewError("agent git repository name is required"))
	}
	for _, segment := range core.Split(configured, "/") {
		if segment == "" || segment == "." || segment == ".." {
			return core.Fail(core.NewError("agent git repository name cannot traverse directories"))
		}
	}
	name := softutils.SanitizeRepo(configured)
	if name == "" || name == "." {
		return core.Fail(core.NewError("agent git repository name is invalid"))
	}
	if err := softutils.ValidateRepo(name); err != nil {
		return core.Fail(core.E("gitserver.normalizeRepositoryName", "invalid repository name", err))
	}
	return core.Ok(name)
}

func (service *softServe) acquireOwner() core.Result {
	created := service.runOperation("owner-mkdir", func() core.Result {
		return core.MkdirAll(service.options.DataPath, 0o700)
	})
	if !created.OK {
		return core.Fail(core.E("gitserver.acquireOwner", "failed to create data directory", created.Err()))
	}
	secured := service.runOperation("owner-chmod", func() core.Result {
		return core.Chmod(service.options.DataPath, 0o700)
	})
	if !secured.OK {
		return core.Fail(core.E("gitserver.acquireOwner", "failed to secure data directory", secured.Err()))
	}
	pid := service.options.PID()
	if pid <= 0 {
		return core.Fail(core.NewError("agent git server owner PID must be positive"))
	}
	record := ownerRecord{PID: pid, StartedAt: core.Now().UTC()}
	receipt := core.JSONMarshalString(record)
	ownerPath := core.PathJoin(service.options.DataPath, ownerFilename)
	for attempt := 0; attempt < 2; attempt++ {
		opened := service.runOperation("owner-open", func() core.Result {
			return core.OpenFile(ownerPath, core.O_CREATE|core.O_EXCL|core.O_NOFOLLOW|core.O_WRONLY, 0o600)
		})
		if opened.OK {
			file := opened.Value.(*core.OSFile)
			written := service.runOperation("owner-write", func() core.Result {
				return core.WriteString(file, receipt)
			})
			synced := service.runOperation("owner-sync", func() core.Result {
				return core.ResultOf(nil, file.Sync())
			})
			closed := core.ResultOf(nil, file.Close())
			if closed.OK && service.operations.before != nil {
				closed = service.operations.before("owner-close")
			}
			if !written.OK || !synced.OK || !closed.OK {
				removed := core.Remove(ownerPath)
				if !removed.OK {
					return core.Fail(core.E("gitserver.acquireOwner", "failed to clean incomplete owner receipt", removed.Err()))
				}
				failure := written
				message := "failed to write owner receipt"
				if failure.OK {
					failure = synced
					message = "failed to sync owner receipt"
				}
				if failure.OK {
					failure = closed
					message = "failed to close owner receipt"
				}
				return core.Fail(core.E("gitserver.acquireOwner", message, failure.Err()))
			}
			service.ownerReceipt = receipt
			service.owned = true
			return core.Ok(nil)
		}
		if !core.IsExist(opened.Err()) {
			return core.Fail(core.E("gitserver.acquireOwner", "failed to create owner receipt", opened.Err()))
		}
		existing := service.runOperation("owner-read", func() core.Result {
			return core.ReadFile(ownerPath)
		})
		if !existing.OK {
			return core.Fail(core.E("gitserver.acquireOwner", "failed to read existing owner receipt", existing.Err()))
		}
		var owner ownerRecord
		decoded := core.JSONUnmarshal(existing.Value.([]byte), &owner)
		if !decoded.OK || owner.PID <= 0 || owner.StartedAt.IsZero() {
			return core.Fail(core.NewError("agent git server owner receipt is malformed; refusing unsafe recovery"))
		}
		if service.options.ProcessAlive(owner.PID) {
			return core.Fail(core.Errorf("agent git server data is owned by live PID %d since %s", owner.PID, owner.StartedAt.UTC().Format(time.RFC3339)))
		}
		removed := service.runOperation("owner-remove-stale", func() core.Result {
			return core.Remove(ownerPath)
		})
		if !removed.OK {
			return core.Fail(core.E("gitserver.acquireOwner", "failed to remove stale owner receipt", removed.Err()))
		}
	}
	return core.Fail(core.NewError("agent git server owner receipt could not be acquired"))
}

func (service *softServe) releaseOwner() core.Result {
	if !service.owned {
		return core.Ok(nil)
	}
	ownerPath := core.PathJoin(service.options.DataPath, ownerFilename)
	existing := service.runOperation("owner-release-read", func() core.Result {
		return core.ReadFile(ownerPath)
	})
	if !existing.OK {
		if core.IsNotExist(existing.Err()) {
			service.owned = false
			service.ownerReceipt = ""
			return core.Ok(nil)
		}
		return core.Fail(core.E("gitserver.releaseOwner", "failed to read owner receipt", existing.Err()))
	}
	if string(existing.Value.([]byte)) != service.ownerReceipt {
		return core.Fail(core.NewError("agent git server owner receipt changed; refusing to remove another owner's lock"))
	}
	removed := service.runOperation("owner-release-remove", func() core.Result {
		return core.Remove(ownerPath)
	})
	if !removed.OK {
		return core.Fail(core.E("gitserver.releaseOwner", "failed to remove owner receipt", removed.Err()))
	}
	service.owned = false
	service.ownerReceipt = ""
	return core.Ok(nil)
}

func (service *softServe) startRuntime(ctx context.Context) core.Result {
	listenResult := service.runOperation("listen", func() core.Result {
		return core.NetListen("tcp", service.options.ListenAddress)
	})
	if !listenResult.OK {
		return core.Fail(core.E("gitserver.startRuntime", "failed to listen on private loopback address", listenResult.Err()))
	}
	service.listener = listenResult.Value.(core.Listener)
	service.address = service.listener.Addr().String()
	service.publicURL = core.Concat("ssh://", service.address)

	sshPath := core.PathJoin(service.options.DataPath, "ssh")
	created := service.runOperation("ssh-mkdir", func() core.Result {
		return core.MkdirAll(sshPath, 0o700)
	})
	if !created.OK {
		return core.Fail(core.E("gitserver.startRuntime", "failed to create SSH directory", created.Err()))
	}
	secured := service.runOperation("ssh-chmod", func() core.Result {
		return core.Chmod(sshPath, 0o700)
	})
	if !secured.OK {
		return core.Fail(core.E("gitserver.startRuntime", "failed to secure SSH directory", secured.Err()))
	}
	clientKeyPath := core.PathJoin(sshPath, clientKeyName)
	clientKeyResult := service.runOperation("client-key", func() core.Result {
		clientKey, keyErr := keygen.New(clientKeyPath, keygen.WithKeyType(keygen.Ed25519), keygen.WithWrite())
		return core.ResultOf(clientKey, keyErr)
	})
	if !clientKeyResult.OK {
		return core.Fail(core.E("gitserver.startRuntime", "failed to create SSH client key", clientKeyResult.Err()))
	}
	clientKey := clientKeyResult.Value.(*keygen.KeyPair)
	secured = service.runOperation("client-key-chmod", func() core.Result {
		return core.Chmod(clientKeyPath, 0o600)
	})
	if !secured.OK {
		return core.Fail(core.E("gitserver.startRuntime", "failed to secure SSH client key", secured.Err()))
	}

	configuration := softconfig.DefaultConfig()
	configuration.Name = "LEM Internal Git"
	configuration.DataPath = service.options.DataPath
	configuration.SSH.Enabled = true
	configuration.SSH.ListenAddr = service.address
	configuration.SSH.PublicURL = service.publicURL
	configuration.SSH.KeyPath = core.PathJoin(sshPath, hostKeyName)
	configuration.SSH.ClientKeyPath = clientKeyPath
	configuration.Git.Enabled = false
	configuration.HTTP.Enabled = false
	configuration.Stats.Enabled = false
	configuration.LFS.Enabled = false
	configuration.LFS.SSHEnabled = false
	configuration.Jobs.MirrorPull = ""
	configuration.DB.Driver = "sqlite"
	configuration.DB.DataSource = core.Concat(
		core.PathJoin(service.options.DataPath, softServeDBName),
		"?_pragma=busy_timeout(5000)&_pragma=foreign_keys(1)",
	)
	configuration.InitialAdminKeys = []string{clientKey.AuthorizedKey()}
	validated := service.runOperation("config-validate", func() core.Result {
		return core.ResultOf(nil, configuration.Validate())
	})
	if !validated.OK {
		return core.Fail(core.E("gitserver.startRuntime", "failed to validate Soft Serve configuration", validated.Err()))
	}

	loggerResult := service.runOperation("logger", func() core.Result {
		logger, logFile, loggerErr := softlog.NewLogger(configuration)
		if loggerErr != nil {
			return core.Fail(loggerErr)
		}
		return core.Ok(loggerResources{logger: logger, file: logFile})
	})
	if !loggerResult.OK {
		return core.Fail(core.E("gitserver.startRuntime", "failed to create Soft Serve logger", loggerResult.Err()))
	}
	loggerBundle := loggerResult.Value.(loggerResources)
	service.logFile = loggerBundle.file
	runtimeContext := softconfig.WithContext(ctx, configuration)
	runtimeContext = charmlog.WithContext(runtimeContext, loggerBundle.logger)
	databaseResult := service.runOperation("database-open", func() core.Result {
		database, databaseErr := softdb.Open(runtimeContext, configuration.DB.Driver, configuration.DB.DataSource)
		return core.ResultOf(database, databaseErr)
	})
	if !databaseResult.OK {
		return core.Fail(core.E("gitserver.startRuntime", "failed to open Soft Serve database", databaseResult.Err()))
	}
	database := databaseResult.Value.(*softdb.DB)
	service.database = database
	migrated := service.runOperation("database-migrate", func() core.Result {
		return core.ResultOf(nil, migrate.Migrate(runtimeContext, database))
	})
	if !migrated.OK {
		return core.Fail(core.E("gitserver.startRuntime", "failed to migrate Soft Serve database", migrated.Err()))
	}
	runtimeContext = softdb.WithContext(runtimeContext, database)
	dataStore := softdatabase.New(runtimeContext, database)
	runtimeContext = softstore.WithContext(runtimeContext, dataStore)
	backend := softbackend.New(runtimeContext, configuration, database, dataStore)
	service.backend = backend
	runtimeContext = softbackend.WithContext(runtimeContext, backend)
	keyless := service.runOperation("disable-keyless", func() core.Result {
		return core.ResultOf(nil, backend.SetAllowKeyless(runtimeContext, false))
	})
	if !keyless.OK {
		return core.Fail(core.E("gitserver.startRuntime", "failed to disable keyless access", keyless.Err()))
	}
	anonymous := service.runOperation("disable-anonymous", func() core.Result {
		return core.ResultOf(nil, backend.SetAnonAccess(runtimeContext, access.NoAccess))
	})
	if !anonymous.OK {
		return core.Fail(core.E("gitserver.startRuntime", "failed to disable anonymous access", anonymous.Err()))
	}

	serverResult := service.runOperation("ssh-server", func() core.Result {
		server, serverErr := softssh.NewSSHServer(runtimeContext)
		return core.ResultOf(server, serverErr)
	})
	if !serverResult.OK {
		return core.Fail(core.E("gitserver.startRuntime", "failed to create Soft Serve SSH server", serverResult.Err()))
	}
	server := serverResult.Value.(*softssh.SSHServer)
	service.server = server
	hostKeyResult := service.runOperation("host-key", func() core.Result {
		hostKey, hostKeyErr := keygen.New(configuration.SSH.KeyPath, keygen.WithKeyType(keygen.Ed25519))
		return core.ResultOf(hostKey, hostKeyErr)
	})
	if !hostKeyResult.OK {
		return core.Fail(core.E("gitserver.startRuntime", "failed to load SSH host key", hostKeyResult.Err()))
	}
	hostKey := hostKeyResult.Value.(*keygen.KeyPair)
	knownHostsPath := core.PathJoin(sshPath, knownHostsName)
	knownHostsLine := knownhosts.Line([]string{service.address}, hostKey.PublicKey())
	written := service.runOperation("known-hosts-write", func() core.Result {
		return core.WriteFile(knownHostsPath, []byte(core.Concat(knownHostsLine, "\n")), 0o600)
	})
	if !written.OK {
		return core.Fail(core.E("gitserver.startRuntime", "failed to write SSH known hosts", written.Err()))
	}
	secured = service.runOperation("known-hosts-chmod", func() core.Result {
		return core.Chmod(knownHostsPath, 0o600)
	})
	if !secured.OK {
		return core.Fail(core.E("gitserver.startRuntime", "failed to secure SSH known hosts", secured.Err()))
	}

	done := make(chan struct{})
	service.serveDone = done
	listener := service.listener
	go func() {
		service.setServeResult(core.ResultOf(nil, server.Serve(listener)))
		close(done)
		service.mu.Lock()
		if service.state == "running" {
			service.state = "failed"
		}
		service.mu.Unlock()
	}()
	ready := service.runOperation("ssh-ready", func() core.Result {
		client, clientErr := cryptossh.Dial("tcp", service.address, &cryptossh.ClientConfig{
			User:            "admin",
			Auth:            []cryptossh.AuthMethod{cryptossh.PublicKeys(clientKey.Signer())},
			HostKeyCallback: cryptossh.FixedHostKey(hostKey.PublicKey()),
			Timeout:         service.options.ShutdownTimeout,
		})
		if clientErr != nil {
			return core.Fail(clientErr)
		}
		return core.ResultOf(nil, client.Close())
	})
	if !ready.OK {
		return core.Fail(core.E("gitserver.startRuntime", "SSH readiness handshake failed", ready.Err()))
	}
	return core.Ok(nil)
}

func (service *softServe) cleanupStart() core.Result {
	failures := make([]string, 0, 4)
	if service.server != nil {
		if err := service.server.Close(); err != nil && !core.Is(err, charmssh.ErrServerClosed) {
			failures = append(failures, core.Concat("SSH close: ", err.Error()))
		}
	}
	if service.listener != nil {
		if err := service.listener.Close(); err != nil {
			if !core.Is(err, net.ErrClosed) {
				failures = append(failures, core.Concat("listener close: ", err.Error()))
			}
		}
	}
	if service.database != nil {
		if err := service.database.Close(); err != nil {
			failures = append(failures, core.Concat("database close: ", err.Error()))
		}
	}
	if service.logFile != nil {
		if err := service.logFile.Close(); err != nil {
			failures = append(failures, core.Concat("log close: ", err.Error()))
		}
	}
	if released := service.releaseOwner(); !released.OK {
		failures = append(failures, released.Error())
	}
	service.listener = nil
	service.server = nil
	service.database = nil
	service.backend = nil
	service.logFile = nil
	service.serveDone = nil
	service.setServeResult(core.Ok(nil))
	service.address = ""
	service.publicURL = ""
	if len(failures) > 0 {
		return core.Fail(core.NewError(core.Join("; ", failures...)))
	}
	return core.Ok(nil)
}

func (service *softServe) disableCommandHooks(name string) core.Result {
	for _, hook := range []string{"pre-receive", "update", "post-receive", "post-update"} {
		path := core.PathJoin(service.options.DataPath, "repos", core.Concat(name, ".git"), "hooks", core.Concat(hook, ".d"), "soft-serve")
		removed := core.Remove(path)
		if !removed.OK && !core.IsNotExist(removed.Err()) {
			return core.Fail(core.E("gitserver.disableCommandHooks", core.Concat("failed to disable embedded ", hook, " command hook"), removed.Err()))
		}
	}
	return core.Ok(nil)
}

func (service *softServe) repository(name string) Repository {
	sshPath := core.PathJoin(service.options.DataPath, "ssh")
	return Repository{
		Name:           name,
		CloneURL:       core.Concat(service.publicURL, "/", name),
		IdentityFile:   core.PathJoin(sshPath, clientKeyName),
		KnownHostsFile: core.PathJoin(sshPath, knownHostsName),
	}
}

func (service *softServe) failureReason() string {
	serveResult := service.currentServeResult()
	if !serveResult.OK && serveResult.Err() != nil && !core.Is(serveResult.Err(), charmssh.ErrServerClosed) {
		return serveResult.Error()
	}
	if service.state == "closing" {
		return "private Git service is closing"
	}
	if service.state == "starting" {
		return "private Git service is starting"
	}
	return "private Git service is unavailable"
}

func (service *softServe) setServeResult(result core.Result) {
	service.serveErrMu.Lock()
	service.serveResult = result
	service.serveErrMu.Unlock()
}

func (service *softServe) currentServeResult() core.Result {
	service.serveErrMu.Lock()
	defer service.serveErrMu.Unlock()
	return service.serveResult
}

func (service *softServe) runOperation(name string, operation func() core.Result) core.Result {
	if service.operations.before != nil {
		if allowed := service.operations.before(name); !allowed.OK {
			return allowed
		}
	}
	return operation()
}

func (service *softServe) runPostOperation(name string, result core.Result) core.Result {
	if !result.OK || service.operations.before == nil {
		return result
	}
	return service.operations.before(name)
}
