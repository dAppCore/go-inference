// SPDX-Licence-Identifier: EUPL-1.2

package driver

import (
	"slices"
	"testing"
	"time"

	core "dappco.re/go"
	coreprocess "dappco.re/go/process"
)

// --- NewService ---

func TestDriver_NewService_Good(t *testing.T) {
	proc := benchProcSvc(t)
	svc := NewService(proc, nil)
	if svc == nil {
		t.Fatal("NewService returned nil")
	}
	if svc.proc != proc {
		t.Fatal("NewService did not retain the process service")
	}
	if svc.served == nil || svc.everReady == nil || svc.restartLog == nil {
		t.Fatal("NewService left a tracking map nil")
	}
	if got := svc.Status(); len(got) != 0 {
		t.Fatalf("fresh Service.Status() = %v, want empty", got)
	}
}

// TestDriver_NewService_Bad proves NewService gives each call its own
// tracking maps — mutating one Service's served set must never leak into a
// second Service built from the same process supervisor.
func TestDriver_NewService_Bad(t *testing.T) {
	proc := benchProcSvc(t)
	first := NewService(proc, nil)
	second := NewService(proc, nil)

	first.served[RuntimeMLX] = &Served{Runtime: RuntimeMLX}
	if _, ok := second.served[RuntimeMLX]; ok {
		t.Fatal("NewService shared the served map across two instances built from the same proc")
	}
}

// TestDriver_NewService_Ugly covers a ServiceRuntime whose Core is nil (never
// registered against a Core app) — NewService must skip RegisterAction
// rather than panic dereferencing a nil Core.
func TestDriver_NewService_Ugly(t *testing.T) {
	proc := &coreprocess.Service{
		ServiceRuntime: core.NewServiceRuntime[coreprocess.Options](nil, coreprocess.Options{}),
	}
	svc := NewService(proc, nil)
	if svc == nil {
		t.Fatal("NewService returned nil")
	}
	if svc.served == nil || svc.everReady == nil || svc.restartLog == nil {
		t.Fatal("NewService left a tracking map nil even with no Core to register against")
	}
}

// --- Serve ---

func TestDriver_Service_Serve_Good(t *testing.T) {
	newHealthyDriver(t, RuntimeMLX)
	addr := newHealthServer(t, true)
	proc := benchProcSvc(t)
	svc := NewService(proc, nil)
	t.Cleanup(func() { svc.Stop(RuntimeMLX) })

	r := svc.Serve(ServeRequest{Runtime: RuntimeMLX, Addr: addr, Model: "lthn/LEM-Gemma3-1B"})
	if !r.OK {
		t.Fatalf("Serve failed: %v", r.Value)
	}
	served, ok := r.Value.(Served)
	if !ok {
		t.Fatalf("Serve returned %T, want Served", r.Value)
	}
	if !served.Ready || !served.Running {
		t.Fatalf("Serve returned %+v, want Ready+Running", served)
	}
	if served.Addr != addr || served.Model != "lthn/LEM-Gemma3-1B" || served.Runtime != RuntimeMLX {
		t.Fatalf("Serve returned %+v, want it echoing the request", served)
	}

	// A successful serve also persists the choice for LastServed.
	last, ok := svc.LastServed()
	if !ok || last.Model != "lthn/LEM-Gemma3-1B" || last.Runtime != RuntimeMLX {
		t.Fatalf("LastServed() = %+v, %t, want the just-served request", last, ok)
	}
}

func TestDriver_Service_Serve_Bad(t *testing.T) {
	svc := &Service{served: map[string]*Served{}, everReady: map[string]bool{}, restartLog: map[string][]time.Time{}}
	r := svc.Serve(ServeRequest{Runtime: "bogus"})
	if r.OK {
		t.Fatal("Serve with an unknown runtime succeeded, want refusal")
	}
	if !core.Contains(r.Error(), "unknown runtime") {
		t.Fatalf("Serve error = %q, want it naming the unknown runtime", r.Error())
	}
}

// TestDriver_Serve_Ugly covers the spawned-but-never-ready path: the process
// starts fine but nothing answers /v1/health, so Serve must time out and
// fail even though the driver stays tracked.
func TestDriver_Service_Serve_Ugly(t *testing.T) {
	newHealthyDriver(t, RuntimeMLX)
	shrinkReadyWait(t, 300*time.Millisecond, 50*time.Millisecond)
	addr := freeDeadAddr(t)

	proc := benchProcSvc(t)
	svc := NewService(proc, nil)
	t.Cleanup(func() { svc.Stop(RuntimeMLX) })

	r := svc.Serve(ServeRequest{Runtime: RuntimeMLX, Addr: addr})
	if r.OK {
		t.Fatal("Serve with a never-ready driver succeeded, want a readiness failure")
	}
	if !core.Contains(r.Error(), "not ready") {
		t.Fatalf("Serve error = %q, want it reporting not-ready", r.Error())
	}
	found := false
	for _, sv := range svc.Status() {
		if sv.Runtime == RuntimeMLX {
			found = true
		}
	}
	if !found {
		t.Fatal("a spawned-but-never-ready driver was not left tracked for Status/Stop")
	}
}

// --- spawn ---

func TestDriver_Spawn_Good(t *testing.T) {
	newHealthyDriver(t, RuntimeMLX)
	proc := benchProcSvc(t)
	s := &Service{proc: proc, served: map[string]*Served{}, everReady: map[string]bool{}, restartLog: map[string][]time.Time{}}

	r := s.spawn(RuntimeMLX, runtimeBinary[RuntimeMLX], "127.0.0.1:9", ServeRequest{Runtime: RuntimeMLX, Model: "m"})
	if !r.OK {
		t.Fatalf("spawn failed: %v", r.Value)
	}
	p, ok := r.Value.(*coreprocess.Process)
	if !ok || p == nil {
		t.Fatalf("spawn returned %T, want *coreprocess.Process", r.Value)
	}
	t.Cleanup(func() { _ = proc.Kill(p.ID) })

	sv := s.served[RuntimeMLX]
	if sv == nil || !sv.Running || sv.ProcessID != p.ID || sv.Model != "m" {
		t.Fatalf("spawn left served state = %+v, want a tracked running entry matching the request", sv)
	}
}

func TestDriver_Spawn_Bad(t *testing.T) {
	proc := benchProcSvc(t)
	pid := benchSleepProc(t, proc)
	s := &Service{proc: proc, served: map[string]*Served{
		RuntimeMLX: {Runtime: RuntimeMLX, Model: "already-running", ProcessID: pid, Running: true},
	}, everReady: map[string]bool{}, restartLog: map[string][]time.Time{}}

	r := s.spawn(RuntimeMLX, runtimeBinary[RuntimeMLX], "127.0.0.1:9", ServeRequest{Runtime: RuntimeMLX, Model: "new-model"})
	if r.OK {
		t.Fatal("spawn over an already-serving runtime succeeded, want refusal")
	}
	if !core.Contains(r.Error(), "already serving") {
		t.Fatalf("spawn error = %q, want it naming the conflict", r.Error())
	}
}

func TestDriver_Spawn_Ugly(t *testing.T) {
	isolateDriverLookup(t)
	proc := benchProcSvc(t)
	s := &Service{proc: proc, served: map[string]*Served{}, everReady: map[string]bool{}, restartLog: map[string][]time.Time{}}

	r := s.spawn(RuntimeMLX, runtimeBinary[RuntimeMLX], "127.0.0.1:9", ServeRequest{Runtime: RuntimeMLX})
	if r.OK {
		t.Fatal("spawn with no resolvable binary succeeded, want refusal")
	}
	if !core.Contains(r.Error(), "not found") {
		t.Fatalf("spawn error = %q, want it reporting the missing binary", r.Error())
	}
}

// --- swapOrPass ---

func TestDriver_SwapOrPass_Good(t *testing.T) {
	proc := benchProcSvc(t)
	pid := benchSleepProc(t, proc)
	s := &Service{proc: proc, served: map[string]*Served{
		RuntimeMLX: {Runtime: RuntimeMLX, Model: "same-model", ProcessID: pid, Running: true, Addr: "127.0.0.1:9", Ready: true},
	}}

	res := s.swapOrPass(RuntimeMLX, ServeRequest{Runtime: RuntimeMLX, Model: "same-model"})
	if res == nil {
		t.Fatal("swapOrPass on a same-model request returned nil, want the current Served snapshot")
	}
	if !res.OK {
		t.Fatalf("swapOrPass same-model result not OK: %v", res.Value)
	}
	sv, ok := res.Value.(Served)
	if !ok || sv.ProcessID != pid {
		t.Fatalf("swapOrPass returned %+v, want the untouched current Served", res.Value)
	}
	r := proc.Get(pid)
	if !r.OK || !r.Value.(*coreprocess.Process).IsRunning() {
		t.Fatal("swapOrPass killed a process serving the SAME model it was asked for")
	}
}

func TestDriver_SwapOrPass_Bad(t *testing.T) {
	s := &Service{served: map[string]*Served{}}
	res := s.swapOrPass(RuntimeMLX, ServeRequest{Runtime: RuntimeMLX, Model: "anything"})
	if res != nil {
		t.Fatalf("swapOrPass with nothing served returned %+v, want nil (proceed to cold-start)", res)
	}
}

// TestDriver_SwapOrPass_Ugly covers the drain-and-replace path: a different
// model is live, so the old driver must be killed and awaited before
// swapOrPass hands back control to Serve for the cold start.
func TestDriver_SwapOrPass_Ugly(t *testing.T) {
	proc := benchProcSvc(t)
	pid := benchSleepProc(t, proc)
	s := &Service{proc: proc, served: map[string]*Served{
		RuntimeMLX: {Runtime: RuntimeMLX, Model: "old-model", ProcessID: pid, Running: true},
	}, everReady: map[string]bool{RuntimeMLX: true}, restartLog: map[string][]time.Time{}}

	res := s.swapOrPass(RuntimeMLX, ServeRequest{Runtime: RuntimeMLX, Model: "new-model"})
	if res != nil {
		t.Fatalf("swapOrPass on a model change returned %+v, want nil (proceed to cold-start)", res)
	}
	if _, ok := s.served[RuntimeMLX]; ok {
		t.Fatal("swapOrPass left the old entry tracked after a model change")
	}
	r := proc.Get(pid)
	if !r.OK {
		t.Fatal("old process vanished entirely instead of just exiting")
	}
	if r.Value.(*coreprocess.Process).IsRunning() {
		t.Fatal("swapOrPass did not kill the old, differently-modelled driver before returning")
	}
}

// --- Stop ---

func TestDriver_Service_Stop_Good(t *testing.T) {
	proc := benchProcSvc(t)
	pid := benchSleepProc(t, proc)
	s := &Service{proc: proc, served: map[string]*Served{
		RuntimeMLX: {Runtime: RuntimeMLX, ProcessID: pid, Running: true},
	}, everReady: map[string]bool{RuntimeMLX: true}, restartLog: map[string][]time.Time{RuntimeMLX: {time.Now()}}}

	r := s.Stop(RuntimeMLX)
	if !r.OK || r.Value.(string) != RuntimeMLX {
		t.Fatalf("Stop = %+v, want Ok(%q)", r, RuntimeMLX)
	}
	if _, ok := s.served[RuntimeMLX]; ok {
		t.Fatal("Stop left the runtime tracked")
	}
	if _, ok := s.restartLog[RuntimeMLX]; ok {
		t.Fatal("Stop left a stale restart log entry")
	}
	if !waitUntil(2*time.Second, 10*time.Millisecond, func() bool {
		pr := proc.Get(pid)
		return !pr.OK || !pr.Value.(*coreprocess.Process).IsRunning()
	}) {
		t.Fatal("Stop did not actually kill the process within 2s")
	}
}

func TestDriver_Service_Stop_Bad(t *testing.T) {
	s := &Service{served: map[string]*Served{}}
	r := s.Stop(RuntimeMLX)
	if r.OK {
		t.Fatal("Stop with nothing served for the runtime succeeded, want refusal")
	}
	if !core.Contains(r.Error(), "no driver serving") {
		t.Fatalf("Stop error = %q, want it reporting nothing served", r.Error())
	}
}

// TestDriver_Stop_Ugly covers stopping a runtime whose tracked process has
// already exited on its own — Kill on an already-dead process must still be
// treated as a successful stop, not surfaced as an error.
func TestDriver_Service_Stop_Ugly(t *testing.T) {
	dir := t.TempDir()
	quick := core.PathJoin(dir, "quick")
	if r := core.WriteFile(quick, []byte("#!/bin/sh\nexit 0\n"), 0o755); !r.OK {
		t.Fatalf("write quick-exit script: %v", r.Value)
	}
	proc := benchProcSvc(t)
	sr := proc.StartWithOptions(core.Background(), coreprocess.RunOptions{Command: quick, Detach: true, KillGroup: true})
	if !sr.OK {
		t.Fatalf("spawn quick-exit script: %v", sr.Value)
	}
	p := sr.Value.(*coreprocess.Process)
	_ = proc.Wait(p.ID) // block until it has actually finished on its own

	s := &Service{proc: proc, served: map[string]*Served{
		RuntimeMLX: {Runtime: RuntimeMLX, ProcessID: p.ID, Running: true},
	}, everReady: map[string]bool{}, restartLog: map[string][]time.Time{}}

	r := s.Stop(RuntimeMLX)
	if !r.OK {
		t.Fatalf("Stop over an already-exited process failed: %v", r.Value)
	}
}

// --- Status ---

func TestDriver_Service_Status_Good(t *testing.T) {
	proc := benchProcSvc(t)
	pid := benchSleepProc(t, proc)
	s := &Service{proc: proc, served: map[string]*Served{
		RuntimeMLX: {Runtime: RuntimeMLX, ProcessID: pid, Addr: "127.0.0.1:1", Running: true, Ready: true},
	}}

	out := s.Status()
	if len(out) != 1 {
		t.Fatalf("Status() returned %d entries, want 1", len(out))
	}
	if !out[0].Running || !out[0].Ready || out[0].Addr != "127.0.0.1:1" {
		t.Fatalf("Status() = %+v, want the live entry reflected faithfully", out[0])
	}
}

// TestDriver_Status_Bad covers a stale tracked entry whose process exited on
// its own — Status must correct both Running and Ready rather than trusting
// the last-known snapshot.
func TestDriver_Service_Status_Bad(t *testing.T) {
	dir := t.TempDir()
	quick := core.PathJoin(dir, "quick")
	if r := core.WriteFile(quick, []byte("#!/bin/sh\nexit 0\n"), 0o755); !r.OK {
		t.Fatalf("write quick-exit script: %v", r.Value)
	}
	proc := benchProcSvc(t)
	sr := proc.StartWithOptions(core.Background(), coreprocess.RunOptions{Command: quick, Detach: true, KillGroup: true})
	if !sr.OK {
		t.Fatalf("spawn quick-exit script: %v", sr.Value)
	}
	p := sr.Value.(*coreprocess.Process)
	_ = proc.Wait(p.ID)

	s := &Service{proc: proc, served: map[string]*Served{
		RuntimeMLX: {Runtime: RuntimeMLX, ProcessID: p.ID, Running: true, Ready: true},
	}}
	out := s.Status()
	if len(out) != 1 {
		t.Fatalf("Status() returned %d entries, want 1", len(out))
	}
	if out[0].Running {
		t.Fatal("Status() reported Running=true for a process that already exited")
	}
	if out[0].Ready {
		t.Fatal("Status() reported Ready=true for a non-running driver — stale readiness must be corrected")
	}
}

// TestDriver_Service_Status_Ugly covers independence across entries: a live
// runtime and a stale (already-exited) one tracked together must each be
// corrected on their own — the live entry must not be dragged down by the
// stale one, and vice versa.
func TestDriver_Service_Status_Ugly(t *testing.T) {
	dir := t.TempDir()
	quick := core.PathJoin(dir, "quick")
	if r := core.WriteFile(quick, []byte("#!/bin/sh\nexit 0\n"), 0o755); !r.OK {
		t.Fatalf("write quick-exit script: %v", r.Value)
	}
	proc := benchProcSvc(t)
	sr := proc.StartWithOptions(core.Background(), coreprocess.RunOptions{Command: quick, Detach: true, KillGroup: true})
	if !sr.OK {
		t.Fatalf("spawn quick-exit script: %v", sr.Value)
	}
	stale := sr.Value.(*coreprocess.Process)
	_ = proc.Wait(stale.ID)
	livePID := benchSleepProc(t, proc)

	s := &Service{proc: proc, served: map[string]*Served{
		RuntimeMLX:  {Runtime: RuntimeMLX, ProcessID: stale.ID, Running: true, Ready: true},
		RuntimeCUDA: {Runtime: RuntimeCUDA, ProcessID: livePID, Running: true, Ready: true},
	}}
	out := s.Status()
	if len(out) != 2 {
		t.Fatalf("Status() returned %d entries, want 2", len(out))
	}
	for _, sv := range out {
		switch sv.Runtime {
		case RuntimeMLX:
			if sv.Running || sv.Ready {
				t.Fatalf("Status() left the stale mlx entry = %+v, want Running=false, Ready=false", sv)
			}
		case RuntimeCUDA:
			if !sv.Running || !sv.Ready {
				t.Fatalf("Status() dragged the live cuda entry down = %+v, want it untouched", sv)
			}
		}
	}
}

// --- Models ---

func TestDriver_Service_Models_Good(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	modelsDir := core.PathJoin(home, "Lethean", "lem", "models")
	profilesDir := core.PathJoin(home, "Lethean", "conf", "models")
	if r := core.MkdirAll(modelsDir, 0o755); !r.OK {
		t.Fatalf("mkdir models: %v", r.Value)
	}
	if r := core.MkdirAll(profilesDir, 0o755); !r.OK {
		t.Fatalf("mkdir profiles: %v", r.Value)
	}
	if r := core.WriteFile(core.PathJoin(modelsDir, "gemma-3-1b"), []byte("x"), 0o644); !r.OK {
		t.Fatalf("seed model: %v", r.Value)
	}
	if r := core.WriteFile(core.PathJoin(modelsDir, ".hidden"), []byte("x"), 0o644); !r.OK {
		t.Fatalf("seed dotfile: %v", r.Value)
	}
	if r := core.WriteFile(core.PathJoin(profilesDir, "default.json"), []byte("{}"), 0o644); !r.OK {
		t.Fatalf("seed profile: %v", r.Value)
	}

	svc := &Service{}
	r := svc.Models()
	if !r.OK {
		t.Fatalf("Models failed: %v", r.Value)
	}
	cat := r.Value.(Catalogue)
	if !slices.Equal(cat.Models, []string{"gemma-3-1b"}) {
		t.Fatalf("Catalogue.Models = %v, want just the visible model, dotfile excluded", cat.Models)
	}
	if !slices.Equal(cat.Profiles, []string{"default.json"}) {
		t.Fatalf("Catalogue.Profiles = %v, want the one profile", cat.Profiles)
	}
}

func TestDriver_Service_Models_Bad(t *testing.T) {
	t.Setenv("HOME", "")
	svc := &Service{}
	r := svc.Models()
	if r.OK {
		t.Fatal("Models succeeded with no resolvable home directory, want failure")
	}
}

// TestDriver_Service_Models_Ugly covers a resolvable home with neither the
// models nor the profiles directory present — an empty catalogue is a valid
// answer, never an error (listNames treats a missing dir as "nothing here").
func TestDriver_Service_Models_Ugly(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	svc := &Service{}
	r := svc.Models()
	if !r.OK {
		t.Fatalf("Models failed over a resolvable home with no populated dirs: %v", r.Value)
	}
	cat := r.Value.(Catalogue)
	if len(cat.Models) != 0 || len(cat.Profiles) != 0 {
		t.Fatalf("Catalogue = %+v, want both empty when neither dir exists", cat)
	}
}

// --- onProcessEvent ---

func TestDriver_OnProcessEvent_Good(t *testing.T) {
	isolateDriverLookup(t) // any restart attempt handleExit fires must fail harmlessly
	proc := benchProcSvc(t)
	s := &Service{proc: proc, served: map[string]*Served{
		RuntimeMLX: {Runtime: RuntimeMLX, ProcessID: "tracked-1", Model: "m", Running: true},
	}, everReady: map[string]bool{RuntimeMLX: true}, restartLog: map[string][]time.Time{}}

	res := s.onProcessEvent(nil, coreprocess.ActionProcessExited{ID: "tracked-1"})
	if !res.OK {
		t.Fatalf("onProcessEvent = %+v, want Ok", res)
	}
	if s.served[RuntimeMLX].Running {
		t.Fatal("onProcessEvent did not route ActionProcessExited through to handleExit")
	}
}

func TestDriver_OnProcessEvent_Bad(t *testing.T) {
	proc := benchProcSvc(t)
	s := &Service{proc: proc, served: map[string]*Served{
		RuntimeMLX: {Runtime: RuntimeMLX, ProcessID: "tracked-1", Running: true},
	}, everReady: map[string]bool{RuntimeMLX: true}, restartLog: map[string][]time.Time{}}

	res := s.onProcessEvent(nil, coreprocess.ActionProcessOutput{ID: "tracked-1", Line: "hello"})
	if !res.OK {
		t.Fatalf("onProcessEvent(unrelated message) = %+v, want Ok(nil) no-op", res)
	}
	if !s.served[RuntimeMLX].Running {
		t.Fatal("onProcessEvent acted on a non-exit message")
	}
}

// --- handleExit ---

// TestDriver_HandleExit_Good exercises the full crash-restart loop for real:
// serve a driver, kill its process WITHOUT going through Stop (so the exit
// reads as a crash), and confirm the exit action drives an automatic
// re-serve on the same (model, profile, addr).
func TestDriver_HandleExit_Good(t *testing.T) {
	newHealthyDriver(t, RuntimeMLX)
	addr := newHealthServer(t, true)
	proc := benchProcSvc(t)
	svc := NewService(proc, nil)
	t.Cleanup(func() { svc.Stop(RuntimeMLX) })

	r := svc.Serve(ServeRequest{Runtime: RuntimeMLX, Addr: addr, Model: "demo/model"})
	if !r.OK {
		t.Fatalf("initial Serve failed: %v", r.Value)
	}
	first := r.Value.(Served)

	if kr := proc.Kill(first.ProcessID); !kr.OK {
		t.Fatalf("simulated crash kill failed: %v", kr.Value)
	}

	var restarted Served
	ok := waitUntil(5*time.Second, 20*time.Millisecond, func() bool {
		for _, sv := range svc.Status() {
			if sv.Runtime == RuntimeMLX && sv.ProcessID != first.ProcessID && sv.Running {
				restarted = sv
				return true
			}
		}
		return false
	})
	if !ok {
		t.Fatal("driver was not auto-restarted after a simulated crash")
	}
	if restarted.ProcessID == first.ProcessID {
		t.Fatal("restarted entry still carries the crashed process id")
	}
}

func TestDriver_HandleExit_Bad(t *testing.T) {
	proc := benchProcSvc(t)
	s := &Service{proc: proc, served: map[string]*Served{
		RuntimeMLX: {Runtime: RuntimeMLX, ProcessID: "tracked-1", Running: true},
	}, everReady: map[string]bool{RuntimeMLX: true}, restartLog: map[string][]time.Time{}}

	s.handleExit("some-foreign-pid")

	if !s.served[RuntimeMLX].Running {
		t.Fatal("handleExit mutated a tracked entry in response to a foreign/unknown process id")
	}
	if len(s.restartLog[RuntimeMLX]) != 0 {
		t.Fatal("handleExit recorded a restart for a foreign process id")
	}
}

// TestDriver_HandleExit_Ugly covers the restart-storm guard: repeated crashes
// of the same tracked entry stop restarting once the budget is spent.
func TestDriver_HandleExit_Ugly(t *testing.T) {
	isolateDriverLookup(t) // restart attempts must fail harmlessly, not launch a real driver
	proc := benchProcSvc(t)
	s := &Service{proc: proc, served: map[string]*Served{
		RuntimeMLX: {Runtime: RuntimeMLX, ProcessID: "crashy", Model: "m", Running: true},
	}, everReady: map[string]bool{RuntimeMLX: true}, restartLog: map[string][]time.Time{}}

	for i := 0; i < maxRestarts; i++ {
		s.handleExit("crashy")
	}
	if got := len(s.restartLog[RuntimeMLX]); got != maxRestarts {
		t.Fatalf("restartLog has %d entries after %d allowed crashes, want %d", got, maxRestarts, maxRestarts)
	}

	s.handleExit("crashy") // one more, over budget
	if got := len(s.restartLog[RuntimeMLX]); got != maxRestarts {
		t.Fatalf("restartLog has %d entries after the over-budget crash, want it capped at %d", got, maxRestarts)
	}
}

// --- trackedByPID ---

func TestDriver_TrackedByPID_Good(t *testing.T) {
	s := &Service{served: map[string]*Served{
		RuntimeMLX:  {Runtime: RuntimeMLX, ProcessID: "pid-mlx"},
		RuntimeCUDA: {Runtime: RuntimeCUDA, ProcessID: "pid-cuda"},
	}}
	rt, sv := s.trackedByPID("pid-cuda")
	if rt != RuntimeCUDA || sv == nil || sv.ProcessID != "pid-cuda" {
		t.Fatalf("trackedByPID(pid-cuda) = (%q, %+v), want the cuda entry", rt, sv)
	}
}

func TestDriver_TrackedByPID_Bad(t *testing.T) {
	s := &Service{served: map[string]*Served{
		RuntimeMLX: {Runtime: RuntimeMLX, ProcessID: "pid-mlx"},
	}}
	rt, sv := s.trackedByPID("no-such-pid")
	if rt != "" || sv != nil {
		t.Fatalf("trackedByPID(unknown) = (%q, %+v), want (\"\", nil)", rt, sv)
	}
}

// --- allowRestart ---

func TestDriver_AllowRestart_Good(t *testing.T) {
	s := &Service{restartLog: map[string][]time.Time{}}
	for i := 0; i < maxRestarts; i++ {
		if !s.allowRestart(RuntimeMLX) {
			t.Fatalf("allowRestart refused call %d, want it allowed within budget", i+1)
		}
	}
	if got := len(s.restartLog[RuntimeMLX]); got != maxRestarts {
		t.Fatalf("restartLog has %d entries, want %d after %d allowed calls", got, maxRestarts, maxRestarts)
	}
}

func TestDriver_AllowRestart_Bad(t *testing.T) {
	s := &Service{restartLog: map[string][]time.Time{}}
	for i := 0; i < maxRestarts; i++ {
		s.allowRestart(RuntimeMLX)
	}
	if s.allowRestart(RuntimeMLX) {
		t.Fatal("allowRestart allowed a call past the restart-storm budget")
	}
	if got := len(s.restartLog[RuntimeMLX]); got != maxRestarts {
		t.Fatalf("restartLog grew past the cap to %d entries, want it to stay at %d", got, maxRestarts)
	}
}

// TestDriver_AllowRestart_Ugly covers window pruning: entries older than
// restartWindow must not count against the budget.
func TestDriver_AllowRestart_Ugly(t *testing.T) {
	stale := time.Now().Add(-restartWindow - time.Minute)
	s := &Service{restartLog: map[string][]time.Time{
		RuntimeMLX: {stale, stale, stale},
	}}
	if !s.allowRestart(RuntimeMLX) {
		t.Fatal("allowRestart refused after stale entries should have been pruned out of the window")
	}
	if got := len(s.restartLog[RuntimeMLX]); got != 1 {
		t.Fatalf("restartLog has %d entries after pruning + one fresh allow, want exactly 1", got)
	}
}

// --- running ---

func TestDriver_Running_Good(t *testing.T) {
	proc := benchProcSvc(t)
	pid := benchSleepProc(t, proc)
	s := &Service{proc: proc}
	if !s.running(pid) {
		t.Fatal("running() reported false for a live process")
	}
}

func TestDriver_Running_Bad(t *testing.T) {
	proc := benchProcSvc(t)
	s := &Service{proc: proc}
	if s.running("no-such-process") {
		t.Fatal("running() reported true for an unknown process id")
	}
}

func TestDriver_Running_Ugly(t *testing.T) {
	dir := t.TempDir()
	quick := core.PathJoin(dir, "quick")
	if r := core.WriteFile(quick, []byte("#!/bin/sh\nexit 0\n"), 0o755); !r.OK {
		t.Fatalf("write quick-exit script: %v", r.Value)
	}
	proc := benchProcSvc(t)
	sr := proc.StartWithOptions(core.Background(), coreprocess.RunOptions{Command: quick, Detach: true, KillGroup: true})
	if !sr.OK {
		t.Fatalf("spawn quick-exit script: %v", sr.Value)
	}
	p := sr.Value.(*coreprocess.Process)
	_ = proc.Wait(p.ID)

	s := &Service{proc: proc}
	if s.running(p.ID) {
		t.Fatal("running() reported true for a process that already exited on its own")
	}
}

// --- serveArgs ---

func TestDriver_ServeArgs_Good(t *testing.T) {
	req := ServeRequest{Model: "org/model", Context: 8192, Profile: "balanced", NoAutoProfile: true}
	got := serveArgs(req, "127.0.0.1:36911")
	want := []string{"serve", "--addr", "127.0.0.1:36911", "--model", "org/model", "--context", "8192", "--profile", "balanced", "--no-auto-profile"}
	if !slices.Equal(got, want) {
		t.Fatalf("serveArgs = %v, want %v", got, want)
	}
}

func TestDriver_ServeArgs_Bad(t *testing.T) {
	got := serveArgs(ServeRequest{}, "127.0.0.1:36911")
	want := []string{"serve", "--addr", "127.0.0.1:36911"}
	if !slices.Equal(got, want) {
		t.Fatalf("serveArgs(empty request) = %v, want the bare model-less serve invocation %v", got, want)
	}
}

// TestDriver_ServeArgs_Ugly proves the optional flags toggle independently
// rather than being coupled to Model/Profile being set.
func TestDriver_ServeArgs_Ugly(t *testing.T) {
	got := serveArgs(ServeRequest{NoAutoProfile: true}, "127.0.0.1:1")
	want := []string{"serve", "--addr", "127.0.0.1:1", "--no-auto-profile"}
	if !slices.Equal(got, want) {
		t.Fatalf("serveArgs(NoAutoProfile only) = %v, want %v", got, want)
	}
}

// --- resolveDriverBinary ---

func TestDriver_ResolveDriverBinary_Good(t *testing.T) {
	dir := isolateDriverLookup(t)
	want := writeFakeDriver(t, dir, "lthn-mlx")
	if got := resolveDriverBinary("lthn-mlx"); got != want {
		t.Fatalf("resolveDriverBinary = %q, want %q (the CORE_AI_DRIVER_DIR candidate)", got, want)
	}
}

func TestDriver_ResolveDriverBinary_Bad(t *testing.T) {
	isolateDriverLookup(t)
	if got := resolveDriverBinary("lthn-mlx"); got != "lthn-mlx" {
		t.Fatalf("resolveDriverBinary = %q, want the bare name (PATH-fallback signal)", got)
	}
}

// TestDriver_ResolveDriverBinary_Ugly proves CORE_AI_DRIVER_DIR takes
// precedence over ~/Lethean/bin when both carry a same-named candidate.
func TestDriver_ResolveDriverBinary_Ugly(t *testing.T) {
	dir := isolateDriverLookup(t)
	want := writeFakeDriver(t, dir, "lthn-mlx")

	lethBin := core.PathJoin(core.Env("HOME"), "Lethean", "bin")
	if r := core.MkdirAll(lethBin, 0o755); !r.OK {
		t.Fatalf("mkdir ~/Lethean/bin: %v", r.Value)
	}
	writeFakeDriver(t, lethBin, "lthn-mlx")

	if got := resolveDriverBinary("lthn-mlx"); got != want {
		t.Fatalf("resolveDriverBinary = %q, want the CORE_AI_DRIVER_DIR candidate %q to win", got, want)
	}
}

// --- waitDriverReady ---

func TestDriver_WaitDriverReady_Good(t *testing.T) {
	addr := newHealthServer(t, true)
	ready, reason := waitDriverReady(addr, 2*time.Second)
	if !ready {
		t.Fatalf("waitDriverReady = false (%s), want true against a healthy server", reason)
	}
	if reason != "" {
		t.Fatalf("waitDriverReady reason = %q on success, want empty", reason)
	}
}

func TestDriver_WaitDriverReady_Bad(t *testing.T) {
	addr := newHealthServer(t, false) // always 503
	ready, reason := waitDriverReady(addr, 300*time.Millisecond)
	if ready {
		t.Fatal("waitDriverReady = true against a server that never answers 200")
	}
	if reason == "" {
		t.Fatal("waitDriverReady returned no reason for the timeout")
	}
}

func TestDriver_WaitDriverReady_Ugly(t *testing.T) {
	addr := freeDeadAddr(t) // nothing listening at all
	ready, reason := waitDriverReady(addr, 300*time.Millisecond)
	if ready {
		t.Fatal("waitDriverReady = true against an address with nothing listening")
	}
	if reason == "" {
		t.Fatal("waitDriverReady returned no reason for a connection failure")
	}
}

// --- listNames ---

func TestDriver_ListNames_Good(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"alpha", "beta", ".hidden"} {
		if r := core.WriteFile(core.PathJoin(dir, name), []byte("x"), 0o644); !r.OK {
			t.Fatalf("seed %s: %v", name, r.Value)
		}
	}
	got := listNames(dir)
	slices.Sort(got)
	want := []string{"alpha", "beta"}
	if !slices.Equal(got, want) {
		t.Fatalf("listNames = %v, want dotfiles excluded: %v", got, want)
	}
}

func TestDriver_ListNames_Bad(t *testing.T) {
	if got := listNames(core.PathJoin(t.TempDir(), "does-not-exist")); got != nil {
		t.Fatalf("listNames(missing dir) = %v, want nil", got)
	}
}

func TestDriver_ListNames_Ugly(t *testing.T) {
	dir := t.TempDir()
	got := listNames(dir)
	if got == nil {
		t.Fatal("listNames(empty existing dir) = nil, want a non-nil empty slice")
	}
	if len(got) != 0 {
		t.Fatalf("listNames(empty dir) = %v, want empty", got)
	}
}

// --- servePersistPath ---

func TestDriver_ServePersistPath_Good(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	want := core.PathJoin(home, "Lethean", "lem", "lthn-ai-serve.json")
	if got := servePersistPath(); got != want {
		t.Fatalf("servePersistPath = %q, want %q", got, want)
	}
}

func TestDriver_ServePersistPath_Bad(t *testing.T) {
	t.Setenv("HOME", "")
	if got := servePersistPath(); got != "" {
		t.Fatalf("servePersistPath = %q, want empty when the home dir can't resolve", got)
	}
}

// --- persistServe ---

func TestDriver_PersistServe_Good(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	persistServe(persistedServe{Runtime: RuntimeMLX, Model: "org/model", Profile: "balanced"})

	data := core.ReadFile(servePersistPath())
	if !data.OK {
		t.Fatal("persistServe did not write the serve-state file")
	}
	var got persistedServe
	if r := core.JSONUnmarshal(data.Value.([]byte), &got); !r.OK {
		t.Fatalf("persisted file did not parse: %v", r.Value)
	}
	if got.Runtime != RuntimeMLX || got.Model != "org/model" || got.Profile != "balanced" {
		t.Fatalf("persisted = %+v, want it echoing what was persisted", got)
	}
}

func TestDriver_PersistServe_Bad(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	persistServe(persistedServe{Runtime: RuntimeMLX, Model: ""}) // model-less: nothing to restore

	if core.Stat(servePersistPath()).OK {
		t.Fatal("persistServe wrote a file for a model-less serve, want a silent no-op")
	}
}

// --- LastServed ---

func TestDriver_Service_LastServed_Good(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	persistServe(persistedServe{Runtime: RuntimeCUDA, Model: "org/model", Profile: "p1"})

	svc := &Service{}
	req, ok := svc.LastServed()
	if !ok {
		t.Fatal("LastServed() ok=false after a successful persist")
	}
	if req.Runtime != RuntimeCUDA || req.Model != "org/model" || req.Profile != "p1" {
		t.Fatalf("LastServed() = %+v, want it echoing the persisted request", req)
	}
}

func TestDriver_Service_LastServed_Bad(t *testing.T) {
	t.Setenv("HOME", t.TempDir()) // resolves fine, but nothing was ever persisted
	svc := &Service{}
	if _, ok := svc.LastServed(); ok {
		t.Fatal("LastServed() ok=true with nothing persisted")
	}
}

func TestDriver_Service_LastServed_Ugly(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	path := servePersistPath()
	if r := core.MkdirAll(core.PathDir(path), 0o755); !r.OK {
		t.Fatalf("mkdir: %v", r.Value)
	}
	if r := core.WriteFile(path, []byte("not json"), 0o644); !r.OK {
		t.Fatalf("seed corrupt file: %v", r.Value)
	}
	svc := &Service{}
	if _, ok := svc.LastServed(); ok {
		t.Fatal("LastServed() ok=true over a corrupt persisted file")
	}
}
