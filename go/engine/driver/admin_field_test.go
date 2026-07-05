// SPDX-Licence-Identifier: EUPL-1.2

package driver

import (
	"testing"
	"time"

	core "dappco.re/go"
	coreprocess "dappco.re/go/process"
)

// TestDownloadLane_FieldExercise walks the EXACT path the Models pane's Get
// button takes: spawn the real engine model-less via the driver, allowlist
// a curated repo, kick the engine-side HF download, poll to done, verify
// the weights landed. Green-unit-tests are a hypothesis; this exercises the
// user path (a real ~0.8GB HuggingFace pull into ~/Lethean/lem/models).
//
// Gated: opt in with LEM_FIELD_DOWNLOAD=1 and point CORE_AI_DRIVER_DIR at a
// built lthn-mlx (e.g. ~/Code/core/go-mlx/bin). The downloaded model is
// deliberately KEPT — it's the curated catalogue's smallest entry and
// immediately useful to the app.
//
//	LEM_FIELD_DOWNLOAD=1 CORE_AI_DRIVER_DIR=$HOME/Code/core/go-mlx/bin \
//	  go test -run TestDownloadLane_FieldExercise -v -timeout 15m ./pkg/driver/
func TestDownloadLane_FieldExercise(t *testing.T) {
	if core.Env("LEM_FIELD_DOWNLOAD") != "1" {
		t.Skip("field exercise — set LEM_FIELD_DOWNLOAD=1 (real engine spawn + ~0.8GB HF download)")
	}
	const repo = "mlx-community/gemma-3-1b-it-4bit"

	procConclave := core.New(core.WithName("process", coreprocess.NewService(coreprocess.Options{})))
	procSvc, ok := core.ServiceFor[*coreprocess.Service](procConclave, "process")
	if !ok {
		t.Fatal("process supervisor not registered")
	}
	svc := NewService(procSvc, nil)

	if r := AllowRepo(repo); !r.OK {
		t.Fatalf("AllowRepo: %v", r.Value)
	}

	if r := svc.Serve(ServeRequest{Runtime: RuntimeMLX, Model: ""}); !r.OK {
		t.Fatalf("Serve (model-less): %v", r.Value)
	}
	defer svc.Stop(RuntimeMLX)

	// The engine may still be binding — retry the kickoff briefly, exactly
	// like EngineService.DownloadCurated does.
	var kick core.Result
	for attempt := 0; attempt < 30; attempt++ {
		kick = svc.DownloadModel(RuntimeMLX, repo, "main")
		if kick.OK {
			break
		}
		time.Sleep(500 * time.Millisecond)
	}
	if !kick.OK {
		t.Fatalf("DownloadModel never reached the engine: %v", kick.Value)
	}
	job := kick.Value.(DownloadJob)
	if job.ID == "" {
		t.Fatalf("kickoff returned no job id: %+v", job)
	}
	t.Logf("download job %s started for %s", job.ID, repo)

	deadline := time.Now().Add(12 * time.Minute)
	for {
		if time.Now().After(deadline) {
			t.Fatalf("download did not finish in time; last: %+v", job)
		}
		time.Sleep(2 * time.Second)
		r := svc.DownloadJobStatus(RuntimeMLX, job.ID)
		if !r.OK {
			t.Fatalf("poll: %v", r.Value)
		}
		job = r.Value.(DownloadJob)
		if job.Status == "failed" {
			t.Fatalf("download failed: %s", job.Error)
		}
		if job.Status == "done" {
			break
		}
	}

	if job.DestPath == "" {
		t.Fatal("done job carries no dest path")
	}
	if stat := core.Stat(job.DestPath); !stat.OK {
		t.Fatalf("dest path %s missing after done", job.DestPath)
	}
	t.Logf("FIELD VERIFIED: %s → %s (%d bytes, %d files)",
		repo, job.DestPath, job.BytesTotal, job.FileCount)
}
