// SPDX-Licence-Identifier: EUPL-1.2

package driver

import (
	"time"

	core "dappco.re/go"
	coreprocess "dappco.re/go/process"
)

func ExampleNewService() {
	app := core.New(core.WithName("process", coreprocess.NewService(coreprocess.Options{})))
	proc, _ := core.ServiceFor[*coreprocess.Service](app, "process")
	svc := NewService(proc, nil)

	core.Println(len(svc.Status()))
	// Output:
	// 0
}

// ExampleService_Serve shows the input-validation refusal — an unknown
// runtime is rejected before anything is spawned.
func ExampleService_Serve() {
	svc := &Service{served: map[string]*Served{}, everReady: map[string]bool{}, restartLog: map[string][]time.Time{}}
	r := svc.Serve(ServeRequest{Runtime: "bogus"})

	core.Println(r.OK)
	// Output:
	// false
}

// ExampleService_LastServed shows the nothing-persisted shape: with no prior
// serve recorded, ok is false.
func ExampleService_LastServed() {
	prevHome := core.Getenv("HOME")
	tmp := core.MkdirTemp("", "driver-example-*")
	if !tmp.OK {
		core.Println(false)
		return
	}
	defer core.RemoveAll(tmp.Value.(string))
	defer core.Setenv("HOME", prevHome)
	core.Setenv("HOME", tmp.Value.(string))

	svc := &Service{}
	_, ok := svc.LastServed()

	core.Println(ok)
	// Output:
	// false
}

// ExampleService_Stop shows the nothing-served shape: stopping a runtime that
// isn't tracked refuses rather than silently succeeding.
func ExampleService_Stop() {
	svc := &Service{served: map[string]*Served{}}
	r := svc.Stop(RuntimeMLX)

	core.Println(r.OK)
	// Output:
	// false
}

func ExampleService_Status() {
	svc := &Service{}

	core.Println(len(svc.Status()))
	// Output:
	// 0
}

// ExampleService_Models shows the empty-catalogue shape: a resolvable HOME
// with no models or profiles directory yet is a valid, non-error answer.
func ExampleService_Models() {
	prevHome := core.Getenv("HOME")
	tmp := core.MkdirTemp("", "driver-example-*")
	if !tmp.OK {
		core.Println(false)
		return
	}
	defer core.RemoveAll(tmp.Value.(string))
	defer core.Setenv("HOME", prevHome)
	core.Setenv("HOME", tmp.Value.(string))

	svc := &Service{}
	r := svc.Models()
	cat := r.Value.(Catalogue)

	core.Println(r.OK)
	core.Println(len(cat.Models))
	core.Println(len(cat.Profiles))
	// Output:
	// true
	// 0
	// 0
}
