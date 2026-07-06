// SPDX-Licence-Identifier: EUPL-1.2

package driver

import (
	core "dappco.re/go"
	"github.com/gin-gonic/gin"
)

func ExampleNewInferenceProvider() {
	p := NewInferenceProvider(nil)

	core.Println(p.Name())
	core.Println(p.BasePath())
	// Output:
	// inference
	// /v1
}

func ExampleInferenceProvider_Name() {
	p := NewInferenceProvider(nil)

	core.Println(p.Name())
	// Output:
	// inference
}

func ExampleInferenceProvider_BasePath() {
	p := NewInferenceProvider(nil)

	core.Println(p.BasePath())
	// Output:
	// /v1
}

func ExampleInferenceProvider_RegisterRoutes() {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	p := NewInferenceProvider(&Service{})
	p.RegisterRoutes(router.Group(p.BasePath()))

	core.Println(len(router.Routes()))
	// Output:
	// 4
}

func ExampleInferenceProvider_Describe() {
	p := NewInferenceProvider(nil)
	descriptions := p.Describe()

	core.Println(len(descriptions))
	core.Println(descriptions[0].Summary)
	// Output:
	// 4
	// Create a chat completion
}

// ExampleService_Target shows the pre-serve shape: with nothing served, ok is
// false and the loopback address/model key are both empty.
func ExampleService_Target() {
	s := &Service{}
	_, _, ok := s.Target()

	core.Println(ok)
	// Output:
	// false
}

// ExampleService_WaitCapacity shows the no-limiter shape: WaitCapacity is a
// no-op and returns immediately when the host has no rate limiter configured.
func ExampleService_WaitCapacity() {
	s := &Service{}
	err := s.WaitCapacity(core.Background(), "org/model", 100)

	core.Println(err == nil)
	// Output:
	// true
}

// ExampleService_Record shows the no-limiter shape: Record is a no-op when the
// host has no rate limiter configured.
func ExampleService_Record() {
	s := &Service{}
	s.Record("org/model", 10, 20)

	core.Println(s.limiter == nil)
	// Output:
	// true
}
