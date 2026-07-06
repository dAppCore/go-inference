// SPDX-Licence-Identifier: EUPL-1.2

package driver

import (
	core "dappco.re/go"
	"github.com/gin-gonic/gin"
)

func ExampleNewProvider() {
	p := NewProvider(nil)

	core.Println(p.Name())
	core.Println(p.BasePath())
	// Output:
	// driver
	// /v1/driver
}

func ExampleProvider_Name() {
	p := NewProvider(nil)

	core.Println(p.Name())
	// Output:
	// driver
}

func ExampleProvider_BasePath() {
	p := NewProvider(nil)

	core.Println(p.BasePath())
	// Output:
	// /v1/driver
}

func ExampleProvider_RegisterRoutes() {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	p := NewProvider(&Service{})
	p.RegisterRoutes(router.Group(p.BasePath()))

	core.Println(len(router.Routes()))
	// Output:
	// 4
}

func ExampleProvider_Describe() {
	p := NewProvider(nil)
	descriptions := p.Describe()

	core.Println(len(descriptions))
	core.Println(descriptions[0].Summary)
	// Output:
	// 4
	// List loadable models and serve profiles
}
