package api

import (
	core "dappco.re/go"
	"github.com/gin-gonic/gin"
)

func ExampleNewProvider() {
	provider := NewProvider()

	core.Println(provider.Name())
	core.Println(provider.BasePath())
	// Output:
	// ai
	// /v1
}

func ExampleNew() {
	provider := New()

	core.Println(provider.Name())
	// Output:
	// ai
}

func ExampleAIProvider_Name() {
	provider := NewProvider()

	core.Println(provider.Name())
	// Output:
	// ai
}

func ExampleAIProvider_BasePath() {
	provider := NewProvider()

	core.Println(provider.BasePath())
	// Output:
	// /v1
}

func ExampleAIProvider_RegisterRoutes() {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	provider := NewProvider()
	provider.RegisterRoutes(router.Group(provider.BasePath()))

	core.Println(len(router.Routes()))
	// Output:
	// 6
}

func ExampleAIProvider_Describe() {
	provider := NewProvider()
	descriptions := provider.Describe()

	core.Println(len(descriptions))
	core.Println(descriptions[0].Summary)
	// Output:
	// 6
	// Create a text embedding
}
