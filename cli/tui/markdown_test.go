// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"strings"
	"testing"

	"github.com/charmbracelet/x/ansi"
)

func TestMarkdownRenderer_Good(t *testing.T) {
	renderer := newMarkdownRenderer("midnight")
	markdown := "# Heading\n\n" +
		"A deliberately long sentence that should wrap differently when the transcript becomes narrow.\n\n" +
		"- first item\n- **bold item** with `inline()` code\n\n" +
		"```go\nfunc main() { println(\"hello\") }\n```\n\n" +
		"[OpenAI](https://openai.com) :sparkles:"

	wide := renderer.Render("turn-wide", markdown, 72)
	narrow := renderer.Render("turn-narrow", markdown, 32)
	for name, rendered := range map[string]string{"wide": wide, "narrow": narrow} {
		plain := ansi.Strip(rendered)
		for _, text := range []string{"Heading", "first item", "bold item", "inline()", "func main()", "OpenAI", "✨"} {
			if !strings.Contains(plain, text) {
				t.Fatalf("%s render missing %q:\n%s", name, text, plain)
			}
		}
	}
	if strings.Count(ansi.Strip(narrow), "\n") <= strings.Count(ansi.Strip(wide), "\n") {
		t.Fatal("narrow Markdown did not wrap to more lines than wide Markdown")
	}
}

func TestMarkdownRenderer_Bad(t *testing.T) {
	renderer := newMarkdownRenderer("midnight")
	const markdown = "**keep this text**"
	if got := renderer.Render("turn-invalid", markdown, 0); got != markdown {
		t.Fatalf("invalid-width fallback = %q, want original plain text", got)
	}
}

func TestMarkdownRenderer_Ugly(t *testing.T) {
	renderer := newMarkdownRenderer("midnight")
	first := renderer.Render("turn-cache", "## Cached\n\ncontent", 48)
	before := renderer.Stats()
	second := renderer.Render("turn-cache", "## Cached\n\ncontent", 48)
	after := renderer.Stats()
	if first != second {
		t.Fatal("cached render changed output")
	}
	if after.Hits != before.Hits+1 || after.Misses != before.Misses {
		t.Fatalf("cache stats before=%+v after=%+v, want one hit", before, after)
	}
}
