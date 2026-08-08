// SPDX-Licence-Identifier: EUPL-1.2

// Package pathx splits paths on either separator convention.
//
// core.PathBase and core.PathDir match only the platform's own separator
// (core.Env("DS")), so on Windows a '/'-separated path reads as having no
// separator at all: PathBase("/models/gemma3-1b") hands back the whole string
// and PathDir("/models/x/adapter.safetensors") hands back ".". Go itself
// accepts '/' in paths on every platform — it arrives from config files, CLI
// flags, HTTP model ids and Hugging Face refs — so any code deriving a name or
// a parent directory from a caller-supplied path needs both conventions.
//
// Base and Dir here are the both-separator forms. They do not clean, resolve
// or otherwise normalise: callers that need a real filesystem path still build
// it with core.PathJoin.
package pathx

import core "dappco.re/go"

// Base returns the last element of p, treating both '/' and the platform
// separator as boundaries. Trailing separators are ignored. An empty path, or
// one that is nothing but separators, returns "".
//
// '\' is a boundary only where the platform says so: on POSIX it is an
// ordinary filename character, and treating it as a separator there would
// corrupt legitimate names.
//
//	pathx.Base("/models/gemma3-1b")     // "gemma3-1b" — every platform
//	pathx.Base("org/model-a")           // "model-a"   — every platform
//	pathx.Base(`C:\models\gemma3-1b\`)  // "gemma3-1b" on Windows
func Base(p string) string {
	p = trimTrailingSeparators(p)
	if p == "" {
		return ""
	}
	if i := lastSeparator(p); i >= 0 {
		return p[i+1:]
	}
	return p
}

// Dir returns all but the last element of p, treating both '/' and the
// platform separator as boundaries. A path with no separator returns "",
// letting callers distinguish "no parent recorded" from the "." that
// core.PathDir reports.
//
//	pathx.Dir("/models/my-lora/adapter.safetensors")    // "/models/my-lora"
//	pathx.Dir("adapter.safetensors")                    // ""
//	pathx.Dir(`C:\models\my-lora\adapter.safetensors`)  // `C:\models\my-lora` on Windows
func Dir(p string) string {
	p = trimTrailingSeparators(p)
	i := lastSeparator(p)
	if i < 0 {
		return ""
	}
	if i == 0 {
		return p[:1] // a root-anchored path keeps its leading separator
	}
	return p[:i]
}

// Join appends child to dir using the separator dir is already written with,
// falling back to the platform's own when dir carries none.
//
// Unlike core.PathJoin it does not clean the result — callers on a hot path
// feed already-canonical directories — and it preserves the caller's
// convention rather than rewriting a '/'-spelled Windows path into a mixed
// one. An empty dir yields the bare child, matching PathJoin's "empty root =
// relative result".
//
//	pathx.Join("/models/my-lora", "adapter_config.json")   // "/models/my-lora/adapter_config.json"
//	pathx.Join("/models/my-lora/", "adapter_config.json")  // "/models/my-lora/adapter_config.json"
func Join(dir, child string) string {
	if dir == "" {
		return child
	}
	separator := separatorOf(dir)
	if core.HasSuffix(dir, separator) {
		return dir + child
	}
	return dir + separator + child
}

// separatorOf reports the separator p is written with — the last one it
// contains — or the platform's own when it contains none.
func separatorOf(p string) string {
	if i := lastSeparator(p); i >= 0 {
		return p[i : i+1]
	}
	if separator := core.Env("DS"); separator != "" {
		return separator
	}
	return "/"
}

// lastSeparator reports the index of the final '/' or platform separator in p,
// or -1 when p carries neither.
func lastSeparator(p string) int {
	best := core.LastIndex(p, "/")
	if separator := core.Env("DS"); separator != "" && separator != "/" {
		if i := core.LastIndex(p, separator); i > best {
			best = i
		}
	}
	return best
}

// trimTrailingSeparators drops any run of trailing separators, so that a path
// written with a trailing slash names the same element as one without.
func trimTrailingSeparators(p string) string {
	separator := core.Env("DS")
	for len(p) > 1 {
		tail := p[len(p)-1:]
		if tail != "/" && (separator == "" || tail != separator) {
			break
		}
		p = p[:len(p)-1]
	}
	return p
}
