// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"flag"
	"io"

	core "dappco.re/go"
)

// printFlagBlock writes fs's flags in the house help shape — GNU-style long
// options (--name), usage indented beneath, the default appended when one
// exists. Every verb's Usage func renders through this one printer so the
// presentation cannot drift back into a -short/--long mix. (Go's flag parser
// accepts a single dash for the same names; the double is what we PRESENT.)
func printFlagBlock(w io.Writer, fs *flag.FlagSet) {
	fs.VisitAll(func(f *flag.Flag) {
		if f.DefValue == "" {
			core.WriteString(w, core.Sprintf("  --%s\n\t%s\n", f.Name, f.Usage))
			return
		}
		core.WriteString(w, core.Sprintf("  --%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
	})
}
