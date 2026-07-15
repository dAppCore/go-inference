rootProject.name = "lem-sdk-kotlin-example"

// The client is GENERATED locally by `task sdk` (build/sdk is gitignored) —
// this composite build points straight at that output via includeBuild, so
// there's no publishToMavenLocal step. A published SDK would swap this for a
// normal `implementation("re.dappco:lem-sdk:x.y.z")` dependency.
includeBuild("../../../build/sdk/kotlin") {
    dependencySubstitution {
        substitute(module("re.dappco:lem-sdk")).using(project(":"))
    }
}
