// swift-tools-version:6.0
import PackageDescription

// The client is GENERATED locally by `task sdk` (build/sdk is gitignored) —
// this demo depends on that output by relative path; a published SDK would
// be a normal package registry / git dependency instead.
let package = Package(
    name: "LemSwiftExample",
    platforms: [.macOS(.v13)],
    dependencies: [
        // `GeneratedSDK` is a symlink to ../../../build/sdk/swift (gitignored,
        // written by `task sdk`). SwiftPM derives a local path dependency's
        // package identity from the directory's own basename, and that
        // basename would otherwise be "swift" - colliding with this very
        // package's directory (also named swift/) and making the graph
        // unresolvable ("product ... not found in package 'swift'"). The
        // symlink gives the dependency a distinct name without renaming
        // either fixed directory.
        .package(path: "GeneratedSDK"),
    ],
    targets: [
        .executableTarget(
            name: "LemSwiftExample",
            dependencies: [
                .product(name: "LemSDK", package: "GeneratedSDK"),
            ]
        ),
    ]
)
