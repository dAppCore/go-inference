plugins {
    kotlin("jvm") version "2.2.20"
    application
}

repositories {
    mavenCentral()
}

dependencies {
    implementation("re.dappco:lem-sdk:0.1.0")

    // The generated client declares its transport (okhttp) and JSON (moshi)
    // libraries as `implementation`, not `api` — Gradle doesn't leak them to
    // a composite-build consumer even though the client's own public methods
    // take a Call.Factory. Redeclared here, version-pinned to match
    // build/sdk/kotlin/build.gradle, because this example talks raw OkHttp +
    // the SDK's own Moshi instance directly (see README Friction).
    implementation("com.squareup.okhttp3:okhttp:5.1.0")
    implementation("com.squareup.moshi:moshi-kotlin:1.15.2")
}

application {
    mainClass.set("re.dappco.lem.example.MainKt")
}
