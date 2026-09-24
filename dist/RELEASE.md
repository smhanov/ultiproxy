# Release procedure (T008)

`dist/install.sh` downloads its tarballs from
`https://github.com/smhanov/ultiproxy/releases/latest/download/`, so a
release is only "done" when the GitHub Release carries the assets. The
`.github/workflows/release.yml` workflow automates this: pushing a tag
`v*` runs the repo gates (`go build ./...`, `go test ./...`,
`go vet ./...`), cross-builds the four targets install.sh supports
(linux/darwin x amd64->x86_64/arm64), stamps the tag into the binary via
`-ldflags "-X main.version=<tag>"`, and uploads
`ultiproxy-<os>-<arch>.tar.gz` + `.sha256` sidecars. A final `verify` job
downloads the assets back and checks them the way install.sh parses them.

Cross-builds use `CGO_ENABLED=0`: go.mod's sqlite driver is
`modernc.org/sqlite` (pure Go), so no cgo cross toolchain is needed.

## Publishing a release (manual)

1. Run the gates locally and confirm green:
   `go build ./... && go test ./... && go vet ./...`
2. Pick the next tag (`v0.1.1`, or next minor). The tag — `v` prefix
   included — is what `/healthz` will report as `version`, so tag exactly
   what you want users to see.
3. Tag and push (this is what triggers the workflow; nothing else does):
   ```bash
   git tag v0.1.1
   git push origin v0.1.1
   ```
4. Watch the run: `gh run list --workflow release.yml` /
   `gh run watch <id>`. All three jobs (gates, 4x build, verify) must be
   green. Do NOT mark the release as prerelease: `releases/latest/download`
   resolves to the newest non-prerelease tag, and install.sh depends on it.
5. Verify the version stamp (AC4) on an installed binary:
   ```bash
   curl -s http://localhost:9050/healthz   # {"status":"ok","version":"v0.1.1",...}
   ultiproxy version                        # ultiproxy v0.1.1
   ```
6. Verify the installer end-to-end (AC3) on a clean machine for at least
   linux-x86_64 and darwin-arm64 (over successive releases is fine):
   ```bash
   curl -fsSL https://raw.githubusercontent.com/smhanov/ultiproxy/main/dist/install.sh | sh
   ultiproxy serve & sleep 2
   curl -s http://localhost:9050/healthz
   ```

## Recovery

The installer is stateless: if a release is bad, delete the Release (and
the tag) on GitHub, delete the local tag (`git tag -d vX.Y.Z`), fix,
retag, and push again.
