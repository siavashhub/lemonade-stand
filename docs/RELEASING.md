# Releasing Lemonade Stand

This guide covers how versions work in the app and the exact flow for cutting a
release. Releases use [Semantic Versioning](https://semver.org): `MAJOR.MINOR.PATCH`.

## How the version shows up in the app

Hover the **Lemonade Stand** brand text in the top bar to see the version:

- **`dev`** — a local development run (`npm run dev`) or an unpackaged preview.
  The app detects this via Electron's `app.isPackaged` and never claims a
  version it wasn't released as.
- **`vX.Y.Z`** — an installed/packaged build. The value comes from the `version`
  field in [`package.json`](../package.json), which is baked into the installer
  by `electron-builder`.

The version is resolved in the main process (`app:version` IPC in
[`src/main/index.ts`](../src/main/index.ts)) and displayed as the brand tooltip
in [`src/renderer/App.tsx`](../src/renderer/App.tsx).

## The release gate

A tag can only become a release if a matching notes file exists. The gate
([`scripts/check-release.mjs`](../scripts/check-release.mjs)) enforces three
things:

1. `package.json` `version` is a valid semantic version.
2. `releases/vX.Y.Z.md` exists and is non-empty.
3. If a tag is supplied, it equals `vX.Y.Z` — so the tag, the package version and
   the notes file can never drift apart.

The gate runs both locally (`npm run check-release`) and as the first CI job, so
a release is impossible without its notes file.

## Starting work on a new sprint
Create a new release branch if not exists
```powershell
git checkout -b release/0.2
git push --set-upstream origin release/0.2
```

work on the release branch like below example:
```powershell
git add package.json
git commit -m "add: updated package file"
git push 
```

## Cutting a release
When ready to create a cut follow below:

### 1. Bump the version

Update `version` in `package.json` to the new semantic version (e.g. `0.2.0`).
You can do it by hand or with npm (which also creates a matching git tag —
remove `--git-tag-version` if you'd rather tag manually):

```powershell
npm version 0.2.0 --no-git-tag-version
```

Other examples:
- npm version patch --no-git-tag-version
- npm version minor --no-git-tag-version   

### 2. Write the release notes

Copy the template and fill it in. The filename **must** be `v<version>.md`:

```powershell
Copy-Item releases/TEMPLATE.md releases/v0.2.0.md
# edit releases/v0.2.0.md — this text becomes the GitHub Release body
```

### 3. Validate locally (the gate)

```powershell
npm run check-release      # confirms notes exist and version is valid
npm run typecheck
npm run audit              # fails on any high/critical dependency vulnerability
```

Optionally build the installer for your OS to smoke-test packaging:

```powershell
npm run package:win        # or package:mac / package:linux
# artifacts land in dist/
```

### 4. Commit, tag, and push

```powershell
git add releases/v0.2.0.md
git commit -m "add: Release v0.2.0"
git push 
```

Create a PR into main and merge.

Pull the changes from main and tag to cut a release:
```powershell
git checkout main
git pull
git tag v0.2.0
git push origin v0.2.0
```

### 5. The pipeline takes over

Pushing the `v*` tag triggers [`.github/workflows/release.yml`](../.github/workflows/release.yml):

1. **gate** — runs `check-release.mjs "<tag>"` and a type-check.
2. **audit** — scans npm dependencies with `npm audit` and fails on any
   high/critical advisory.
3. **build** — packages installers on Windows, macOS, and Linux in parallel.
4. **release** — creates the GitHub Release named after the tag, using
   `releases/<tag>.md` as the body, and attaches every installer.

Both **gate** and **audit** must pass before any build runs. If either fails
(missing notes, tag mismatch, or a vulnerable dependency), no build runs and no
release is created — fix the issue, then retag.

> A vulnerability found here means a dependency needs updating. Run
> `npm audit fix` (or bump the offending package), commit, and retag.

## Hotfix
If fixes can be done on on new release then continue usual flow, if not 
and a patch/hotfix is needed for previous release:

```powershell
git checkout release/0.2
git checkout -b hotfix/0.2.1 v0.2.0     # <-- off the TAG, not main
```

PR into release/0.2
```powershell
git checkout release/0.2
git tag v0.2.1
```

Finally cherry pick hotfix commits to main:

forward-port the hotfix to main via a temp branch
```powershell
git checkout main
git pull
git checkout -b forwardport/v0.2.1        # temp branch
git cherry-pick <hotfix-commit-sha>       # or a range: <sha1>^..<sha2>
git push --set-upstream origin forwardport/v0.2.1
```

open PR: forwardport/v0.2.1  --->  main, then merge (squash is fine)

## Local packaging (no release)

To build installers without tagging:

```powershell
npm run package            # current OS, output in dist/
```

## Quick reference

| Command | Purpose |
| --- | --- |
| `npm run dev` | Run the app locally (version shows `dev`). |
| `npm run check-release` | Validate the release gate for the current `package.json` version. |
| `npm run audit` | Fail on any high/critical npm dependency vulnerability. |
| `npm run package` | Build installers for the current OS into `dist/`. |
| `npm run package:win\|mac\|linux` | Build installers for a specific OS. |
| `git tag vX.Y.Z && git push --tags` | Trigger the release pipeline. |
