// Release gate. Fails (non-zero exit) unless the repo is in a releasable state:
//
//   1. package.json has a valid semantic version.
//   2. A matching release-notes file exists at releases/v<version>.md and is
//      non-empty.
//   3. When a tag is supplied (CI passes the pushed tag, or you can pass one as
//      the first CLI arg / RELEASE_TAG env var), it equals `v<version>` so the
//      tag, the package version and the notes file can never drift apart.
//
// Run locally before tagging:   npm run check-release
// Or validate a specific tag:   node scripts/check-release.mjs v0.0.1
//
// The GitHub Actions release workflow runs this first, so a release cannot be
// built or published without its notes file.

import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join, resolve } from 'node:path'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')

// Strict semver (major.minor.patch, optional -prerelease and +build).
const SEMVER =
  /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z-.]+)?(?:\+[0-9A-Za-z-.]+)?$/

function fail(message) {
  console.error(`\u2717 Release gate failed: ${message}`)
  process.exit(1)
}

const pkg = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8'))
const version = pkg.version

if (!SEMVER.test(version)) {
  fail(`package.json version "${version}" is not a valid semantic version.`)
}

// The tag being released, if any: first CLI arg wins, else RELEASE_TAG env var.
const tag = (process.argv[2] ?? process.env.RELEASE_TAG ?? '').trim()
if (tag) {
  const expected = `v${version}`
  if (tag !== expected) {
    fail(
      `tag "${tag}" does not match package.json version (expected "${expected}"). ` +
        `Bump package.json to ${tag.replace(/^v/, '')} or retag.`
    )
  }
}

const notesPath = join(root, 'releases', `v${version}.md`)
let notes
try {
  notes = readFileSync(notesPath, 'utf8')
} catch {
  fail(
    `missing release notes. Create releases/v${version}.md describing this ` +
      `release before tagging (see releases/TEMPLATE.md).`
  )
}

if (notes.trim().length === 0) {
  fail(`release notes releases/v${version}.md is empty.`)
}

console.log(`\u2713 Release gate passed for v${version} (releases/v${version}.md).`)
