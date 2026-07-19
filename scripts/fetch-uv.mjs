// Download the `uv` / `uvx` binaries for the CURRENT platform into
// resources/bin, so the packaged app can launch the uvx-based MCP servers
// (Web Fetch, Git, Time, SQLite) without the user installing Python/uv.
//
// electron-builder bundles resources/bin as an extraResource (see
// electron-builder.yml), and the MCP manager prepends that folder to each
// stdio server's PATH at spawn time. Each OS release build runs this script,
// so every installer carries only its own platform's binary.
//
// Run manually any time with:  node scripts/fetch-uv.mjs [--force]
import { execFileSync } from 'node:child_process'
import { existsSync, mkdirSync, mkdtempSync, readdirSync, copyFileSync, rmSync, statSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { basename, dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

// Pin a known-good uv version for reproducible builds. Bump deliberately.
const UV_VERSION = '0.11.29'

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..')
const OUT_DIR = join(ROOT, 'resources', 'bin')
const FORCE = process.argv.includes('--force')

// Map Node's platform/arch to uv's release "target triple" and archive kind.
function target() {
  const { platform, arch } = process
  const a = arch === 'arm64' ? 'aarch64' : arch === 'x64' ? 'x86_64' : arch
  if (platform === 'win32') return { triple: `${a}-pc-windows-msvc`, ext: 'zip', exe: '.exe' }
  if (platform === 'darwin') return { triple: `${a}-apple-darwin`, ext: 'tar.gz', exe: '' }
  if (platform === 'linux') return { triple: `${a}-unknown-linux-gnu`, ext: 'tar.gz', exe: '' }
  throw new Error(`Unsupported platform for uv bundling: ${platform}/${arch}`)
}

// Recursively find files named uv/uvx (optionally with .exe) under a dir.
function findBinaries(dir, exe) {
  const wanted = new Set([`uv${exe}`, `uvx${exe}`])
  const found = []
  for (const name of readdirSync(dir)) {
    const full = join(dir, name)
    if (statSync(full).isDirectory()) found.push(...findBinaries(full, exe))
    else if (wanted.has(name)) found.push(full)
  }
  return found
}

async function main() {
  const { triple, ext, exe } = target()

  mkdirSync(OUT_DIR, { recursive: true })
  const already = existsSync(join(OUT_DIR, `uvx${exe}`)) && existsSync(join(OUT_DIR, `uv${exe}`))
  if (already && !FORCE) {
    console.log(`[fetch-uv] uv/uvx already present in ${OUT_DIR}; skipping (use --force to refresh)`)
    return
  }

  const asset = `uv-${triple}.${ext}`
  const url = `https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/${asset}`
  console.log(`[fetch-uv] downloading ${url}`)

  const res = await fetch(url)
  if (!res.ok) throw new Error(`download failed: ${res.status} ${res.statusText} for ${url}`)
  const buf = Buffer.from(await res.arrayBuffer())

  const work = mkdtempSync(join(tmpdir(), 'uv-fetch-'))
  try {
    const archive = join(work, asset)
    const { writeFileSync } = await import('node:fs')
    writeFileSync(archive, buf)

    // bsdtar (Windows 10+, macOS, Linux) extracts both .zip and .tar.gz.
    execFileSync('tar', ['-xf', archive, '-C', work], { stdio: 'inherit' })

    const bins = findBinaries(work, exe)
    if (bins.length === 0) throw new Error(`no uv/uvx binaries found inside ${asset}`)
    for (const src of bins) {
      const dest = join(OUT_DIR, basename(src))
      copyFileSync(src, dest)
      if (exe === '') execFileSync('chmod', ['+x', dest])
      console.log(`[fetch-uv] installed ${dest}`)
    }
  } finally {
    rmSync(work, { recursive: true, force: true })
  }
  console.log(`[fetch-uv] done (uv ${UV_VERSION}, ${triple}) -> ${OUT_DIR}`)
}

main().catch((err) => {
  console.error('[fetch-uv] failed:', err)
  process.exit(1)
})
