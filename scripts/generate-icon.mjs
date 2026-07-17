// Rasterizes resources/icon.svg into the PNG and ICO assets used for the
// Electron window / Windows taskbar icon. Run with: npm run icons
import { readFile, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import sharp from 'sharp'
import pngToIco from 'png-to-ico'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const resources = join(root, 'resources')
const svgPath = join(resources, 'icon.svg')

const svg = await readFile(svgPath)

// Main window icon (Electron accepts PNG on Windows/Linux).
const png256 = await sharp(svg, { density: 384 }).resize(256, 256).png().toBuffer()
await writeFile(join(resources, 'icon.png'), png256)

// Multi-size .ico for crisp taskbar / packaged-exe rendering.
const icoSizes = [16, 24, 32, 48, 64, 128, 256]
const icoPngs = await Promise.all(
  icoSizes.map((size) => sharp(svg, { density: 384 }).resize(size, size).png().toBuffer())
)
await writeFile(join(resources, 'icon.ico'), await pngToIco(icoPngs))

console.log('Wrote resources/icon.png and resources/icon.ico')
