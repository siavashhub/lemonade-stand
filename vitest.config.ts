import { resolve } from 'node:path'
import { defineConfig } from 'vitest/config'

// Vitest runs the main-process (Node) logic and MCP integration tests. The
// renderer/Electron layers aren't exercised here — these tests cover the parts
// that can run headless in CI without a GPU, a model, or a display.
export default defineConfig({
  resolve: {
    alias: {
      // Mirror the tsconfig `@shared/*` path mapping so tests can import the
      // same shared types the app uses.
      '@shared': resolve(__dirname, 'src/shared')
    }
  },
  test: {
    environment: 'node',
    // Integration tests spawn real MCP subprocesses (npx/uvx) whose first run
    // may download a package; give them room rather than the 5s default.
    testTimeout: 60_000,
    hookTimeout: 60_000,
    include: ['test/**/*.test.ts']
  }
})
