import type { RendererApi } from '@shared/types'

// Makes `window.api` visible to the renderer's TypeScript. The concrete
// implementation is injected by src/preload/index.ts via contextBridge.
declare global {
  interface Window {
    api: RendererApi
  }
}

export {}
