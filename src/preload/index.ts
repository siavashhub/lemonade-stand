import { contextBridge, ipcRenderer } from 'electron'
import type {
  AgentEvent,
  ApprovalDecision,
  ChatMessage,
  RendererApi
} from '@shared/types'

// The only bridge between the sandboxed renderer and the Node-privileged main
// process. Keep this surface small and explicit — the renderer can do nothing
// the methods below don't allow.
const api: RendererApi = {
  // Read from the --app-debug=<0|1> argument injected via webPreferences
  // additionalArguments, so the renderer knows the debug flag synchronously.
  debug: process.argv.some((a) => a === '--app-debug=1'),

  sendMessage(messages: ChatMessage[]): Promise<void> {
    return ipcRenderer.invoke('agent:send', messages)
  },

  cancelMessage(): void {
    ipcRenderer.send('agent:cancel')
  },

  listTools() {
    return ipcRenderer.invoke('agent:list-tools')
  },

  getThinkingPhrases() {
    return ipcRenderer.invoke('agent:thinking-phrases')
  },

  onAgentEvent(handler: (event: AgentEvent) => void): () => void {
    const listener = (_event: unknown, payload: AgentEvent): void => handler(payload)
    ipcRenderer.on('agent:event', listener)
    return () => ipcRenderer.removeListener('agent:event', listener)
  },

  respondApproval(id: string, decision: ApprovalDecision): void {
    ipcRenderer.send('agent:approve', id, decision)
  },

  setSpeak(enabled: boolean): Promise<boolean> {
    return ipcRenderer.invoke('agent:set-speak', enabled)
  },

  getSpeak(): Promise<boolean> {
    return ipcRenderer.invoke('agent:get-speak')
  },

  transcribe(audioBase64: string, mimeType: string): Promise<string> {
    return ipcRenderer.invoke('agent:transcribe', audioBase64, mimeType)
  },

  cancelTranscribe(): void {
    ipcRenderer.send('agent:cancel-transcribe')
  },

  getAppVersion(): Promise<string> {
    return ipcRenderer.invoke('app:version')
  },

  checkHealth(): Promise<boolean> {
    return ipcRenderer.invoke('agent:check-health')
  },

  getConnection() {
    return ipcRenderer.invoke('agent:get-connection')
  },

  setConnection(opts: { baseUrl: string; apiKey: string }) {
    return ipcRenderer.invoke('agent:set-connection', opts)
  },

  getContextInfo() {
    return ipcRenderer.invoke('agent:context-info')
  },

  setContextSize(ctxSize: number) {
    return ipcRenderer.invoke('agent:set-context', ctxSize)
  },

  listModels() {
    return ipcRenderer.invoke('agent:list-models')
  },

  loadModel(id: string, ctxSize?: number) {
    return ipcRenderer.invoke('agent:load-model', id, ctxSize)
  },

  listCatalog() {
    return ipcRenderer.invoke('catalog:list')
  },

  listServers() {
    return ipcRenderer.invoke('servers:list')
  },

  configureServer(id: string, opts: { enabled: boolean; path?: string }) {
    return ipcRenderer.invoke('servers:configure', id, opts)
  },

  removeServer(id: string) {
    return ipcRenderer.invoke('servers:remove', id)
  },

  pickPath(kind: 'folder' | 'file') {
    return ipcRenderer.invoke('dialog:pick-path', kind)
  },

  minimizeWindow(): void {
    ipcRenderer.send('window:minimize')
  },

  toggleMaximizeWindow(): void {
    ipcRenderer.send('window:toggle-maximize')
  },

  closeWindow(): void {
    ipcRenderer.send('window:close')
  }
}

contextBridge.exposeInMainWorld('api', api)
