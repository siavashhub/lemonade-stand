import type { Pitcher } from '@shared/types'
import { readPitchers } from '../config'

// Compute the next epoch-ms occurrence of a local "HH:MM" from `now`.
function nextDailyAt(at: string, now = new Date()): number {
  const [h, m] = at.split(':').map((n) => parseInt(n, 10))
  const next = new Date(now)
  next.setHours(h, m, 0, 0)
  if (next.getTime() <= now.getTime()) next.setDate(next.getDate() + 1)
  return next.getTime()
}

// True when a daily Pitcher's most recent scheduled time has already passed and
// we have no record of pouring it since, i.e. the app was closed at that time.
function missedDaily(p: Pitcher, now = new Date()): boolean {
  if (p.trigger.type !== 'daily') return false
  const [h, m] = p.trigger.at.split(':').map((n) => parseInt(n, 10))
  const todaysFire = new Date(now)
  todaysFire.setHours(h, m, 0, 0)
  if (todaysFire.getTime() > now.getTime()) return false // hasn't come round yet today
  return (p.lastRunAt ?? 0) < todaysFire.getTime()
}

/**
 * Owns the timers for daily Pitchers plus the launch-time on-open / catch-up
 * runs. Pours are handed off to an injected `pour` callback the main process
 * implements (it has the agent, history store, and window). Runs are serialized
 * through a queue and deferred while an interactive agent turn is active, so a
 * scheduled pour never collides with the user's own chat.
 */
export class PitcherScheduler {
  private timers = new Map<string, NodeJS.Timeout>()
  private queue: string[] = []
  private draining = false
  private stopped = false

  constructor(
    private readonly cwd: string,
    private readonly pour: (p: Pitcher) => Promise<void>,
    private readonly isBusy: () => boolean
  ) {}

  /** Call once after the window exists. Runs on-open + any missed dailies, then
   * arms timers for future daily fires. */
  start(): void {
    this.stopped = false
    const pitchers = readPitchers(this.cwd).filter((p) => p.enabled)
    for (const p of pitchers) {
      if (p.trigger.type === 'on-open') this.enqueue(p.id)
      else if (missedDaily(p)) this.enqueue(p.id)
    }
    this.arm(pitchers)
  }

  /** Re-read from disk and rebuild all timers (after a save or delete). */
  reload(): void {
    for (const t of this.timers.values()) clearTimeout(t)
    this.timers.clear()
    this.arm(readPitchers(this.cwd).filter((p) => p.enabled))
  }

  /** Tear down every timer (on quit). */
  stop(): void {
    this.stopped = true
    for (const t of this.timers.values()) clearTimeout(t)
    this.timers.clear()
  }

  private arm(pitchers: Pitcher[]): void {
    for (const p of pitchers) {
      if (p.trigger.type === 'daily') this.scheduleDaily(p)
    }
  }

  private scheduleDaily(p: Pitcher): void {
    if (p.trigger.type !== 'daily') return
    const existing = this.timers.get(p.id)
    if (existing) clearTimeout(existing)
    const delay = nextDailyAt(p.trigger.at) - Date.now()
    // setTimeout (not setInterval) so we recompute after each fire, which is
    // robust to clock drift, DST, and the machine sleeping through a fire.
    const timer = setTimeout(() => {
      this.enqueue(p.id)
      this.scheduleDaily(p) // re-arm for tomorrow
    }, Math.max(0, delay))
    this.timers.set(p.id, timer)
  }

  /** Queue a pour by id. Shared by timers and the manual "Pour now" path so
   * every run goes through the same serialized, busy-aware drain. */
  enqueue(id: string): void {
    if (!this.queue.includes(id)) this.queue.push(id)
    void this.drain()
  }

  private async drain(): Promise<void> {
    if (this.draining) return
    this.draining = true
    try {
      while (this.queue.length && !this.stopped) {
        // Never collide with an interactive turn; wait it out.
        if (this.isBusy()) {
          await new Promise((r) => setTimeout(r, 2000))
          continue
        }
        const id = this.queue.shift()!
        const p = readPitchers(this.cwd).find((x) => x.id === id)
        if (!p || !p.enabled) continue
        try {
          await this.pour(p)
        } catch {
          // pour() is expected to record its own failure; never let one bad
          // pour break the drain loop for the rest of the queue.
        }
      }
    } finally {
      this.draining = false
    }
  }
}
