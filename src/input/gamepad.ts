/** Gamepad API -> [vx, vy, wz] velocity command + FSM/dance button combos.
 *
 * Uses the browser's *standard mapping* (the Gamepad-API equivalent of SDL's
 * controller DB): axes[0/1] = left stick X/Y, axes[2/3] = right stick X/Y;
 * button indices are fixed across pads (0 A, 1 B, 2 X, 3 Y, 4 LB, 5 RB,
 * 6 LT, 7 RT, 8 select/back, 9 start, 12-15 dpad).
 *
 * Combos mirror the on-robot controller bindings where they make sense here:
 *   LT held + dpad-up      -> Stance          (robot: L2_hold + up)
 *   LT held + B            -> Damping         (robot: L2_hold + B)
 *   LT held + Y            -> ZeroTorque      (robot: L2_hold + Y from Damping)
 *   RT held + A            -> Policy          (robot: R2_hold + A enters loco)
 *   RT held + dpad-left/right -> prev/next policy
 */
import type { Command } from "../core/types";

export const BTN = {
  A: 0, B: 1, X: 2, Y: 3, LB: 4, RB: 5, LT: 6, RT: 7,
  SELECT: 8, START: 9, L3: 10, R3: 11,
  UP: 12, DOWN: 13, LEFT: 14, RIGHT: 15,
} as const;

export type ComboAction =
  | { kind: "state"; state: "zero_torque" | "damping" | "lock" | "stance" | "policy" }
  | { kind: "cycle"; step: 1 | -1 }
  | { kind: "reset" };

const COMBOS: { hold: number; press: number; action: ComboAction }[] = [
  { hold: BTN.LT, press: BTN.UP, action: { kind: "state", state: "stance" } },
  { hold: BTN.LT, press: BTN.B, action: { kind: "state", state: "damping" } },
  { hold: BTN.LT, press: BTN.Y, action: { kind: "state", state: "zero_torque" } },
  { hold: BTN.LT, press: BTN.X, action: { kind: "state", state: "lock" } },
  { hold: BTN.RT, press: BTN.A, action: { kind: "state", state: "policy" } },
  { hold: BTN.RT, press: BTN.RIGHT, action: { kind: "cycle", step: 1 } },
  { hold: BTN.RT, press: BTN.LEFT, action: { kind: "cycle", step: -1 } },
  { hold: BTN.RT, press: BTN.B, action: { kind: "reset" } },
];

export class GamepadInput {
  deadzone = 0.08;
  scale: Command = [1.0, 1.0, 1.0];
  private prevButtons: boolean[] = [];
  // Rest position = per-axis median of a QUIET 30-poll window (~0.5 s where
  // every axis stays within a small band). Chrome only exposes a pad after a
  // button press, so the first polls usually happen mid-gesture — sampling
  // them blindly bakes the user's stick deflection into the neutral and the
  // robot then drives itself on a stale phantom command after release. The
  // sliding quiet-window arms as soon as the sticks are left alone. Until
  // armed, poll() returns null and the keyboard command stays in charge.
  private neutral: number[] | null = null;
  private neutralIndex = -1;
  private samples: number[][] = [];
  connected = false;
  name = "";

  constructor() {
    window.addEventListener("gamepadconnected", (e) => {
      this.connected = true;
      this.name = (e as GamepadEvent).gamepad.id;
    });
    window.addEventListener("gamepaddisconnected", () => {
      this.connected = false; this.name = "";
    });
  }

  /** Forget the sampled rest position and re-arm (~0.5 s, hands off the pad).
   * Wired to Space/Stop in the app — the manual fix for any leftover drift. */
  rearm() { this.neutral = null; this.samples = []; }

  /** Raw state for the debug panel: every axis (raw / neutral / corrected)
   * and every button. Null when no pad is present. */
  debugState(): {
    name: string; mapping: string; armed: boolean;
    axes: { raw: number; neutral: number; out: number }[];
    buttons: { pressed: boolean; value: number }[];
  } | null {
    const g = this.pad();
    if (!g) return null;
    return {
      name: g.id,
      mapping: g.mapping || "raw",
      armed: this.neutral !== null,
      axes: g.axes.map((raw, i) => {
        const neutral = this.neutral?.[i] ?? 0;
        return { raw, neutral, out: this.neutral ? this.dz(raw - neutral) : 0 };
      }),
      buttons: g.buttons.map((b) => ({ pressed: b.pressed, value: b.value })),
    };
  }

  get armed(): boolean { return this.neutral !== null; }

  private pad(): Gamepad | null {
    for (const g of navigator.getGamepads())
      if (g && g.connected && g.mapping === "standard") return g;
    for (const g of navigator.getGamepads()) if (g && g.connected) return g;
    return null;
  }

  private dz(v: number): number { return Math.abs(v) < this.deadzone ? 0 : v; }

  /** Poll once per control step. Returns the velocity command and fires
   * `onAction` for each combo whose press-button went down this step. */
  poll(onAction: (a: ComboAction) => void): Command | null {
    const g = this.pad();
    if (!g) {
      this.prevButtons = []; this.neutral = null; this.neutralIndex = -1;
      this.samples = [];
      return null;
    }
    if (this.neutralIndex !== g.index) {
      this.neutral = null; this.neutralIndex = g.index; this.samples = [];
    }
    if (this.neutral === null) {
      this.samples.push([...g.axes]);
      if (this.samples.length > 30) this.samples.shift();  // sliding window
      if (this.samples.length < 30) return null;
      const n = g.axes.length;
      const med: number[] = [];
      let quiet = true;
      for (let i = 0; i < n; i++) {
        const vals = this.samples.map((s) => s[i] ?? 0).sort((a, b) => a - b);
        med.push(vals[Math.floor(vals.length / 2)]);
        if (vals[vals.length - 1] - vals[0] > 0.06) quiet = false;
      }
      if (!quiet) return null;          // sticks moving — wait for a calm window
      this.neutral = med;
      this.samples = [];
    }
    const ax = (i: number) => (g.axes[i] ?? 0) - (this.neutral![i] ?? 0);
    // sticks: up = -Y on the pad; forward = +vx, left = +vy, ccw = +wz
    const vx = -this.dz(ax(1)) * this.scale[0];
    const vy = -this.dz(ax(0)) * this.scale[1];
    const wz = -this.dz(ax(2)) * this.scale[2];
    const down = g.buttons.map((b) => b.pressed || (typeof b.value === "number" && b.value > 0.6));
    for (const c of COMBOS) {
      if (down[c.hold] && down[c.press] && !this.prevButtons[c.press])
        onAction(c.action);
    }
    this.prevButtons = down;
    return [vx, vy, wz];
  }
}
