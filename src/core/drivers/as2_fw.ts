/** Unitree AS2 firmware policies, driven the way the firmware drives them.
 *
 * Source: AS2 OTA 2.0.2.0, /unitree/module/ai_sport/ai_sport 1.0.0.17 (one executable, MNN linked statically,
 * Virbox "headers" mode = plaintext code with stripped locals; decompiled with IDA) and the plain YAML configs +
 * MNN nets in ai_sport/model. Every obs layout / command filter / state below was checked against bench MNN-I/O
 * captures (orange_pi_workbench tools/ai_sport_probe/as2mnnhook.c + as2_probe.py); notes in
 * docs/AS2_FIRMWARE_POLICIES.md.
 *
 * One driver = a small copy of ai_sport's controller: the pack's home state plus the states it always has at hand
 * (0 PASSIVE, 1 STAND_DOWN, 2 STAND_UP, 3 NORMAL_WALK, 12 RECOVERY — the walk / recovery nets ride along as aux
 * models). Remote combos request a state exactly as ai_sport's rule table does; the request goes through the
 * firmware's transition table (allowed edges + guards, refused requests are dropped). A switch to another policy
 * state hands over to that state's pack (`as2.packs`), like the robot's FSM change (the worker keeps the robot state).
 *
 * Common to the RL states (50 Hz): obs frame (42) = [gyro·0.25, gravity, (q − default)·1, dq·0.05, last action] in
 * the urdf / web-model order FL FR RL RR; net input = [current obs, history oldest → newest] where the history holds
 * the previous ticks' clipped obs[0:42] (filled with the entry frame on state entry); action clipped to ±clip_act,
 * target = default + 0.25·action, PD hip / thigh kp 40 kd 1, calf kp 60 kd 1.5; most states then clamp each target so
 * |kp(q* − q) − kd·dq| ≤ state_checker torque_limit (60 / 60 / 90 N·m).
 * Sticks (dead zone 0.05): raw command = (ly, −lx, −rx); the web command is stick × the scale the driver publishes in
 * `velRange`, so the driver recovers the stick as command / scale. The worker does not ramp these packs.
 */
import { clip, matFromWxyz, matTVec } from "../math";
import type { Command, EvalConfig, RobotSpec } from "../types";
import type { Engine } from "../../sim/engine";
import { OnnxSession } from "../policy";
import { assetUrl } from "../assets";
import { BaseDriver } from "./base";

/** SDK motor order (FR, FL, RR, RL) <-> model / urdf order (FL, FR, RL, RR); an involution */
const PERM = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8];
const f32 = Math.fround;
const sdk2m = (a: ArrayLike<number>): Float64Array => { const o = new Float64Array(12); for (let i = 0; i < 12; i++) o[PERM[i]] = a[i]; return o; };
const m2sdk = sdk2m;                                   // involution
const legs = (h: number, t: number, c: number) => [h, t, c, h, t, c, h, t, c, h, t, c];
const frac = (x: number) => x - Math.trunc(x);
/** C `(int)(a / b)` (the firmware's loop counts, e.g. (int)(1.2 / 0.02) = 59) */
const nTicks = (a: number, b: number) => Math.trunc(a / b);

// ---- firmware states (rt/sportmodestate .mode) ---------------------------------------------------------------
export const S: Record<string, number> = {
  PASSIVE: 0, STAND_DOWN: 1, STAND_UP: 2, WALK: 3, SPORT: 4, CLIMB: 5, LSIDE: 6, RSIDE: 7, HANDSTAND: 8, BIPED: 9,
  FFLIP: 10, BFLIP: 11, RECOVERY: 12, BASE_HEIGHT: 13, POSE: 14, EULER: 15,
  BAI_NIAN: 16, BI_XIN: 17, CONTENT: 18, DANCE1: 19, DANCE2: 20, HANDSHAKE: 21, STRETCH: 22, SIT: 23,
  FRONT_JUMP: 24, PUSHUP: 25, UP_JUMP: 26,
};
export const AS2_STATE_NAME: Record<number, string> = Object.fromEntries(Object.entries(S).map(([k, v]) => [v, k]));
const MIMICS = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26];
const SIDED = [6, 7, 8, 9];
const SCHEME_STATE: Record<string, number> = {
  normal_walk: 3, sport_mode: 4, climb_mode: 5, left_side_gait: 6, right_side_gait: 7, handstand: 8, biped_stand: 9,
  front_flip: 10, back_flip: 11, recovery: 12, base_height: 13, euler: 15,
};

// ---- leg kinematics (ai_sport: hip origins ±0.2079 / ±0.06, hip offset 0.1044, thigh 0.212, calf 0.21348) -------
const HIP = [[0.2079, -0.06, 0], [0.2079, 0.06, 0], [-0.2079, -0.06, 0], [-0.2079, 0.06, 0]];   // SDK legs FR FL RR RL
const SIDE = [-1, 1, -1, 1];
/** foot table (controller.yaml / state_checker.yaml [x, y, h]) -> per SDK leg foot position in the body frame */
function feet(p: number[]): number[][] {
  const [x, y, h] = p;
  return [[x, -y, -h], [x, y, -h], [-x, -y, -h], [-x, y, -h]];
}
/** sub_1DCFE0: foot position (body frame) -> [hip, thigh, calf] of SDK leg i */
function legIK(foot: number[], i: number): number[] {
  const dx = foot[0] - HIP[i][0], dy = foot[1] - HIP[i][1], dz = foot[2] - HIP[i][2];
  const L = Math.sqrt(dx * dx + dy * dy + dz * dz);
  const lx = Math.sqrt(L * L - 0.01089936);
  const s = SIDE[i] * 0.1044;
  const r = Math.sqrt(dz * dz + dy * dy - s * s);
  const hip = Math.atan2(r * dy + s * dz, -(r * dz - s * dy));
  const ck = clip((0.0905177104 - lx * lx) / 0.09051552, -1, 1);
  const calf = Math.acos(ck) - 3.14159265;
  const a = -0.212 - Math.cos(calf) * 0.21348, b = Math.sin(calf) * -0.21348;
  const p = -(Math.cos(hip) * dz - dy * Math.sin(hip));
  const thigh = Math.atan2(f32(f32(dx * a) + f32(p * b)), f32(-f32(f32(p * a) - f32(dx * b))));
  return [hip, thigh, calf];
}
/** sub_1DCE40: [hip, thigh, calf] of SDK leg i -> foot position (body frame), float math like the firmware */
function legFK(q: ArrayLike<number>, i: number): number[] {
  const s = SIDE[i] * 0.1044;
  const c0 = Math.cos(q[0]), s0 = Math.sin(q[0]), c1 = Math.cos(q[1]), s1 = Math.sin(q[1]);
  const c2 = f32(Math.cos(q[2])), s2 = Math.sin(q[2]);
  const c12 = -f32(f32(s1 * s2) - f32(c1 * c2));
  const x = f32(f32(s1 * -0.212) + f32(f32(f32(s2 * c1) + f32(s1 * c2)) * -0.21348));
  const y = f32(f32(f32(s * c0) + f32(f32(s0 * 0.21348) * c12)) - f32(f32(c1 * -0.212) * s0));
  const z = f32(f32(f32(s * s0) + f32(f32(c0 * -0.21348) * c12)) + f32(f32(c0 * -0.212) * c1));
  return [x + HIP[i][0], y + HIP[i][1], z + HIP[i][2]];
}
/** SDK-order joint targets for 4 foot positions */
function ikAll(f: number[][]): Float64Array {
  const o = new Float64Array(12);
  for (let i = 0; i < 4; i++) o.set(legIK(f[i], i), 3 * i);
  return o;
}

// ---- config --------------------------------------------------------------------------------------------------
export interface As2FwCfg {
  firmware: true;
  fsm_id: number;
  /** firmware class of the home state: normal_walk | sport_mode | climb_mode | left_side_gait | right_side_gait |
   *  handstand | biped_stand | front_flip | back_flip | recovery | base_height | euler | mimic */
  scheme: string;
  packs?: Record<string, string>;
  base: Record<string, number | number[]>;            // model/rl_base.yaml
  controller: Record<string, number[]>;               // model/controller.yaml
  state_checker: Record<string, number | number[]>;   // model/state_checker.yaml
  cfg: Record<string, Record<string, unknown>>;       // per-net yaml (normal_walk, sport_mode, …)
  mimic?: Record<string, unknown>;                    // mimic.yaml shared fields + this pack's motion
  aux_models?: Record<string, string>;
  [k: string]: unknown;
}
export const as2fw = (cfg: EvalConfig): As2FwCfg => cfg.as2 as unknown as As2FwCfg;

/** a blocking firmware routine: one item per 20 ms tick, a target or a tick that runs a net */
type Gen = Generator<Float64Array | (() => Promise<Float64Array>), void, unknown>;

interface Robot { q: Float64Array; dq: Float64Array; qs: Float64Array; dqs: Float64Array; grav: number[]; gyro: number[]; quat: Float64Array }

/** history ring of obs frames, oldest first */
class Ring {
  rows: Float32Array[] = [];
  constructor(readonly n: number) {}
  fill(f: Float32Array) { this.rows = Array.from({ length: this.n }, () => f.slice()); }
  push(f: Float32Array) { this.rows.push(f.slice()); while (this.rows.length > this.n) this.rows.shift(); }
}

/** One ai_sport RL object (sub_10BC10 / sub_10AF30): its own phase clock, last action, history. */
class Rl {
  t = 0;                                   // phase clock (s)
  last = new Float32Array(12);
  ring: Ring;
  readonly dflt: Float64Array;             // default angle, model order
  readonly kp: Float64Array; readonly kd: Float64Array;   // model order
  constructor(readonly role: string, readonly cur: number, nHist: number, readonly b: Record<string, number | number[]>,
              dflt?: number[]) {
    this.ring = new Ring(nHist);
    this.dflt = Float64Array.from(dflt ?? (b.default_angle as number[]));
    const kp = b.kp as number, kd = b.kd as number, kpc = (b.kp_calf ?? kp) as number, kdc = (b.kd_calf ?? kd) as number;
    this.kp = Float64Array.from(legs(kp, kp, kpc)); this.kd = Float64Array.from(legs(kd, kd, kdc));
  }
  get nIn() { return this.cur + 42 * this.ring.n; }
  /** obs frame [0:42] */
  frame(r: Robot, o: Float32Array) {
    const b = this.b, sa = b.obs_ang_vel_scale as number, sq = b.obs_joint_pos_scale as number, sv = b.obs_joint_vel_scale as number;
    for (let i = 0; i < 3; i++) { o[i] = sa * r.gyro[i]; o[3 + i] = r.grav[i]; }
    for (let i = 0; i < 12; i++) {
      o[6 + i] = (r.q[i] - this.dflt[i]) * sq; o[18 + i] = sv * r.dq[i]; o[30 + i] = this.last[i];
    }
  }
}

export class As2FwDriver extends BaseDriver {
  readonly g: As2FwCfg;
  readonly home: number;
  nextPolicy?: string;

  state: number = S.WALK;
  private entry = true;
  private tick = 0;
  private level = 1;                        // speed level (state+2148): 1 = high
  private autoRec = true;                   // state+2152 (Y+B toggles)
  private stick = [0, 0, 0];                // raw (ly, −lx, −rx)
  private lx = 0; private ly = 0; private rx = 0;
  private pending: string[] = [];
  private held: Record<string, number> = {};   // key -> ticks it stays "pressed" (Up / Down / L1+Up …)
  private script: Gen | null = null;   // blocking transition action / exit routine
  private height = 0.35;                    // base-height command (state+2168)
  private scaleV = [1, 1, 1];
  private flipDone = false;
  activeNet = "";
  private roleNow = "";
  repeat = false;

  // RL objects (one per net, own history / clock / filters)
  private rl: Record<string, Rl> = {};
  private cmdRaw = [0, 0, 0]; private cmdF = [0, 0, 0]; private cmdO = [0, 0, 0];   // raw / filtered / +API
  private posture = [0, 0];                 // sided: [pitch (664), roll (672)]
  private eul = [0, 0, 0];                  // euler filtered targets
  private moving = false; private nomTimer = 0; private nomFlag = false;
  private rec = { t: 0 };
  private mim = { t: 0, frame: 0, start: 0, end: 0, T: 0.2, q0: new Float64Array(12) as Float64Array<ArrayBufferLike>,
                  kp0: new Float64Array(12) as Float64Array<ArrayBufferLike>, kd0: new Float64Array(12) as Float64Array<ArrayBufferLike>, R0: [1, 0, 0, 0, 1, 0, 0, 0, 1], out: {} as Record<string, Float32Array>,
                  last: new Float32Array(12) };

  constructor(cfg: EvalConfig, spec: RobotSpec, eng: Engine, readonly nets: Record<string, OnnxSession>) {
    super(cfg, spec, eng);
    this.g = as2fw(cfg);
    this.home = this.g.scheme === "mimic" ? this.g.fsm_id : (SCHEME_STATE[this.g.scheme] ?? this.g.fsm_id);
    const b = this.g.base, c = this.g.cfg;
    const nh = (b.buffer_size as number) ?? 6;
    const mk = (role: string, cur: number, n = nh, extra?: Record<string, unknown>, dflt?: number[]) =>
      (this.rl[role] = new Rl(role, cur, n, { ...b, ...(extra as Record<string, number>) }, dflt));
    mk("walk", 51); mk("recovery", 51);
    const home = this.g.scheme;
    if (home === "sport_mode") mk("main", 51);
    if (home === "climb_mode") mk("main", 51, nh, {}, c.climb_mode.default_angle as number[]);
    if (home === "base_height") mk("main", 52);
    if (home === "euler") mk("main", 45);
    if (SIDED.includes(this.home)) mk("main", (c.sided_gait.current_obs_dim as number) ?? 47, (c.sided_gait.buffer_size as number) ?? 4);
    if (home === "front_flip" || home === "back_flip") {
      const fl = c.flip as Record<string, number>;
      mk("main", 46, 4, { clip_act: fl.clip_act, clip_obs: fl.clip_obs }, fl.default_angle as unknown as number[]);
    }
    this.default = Float64Array.from(this.rl.walk.dflt);
    this.kp = Float64Array.from(this.rl.walk.kp); this.kd = Float64Array.from(this.rl.walk.kd);
    this.needsCommand = true;
    this.stackDepth = 1;
    this.frameLayout = [["gyro", 3], ["gravity", 3], ["q", 12], ["dq", 12], ["last_action", 12], ["extra", 9]];
    this.state = this.home;
    this.publishRange();
  }

  // ---------------------------------------------------------------- reset / seat
  override reset() {
    const st = this.home;
    if (st === S.RECOVERY) {
      // shown from a fall: on the back, legs folded (RECOVERY is entered from PASSIVE / STAND_DOWN)
      this.seat(sdk2m(ikAll(feet(this.g.controller.foot_lie_down))), [0, 1, 0, 0], this.eng.allBodyIds());
    } else this.seat(this.rl.walk.dflt, [1, 0, 0, 0], this.feet);
    this.state = st; this.entry = true; this.tick = 0; this.level = 1; this.done = false; this.nextPolicy = undefined;
    this.pending = []; this.held = {}; this.script = null; this.flipDone = false;
  }

  // ---------------------------------------------------------------- remote
  override remoteCombos(): string[] {
    return ["L2+B", "L2+A", "Start", "L2+Start", "L2+X", "R1+X", "X X", "Y Y", "R1 R1", "R2 R2", "Up Up", "Down Down",
      "L1+A", "R2+Y", "R2+X", "L1+B", "L1+X", "R2+B", "R2+A", "R1+B", "R1+A", "L1+Y", "R1+Y", "Select", "L1+Up",
      "L1+Down", "Up", "Down", "Y+B"];
  }
  override remote(combo: string) { this.pending.push(combo); }

  /** ai_sport's remote rule table (sub_196C50 rules, first match wins) -> requested state */
  private event(combo: string) {
    const R: Record<string, number> = {
      "L2+B": S.PASSIVE, "Start": S.WALK, "L2+Start": S.SPORT, "L2+X": S.RECOVERY, "R1+X": S.CLIMB,
      "X X": S.LSIDE, "Y Y": S.RSIDE, "R1 R1": S.HANDSTAND, "R2 R2": S.BIPED, "Up Up": S.FFLIP, "Down Down": S.BFLIP,
      "L1+A": S.BAI_NIAN, "R2+Y": S.BI_XIN, "R2+X": S.CONTENT, "L1+B": S.DANCE1, "L1+X": S.DANCE2, "R2+B": S.HANDSHAKE,
      "R2+A": S.STRETCH, "R1+B": S.SIT, "R1+A": S.FRONT_JUMP, "L1+Y": S.PUSHUP, "R1+Y": S.UP_JUMP, "Select": S.EULER,
    };
    if (combo === "L2+A") { this.request(this.state === S.STAND_UP ? S.STAND_DOWN : S.STAND_UP); return; }
    if (combo === "Y+B") { this.autoRec = !this.autoRec; return; }
    if (combo === "Up" || combo === "Down") { this.held[combo] = 6; return; }          // a short press (0.12 s)
    if (combo === "L1+Up" || combo === "L1+Down") {
      this.held[combo] = 6;
      this.request(S.BASE_HEIGHT);
      return;
    }
    // the flip packs: the firmware has no edge into 10 / 11 (see docs) — the forced pack re-runs its flip
    if ((combo === "Up Up" && this.home === S.FFLIP) || (combo === "Down Down" && this.home === S.BFLIP)) {
      if (this.state === S.WALK || this.state === this.home) this.switchTo(this.home);
      return;
    }
    // a mimic pack re-plays its motion with its own combo (the firmware: from NORMAL_WALK)
    const tgt = R[combo];
    if (tgt !== undefined) this.request(tgt);
  }

  // ---------------------------------------------------------------- transition table (sub_196C50)
  private upright(st: Robot) { return st.grav[2] < -0.5; }
  private allowed(from: number, to: number, st: Robot): boolean {
    const P = S.PASSIVE, D = S.STAND_DOWN, U = S.STAND_UP, W = S.WALK;
    const toRec = (f: number) => f <= 1;                              // guard 106FE0: current state <= STAND_DOWN
    switch (from) {
      case P: return to === U || (to === S.RECOVERY && toRec(from));
      case D: return to === P || to === U || (to === S.RECOVERY && toRec(from));
      case U: return to === P || to === D || ([W, S.SPORT, S.CLIMB].includes(to) && this.upright(st));
      case W:
        if ([P, D, U].includes(to)) return true;
        if ([S.SPORT, S.CLIMB, S.EULER, S.BASE_HEIGHT].includes(to)) return this.upright(st);
        if (to === S.RECOVERY) return toRec(from);
        return SIDED.includes(to) || MIMICS.includes(to);
      case S.SPORT: case S.CLIMB: case S.BASE_HEIGHT:
        if ([P, D, U].includes(to)) return true;
        if (to === S.RECOVERY) return toRec(from);
        if (from === S.BASE_HEIGHT) return to === W;
        return [W, S.SPORT, S.CLIMB].includes(to) && to !== from && this.upright(st);
      case S.RECOVERY: return to === P || to === W;
      case S.FFLIP: case S.BFLIP: return to === W ? this.flipDone : (to === S.RECOVERY && toRec(from));
      case S.EULER: return to === P || to === W;
      default:
        if (SIDED.includes(from)) return to === P || to === W || (to === S.RECOVERY && toRec(from));
        if (MIMICS.includes(from)) {
          if (to === P) return true;
          if (to === W) return [S.SIT, S.PUSHUP, S.UP_JUMP].includes(from)        // no guard (Start leaves SIT)
            || (this.upright(st) && this.mim.frame >= this.mim.end);
          return to === S.RECOVERY && toRec(from);
        }
        return false;
    }
  }
  private st0: Robot | null = null;
  /** a remote / FSM request: refused unless the firmware's transition table allows it */
  private request(to: number) {
    if (to === this.state || this.script) return;
    const st = this.st0 ?? this.robot();
    if (!this.allowed(this.state, to, st)) return;
    this.switchTo(to);
  }

  /** perform the transition: exit routine of the current state, transition action, then the target's enter */
  private switchTo(to: number) {
    const from = this.state;
    const local = [S.PASSIVE, S.STAND_DOWN, S.STAND_UP, S.WALK, S.RECOVERY, this.home];
    if (!local.includes(to)) {
      const pack = this.g.packs?.[String(to)];
      if (pack) {
        // exit routines that run before the handover (sided gait settle, euler / base-height settle, sit stand-up)
        this.nextPolicy = pack; this.done = true;
      }
      return;
    }
    let scr: Gen | null = null;
    if (SIDED.includes(from) && to !== S.PASSIVE && to !== S.RECOVERY) scr = this.sidedExit();
    else if (from === S.EULER && to !== S.PASSIVE && to !== S.RECOVERY) scr = this.eulerExit();
    else if (from === S.BASE_HEIGHT && to !== S.PASSIVE && to !== S.RECOVERY) scr = this.baseHeightExit();
    else if (from === S.SIT && to !== S.PASSIVE && to !== S.RECOVERY) scr = this.sitExit();
    const act = this.transitionAction(from, to);
    if (scr || act) {
      const self = this;
      this.script = (function* () { if (scr) yield* scr; if (act) yield* act; self.state = to; self.entry = true; })();
      return;
    }
    this.state = to; this.entry = true;
    this.publishRange();
  }
  private transitionAction(from: number, to: number): Gen | null {
    if ((from === S.PASSIVE || from === S.STAND_DOWN) && to === S.STAND_UP) return this.standUpFromFloor();
    if (from === S.STAND_UP && to === S.STAND_DOWN) return this.feetBlend("foot_stand_up", "foot_lie_down", legs(150, 150, 200), legs(10, 10, 12));
    if ([S.WALK, S.SPORT, S.CLIMB].includes(from) && to === S.STAND_DOWN)
      return this.feetBlend("foot_stand_up", "foot_lie_down", legs(40, 40, 60), legs(5, 5, 5));
    if ([S.WALK, S.SPORT, S.CLIMB, S.BASE_HEIGHT].includes(from) && to === S.STAND_UP) return this.toStandUp();
    return null;
  }

  // ---------------------------------------------------------------- robot state
  private robot(): Robot {
    const [q, dq] = this.readQ();
    const R = matFromWxyz(this.baseQuat());
    return { q, dq, qs: m2sdk(q), dqs: m2sdk(dq), grav: matTVec(R, [0, 0, -1]), gyro: Array.from(this.baseAngVel()),
             quat: this.baseQuat() };
  }

  // ---------------------------------------------------------------- web command -> sticks
  private publishRange() {
    const c = this.g.cfg, s = this.state;
    let m = [1, 1, 1];
    const lv = this.level;
    // full stick = the clamped command the state reaches (the web command is divided back by this scale)
    if (s === S.WALK) { const w = c.normal_walk as Record<string, number[]>; m = [w.vx_limit[lv], w.vy_limit[lv], w.vyaw_limit[lv]]; }
    else if (s === S.BASE_HEIGHT) { const w = c.base_height as Record<string, number[]>; m = [w.vx_limit[0], w.vy_limit[0], w.vyaw_limit[0]]; }
    else if (s === S.SPORT) { const w = c.sport_mode as Record<string, number>; m = [w.vx_limit, w.vy_limit, w.vyaw_limit]; }
    else if (s === S.CLIMB) m = (c.climb_mode.cmd_limit as number[]).slice();
    else if (SIDED.includes(s)) m = (c.sided_gait.cmd_limit as number[]).slice();
    else if (s === S.EULER) m = [1, 1, 1];
    this.scaleV = m;
    this.velRange = [-m[0], m[0], -m[1], m[1], -m[2], m[2]];
  }

  // ---------------------------------------------------------------- main step
  async step(command: Command): Promise<Float64Array> {
    // web command -> sticks (ly, lx, rx), dead zone 0.05 like ai_sport's remote decode
    const dz = (v: number) => (Math.abs(v) > 0.05 ? v : 0);
    this.ly = dz(clip(command[0] / (this.scaleV[0] || 1), -1, 1));
    this.lx = dz(clip(-command[1] / (this.scaleV[1] || 1), -1, 1));
    this.rx = dz(clip(-command[2] / (this.scaleV[2] || 1), -1, 1));
    this.stick = [this.ly, -this.lx, -this.rx];
    const st = this.robot();
    this.st0 = st;
    while (this.pending.length) this.event(this.pending.shift()!);
    for (const k of Object.keys(this.held)) if (--this.held[k] < 0) delete this.held[k];
    this.tick++;
    let tgt: Float64Array;
    if (this.script) {
      const r = this.script.next();
      if (r.done) { this.script = null; this.publishRange(); tgt = await this.runState(st); }
      else tgt = typeof r.value === "function" ? await r.value() : r.value;
    } else tgt = await this.runState(st);
    this.lastTgt = tgt;
    this.activeNet = `${this.state} ${AS2_STATE_NAME[this.state] ?? ""}${this.roleNow ? " · " + this.roleNow : ""}`;
    this.roleNow = "";
    this.st0 = null;
    return tgt;
  }
  private lastTgt: Float64Array<ArrayBufferLike> = new Float64Array(12);

  private async runState(st: Robot): Promise<Float64Array> {
    const s = this.state, entry = this.entry;
    this.entry = false;
    if (entry) this.publishRange();
    switch (s) {
      case S.PASSIVE: case S.STAND_DOWN: return this.damping(st);
      case S.STAND_UP: return this.standHold();
      case S.WALK: return this.walkLike(st, entry, "walk");
      case S.SPORT: return this.walkLike(st, entry, "sport");
      case S.BASE_HEIGHT: return this.walkLike(st, entry, "base_height");
      case S.CLIMB: return this.climb(st, entry);
      case S.EULER: return this.euler(st, entry);
      case S.RECOVERY: return this.recovery(st, entry);
      case S.FFLIP: case S.BFLIP: return this.flip(st, entry);
      default:
        if (SIDED.includes(s)) return this.sided(st, entry);
        if (MIMICS.includes(s)) return this.mimic(st, entry);
        return this.damping(st);
    }
  }

  // ---------------------------------------------------------------- shared helpers
  private setGains(kpS: number[], kdS: number[]) { this.kp = sdk2m(kpS); this.kd = sdk2m(kdS); }
  /** sub_10B5B0: clamp each target so |kp(q* − q) − kd·dq| ≤ torque_limit (SDK table, order-independent per leg) */
  private torqueClamp(t: Float64Array, st: Robot) {
    const lim = this.g.state_checker.torque_limit as number[];
    for (let i = 0; i < 12; i++) {
      const L = lim[PERM[i]], kp = this.kp[i], kd = this.kd[i];
      if (!(kp > 0)) continue;
      const v = st.dq[i] * kd + st.q[i] * kp;
      t[i] = clip(t[i], (v - L) / kp, (v + L) / kp);
    }
  }
  /** fall check of the RL states: flipped past `th` (gravity z) -> RECOVERY (auto-recovery on) else PASSIVE */
  private fall(st: Robot, th: number): boolean {
    if (st.grav[2] > th) { this.request(this.autoRec ? S.RECOVERY : S.PASSIVE); return true; }
    return false;
  }
  private async net(role: string, input: Float32Array): Promise<Float32Array> {
    const s = this.nets[role];
    if (!s) throw new Error(`${this.name}: net ${role} missing`);
    this.roleNow = this.roleNow ? `${this.roleNow}+${role}` : role;
    return s.run({ [s.inputNames[0]]: { data: input, dims: [1, input.length] } });
  }
  /** sub_10BC10: clock, obs (+extra), clip, [obs, history], net, push, clip action, target */
  private async rlStep(o: Rl, st: Robot, extra: (ob: Float32Array) => void, netRole: string, scale?: number): Promise<Float64Array> {
    o.t += 0.02;
    const ob = new Float32Array(o.cur);
    o.frame(st, ob); extra(ob);
    const co = o.b.clip_obs as number;
    for (let i = 0; i < ob.length; i++) ob[i] = clip(ob[i], -co, co);
    const inp = new Float32Array(o.nIn);
    inp.set(ob, 0);
    o.ring.rows.forEach((r, k) => inp.set(r, o.cur + 42 * k));
    const act = await this.net(netRole, inp);
    o.ring.push(ob.subarray(0, 42));
    const ca = o.b.clip_act as number, as = scale ?? (o.b.action_scale as number);
    const t = new Float64Array(12);
    for (let i = 0; i < 12; i++) { const a = clip(act[i], -ca, ca); o.last[i] = a; t[i] = o.dflt[i] + as * a; }
    this.lastFrame = ob; this.lastAction = o.last.slice();
    if (this.frameLayout[5]?.[1] !== ob.length - 42 || this.frameLayout[0][0] !== "gyro")
      this.frameLayout = [["gyro", 3], ["gravity", 3], ["q", 12], ["dq", 12], ["last_action", 12], ["extra", ob.length - 42]];
    this.kp = Float64Array.from(o.kp); this.kd = Float64Array.from(o.kd);
    return t;
  }
  /** sub_10AF30: clock 0, last action 0, history = N copies of the entry frame */
  private rlReset(o: Rl, st: Robot, extra: (ob: Float32Array) => void) {
    o.t = 0; o.last.fill(0);
    const ob = new Float32Array(o.cur);
    o.frame(st, ob); extra(ob);
    o.ring.fill(ob.slice(0, 42));
  }

  // ---------------------------------------------------------------- scripted states
  private damping(st: Robot): Float64Array {                 // 104A50 / 104960: q* = q, kp 0, kd 3
    this.kp = new Float64Array(12); this.kd = new Float64Array(12).fill(3);
    return Float64Array.from(st.q);
  }
  private standHold(): Float64Array {                         // 1047A0: IK(foot_stand_up), kp 150/150/200 kd 10/10/12
    this.setGains(legs(150, 150, 200), legs(10, 10, 12));
    return sdk2m(ikAll(feet(this.g.controller.foot_stand_up)));
  }
  /** 106250 (PASSIVE / STAND_DOWN -> STAND_UP): 1.5 s from the measured pose to IK(foot_stand_down) (blend of the
   * live q), then 1.2 s moving the feet stand_down -> stand_up; kp 150/150/200 kd 10/10/12 */
  private *standUpFromFloor(): Gen {
    const C = this.g.controller, dt = 0.02;
    this.setGains(legs(150, 150, 200), legs(10, 10, 12));
    const down = ikAll(feet(C.foot_stand_down));
    let acc = 0;
    for (let k = 0; k < nTicks(1.5, dt); k++) {
      acc += dt; const r = acc / 1.5;
      const qs = this.robot().qs, o = new Float64Array(12);
      for (let i = 0; i < 12; i++) o[i] = (1 - r) * qs[i] + r * down[i];
      this.setGains(legs(150, 150, 200), legs(10, 10, 12));
      yield sdk2m(o);
    }
    yield* this.feetBlend("foot_stand_down", "foot_stand_up", legs(150, 150, 200), legs(10, 10, 12), 1.2);
  }
  /** 1066D0 / 106980: feet linearly from table a to table b over 1.2 s, IK each tick */
  private *feetBlend(a: string, b: string, kp: number[], kd: number[], T = 1.2): Gen {
    const C = this.g.controller, fa = feet(C[a]), fb = feet(C[b]), dt = 0.02;
    let acc = 0;
    for (let k = 0; k < nTicks(T, dt); k++) {
      acc += dt; const r = acc / T;
      const f = fa.map((p, i) => p.map((v, j) => (1 - r) * v + r * fb[i][j]));
      this.setGains(kp, kd);
      yield sdk2m(ikAll(f));
    }
  }
  /** 106C20 (walking states -> STAND_UP): 2 s, r = k·dt/2 (k from 0): target and gains blend from the previous
   * command toward IK(foot_stand_up) / 150-150-200, 10-10-12 */
  private *toStandUp(): Gen {
    const up = ikAll(feet(this.g.controller.foot_stand_up)), dt = 0.02;
    const KP = legs(150, 150, 200), KD = legs(10, 10, 12);
    let t = m2sdk(this.lastTgt), kp = m2sdk(this.kp), kd = m2sdk(this.kd);
    for (let k = 0; k < nTicks(2.0, dt); k++) {
      const r = k * dt * 0.5;
      t = t.map((v, i) => (1 - r) * v + r * up[i]);
      kp = kp.map((v, i) => (1 - r) * v + r * KP[i]); kd = kd.map((v, i) => (1 - r) * v + r * KD[i]);
      this.kp = sdk2m(kp); this.kd = sdk2m(kd);
      yield sdk2m(t);
    }
  }

  // ---------------------------------------------------------------- walk-like (3 NORMAL_WALK, 4 SPORT, 13 BASE_HEIGHT)
  private async walkLike(st: Robot, entry: boolean, kind: "walk" | "sport" | "base_height"): Promise<Float64Array> {
    const c = this.g.cfg, o = kind === "walk" ? this.rl.walk : this.rl.main;
    const W = kind === "walk" ? c.normal_walk : kind === "sport" ? c.sport_mode : c.base_height;
    const role = kind === "walk" ? "walk" : "main";
    if (entry) {
      this.cmdRaw = [0, 0, 0]; this.cmdF = [0, 0, 0]; this.cmdO = [0, 0, 0]; this.moving = false;
      this.level = kind === "walk" ? 1 : 0; this.nomTimer = 0; this.nomFlag = false;
      if (kind === "base_height") this.height = 0.35;      // controller ctor: body height 0.35
      this.publishRange();
    }
    // fall: > 0.5 -> recovery (auto) / passive
    this.fall(st, 0.5);
    // speed level (NORMAL_WALK / SPORT: Up / Down without L1); base height: L1+Up / L1+Down ±0.005 per tick
    if (kind === "base_height") {
      if (this.held["L1+Up"] !== undefined) this.height += 0.005;
      if (this.held["L1+Down"] !== undefined) this.height -= 0.005;
    } else {
      const prev = this.level;
      if (this.held.Up !== undefined) this.level = 1; else if (this.held.Down !== undefined) this.level = 0;
      if (prev !== this.level) this.publishRange();
    }
    this.cmdRaw = [...this.stick];
    // moving flag (feet off the nominal stance in the 1 s state-checker window, or |previous command| > 0.05)
    const chk = this.g.state_checker;
    this.nomTimer += 0.02;
    if (this.nomTimer > (chk.nominal_check_time as number)) {
      const nom = feet(chk.foot_pos_nominal as number[]);
      let n = 0;
      for (let i = 0; i < 4; i++) {
        const f = legFK(st.qs.subarray(3 * i, 3 * i + 3), i);
        if (Math.abs(f[0] - nom[i][0]) > (chk.nominal_threshold_x as number)) n++;
        if (Math.abs(f[1] - nom[i][1]) > (chk.nominal_threshold_y as number)) n++;
      }
      this.nomTimer = 0; this.nomFlag = n > 0;
    }
    this.moving = this.nomFlag || Math.hypot(...this.cmdO) > 0.05;
    const extra = (ob: Float32Array) => this.walkExtra(ob, kind, W, o);
    if (entry) this.rlReset(o, st, () => {});
    const t = await this.rlStep(o, st, extra, role);
    this.torqueClamp(t, st);
    return t;
  }
  /** obs[42:] of 109530 (walk), 10A770 (sport), 109300 (base height) */
  private walkExtra(ob: Float32Array, kind: string, W: Record<string, unknown>, o: Rl) {
    const lv = kind === "walk" ? this.level : 0;
    let freq: number, fh: number;
    if (kind === "sport") { freq = W.gait_freq as number; fh = W.foot_height as number; }
    else { freq = lv === 1 ? (W.gait_freq_high as number) : (W.gait_freq_low as number); fh = W.foot_height as number; }
    const fs = W.obs_foot_height_scale as number;
    if (this.moving) {
      const ph = freq * o.t;
      const a = f32(frac(ph)), b = f32((ph + 0.5) - Math.trunc(ph + 0.5));
      ob[42] = a; ob[43] = b; ob[44] = b; ob[45] = a; ob[46] = freq; ob[47] = fh * fs;
    } else { ob[42] = ob[43] = ob[44] = ob[45] = 0.75; ob[46] = freq; ob[47] = 0; }
    const flt = W.vel_filter as number[], [rx, ry, rw] = this.cmdRaw;
    let vx: number, vy: number, wz: number, lim: number[];
    if (kind === "sport") {
      const vxl = W.vx_limit as number;
      this.cmdF[0] = (1 - flt[0]) * this.cmdF[0] + flt[0] * rx * vxl;
      this.cmdF[1] = (1 - flt[1]) * this.cmdF[1] + flt[1] * ry * vxl;          // firmware: vy scaled by vx_limit too
      this.cmdF[2] = (1 - flt[2]) * this.cmdF[2] + flt[2] * rw * (W.vyaw_limit as number);
      this.cmdO = [...this.cmdF];
      vx = clip(this.cmdO[0], -vxl, vxl); vy = clip(this.cmdO[1], -(W.vy_limit as number), W.vy_limit as number);
      wz = clip(this.cmdO[2], -(W.vyaw_limit as number), W.vyaw_limit as number);
    } else {
      const vxl = (W.vx_limit as number[])[lv], vyl = (W.vy_limit as number[])[lv], vwl = (W.vyaw_limit as number[])[lv];
      this.cmdF[0] = (1 - flt[0]) * this.cmdF[0] + rx * vxl * flt[0];
      this.cmdF[1] = (1 - flt[1]) * this.cmdF[1] + ry * vxl * flt[1];          // firmware: vy scaled by vx_limit
      this.cmdF[2] = (1 - flt[2]) * this.cmdF[2] + vwl * rw * flt[2];
      this.cmdO = [...this.cmdF];
      lim = [vxl, Math.abs(this.cmdO[2]) > 1.5 ? 0 : vyl, Math.abs(this.cmdO[0]) > 1.0 ? 2.0 : vwl];
      vx = clip(this.cmdO[0], -lim[0], lim[0]); vy = clip(this.cmdO[1], -lim[1], lim[1]); wz = clip(this.cmdO[2], -lim[2], lim[2]);
    }
    const b = this.g.base;
    ob[48] = vx * (b.obs_lin_vel_scale as number); ob[49] = vy * (b.obs_lin_vel_scale as number);
    ob[50] = wz * (b.obs_ang_vel_scale as number);
    if (kind === "base_height") {
      const r = W.base_height_range as number[];
      this.height = clip(this.height, r[0], r[1]);
      ob[51] = this.height;
    }
  }

  // ---------------------------------------------------------------- 5 CLIMB_MODE
  private async climb(st: Robot, entry: boolean): Promise<Float64Array> {
    const C = this.g.cfg.climb_mode, o = this.rl.main, lim = C.cmd_limit as number[], a = C.cmd_filter as number[];
    if (entry) { this.cmdRaw = [0, 0, 0]; this.cmdF = [0, 0, 0]; this.rlReset(o, st, () => {}); }
    this.fall(st, -0.1);
    this.cmdRaw = [...this.stick];
    const extra = (ob: Float32Array) => {
      for (let i = 0; i < 3; i++) this.cmdF[i] = (1 - a[i]) * this.cmdF[i] + a[i] * this.cmdRaw[i] * lim[i];
      ob[42] = ob[43] = ob[44] = ob[45] = 0.75; ob[46] = 1.5; ob[47] = 0;
      const b = this.g.base;
      ob[48] = clip(this.cmdF[0], -lim[0], lim[0]) * (b.obs_lin_vel_scale as number);
      ob[49] = clip(this.cmdF[1], -lim[1], lim[1]) * (b.obs_lin_vel_scale as number);
      ob[50] = clip(this.cmdF[2], -lim[2], lim[2]) * (b.obs_ang_vel_scale as number);
    };
    return this.rlStep(o, st, extra, "main");
  }

  // ---------------------------------------------------------------- 15 EULER
  private eulerExtra = (ob: Float32Array) => {
    const E = this.g.cfg.euler, up = E.euler_limit_upper as number[], lo = E.euler_limit_lower as number[];
    const a = E.euler_filter as number[], sc = E.obs_euler_cmd_scale as number;
    const tg = [this.lx * up[0], this.ly * up[1], -this.rx * up[2]];
    for (let i = 0; i < 3; i++) {
      this.eul[i] = a[i] * tg[i] + (1 - a[i]) * this.eul[i];
      ob[42 + i] = clip(this.eul[i], lo[i], up[i]) * sc;
    }
  };
  private async euler(st: Robot, entry: boolean): Promise<Float64Array> {
    const o = this.rl.main;
    if (entry) { this.eul = [0, 0, 0]; this.rlReset(o, st, () => {}); }
    this.fall(st, 0.5);
    return this.rlStep(o, st, this.eulerExtra, "main");
  }
  /** 1059C0: up to 0.8 s of the net with the sticks released (targets decay with the 0.1 filter) */
  private *eulerExit(): Gen {
    for (let k = 0; k < nTicks(0.8, 0.02); k++) {
      if (Math.hypot(...this.eul) < 1e-12) break;
      this.lx = this.ly = this.rx = 0;
      yield () => this.rlStep(this.rl.main, this.robot(), this.eulerExtra, "main");
    }
  }

  /** 105E30: ease the height command back toward 0.35 (−10 % of the error per tick, ≤ 1 s) */
  private *baseHeightExit(): Gen {
    let near = 0;
    for (let k = 0; k < nTicks(1.0, 0.02); k++) {
      const e = this.height - 0.35;
      if (Math.abs(e) < 0.005) near++;
      if (near > 10) break;
      this.height -= e * 0.1;
      this.cmdRaw = [0, 0, 0];
      const W = this.g.cfg.base_height, o = this.rl.main;
      yield async () => {
        const st = this.robot(), t = await this.rlStep(o, st, (ob) => this.walkExtra(ob, "base_height", W, o), "main");
        this.torqueClamp(t, st); return t;
      };
    }
  }

  // ---------------------------------------------------------------- 6-9 sided gait
  private sidedExtra = (ob: Float32Array) => {
    const G = this.g.cfg.sided_gait, lim = G.cmd_limit as number[], step = G.cmd_filter as number[];
    for (let i = 0; i < 3; i++) this.cmdF[i] += clip(this.cmdRaw[i] * lim[i] - this.cmdF[i], -step[i], step[i]);
    const b = this.g.base;
    ob[42] = clip(this.cmdF[0], -lim[0], lim[0]) * (b.obs_lin_vel_scale as number);
    ob[43] = clip(this.cmdF[1], -lim[1], lim[1]) * (b.obs_lin_vel_scale as number);
    ob[44] = clip(this.cmdF[2], -lim[2], lim[2]) * (b.obs_ang_vel_scale as number);
    this.posture[1] = clip(this.posture[1], -(G.max_roll_cmd as number), G.max_roll_cmd as number);
    this.posture[0] = clip(this.posture[0], -(G.max_pitch_cmd as number), G.max_pitch_cmd as number);
    ob[45] = this.posture[1]; ob[46] = this.posture[0];
  };
  private async sided(st: Robot, entry: boolean): Promise<Float64Array> {
    const G = this.g.cfg.sided_gait, o = this.rl.main, s = this.state;
    if (entry) { this.cmdRaw = [0, 0, 0]; this.cmdF = [0, 0, 0]; this.posture = [0, 0]; this.rlReset(o, st, () => {}); }
    this.fall(st, s === S.HANDSTAND || s === S.BIPED ? 0.2 : 0.0);
    this.cmdRaw = [...this.stick];
    const mr = G.max_roll_cmd as number, mp = G.max_pitch_cmd as number;
    if (s === S.LSIDE) this.posture[1] = -(mr * 0.1 - this.posture[1] * 0.9);
    else if (s === S.RSIDE) this.posture[1] = mr * 0.1 + this.posture[1] * 0.9;
    else if (s === S.HANDSTAND) this.posture[0] = mp * 0.04 + this.posture[0] * 0.96;
    else this.posture[0] = -(mp * 0.04 - this.posture[0] * 0.96);
    const t = await this.rlStep(o, st, this.sidedExtra, "main");
    if (s !== S.HANDSTAND) this.torqueClamp(t, st);
    return t;
  }
  /** 105190: 1 s of the sided net with the sticks released and the posture command decaying ×0.95 per tick */
  private *sidedExit(): Gen {
    for (let k = 0; k < nTicks(1.0, 0.02); k++) {
      this.cmdRaw = [0, 0, 0];
      this.posture = [this.posture[0] * 0.95, this.posture[1] * 0.95];
      yield () => this.rlStep(this.rl.main, this.robot(), this.sidedExtra, "main");
    }
  }

  // ---------------------------------------------------------------- 12 RECOVERY (104B40)
  private async recovery(st: Robot, entry: boolean): Promise<Float64Array> {
    const o = this.rl.recovery, R = this.rec, dt = 0.02;
    const extra = (ob: Float32Array) => { ob[42] = ob[43] = ob[44] = ob[45] = 0.75; ob[46] = 1.8; ob[47] = ob[48] = ob[49] = ob[50] = 0; };
    if (entry) { R.t = 0; this.rlReset(o, st, extra); }
    R.t += dt;
    if (R.t < 0.5) {                                          // relax: q* = q, kp ×0.9 per tick, kd 4
      this.kp = this.kp.map((v) => v * 0.9); this.kd = new Float64Array(12).fill(4);
      const t = Float64Array.from(st.q);
      this.torqueClamp(t, st);
      return t;
    }
    if (R.t < 1.5) {                                          // fold: blend the live q into IK(foot_recovery)
      const r = R.t - 0.5, tg = ikAll(feet(this.g.controller.foot_recovery)), o2 = new Float64Array(12);
      for (let i = 0; i < 12; i++) o2[i] = (1 - r) * st.qs[i] + r * tg[i];
      this.setGains(legs(60, 60, 90), legs(1.5, 1.5, 2.0));
      this.rlReset(o, st, extra);
      const t = sdk2m(o2);
      this.torqueClamp(t, st);
      return t;
    }
    if (R.t >= 3.0) {
      R.t = 0;
      if (st.grav[2] < -0.5) this.request(S.WALK);
      return this.lastTgt;
    }
    const r = Math.min((R.t - 1.5) / 1.3, 1);
    const kp = this.kp, kd = this.kd;
    const t = await this.rlStep(o, st, extra, "recovery");
    this.kp = kp; this.kd = kd;                               // the fold gains stay
    for (let i = 0; i < 12; i++) t[i] = (1 - r) * st.q[i] + r * t[i];
    this.torqueClamp(t, st);
    return t;
  }

  // ---------------------------------------------------------------- 10 / 11 flips (never entered by 2.0.2.0)
  private async flip(st: Robot, entry: boolean): Promise<Float64Array> {
    const F = this.g.cfg.flip as Record<string, number>, o = this.rl.main, dir = this.state === S.FFLIP ? 1 : -1;
    const extra = (ob: Float32Array) => {
      const tau = frac(o.t / F.flip_time);
      const s = 6 * tau ** 5 - 15 * tau ** 4 + 10 * tau ** 3;
      ob[42] = 0; ob[43] = 0; ob[44] = 0; ob[45] = f32(2 * dir * 3.14159265 * s);
    };
    if (entry) { this.flipDone = false; this.rlReset(o, st, extra); }
    if (o.t + 0.02 > F.flip_time) {
      this.flipDone = true;
      this.request(S.WALK);
      return this.lastTgt;
    }
    const t = await this.rlStep(o, st, extra, "main", F.action_scale);
    this.torqueClamp(t, st);
    return t;
  }

  // ---------------------------------------------------------------- 16-26 mimic (sub_10FF60 / sub_10F550 / 1077A0)
  private mimicObs(st: Robot, useOut: boolean): Float32Array {
    const M = this.g.mimic as Record<string, number[]>, m = this.mim, ob = new Float32Array(69);
    // joint-major (IsaacLab) order: hips FL FR RL RR, thighs, calves; model index of (leg l, type t) = 3l + t
    const jm = (j: number) => 3 * (j % 4) + Math.floor(j / 4);
    for (let j = 0; j < 12; j++) {
      ob[j] = useOut ? m.out.joint_pos[j] : st.q[jm(j)];
      ob[12 + j] = useOut ? m.out.joint_vel[j] : st.dq[jm(j)];
      ob[33 + j] = st.q[jm(j)]; ob[45 + j] = st.dq[jm(j)]; ob[57 + j] = m.last[j];
    }
    // anchor orientation: R = R_robot^T · Rz(yaw0) · R_ref, first two columns (ref = net's body_quat_w[0], wxyz)
    const Rr = matFromWxyz(st.quat);
    const qr = useOut ? Array.from(m.out.body_quat_w.subarray(0, 4)) : Array.from(st.quat);
    const Rf = matFromWxyz(qr), Z = m.R0;
    const A = new Array(9).fill(0), Rm = new Array(9).fill(0);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) A[3 * i + j] += Z[3 * i + k] * Rf[3 * k + j];
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) Rm[3 * i + j] += Rr[3 * k + i] * A[3 * k + j];
    ob[24] = Rm[0]; ob[25] = Rm[1]; ob[26] = Rm[3]; ob[27] = Rm[4]; ob[28] = Rm[6]; ob[29] = Rm[7];
    for (let i = 0; i < 3; i++) ob[30 + i] = st.gyro[i];
    const mean = M.obs_mean;
    for (let i = 0; i < 69; i++) ob[i] -= mean[i];
    return ob;
  }
  private async mimicRun(ob: Float32Array): Promise<void> {
    const s = this.nets.main;
    this.roleNow = "main";
    const outs = await s.runAll({ obs: { data: ob, dims: [1, 69] }, time_step: { data: Float32Array.of(this.mim.frame), dims: [1, 1] } });
    this.mim.out = outs;
    this.lastFrame = ob;
    if (this.frameLayout[0][0] !== "ref_q")
      this.frameLayout = [["ref_q", 12], ["ref_dq", 12], ["anchor_6d", 6], ["gyro", 3], ["q", 12], ["dq", 12], ["last_action", 12]];
  }
  private async mimic(st: Robot, entry: boolean): Promise<Float64Array> {
    const M = this.g.mimic as Record<string, unknown>, m = this.mim;
    const kp = M.kp as number[], kd = M.kd as number[], am = M.act_mean as number[], asc = M.act_scale as number[];
    const IDX = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8];      // joint-major j -> SDK slot (dword_4CD1C0)
    if (entry) {
      const [a, b] = M.idx as number[];
      m.start = a; m.end = b; m.frame = a; m.T = M.transition as number; m.t = 0;
      m.q0 = Float64Array.from(st.qs); m.kp0 = m2sdk(this.kp); m.kd0 = m2sdk(this.kd);
      const [w, x, y, z] = st.quat, yaw = Math.atan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z));
      m.R0 = [Math.cos(yaw), -Math.sin(yaw), 0, Math.sin(yaw), Math.cos(yaw), 0, 0, 0, 1];
      m.last = new Float32Array(12);
      await this.mimicRun(this.mimicObs(st, false));
      m.last = new Float32Array(12);
    }
    if (st.grav[2] > 0.5) this.request(this.autoRec ? S.RECOVERY : S.PASSIVE);
    m.t += 0.02;
    await this.mimicRun(this.mimicObs(st, true));
    if (m.t >= m.T) m.frame++;
    const act = m.out.actions;
    m.last = Float32Array.from(act);
    this.lastAction = m.last.slice();
    let sw = 1, sw0 = 0;
    if (m.T > 0) { const x = m.t / m.T; if (x <= 1) { sw = x * x * (3 - 2 * x); sw0 = 1 - sw; } }
    const ts = new Float64Array(12), kps = new Float64Array(12), kds = new Float64Array(12);
    for (let j = 0; j < 12; j++) {
      const slot = IDX[j];
      ts[slot] = (am[j] + act[j] * asc[j]) * sw + m.q0[slot] * sw0;
      kps[j] = kp[j] * sw + m.kp0[j] * sw0; kds[j] = kd[j] * sw + m.kd0[j] * sw0;
    }
    this.kp = sdk2m(kps); this.kd = sdk2m(kds);
    const out = sdk2m(ts);
    if (this.state === S.SIT) {
      // 1072C0: SIT loops its last 50 frames (keeps sitting) until another state is requested; torque clamp
      if (m.frame >= m.end) m.frame = m.end - 50;
      this.torqueClamp(out, st);
    } else if (m.frame >= m.end) this.request(S.WALK);
    return out;
  }
  /** 105600 (SIT -> walking): 0.5 s from the measured pose to IK(foot_stand_up), kp 80/80/120 kd 4/4/6 */
  private *sitExit(): Gen {
    const up = ikAll(feet(this.g.controller.foot_stand_up)), dt = 0.02;
    for (let k = 0; k < nTicks(0.5, dt); k++) {
      const r = 2 * k * dt, qs = this.robot().qs;
      this.setGains(legs(80, 80, 120), legs(4, 4, 6));
      yield sdk2m(qs.map((v, i) => (1 - r) * v + r * up[i]));
    }
  }
}

/** Build the driver: the pack's checkpoint is role "main", `as2.aux_models` the other roles (walk, recovery). */
export async function makeAs2FwDriver(cfg: EvalConfig, spec: RobotSpec, eng: Engine): Promise<BaseDriver> {
  const g = as2fw(cfg);
  const url = assetUrl(cfg.policy.path).replace(/\.mnn$/, ".onnx");
  const nets: Record<string, OnnxSession> = { main: await OnnxSession.load(url) };
  for (const [role, p] of Object.entries(g.aux_models ?? {}))
    if (p) nets[role] = await OnnxSession.load(assetUrl(p).replace(/\.mnn$/, ".onnx"));
  return new As2FwDriver(cfg, spec, eng, nets);
}
