/** Go2 quadruped drivers — TS port of the reference implementation's Go2Driver +
 * Go2FlipDriver (sim/drivers.py). Vision (LIDAR) is not ported yet.
 *
 * Go2Driver covers the go2_locomotion + go2_recovery families: obs are the
 * concatenated term vectors, normalized (raw - ob_mean) * ob_scale and clipped,
 * stacked over `memory_size` frames. Two policy shapes:
 *   - concat: single flat input  p_obs[mem*frame_dim]   (no `cmd` spec)
 *   - split : p_obs[1,mem,frame_dim] + cmd[1,L] -> "act" (gait-phase `cmd` spec)
 *
 * Go2FlipDriver is the one-shot flip maneuver: a ready-hold, an NN launch, then
 * a scripted control-point landing, with per-phase PD gains.
 */
import { clip, concatF32, matFromWxyz, matTVec } from "../math";
import type { OnnxSession } from "../policy";
import type { Command, EvalConfig, Go2Cfg, RobotSpec } from "../types";
import type { Engine } from "../../sim/engine";
import { BaseDriver } from "./base";

const TAU = 2 * Math.PI;

const TERM_DIM = (n: number): Record<string, number> => ({
  velocity_commands: 3, base_ang_vel: 3, projected_gravity: 3,
  joint_pos: n, joint_vel: n, last_action: n,
});

export class Go2Driver extends BaseDriver {
  private readonly aScale: Float64Array;
  private readonly clipAct: number;
  private readonly clipObs: number;
  private readonly obMean: Float32Array;
  private readonly obScale: Float32Array;
  private readonly mem: number;
  private readonly frameDim: number;
  private readonly frameTerms: string[];
  private readonly cmdSpec?: (string | number)[];
  private readonly vr: number[];
  private readonly newestFirst: boolean;
  private readonly gaitF: number;
  private readonly dtGait: number;
  private readonly twist: number;
  private readonly gfMean: number;
  private readonly gfScale: number;
  private readonly gfStand: number;
  private readonly moveEps: number;
  private readonly recovery: boolean;
  private readonly zeroVel: boolean;

  private hist: Float32Array[] = [];
  private phase = 0;
  private lastAct: Float32Array;

  constructor(cfg: EvalConfig, spec: RobotSpec, eng: Engine, readonly sess: OnnxSession) {
    super(cfg, spec, eng);
    const g = cfg.go2 as Go2Cfg;
    this.default = Float64Array.from(g.act_mean);        // policy == model order
    this.aScale = Float64Array.from(g.act_scale);
    this.kp = Float64Array.from(g.kp); this.kd = Float64Array.from(g.kd);
    this.clipAct = g.clip_act ?? 100; this.clipObs = g.clip_obs ?? 100;
    this.obMean = Float32Array.from(g.ob_mean);
    this.obScale = Float32Array.from(g.ob_scale);
    this.mem = g.memory_size; this.frameDim = g.frame_dim;
    const cmdInFrame = g.command_in_frame ?? true;
    this.frameTerms = g.frame_terms ?? [
      ...(cmdInFrame ? ["velocity_commands"] : []),
      "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "last_action"];
    this.cmdSpec = g.cmd;
    this.vr = g.vel_range ?? [-1.0, 1.5, -0.6, 0.6, -1.5, 1.5];
    this.newestFirst = g.history_newest_first ?? !this.cmdSpec;
    this.gaitF = g.base_gait_f ?? 1.7; this.dtGait = g.dt ?? 0.02;
    this.twist = g.twist_cmd_scale ?? 0.2;
    this.gfMean = g.gait_freq_mean ?? this.gaitF; this.gfScale = g.gait_freq_scale ?? 1.0;
    this.gfStand = g.gait_freq_stand ?? 0.0; this.moveEps = g.gait_move_eps ?? 0.05;
    this.recovery = !!g.recovery;
    this.zeroVel = !!g.zero_vel || this.recovery;         // hand_stand / recovery
    this.needsCommand = !this.zeroVel;
    this.groundStart = g.ground_start === "on_back";
    this.stackDepth = this.mem;
    this.lastAct = new Float32Array(this.n);
    const dims = TERM_DIM(this.n);
    this.frameLayout = this.frameTerms.map((t) => [t, dims[t]]);
  }

  override reset() {
    const drop = this.eng.allBodyIds();
    // recovery starts belly-up (roll 180° about x): quat [0,1,0,0]
    this.seat(this.default, this.groundStart ? [0, 1, 0, 0] : [1, 0, 0, 0], drop);
    this.hist = []; this.phase = 0;
    this.lastAct = new Float32Array(this.n);
  }

  private buildCmd(cmd: number[], gf: number): Float32Array {
    const [vx, vy, vyaw] = cmd;
    const vals: Record<string, number> = {
      cos: Math.cos(TAU * this.phase), sin: Math.sin(TAU * this.phase),
      gait_freq_norm: (gf - this.gfMean) * this.gfScale,
      vx, vy_s: vy * this.twist, vyaw_s: vyaw * this.twist,
      vy, vyaw, zero: 0,
    };
    return Float32Array.from(this.cmdSpec!.map((t) =>
      typeof t === "string" ? (vals[t] ?? 0) : t));
  }

  async step(command: Command): Promise<Float64Array> {
    const vr = this.vr;
    let cmd = [clip(command[0], vr[0], vr[1]), clip(command[1], vr[2], vr[3]),
               clip(command[2], vr[4], vr[5])];
    if (this.zeroVel) cmd = [0, 0, 0];
    const [q, qd] = this.readQ();
    const ang = this.baseAngVel();
    const grav = matTVec(matFromWxyz(this.baseQuat()), [0, 0, -1]);
    const tv: Record<string, ArrayLike<number>> = {
      velocity_commands: cmd, base_ang_vel: ang, projected_gravity: grav,
      joint_pos: q, joint_vel: qd, last_action: this.lastAct,
    };
    const raw = concatF32(this.frameTerms.map((t) => tv[t]));
    const frame = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i++)
      frame[i] = clip((raw[i] - this.obMean[i]) * this.obScale[i], -this.clipObs, this.clipObs);
    this.hist.push(frame);
    let buf = this.hist.length < this.mem
      ? [...Array(this.mem - this.hist.length).fill(this.hist[0]), ...this.hist]
      : this.hist.slice(-this.mem);
    if (this.hist.length > this.mem) this.hist = this.hist.slice(-this.mem);
    if (this.newestFirst) buf = [...buf].reverse();
    const pObs = concatF32(buf);

    let action: Float32Array;
    if (this.cmdSpec) {
      const moving = Math.abs(cmd[0]) + Math.abs(cmd[1]) + Math.abs(cmd[2]) > this.moveEps;
      const gf = (moving || this.recovery) ? this.gaitF : this.gfStand;
      this.phase = (this.phase + this.dtGait * gf) % 1.0;
      action = await this.sess.run({
        p_obs: { data: pObs, dims: [1, this.mem, this.frameDim] },
        cmd: { data: this.buildCmd(cmd, gf), dims: [1, this.cmdSpec.length] },
      });
    } else {
      const inName = this.sess.inputNames[0];
      action = await this.sess.run({ [inName]: { data: pObs, dims: [1, pObs.length] } });
    }
    const a = new Float32Array(action.length);
    for (let i = 0; i < a.length; i++) a[i] = clip(action[i], -this.clipAct, this.clipAct);
    this.lastAct = a;
    this.lastFrame = frame; this.lastAction = a;
    const target = new Float64Array(this.n);
    for (let i = 0; i < this.n; i++) target[i] = a[i] * this.aScale[i] + this.default[i];
    return target;
  }
}

// =====================================================================
// Go2 flip: single-frame obs, hybrid NN-launch + scripted control-point landing.
// One-shot — Reset (R) re-triggers.
// =====================================================================
export class Go2FlipDriver extends BaseDriver {
  private readonly aScale: Float64Array;
  private readonly obMean: Float32Array;
  private readonly obScale: Float32Array;
  private readonly readySteps: number;
  private readonly nnLaunch: number;
  private readonly warmup: number;
  private readonly clampN: number;
  private readonly kpReady: Float64Array; private readonly kdReady: Float64Array;
  private readonly kpNn: Float64Array; private readonly kdNn: Float64Array;
  private readonly kpLand: Float64Array; private readonly kdLand: Float64Array;
  private readonly controlPoints: [number, Float64Array][];
  private readonly cpEnd: number;
  private readonly continuous: boolean;
  private readonly cycleSteps: number;

  private stance: Float64Array;
  private lastAct: Float32Array;
  private k = 0; private li = 0;

  constructor(cfg: EvalConfig, spec: RobotSpec, eng: Engine, readonly sess: OnnxSession) {
    super(cfg, spec, eng);
    const g = cfg.go2 as Go2Cfg;
    const fl = (g.flip ?? {}) as Record<string, unknown>;
    const num = (v: unknown, d: number) => (typeof v === "number" ? v : d);
    const g12 = (key: string, dflt: number | number[]): Float64Array => {
      const v = fl[key] ?? dflt;
      return typeof v === "number"
        ? new Float64Array(this.n).fill(v) : Float64Array.from(v as number[]);
    };
    this.default = Float64Array.from(g.act_mean);
    this.aScale = Float64Array.from(g.act_scale);
    this.obMean = Float32Array.from(g.ob_mean);
    this.obScale = Float32Array.from(g.ob_scale);
    this.readySteps = num(fl.ready_steps, 15);
    this.nnLaunch = num(fl.nn_launch_steps, 17);
    this.warmup = num(fl.warmup, 4);
    this.clampN = num(fl.counter_clamp, 51);
    this.kpReady = g12("ready_kp", num(fl.kp_ready, 100));
    this.kdReady = g12("ready_kd", num(fl.kd_ready, 5));
    this.kpNn = g12("kp_nn", [24, 24, 24, 24, 24, 24, 23, 23, 23, 23, 23, 23]);
    this.kdNn = g12("kd_nn", 1.5);
    this.kpLand = g12("kp_land", (fl.kp_fly as number[]) ?? [70, 70, 70, 70, 70, 70, 50, 50, 50, 50, 50, 50]);
    this.kdLand = g12("kd_land", num(fl.kd_fly, 5));
    this.controlPoints = ((fl.control_points as [number, number[]][]) ?? [])
      .map(([s, a]) => [s | 0, Float64Array.from(a)] as [number, Float64Array]);
    this.cpEnd = num(fl.cp_end, 55);
    this.continuous = !!fl.continuous;
    this.cycleSteps = num(fl.cycle_steps, this.cpEnd);
    this.kp = this.kpReady.slice(); this.kd = this.kdReady.slice();
    this.stance = this.default.slice();
    this.lastAct = new Float32Array(this.n);
    this.stackDepth = 1;
    this.frameLayout = [["base_ang_vel", 3], ["projected_gravity", 3], ["zeros", 3],
      ["joint_pos", this.n], ["joint_vel", this.n], ["last_action", this.n],
      ["step_index", 1], ["active_flag", 1]];
  }

  override reset() {
    this.seat(this.default, [1, 0, 0, 0], this.eng.allBodyIds());
    const [q] = this.readQ();
    this.stance = q;
    this.k = 0; this.li = 0;
    this.lastAct = new Float32Array(this.n);
    this.kp = this.kpReady.slice(); this.kd = this.kdReady.slice();
  }

  private landingPose(lc: number): Float64Array {
    const cps = this.controlPoints;
    if (!cps.length) return this.stance;
    if (lc <= cps[0][0]) return cps[0][1];
    for (let i = 0; i < cps.length - 1; i++) {
      const [s0, a0] = cps[i], [s1, a1] = cps[i + 1];
      if (s0 <= lc && lc < s1 && s1 > s0) {
        const t = (lc - s0) / (s1 - s0);
        const out = new Float64Array(this.n);
        for (let j = 0; j < this.n; j++) out[j] = a0[j] + (a1[j] - a0[j]) * t;
        return out;
      }
    }
    return cps[cps.length - 1][1];
  }

  async step(_command: Command): Promise<Float64Array> {
    if (this.k < this.readySteps) {                 // (1) ready-hold
      this.kp = this.kpReady; this.kd = this.kdReady;
      this.k += 1;
      return this.stance;
    }
    const lc = this.continuous ? this.li % this.cycleSteps : this.li;
    const idx = lc < this.warmup ? 0 : Math.min(lc - (this.warmup - 1), this.clampN);
    const flag = lc < this.warmup ? 0 : 1;
    const [q, qd] = this.readQ();
    const ang = this.baseAngVel();
    const grav = matTVec(matFromWxyz(this.baseQuat()), [0, 0, -1]);
    const raw = concatF32([ang, grav, [0, 0, 0], q, qd, this.lastAct, [idx, flag]]);
    const frame = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i++)
      frame[i] = (raw[i] - this.obMean[i]) * this.obScale[i];   // flips don't clip obs
    let target: Float64Array;
    if (lc < this.nnLaunch) {                        // (2) NN launch
      const inName = this.sess.inputNames[0];
      const act = await this.sess.run({ [inName]: { data: frame, dims: [1, frame.length] } });
      this.lastAct = act.slice();
      this.lastFrame = frame; this.lastAction = act;
      this.kp = this.kpNn; this.kd = this.kdNn;
      target = new Float64Array(this.n);
      for (let i = 0; i < this.n; i++) target[i] = act[i] * this.aScale[i] + this.default[i];
    } else {                                          // (3) scripted landing
      this.lastFrame = frame; this.lastAction = this.lastAct;
      this.kp = this.kpLand; this.kd = this.kdLand;
      target = this.landingPose(lc);
    }
    this.k += 1; this.li += 1;
    return target;
  }
}
