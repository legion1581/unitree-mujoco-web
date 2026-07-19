/** Locomotion-family drivers:
 *  - LocomotionConcatDriver: single concatenated obs (locomotion_20dofs,
 *    amp_locomotion, *_gait). Command-driven, stacked history, optional gait clock.
 *  - LocomotionSplitDriver: 3-input onnx (24dof walk / straight_knee).
 *  - AmpSplitDriver: 3-input, subset actuation (amp_motion / amp_motion_22dof).
 *  - ArmsdkDriver: single-input with isaac-order obs subset (arm_sdk_locomotion).
 */
import { clip, concatF32, gaitClock, matFromWxyz, matTVec, rot6d, type GaitParams } from "../math";
import type { OnnxPolicy, OnnxSession } from "../policy";
import type { Command, EvalConfig, RobotSpec } from "../types";
import type { Engine } from "../../sim/engine";
import { BaseDriver, stackFrames } from "./base";

function clampCmd(command: Command, vr: number[]): Float32Array {
  return new Float32Array([
    clip(command[0], vr[0], vr[1]),
    clip(command[1], vr[2], vr[3]),
    clip(command[2], vr[4], vr[5]),
  ]);
}

export class LocomotionConcatDriver extends BaseDriver {
  needsCommand = true;

  private readonly aScale: Float64Array;
  private readonly adi: Int32Array;
  private readonly mem: number;
  private readonly vr: number[];
  private readonly gait?: GaitParams;
  private hist: Float32Array[] = [];
  private lastActionRaw: Float32Array;
  private phase = 0;

  constructor(cfg: EvalConfig, spec: RobotSpec, eng: Engine,
              readonly policy: OnnxPolicy) {
    super(cfg, spec, eng);
    const lc = cfg.locomotion!;
    this.default = Float64Array.from(lc.default_dof_pos);
    this.kp = Float64Array.from(lc.kp); this.kd = Float64Array.from(lc.kd);
    const s = lc.action_scale;
    this.aScale = typeof s === "number"
      ? new Float64Array(this.n).fill(s) : Float64Array.from(s);
    this.adi = Int32Array.from(lc.action_dof_idx!);
    this.mem = lc.memory_size; this.vr = lc.vel_range;
    this.gait = lc.gait as unknown as GaitParams | undefined;
    this.lastActionRaw = new Float32Array(this.n);
    this.actJointNames = Array.from(this.adi, (i) => this.names[i]);
    this.stackDepth = this.mem;
    this.frameLayout = [["base_ang_vel", 3], ["projected_gravity", 3],
      ["velocity_commands", 3], ["joint_pos_rel", this.n],
      ["joint_vel", this.n], ["last_action", this.n]];
    if (this.gait) this.frameLayout.push(["gait_clock", 6]);
  }

  override reset() {
    this.seat(this.default, [1, 0, 0, 0], this.feet);
    this.hist = []; this.phase = 0;
    this.lastActionRaw = new Float32Array(this.n);
  }

  async step(command: Command): Promise<Float64Array> {
    const cmd = clampCmd(command, this.vr);
    const [q, qd] = this.readQ();
    const ang = this.baseAngVel();
    const grav = matTVec(matFromWxyz(this.baseQuat()), [0, 0, -1]);
    const posRel = new Float32Array(this.adi.length);
    const vel = new Float32Array(this.adi.length);
    for (let i = 0; i < this.adi.length; i++) {
      posRel[i] = q[this.adi[i]] - this.default[this.adi[i]];
      vel[i] = qd[this.adi[i]];
    }
    const segs: ArrayLike<number>[] = [ang, grav, cmd, posRel, vel, this.lastActionRaw];
    if (this.gait) {
      const [ph, clk] = gaitClock(this.phase, cmd, this.gait);
      this.phase = ph; segs.push(clk);
    }
    const frame = concatF32(segs);
    this.hist.push(frame);
    const obs = concatF32(stackFrames(this.hist, this.mem));
    if (this.hist.length > this.mem) this.hist = this.hist.slice(-this.mem);
    const action = await this.policy.run(obs);
    this.lastActionRaw = action;
    this.lastFrame = frame; this.lastAction = action;
    const target = Float64Array.from(this.default);
    for (let i = 0; i < this.adi.length; i++)
      target[this.adi[i]] = action[i] * this.aScale[this.adi[i]] + this.default[this.adi[i]];
    return target;
  }
}

export class LocomotionSplitDriver extends BaseDriver {
  needsCommand = true;

  private readonly aScale: Float64Array;
  private readonly mem: number;
  private readonly vr: number[];
  private prevScaled: Float32Array;
  private hist: Float32Array[] = [];

  constructor(cfg: EvalConfig, spec: RobotSpec, eng: Engine,
              readonly sess: OnnxSession) {
    super(cfg, spec, eng);
    const lc = cfg.locomotion!;
    this.default = Float64Array.from(lc.default_dof_pos);
    this.kp = Float64Array.from(lc.kp); this.kd = Float64Array.from(lc.kd);
    const s = lc.action_scale;
    this.aScale = typeof s === "number"
      ? new Float64Array(this.n).fill(s) : Float64Array.from(s);
    this.mem = lc.memory_size; this.vr = lc.vel_range;
    this.prevScaled = new Float32Array(this.n);
    this.stackDepth = this.mem;
    this.frameLayout = [["base_ang_vel", 3], ["orient6d", 6],
      ["joint_pos(abs)", this.n], ["joint_vel", this.n],
      ["prev_action(scaled)", this.n]];
  }

  override reset() {
    this.seat(this.default, [1, 0, 0, 0], this.feet);
    this.hist = []; this.prevScaled = new Float32Array(this.n);
  }

  async step(command: Command): Promise<Float64Array> {
    const cmd = clampCmd(command, this.vr);
    const [q, qd] = this.readQ();
    const o6 = rot6d(matFromWxyz(this.baseQuat()));
    const prop = concatF32([this.baseAngVel(), o6, q, qd, this.prevScaled]);
    this.hist.push(prop);
    const memory = concatF32(stackFrames(this.hist, this.mem));
    if (this.hist.length > this.mem) this.hist = this.hist.slice(-this.mem);
    const action = await this.sess.run({
      memory: { data: memory, dims: [1, memory.length] },
      commands: { data: cmd, dims: [1, 3] },
      proprioception: { data: prop, dims: [1, prop.length] },
    });
    for (let i = 0; i < this.n; i++) this.prevScaled[i] = action[i] * this.aScale[i];
    this.lastFrame = prop; this.lastAction = action;
    const target = new Float64Array(this.n);
    for (let i = 0; i < this.n; i++) target[i] = this.default[i] + this.prevScaled[i];
    return target;
  }
}

export class AmpSplitDriver extends BaseDriver {
  needsCommand = true;

  private readonly aScale: number;
  private readonly dai: Int32Array;
  private readonly mem: number;
  private readonly vr: number[];
  private readonly na: number;
  private prev: Float32Array;
  private hist: Float32Array[] = [];

  constructor(cfg: EvalConfig, spec: RobotSpec, eng: Engine,
              readonly sess: OnnxSession) {
    super(cfg, spec, eng);
    const lc = cfg.locomotion!;
    this.default = Float64Array.from(lc.default_dof_pos);
    this.kp = Float64Array.from(lc.kp); this.kd = Float64Array.from(lc.kd);
    this.aScale = lc.action_scale as number;
    this.dai = Int32Array.from(lc.dof_activated_idx!);
    this.mem = lc.memory_size; this.vr = lc.vel_range;
    this.na = this.dai.length;
    this.prev = new Float32Array(this.na);
    this.stackDepth = this.mem;
    this.actJointNames = Array.from(this.dai, (i) => this.names[i]);
    this.frameLayout = [["base_ang_vel", 3], ["projected_gravity", 3],
      ["joint_pos[dai](abs)", this.na], ["joint_vel[dai]", this.na],
      ["prev_action(raw)", this.na]];
  }

  override reset() {
    this.seat(this.default, [1, 0, 0, 0], this.feet);
    this.hist = []; this.prev = new Float32Array(this.na);
  }

  async step(command: Command): Promise<Float64Array> {
    const cmd = clampCmd(command, this.vr);
    const [q, qd] = this.readQ();
    const grav = matTVec(matFromWxyz(this.baseQuat()), [0, 0, -1]);
    const qs = new Float32Array(this.na), qds = new Float32Array(this.na);
    for (let i = 0; i < this.na; i++) { qs[i] = q[this.dai[i]]; qds[i] = qd[this.dai[i]]; }
    const prop = concatF32([this.baseAngVel(), grav, qs, qds, this.prev]);
    this.hist.push(prop);
    const memory = concatF32(stackFrames(this.hist, this.mem));
    if (this.hist.length > this.mem) this.hist = this.hist.slice(-this.mem);
    const action = await this.sess.run({
      memory: { data: memory, dims: [1, memory.length] },
      commands: { data: cmd, dims: [1, 3] },
      proprioception: { data: prop, dims: [1, prop.length] },
    });
    this.prev = action.slice(0, this.na);
    this.lastFrame = prop; this.lastAction = action;
    const target = Float64Array.from(this.default);
    for (let i = 0; i < this.na; i++)
      target[this.dai[i]] = this.default[this.dai[i]] + action[i] * this.aScale;
    return target;
  }
}

export class ArmsdkDriver extends BaseDriver {
  needsCommand = true;

  private readonly aScale: Float64Array;
  private readonly odi: Int32Array;
  private readonly cdi: Int32Array;
  private readonly mem: number;
  private readonly vr: number[];
  private readonly na: number;
  private lastActionRaw: Float32Array;
  private hist: Float32Array[] = [];

  constructor(cfg: EvalConfig, spec: RobotSpec, eng: Engine,
              readonly policy: OnnxPolicy) {
    super(cfg, spec, eng);
    const lc = cfg.locomotion!;
    this.default = Float64Array.from(lc.default_dof_pos);
    this.kp = Float64Array.from(lc.kp); this.kd = Float64Array.from(lc.kd);
    this.aScale = Float64Array.from(lc.action_scale as number[]);
    this.odi = Int32Array.from(lc.obs_dof_idx!);
    this.cdi = Int32Array.from(lc.control_dof_idx!);
    this.mem = lc.memory_size; this.vr = lc.vel_range;
    this.na = this.cdi.length;
    this.lastActionRaw = new Float32Array(this.na);
    this.stackDepth = this.mem;
    this.actJointNames = Array.from(this.cdi, (i) => this.names[i]);
    this.frameLayout = [["base_ang_vel", 3], ["projected_gravity", 3],
      ["velocity_commands", 3], ["joint_pos_rel[isaac]", this.n],
      ["joint_vel[isaac]", this.n], ["prev_action", this.na]];
  }

  override reset() {
    this.seat(this.default, [1, 0, 0, 0], this.feet);
    this.hist = []; this.lastActionRaw = new Float32Array(this.na);
  }

  async step(command: Command): Promise<Float64Array> {
    const cmd = clampCmd(command, this.vr);
    const [q, qd] = this.readQ();
    const grav = matTVec(matFromWxyz(this.baseQuat()), [0, 0, -1]);
    const posRel = new Float32Array(this.odi.length);
    const vel = new Float32Array(this.odi.length);
    for (let i = 0; i < this.odi.length; i++) {
      posRel[i] = q[this.odi[i]] - this.default[this.odi[i]];
      vel[i] = qd[this.odi[i]];
    }
    const frame = concatF32([this.baseAngVel(), grav, cmd, posRel, vel, this.lastActionRaw]);
    this.hist.push(frame);
    const obs = concatF32(stackFrames(this.hist, this.mem));
    if (this.hist.length > this.mem) this.hist = this.hist.slice(-this.mem);
    const action = await this.policy.run(obs);
    this.lastActionRaw = action;
    this.lastFrame = frame; this.lastAction = action;
    const target = Float64Array.from(this.default);
    for (let i = 0; i < this.na; i++)
      target[this.cdi[i]] = this.default[this.cdi[i]] + action[i] * this.aScale[i];
    return target;
  }
}
