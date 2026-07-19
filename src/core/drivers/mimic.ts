/** mimic_actor / mimic_punch driver: p_obs[1,mem,frame] + cmd (clip row or
 * twist command). The clip CSV *is* the command stream — each row is fed to the
 * `cmd` input verbatim (no interpolation), looping. */
import { clip as clamp, concatF32, matFromWxyz, matTVec } from "../math";
import type { OnnxSession } from "../policy";
import type { Command, EvalConfig, RobotSpec } from "../types";
import type { Engine } from "../../sim/engine";
import { BaseDriver, stackFrames } from "./base";
import type { Traj } from "../assets";

export class MimicDriver extends BaseDriver {
  private readonly angS: number;
  private readonly dposS: number;
  private readonly dvelS: number;
  private readonly aScale: number;
  private readonly memN: number;
  private readonly limIdx: Int32Array;
  private readonly lo: Float64Array;
  private readonly hi: Float64Array;
  private readonly tsc: Float64Array;
  private readonly tof: Float64Array;
  private readonly clipTraj: Traj | null;
  private readonly startJpos: number[] | null;
  private readonly face: string | undefined;
  private lastActionRaw: Float32Array;
  private hist: Float32Array[] = [];
  private k = 0;

  constructor(cfg: EvalConfig, spec: RobotSpec, eng: Engine,
              readonly sess: OnnxSession, clipTraj: Traj | null) {
    super(cfg, spec, eng);
    const mc = cfg.mimic as Record<string, any>;
    this.default = Float64Array.from(mc.default_dof_pos);
    this.kp = Float64Array.from(mc.kp); this.kd = Float64Array.from(mc.kd);
    this.angS = mc.ang_vel_scale ?? 0.2;
    this.dposS = mc.dof_pos_scale ?? 1.0;
    this.dvelS = mc.dof_vel_scale ?? 0.05;
    this.aScale = mc.action_scale ?? 1.0;
    this.memN = mc.memory_size;
    this.limIdx = Int32Array.from(mc.limits_joint_idx ?? []);
    this.lo = Float64Array.from(mc.dof_lower_limits ?? []);
    this.hi = Float64Array.from(mc.dof_upper_limits ?? []);
    this.tsc = Float64Array.from(mc.twist_scale ?? [1, 1, 1]);
    this.tof = Float64Array.from(mc.twist_offset ?? [0, 0, 0]);
    this.clipTraj = clipTraj;
    this.clipDriven = clipTraj !== null;
    this.needsCommand = clipTraj === null;
    this.startJpos = (cfg.start_jpos as number[] | undefined) ?? null;
    this.face = cfg.face;
    this.groundStart = Boolean(this.startJpos);
    this.lastActionRaw = new Float32Array(this.n);
    this.stackDepth = this.memN;
    this.frameLayout = [["base_ang_vel", 3], ["projected_gravity", 3],
      ["joint_pos_rel", this.n], ["joint_vel", this.n], ["last_action", this.n]];
  }

  override reset() {
    if (this.startJpos) {
      // lie on the floor face up/down, then settle under a PD hold
      const half = Math.SQRT1_2;
      const quat = this.face === "down" ? [half, 0, -half, 0] : [half, 0, half, 0];
      this.seat(this.startJpos, quat, this.eng.allBodyIds());
      const hold = new Float64Array(this.n);
      for (let i = 0; i < this.n; i++) hold[i] = this.eng.qpos[this.qadr[i]];
      for (let s = 0; s < 250; s++) {
        this.eng.qfrcApplied.fill(0);
        for (let i = 0; i < this.n; i++) {
          const tq = this.kp[i] * (hold[i] - this.eng.qpos[this.qadr[i]])
                   - this.kd[i] * this.eng.qvel[this.vadr[i]];
          this.eng.qfrcApplied[this.vadr[i]] = clamp(tq, -this.lim[i], this.lim[i]);
        }
        this.eng.step();
      }
    } else {
      this.seat(this.default, [1, 0, 0, 0], this.feet);
    }
    this.hist = []; this.k = 0;
    this.lastActionRaw = new Float32Array(this.n);
  }

  async step(command: Command): Promise<Float64Array> {
    const [q, qd] = this.readQ();
    const grav = matTVec(matFromWxyz(this.baseQuat()), [0, 0, -1]);
    const ang = this.baseAngVel();
    const frame = new Float32Array(3 + 3 + this.n * 3);
    frame[0] = ang[0] * this.angS; frame[1] = ang[1] * this.angS; frame[2] = ang[2] * this.angS;
    frame[3] = grav[0]; frame[4] = grav[1]; frame[5] = grav[2];
    for (let i = 0; i < this.n; i++) {
      frame[6 + i] = (q[i] - this.default[i]) * this.dposS;
      frame[6 + this.n + i] = qd[i] * this.dvelS;
      frame[6 + 2 * this.n + i] = this.lastActionRaw[i];
    }
    this.hist.push(frame);
    const stack = stackFrames(this.hist, this.memN);
    if (this.hist.length > this.memN) this.hist = this.hist.slice(-this.memN);
    const pObs = concatF32(stack);
    let cmd: Float32Array;
    if (this.clipTraj) {
      const { data, nRows, nCols } = this.clipTraj;
      const r = Math.min(this.k, nRows - 1);
      cmd = data.slice(r * nCols, (r + 1) * nCols);
      this.k = (this.k + 1) % nRows;                    // loop the clip
    } else {
      cmd = new Float32Array([
        command[0] * this.tsc[0] + this.tof[0],
        command[1] * this.tsc[1] + this.tof[1],
        command[2] * this.tsc[2] + this.tof[2],
      ]);
    }
    const action = await this.sess.run({
      p_obs: { data: pObs, dims: [1, this.memN, frame.length] },
      cmd: { data: cmd, dims: [1, cmd.length] },
    });
    this.lastActionRaw = action.slice(0, this.n);
    this.lastFrame = frame; this.lastAction = action;
    const target = new Float64Array(this.n);
    for (let i = 0; i < this.n; i++) target[i] = this.default[i] + action[i] * this.aScale;
    for (let j = 0; j < this.limIdx.length; j++) {
      const i = this.limIdx[j];
      target[i] = clamp(target[i], this.lo[i], this.hi[i]);
    }
    return target;
  }
}
