/** Motion-tracking (dances, dances_air) and recovery (qishen/tangxia) driver.
 * Plays a reference clip; obs assembled from cfg.policy.obs_terms. */
import { matFromWxyz, sampleTraj, wxyz, argsort } from "../math";
import { buildObs, type ObsContext } from "../obs";
import type { OnnxPolicy } from "../policy";
import type { Command, EvalConfig, ObsTerm, RobotSpec } from "../types";
import type { Engine } from "../../sim/engine";
import { BaseDriver } from "./base";
import type { Traj } from "../assets";

export class MotionTrackingDriver extends BaseDriver {
  clipDriven = true;

  private readonly fps: number;
  private readonly stepDt: number;
  private readonly t0: number;
  private readonly t1: number;
  private readonly nAct: number;
  private readonly terms: ObsTerm[];
  private readonly jmap: Int32Array;
  private readonly jinv: Int32Array;
  private readonly scale: Float64Array;
  private readonly offset: Float64Array;
  private readonly nCols: number;
  private readonly traj: Traj;
  private readonly torso: number;
  private readonly tref: ReturnType<Engine["makeScratchData"]>;
  private k = 0;
  private lastActionPol: Float32Array;

  constructor(cfg: EvalConfig, spec: RobotSpec, eng: Engine, traj: Traj,
              readonly policy: OnnxPolicy) {
    super(cfg, spec, eng);
    this.groundStart = Boolean(cfg.ground_start);
    const ctl = cfg.control, pol = cfg.policy, g = cfg.gains!;
    this.fps = ctl.fps!; this.stepDt = ctl.step_dt;
    this.t0 = ctl.time_start ?? 0; this.t1 = ctl.time_end ?? Infinity;
    this.nAct = pol.num_act!;
    this.terms = pol.obs_terms!;
    this.jmap = Int32Array.from(cfg.model!.joint_ids_map);
    this.jinv = argsort(this.jmap);
    this.default = Float64Array.from(g.default_joint_pos);   // BFS/policy order
    this.scale = Float64Array.from(g.action_scale);
    this.offset = Float64Array.from(g.action_offset);
    this.kp = Float64Array.from(g.stiffness);
    this.kd = Float64Array.from(g.damping);
    this.nCols = 7 + this.nAct;
    this.traj = traj;
    if (cfg.traj?.order === "policy") {
      // clip stored in policy order → re-gather joint cols to model order
      const { data, nRows, nCols } = traj;
      for (let r = 0; r < nRows; r++) {
        const joints = data.slice(r * nCols + 7, r * nCols + this.nCols);
        for (let j = 0; j < this.nAct; j++)
          data[r * nCols + 7 + j] = joints[this.jinv[j]];
      }
    }
    this.torso = eng.bodyId(spec.anchor_body);
    this.tref = eng.makeScratchData();
    this.frameLayout = this.terms.map((t) => [t.name, t.dim]);
    this.actJointNames = Array.from(this.jinv, (mi) => this.names[mi]);
    this.lastActionPol = new Float32Array(this.nAct);
  }

  /** Reference anchor-body rotation for a clip row (forward on scratch data). */
  private refAnchorR(row: Float64Array): Float64Array {
    const tr = this.tref as any;
    (tr.qpos as Float64Array).fill(0);
    for (let i = 0; i < 3; i++) tr.qpos[this.bqp + i] = row[i];
    const q = wxyz([row[3], row[4], row[5], row[6]]);
    for (let i = 0; i < 4; i++) tr.qpos[this.bqp + 3 + i] = q[i];
    for (let i = 0; i < this.nAct; i++) tr.qpos[this.qadr[i]] = row[7 + i];
    this.eng.forwardData(this.tref);
    const xq = (tr.xquat as Float64Array).subarray(4 * this.torso, 4 * this.torso + 4);
    return matFromWxyz(xq);
  }

  override reset() {
    const r0 = sampleTraj(this.traj.data, this.traj.nRows, this.traj.nCols,
                          this.t0, this.fps);
    const drop = this.groundStart ? this.eng.allBodyIds() : this.feet;
    this.eng.qpos[this.bqp] = r0[0];
    this.eng.qpos[this.bqp + 1] = r0[1];
    this.eng.qpos[this.bqp + 2] = r0[2];
    this.seat(r0.subarray(7, this.nCols), wxyz([r0[3], r0[4], r0[5], r0[6]]), drop);
    this.k = 0;
    this.lastActionPol = new Float32Array(this.nAct);
  }

  async step(_command: Command): Promise<Float64Array> {
    let t = this.t0 + this.k * this.stepDt;
    if (t >= this.t1) { this.k = 0; t = this.t0; }        // loop the clip
    const { data, nRows, nCols } = this.traj;
    const rn = sampleTraj(data, nRows, nCols, t, this.fps);
    const rx = sampleTraj(data, nRows, nCols, t + 1.0 / this.fps, this.fps);
    const refPos = new Float64Array(this.nAct);
    const refVel = new Float64Array(this.nAct);
    for (let i = 0; i < this.nAct; i++) {
      const mi = this.jmap[i];
      refPos[i] = rn[7 + mi];
      refVel[i] = (rx[7 + mi] - refPos[i]) * this.fps;
    }
    const [q, qd] = this.readQ();
    const qBfs = new Float64Array(this.nAct), qdBfs = new Float64Array(this.nAct);
    for (let i = 0; i < this.nAct; i++) { qBfs[i] = q[this.jmap[i]]; qdBfs[i] = qd[this.jmap[i]]; }
    const anchorQuat = this.eng.xquat.subarray(4 * this.torso, 4 * this.torso + 4);
    const c: ObsContext = {
      refPos, refVel, refAnchorR: this.refAnchorR(rn),
      anchorQuatWxyz: anchorQuat, baseQuatWxyz: this.baseQuat(),
      baseAngVel: this.baseAngVel(), qBfs, qdBfs,
      defaultJointPos: this.default, lastAction: this.lastActionPol,
    };
    const obs = buildObs(this.terms, c);
    const action = await this.policy.run(obs);
    this.lastActionPol = action;
    this.lastFrame = obs; this.lastAction = action;
    this.k += 1;
    // action+offset are in policy order; jinv gathers to model order
    const target = new Float64Array(this.n);
    for (let mi = 0; mi < this.n; mi++) {
      const pi = this.jinv[mi];
      target[mi] = action[pi] * this.scale[pi] + this.offset[pi];
    }
    return target;
  }
}
