/** Unified per-step policy drivers for the interactive sim — TS port of the
 * reference Python implementation.
 *
 * A Driver wraps ONE policy so the FSM can run it, switch to it live, and
 * inspect its obs/action:
 *
 *     const drv = await makeDriver(cfg, spec, engine);
 *     drv.reset();                          // set the initial pose
 *     const target = await drv.step(cmd);   // -> model-order PD target
 *
 * The FSM applies a uniform PD law with drv.kp/kd (model order, full width);
 * joints the policy doesn't drive are held at drv.default. `step` is async
 * because onnxruntime-web inference is async.
 */
import type { Engine } from "../../sim/engine";
import { JNT_HINGE } from "../../sim/engine";
import type { Command, EvalConfig, RobotSpec } from "../types";

export abstract class BaseDriver {
  groundStart = false;      // starts lying on the floor (recovery / get-up)
  needsCommand = false;     // consumes the [vx,vy,wz] command
  clipDriven = false;       // plays a fixed reference clip

  readonly name: string;
  readonly n: number;
  readonly names: string[];
  readonly bqp: number;     // base qpos address
  readonly bqv: number;     // base dof address
  readonly qadr: Int32Array;
  readonly vadr: Int32Array;
  readonly lim: Float64Array;
  readonly exq: Int32Array; // hinges NOT in joint_names (held at neutral)
  readonly exv: Int32Array;
  readonly exKp: number;
  readonly exKd: number;
  readonly feet: number[];

  // inspector state (filled by step)
  frameLayout: [string, number][] = [];
  lastFrame: Float32Array = new Float32Array(0);
  lastAction: Float32Array;
  actJointNames: string[];
  stackDepth = 1;

  // PD gains + default, model order, full width — set by subclasses
  kp: Float64Array;
  kd: Float64Array;
  default: Float64Array;

  homeXY: [number, number] = [0, 0];

  constructor(readonly cfg: EvalConfig, readonly spec: RobotSpec, readonly eng: Engine) {
    this.name = cfg.name;
    this.names = spec.joint_names;
    this.n = this.names.length;
    this.bqp = eng.jointQposAdr(spec.floating_base_joint);
    this.bqv = eng.jointDofAdr(spec.floating_base_joint);
    this.qadr = Int32Array.from(this.names.map((x) => eng.jointQposAdr(x)));
    this.vadr = Int32Array.from(this.names.map((x) => eng.jointDofAdr(x)));
    this.lim = Float64Array.from(this.names.map((x) => eng.actuatorLim(x)));
    const inSpec = new Set(this.names);
    const hinges = eng.hinges();
    const exq: number[] = [], exv: number[] = [];
    hinges.names.forEach((nm, i) => {
      if (!inSpec.has(nm)) { exq.push(hinges.qadr[i]); exv.push(hinges.vadr[i]); }
    });
    this.exq = Int32Array.from(exq); this.exv = Int32Array.from(exv);
    this.exKp = spec.extra_kp ?? 20.0;
    this.exKd = spec.extra_kd ?? 1.0;
    this.feet = spec.foot_bodies.map((b) => eng.bodyId(b));
    this.lastAction = new Float32Array(this.n);
    this.actJointNames = [...this.names];
    this.kp = new Float64Array(this.n);
    this.kd = new Float64Array(this.n);
    this.default = new Float64Array(this.n);
  }

  /** Place the base at home, set joints/quat, drop until the lowest geom of
   * `dropBodies` sits 2 cm above z=0, zero velocity. */
  protected seat(qposJoints: ArrayLike<number>, quatWxyz: ArrayLike<number>,
                 dropBodies: Iterable<number>) {
    const { eng } = this;
    eng.qpos[this.bqp] = this.homeXY[0];
    eng.qpos[this.bqp + 1] = this.homeXY[1];
    for (let i = 0; i < this.n; i++) eng.qpos[this.qadr[i]] = qposJoints[i];
    for (let i = 0; i < 4; i++) eng.qpos[this.bqp + 3 + i] = quatWxyz[i];
    eng.forward();
    const zmin = eng.minGeomZ(dropBodies);
    eng.qpos[this.bqp + 2] += 0.02 - zmin;
    eng.qvel.fill(0);
    eng.forward();
  }

  /** Default: seat standing at the default pose, feet on the floor. */
  reset() { this.seat(this.default, [1, 0, 0, 0], this.feet); }

  /** Teleport just the base back home, keep pose + policy running. */
  recenter() {
    const { eng } = this;
    eng.qpos[this.bqp] = this.homeXY[0];
    eng.qpos[this.bqp + 1] = this.homeXY[1];
    eng.qvel[this.bqv] = 0; eng.qvel[this.bqv + 1] = 0;
    eng.forward();
  }

  /** Hold uncontrolled hinges (Air's absent waist/wrists) near neutral. */
  holdExtras(qfrc: Float64Array) {
    const { eng } = this;
    for (let i = 0; i < this.exv.length; i++)
      qfrc[this.exv[i]] =
        this.exKp * (0 - eng.qpos[this.exq[i]]) - this.exKd * eng.qvel[this.exv[i]];
  }

  /** Read current joint pos/vel (model order). */
  protected readQ(): [Float64Array, Float64Array] {
    const q = new Float64Array(this.n), qd = new Float64Array(this.n);
    for (let i = 0; i < this.n; i++) {
      q[i] = this.eng.qpos[this.qadr[i]];
      qd[i] = this.eng.qvel[this.vadr[i]];
    }
    return [q, qd];
  }

  protected baseQuat(): Float64Array {
    return this.eng.qpos.slice(this.bqp + 3, this.bqp + 7);
  }

  protected baseAngVel(): Float64Array {
    return this.eng.qvel.slice(this.bqv + 3, this.bqv + 6);
  }

  /** Once per control step: inference → PD target (model order, full width). */
  abstract step(command: Command): Promise<Float64Array>;
}

/** Oldest-first frame stack: pad by repeating the first frame. */
export function stackFrames(hist: Float32Array[], mem: number): Float32Array[] {
  if (hist.length >= mem) return hist.slice(hist.length - mem);
  return [...Array(mem - hist.length).fill(hist[0]), ...hist];
}

export { JNT_HINGE };
