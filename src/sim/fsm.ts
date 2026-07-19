/** On-robot-style safety state machine — TS port of the reference implementation's
 * sim/fsm.py. Each control step the app asks the FSM for a PD target; every
 * physics substep it recomputes joint torques from the current q/qd (policy at
 * 50 Hz, PD at 500 Hz — a torque held open-loop across substeps topples the
 * robot).
 *
 *   ZeroTorque : motors off (robot collapses) — safe boot / e-stop
 *   Damping    : pure joint damping (robot sags gently)
 *   Lock       : freeze at the current pose (stiff PD to the captured qpos)
 *   Stance     : hold the active policy's home/default pose
 *   Policy     : run the selected policy driver
 */
import { clip } from "../core/math";
import type { Command } from "../core/types";
import type { BaseDriver } from "../core/drivers";
import type { Engine, HingeSet } from "./engine";

export type FsmState = "zero_torque" | "damping" | "lock" | "stance" | "policy";

export const STATES: FsmState[] = ["zero_torque", "damping", "lock", "stance", "policy"];
export const LABELS: Record<FsmState, string> = {
  zero_torque: "ZeroTorque (motors off)",
  damping: "Damping (soft)",
  lock: "Lock (freeze pose)",
  stance: "Stance (hold home)",
  policy: "Policy",
};

export class SimFSM {
  state: FsmState = "damping";
  private readonly hinges: HingeSet;
  private holdPose: Float64Array | null = null;

  constructor(readonly eng: Engine,
              readonly dampKd = 3.0, readonly holdKp = 80.0, readonly holdKd = 4.0) {
    this.hinges = eng.hinges();
  }

  enter(state: FsmState) {
    this.state = state;
    if (state === "lock") {
      const h = this.hinges;
      this.holdPose = Float64Array.from(h.qadr, (a) => this.eng.qpos[a]);
    }
  }

  /** Once per CONTROL step: run policy inference (advances clip/history).
   * Returns the joint target the PD loop tracks (null for torque-only states). */
  async infer(driver: BaseDriver | null, command: Command): Promise<Float64Array | null> {
    if (this.state === "policy" && driver) return driver.step(command);
    if (this.state === "stance" && driver) return driver.default;
    return null;
  }

  /** Every PHYSICS substep: write torques into eng.qfrcApplied. */
  substepTorque(driver: BaseDriver | null, target: Float64Array | null) {
    const { eng } = this;
    const qfrc = eng.qfrcApplied;
    qfrc.fill(0);
    const h = this.hinges;
    if (this.state === "zero_torque") {
      return;
    }
    if (this.state === "damping") {
      for (let i = 0; i < h.vadr.length; i++)
        qfrc[h.vadr[i]] = clip(-this.dampKd * eng.qvel[h.vadr[i]], -h.lim[i], h.lim[i]);
      return;
    }
    if (this.state === "lock") {
      if (!this.holdPose)
        this.holdPose = Float64Array.from(h.qadr, (a) => eng.qpos[a]);
      for (let i = 0; i < h.vadr.length; i++) {
        const tq = this.holdKp * (this.holdPose[i] - eng.qpos[h.qadr[i]])
                 - this.holdKd * eng.qvel[h.vadr[i]];
        qfrc[h.vadr[i]] = clip(tq, -h.lim[i], h.lim[i]);
      }
      return;
    }
    if ((this.state === "stance" || this.state === "policy") && driver && target) {
      for (let i = 0; i < driver.vadr.length; i++) {
        const tq = driver.kp[i] * (target[i] - eng.qpos[driver.qadr[i]])
                 - driver.kd[i] * eng.qvel[driver.vadr[i]];
        qfrc[driver.vadr[i]] = clip(tq, -driver.lim[i], driver.lim[i]);
      }
      driver.holdExtras(qfrc);
    }
  }
}
