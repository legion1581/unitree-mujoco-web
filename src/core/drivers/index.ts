/** Driver factory — builds the right driver for a policy config and
 * loads its policy (and clip, when the family needs one). Mirrors make_driver
 * in the reference Python implementation, plus the async asset loading. */
import { OnnxPolicy, OnnxSession } from "../policy";
import type { EvalConfig, RobotSpec } from "../types";
import type { Engine } from "../../sim/engine";
import { assetUrl, loadTraj } from "../assets";
import { BaseDriver } from "./base";
import { MotionTrackingDriver } from "./motionTracking";
import { ArmsdkDriver, AmpSplitDriver, LocomotionConcatDriver, LocomotionSplitDriver }
  from "./locomotion";
import { MimicDriver } from "./mimic";
import { Go2Driver, Go2FlipDriver } from "./go2";

export type { BaseDriver };

export async function makeDriver(cfg: EvalConfig, spec: RobotSpec, eng: Engine):
    Promise<BaseDriver> {
  const fam = cfg.family;
  const loc = cfg.locomotion;
  const policyUrl = assetUrl(cfg.policy.path);

  if (fam === "go2_locomotion" || fam === "go2_recovery")
    return new Go2Driver(cfg, spec, eng, await OnnxSession.load(policyUrl));
  if (fam === "go2_flip")
    return new Go2FlipDriver(cfg, spec, eng, await OnnxSession.load(policyUrl));
  // go2_vision (LIDAR nav) not ported yet

  if (fam === "motion_tracking" || fam === "recovery") {
    const [policy, traj] = await Promise.all([
      OnnxPolicy.load(policyUrl), loadTraj(cfg.traj!.path)]);
    return new MotionTrackingDriver(cfg, spec, eng, traj, policy);
  }
  if (fam === "mimic") {
    const sess = await OnnxSession.load(policyUrl);
    const clip = cfg.traj?.path ? await loadTraj(cfg.traj.path) : null;
    return new MimicDriver(cfg, spec, eng, sess, clip);
  }
  if (loc?.obs_mode === "armsdk")
    return new ArmsdkDriver(cfg, spec, eng, await OnnxPolicy.load(policyUrl));
  if (loc?.split_frame === "amp")
    return new AmpSplitDriver(cfg, spec, eng, await OnnxSession.load(policyUrl));
  if ((cfg.policy.onnx_inputs ?? 1) === 3)
    return new LocomotionSplitDriver(cfg, spec, eng, await OnnxSession.load(policyUrl));
  if (fam === "locomotion")
    return new LocomotionConcatDriver(cfg, spec, eng, await OnnxPolicy.load(policyUrl));
  throw new Error(`no driver for family '${fam}' (${cfg.name})`);
}
