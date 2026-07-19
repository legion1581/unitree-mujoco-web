/** Unitree Go2 robot spec — 12-DOF quadruped (4 legs × hip/thigh/calf).
 *
 * The MuJoCo model enumerates joints FL, FR, RL, RR, which is the same order
 * the IsaacLab-trained ai policies use, so the policy↔model joint map is the
 * identity (no remap). Actuator names = joint name minus "_joint". Ported from
 * the reference Python implementation.
 */
import type { RobotSpec } from "../core/types";

const JOINT_NAMES = [
  "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
  "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
  "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
  "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
];

export const GO2: RobotSpec = {
  robot: "go2",
  anchor_body: "base_link",
  floating_base_joint: "floating_base",
  foot_bodies: ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
  num_act: 12,
  joint_names: JOINT_NAMES,
  model_dir: "/assets/models/go2",
  fall_height: 0.18,     // base-z below this = fell (quadruped stands ~0.30 m)
  sim_dt: 0.002,
};
