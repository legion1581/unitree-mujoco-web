/** Unitree R1 **Air** spec — the 20-DOF variant (no waist, no wrist_roll).
 * Reuses the 24-DOF Edu model; the absent joints are held near-rigid via
 * extra_kp/extra_kd (the runner holds any hinge not in joint_names). */
import type { RobotSpec } from "../core/types";

export const JOINT_NAMES = [
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
  "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
  "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint", "left_elbow_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint", "right_elbow_joint",
];

export const R1_AIR: RobotSpec = {
  robot: "r1_air",
  anchor_body: "torso_link",
  floating_base_joint: "floating_base_joint",
  foot_bodies: ["left_ankle_roll_link", "right_ankle_roll_link"],
  num_act: 20,
  joint_names: JOINT_NAMES,
  model_dir: "/assets/models/r1",
  fall_height: 0.40,
  sim_dt: 0.002,
  extra_kp: 400.0,
  extra_kd: 10.0,
};
