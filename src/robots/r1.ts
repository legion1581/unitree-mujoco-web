/** Unitree R1 robot spec — the parts NOT in the on-robot config: fixed model
 * joint order, anchor body, model bundle location. Once per robot. */
import type { RobotSpec } from "../core/types";

// 24 controlled joints, in model / trajectory-CSV order (URDF DFS order).
export const JOINT_NAMES = [
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
  "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
  "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
  "waist_roll_joint", "waist_yaw_joint",
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
];

export const R1: RobotSpec = {
  robot: "r1",
  anchor_body: "torso_link",
  floating_base_joint: "floating_base_joint",
  foot_bodies: ["left_ankle_roll_link", "right_ankle_roll_link"],
  num_act: 24,
  joint_names: JOINT_NAMES,
  model_dir: "/assets/models/r1",
  fall_height: 0.40,
  sim_dt: 0.002,
};
