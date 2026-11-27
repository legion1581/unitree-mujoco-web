/** Unitree AS2 robot spec — 12-DOF quadruped (4 legs × hip/thigh/calf), the official unitree_mujoco model
 * (unitree_robots/as2, visual meshes decimated for the web; collision geoms untouched).
 *
 * Joint order = the MJCF body order FL, FR, RL, RR, which is also the "urdf order" ai_sport's policies use (the
 * firmware maps the SDK motor order FR, FL, RR, RL with [3,4,5,0,1,2,9,10,11,6,7,8], like Go2). Actuator names =
 * joint name minus "_joint". */
import type { RobotSpec } from "../core/types";

const JOINT_NAMES = [
  "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
  "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
  "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
  "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
];

export const AS2: RobotSpec = {
  robot: "as2",
  anchor_body: "base_link",
  floating_base_joint: "floating_base_joint",
  foot_bodies: ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
  num_act: 12,
  joint_names: JOINT_NAMES,
  model_dir: "/assets/models/as2",
  fall_height: 0.15,     // base-z below this = fell (stands ~0.34 m; lying ~0.06 m)
  sim_dt: 0.001,
};
