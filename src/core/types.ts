/** Policy-config shapes — mirrors the `config.yaml` carried inside policy
 * packs. Fields we don't consume are allowed through via index signatures. */

export interface ObsTerm { name: string; dim: number }

export interface PolicyCfg {
  path: string;
  num_obs?: number;
  num_act?: number;
  memory_size?: number;
  onnx_inputs?: number;
  obs_terms?: ObsTerm[];
  [k: string]: unknown;
}

/** Go2 policy config block (obs/act normalization, memory, gait, flip).
 * Mirrors the `go2:` config block; unlisted fields pass through. */
export interface Go2Cfg {
  ob_mean: number[]; ob_scale: number[];
  act_mean: number[]; act_scale: number[];
  kp: number[]; kd: number[];
  clip_act?: number; clip_obs?: number;
  memory_size: number; frame_dim: number;
  command_in_frame?: boolean;
  frame_terms?: string[];
  cmd?: (string | number)[];       // split_gait command vector spec
  vel_range?: number[];
  history_newest_first?: boolean;
  base_gait_f?: number; dt?: number; twist_cmd_scale?: number;
  gait_freq_mean?: number; gait_freq_scale?: number; gait_freq_stand?: number;
  gait_move_eps?: number;
  recovery?: boolean;
  ground_start?: string | null;    // null | "on_back"
  zero_vel?: boolean;
  flip?: Record<string, unknown>;  // flip maneuver params (see Go2FlipDriver)
  [k: string]: unknown;
}

export interface EvalConfig {
  name: string;
  robot: string;                     // "r1" | "r1_air" | later "g1" | "go2"
  family: string;                    // motion_tracking | recovery | locomotion | mimic
  control: { fps?: number; step_dt: number; sim_dt: number;
             time_start?: number; time_end?: number };
  policy: PolicyCfg;
  traj?: { path: string; order?: string };
  model?: { joint_ids_map: number[] };
  gains?: {
    default_joint_pos: number[]; action_scale: number[]; action_offset: number[];
    stiffness: number[]; damping: number[];
  };
  locomotion?: {
    default_dof_pos: number[]; kp: number[]; kd: number[];
    action_scale: number | number[];
    action_dof_idx?: number[]; dof_activated_idx?: number[];
    obs_dof_idx?: number[]; control_dof_idx?: number[];
    memory_size: number; vel_range: number[];
    obs_mode?: string; split_frame?: string;
    gait?: Record<string, number>;
  };
  mimic?: Record<string, unknown>;
  go2?: Go2Cfg;
  ground_start?: boolean;
  start_jpos?: number[];
  face?: string;
  control_ui?: { trigger_keys?: string[] };
  [k: string]: unknown;
}

/** One selectable policy — always backed by an imported pack checkpoint. */
export interface PolicyEntry {
  name: string; family: string; robot: string; dof: number;
  group: string;                     // obs-scheme label the UI groups by
  packId: string;                    // owning pack (see src/packs)
  checkpointFile: string;            // in-pack path of this checkpoint's .onnx
}

export interface RobotSpec {
  robot: string;
  anchor_body: string;
  floating_base_joint: string;
  foot_bodies: string[];
  num_act: number;
  joint_names: string[];
  model_dir: string;                 // bundle path holding scene.xml + manifest.json
  fall_height: number;
  sim_dt: number;
  extra_kp?: number;
  extra_kd?: number;
}

export type Command = [number, number, number];   // [vx, vy, wz]
