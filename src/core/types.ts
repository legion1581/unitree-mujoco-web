/** Policy-config shapes — mirrors the `config.yaml` carried inside policy
 * packs. Fields we don't consume are allowed through via index signatures. */

export interface PolicyCfg {
  path: string;
  num_obs?: number;
  num_act?: number;
  [k: string]: unknown;
}

/** Go2 block of a firmware pack (`go2.firmware: true`); the driver reads it as Go2FwCfg (go2_fw.ts). */
export interface Go2Cfg {
  firmware?: boolean;
  fsm_id?: number;
  /** extra networks by role -> path or blob url */
  aux_models?: Record<string, string>;
  [k: string]: unknown;
}

/** AS2 block of a firmware pack (`as2.firmware: true`); the driver reads it as As2FwCfg (as2_fw.ts). */
export interface As2Cfg {
  firmware?: boolean;
  fsm_id?: number;
  /** extra networks by role -> path or blob url */
  aux_models?: Record<string, string>;
  [k: string]: unknown;
}

export interface G1Cfg {
  obs_scheme?: string;
  control_joints: number[];
  num_obs: number;
  num_act: number;
  memory_size: number;
  dof_pos_rel?: boolean;
  default_dof_pos: number[];
  action_scale: number | number[];
  action_clip?: number;
  obs_clip?: number;
  ang_vel_scale?: number;
  dof_pos_scale?: number;
  dof_vel_scale?: number;
  dt?: number;
  cmd_in_obs?: boolean;
  has_phase?: boolean;
  maneuver?: boolean;
  ground_start?: string | null;
  spawn_pitch_deg?: number;
  init_pose?: number[];
  maneuver_period?: number;
  kp?: number[];
  kd?: number[];
  tau?: number[];
  vel_range?: number[];
  motion_clip?: string;
  quat_order?: string;
  root_rel?: string;
  clip_loop?: boolean;
  /** mimic: "entry_yaw" = quaternion input relative to the heading at policy start (ai_sport Cpy mimic) */
  quat_frame?: string;
  /** mimic: step k tracks clip row k + clip_offset */
  clip_offset?: number;
  /** mimic: the FSM exits after this fraction of the clip (ai_sport Cpy mimic: 0.9) */
  exit_after_frac?: number;
  /** policy the robot auto-switches to when this one is done (e.g. "cpy_walk_run_29dofs/actor") */
  next_policy?: string;
  orient?: string;
  stance_cmd_offset?: number[];
  /** ai_sport Cpy* locomotion (801/802/812): extra networks the FSM swaps in, sharing one obs history.
   * `stance` runs while there is no translation command (yaw-only included), `run` from
   * run_switch[0] m/s forward until the command drops to run_switch[1]. Paths as `policy.path`. */
  aux_models?: Record<string, string>;
  /** extra DATA files a driver needs (latent / motion csv, tables), by role -> path or blob url */
  aux_files?: Record<string, string>;
  /** ai_sport FSM id this policy runs as on the robot (shown in the UI) */
  fsm_id?: number;
  /** G1 variants the policy runs on: "air" (machinetype 4), "edu" (5); the picker lists Air first
   * (R1 packs use the same convention, see EvalConfig.r1 / models) */
  models?: string[];
  run_switch?: [number, number];
  /** last_action slot = action * action_scale (the joint offset) instead of the raw output */
  last_action_scaled?: boolean;
  /** firmware command shaping (Cpy* FSMs): per-axis up/down deltas per tick, the first step out of
   * standstill, and the translation speed under which the command snaps back to 0 */
  cmd_ramp?: { up: number[]; down: number[]; start: number[]; snap: number[] };
  /** remote speed levels (decompiled + bench-checked, see docs/G1_FIRMWARE_POLICIES.md): the level sets the
   * full-stick command (and, for FsmHumanLoco, the gait); `combos` maps a remote combo to a level name or to
   * a step of the forward limit (802/801: R2+Left/Right = -/+0.1 m/s, capped at `vx_cap`) */
  speed?: G1SpeedModes;
  /** bm_getup (1001): net slot i drives Unitree joint action_joint_idx[i] (IsaacLab order) */
  action_joint_idx?: number[];
  /** bm_getup (1001): 50 Hz ticks of the move-to-first-frame interpolation / hold, and the reference rows
   * before its end where the firmware hands over to 802 */
  start_frames?: number;
  hold_frames?: number;
  end_frames?: number;
  /** legacy (2024) loco FSMs 200-203 / 210: gait scheduler, filters and obs constants (see G1LegacyMode) */
  legacy?: G1LegacyMode & { walk?: G1LegacyMode };
  [k: string]: unknown;
}

/** Remote speed levels of an ai_sport loco FSM (G1Cfg.speed). */
export interface G1SpeedModes {
  /** level name -> full-stick vx (m/s); optional vy / wz maxima and FsmHumanLoco gait frequency / duty */
  levels: Record<string, { vx: number; vy?: number; wz?: number; freq?: number; duty?: number }>;
  /** level at FSM entry */
  initial: string;
  /** remote combo -> level name, or a signed step (m/s) of the current forward limit */
  combos: Record<string, string | number>;
  /** upper bound of the stepped forward limit (Cpy*: high_speed + 1) */
  vx_cap?: number;
  /** Cpy* backward rule: vx < 0 is the raw stick (so at most -1 m/s at any level), 0 above -0.3 m/s */
  back_stick?: boolean;
}

/** One ai_sport legacy loco FSM (hybrid_run 200, blind_terrain 201, plane_encoder 202, waist_control 203,
 * shot 210), as decoded from the bench MNN captures. Unset fields fall back to the hybrid_run values. */
export interface G1LegacyMode {
  kp?: number[]; kd?: number[]; default_dof_pos?: number[]; tau?: number[];
  frame_dim?: number;                // 40, or 42 (waist_control / shot)
  num_latent?: number;               // encoder output slots appended to the actor input (0 = single net)
  omega_alpha?: number;              // gyro low-pass  w = a*w_new + (1-a)*w
  jvel_alpha?: number;               // joint-velocity low-pass
  ankle_lpf_hz?: number;             // ankle pitch/roll position low-pass (obs only)
  cmd_alpha?: number;                // remote-command low-pass per 10 ms tick
  vx_max?: number;                   // vx_cmd_levels[0]
  delta_phi?: [number, number];      // leg phase offsets, units of pi
  velocity_points?: number[]; duty_factor_points?: number[]; period_points?: number[];
  duty_mean?: number; duty_scale?: number;
  obs_tail?: [number, number];       // constant gait slots 37 (stand height) and 39
  stand_phase?: number;              // raw gait phase (cycles) at FSM entry
  swing?: string;                    // arm swing: clock | leg | none
  rules?: boolean;                   // keep stepping while the base tilts / spins (enable_rules)
  stand_kd_mult?: number;            // stand_damping_multiplier on the legs while standing
  waist_obs?: boolean;               // frame tail [q waist pitch, -q waist roll] (203)
  kick?: boolean;                    // shot: kick clock slots, one gait period then back to `walk`
  kick_upper?: number[][];           // shot: [tick, waist roll, left wrist yaw] upper-body key frames
  kick_delay?: number;               // shot: s after reset before the kick fires
  kick_repeat?: number;              // shot: re-kick every N s while standing (0 = once)
}

export interface EvalConfig {
  name: string;
  robot: string;                     // "r1" | "r1_air" | "g1" | "go2" | "as2" (pack manifest robot)
  family: string;                    // r1_fw | go2_fw | as2_fw | g1_*
  control: { fps?: number; step_dt: number; sim_dt: number; cmd_delay?: number;
             time_start?: number; time_end?: number; duration?: number };
  policy: PolicyCfg;
  go2?: Go2Cfg;
  as2?: As2Cfg;
  g1?: G1Cfg;
  /** R1 block (`firmware: true`, read as R1FwCfg in r1_fw.ts); `models` = R1 variants the policy runs on
   * ("air" = 20-DOF, "edu" = 24-DOF), like g1.models */
  r1?: { models?: string[]; [k: string]: unknown };
  control_ui?: {
    trigger_keys?: string[];
    /** named remote actions, grouped for the panel (e.g. 505: punches / kicks / footwork) */
    actions?: RemoteActionGroup[];
    /** shown in a popup when the policy is selected (how to control it) */
    notice?: { title?: string; text: string };
  };
  [k: string]: unknown;
}

/** A group of named remote-controller actions (a label per combo, e.g. "Left jab" = R1+Up). */
export interface RemoteActionGroup { group: string; items: { combo: string; label: string }[] }

/** One selectable policy — always backed by an imported pack checkpoint. */
export interface PolicyEntry {
  name: string; family: string; robot: string; dof: number;
  group: string;                     // obs-scheme label the UI groups by
  packId: string;                    // owning pack (see src/packs)
  checkpointFile: string;            // in-pack path of this checkpoint's .onnx
  fsmId?: number;                    // ai_sport FSM id this policy runs as on the robot (G1)
  models?: string[];                 // G1 / R1 variants it runs on ("air" / "edu"), as declared by the pack
}

export interface RobotSpec {
  robot: string;
  anchor_body: string;
  /** body + body-frame point the overhead hoist holds (default: anchor_body at its COM) */
  hoist_body?: string;
  hoist_offset?: [number, number, number];
  floating_base_joint: string;
  foot_bodies: string[];
  num_act: number;
  joint_names: string[];
  model_dir: string;                 // bundle path holding scene.xml + manifest.json
  /** scene file in model_dir to compile (default "scene.xml"); variants sharing one bundle (G1 Air) */
  scene?: string;
  fall_height: number;
  sim_dt: number;
  extra_kp?: number;
  extra_kd?: number;
  torque_limit?: number[];
  /** Parallel-linkage ankles driven in MOTOR space by the robot's basic_service (Unitree R1): for each pair
   * [pitch, roll] with the linkage matrix m (row-major 2×2, joint = m · motor), the stiffness stays the
   * joint-space kp (basic_service folds kp·(q_des − q) into the motor targets) but the damping is the motors'
   * own kd_m = Σ_j kd_j·m[j][motor]², i.e. joint damping D = m⁻ᵀ·diag(kd_m)·m⁻¹ (R1: pitch ≈ 1.7× kd, roll
   * ≈ 0.7× kd). Measured on the bench (mcu_ffs wire kp/kd/q of the ankle motors). */
  ankle_pairs?: { joints: [string, string]; m: [number, number, number, number] }[];
}

export type Command = [number, number, number];   // [vx, vy, wz]
