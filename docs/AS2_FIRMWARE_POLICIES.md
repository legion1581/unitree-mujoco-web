# AS2 firmware policies (OTA 2.0.2.0) in the browser sim

Every policy the AS2 2.0.2.0 firmware ships, driven the way its `ai_sport` drives it, checked against the
emulated-MCU bench (orange_pi_workbench, slot `As2/Air_Edu/2.0.2/edu-secdev`). Driver `src/core/drivers/as2_fw.ts`,
importer `scripts/import_as2_ai_sport.py`, registry `registry/as2` (23 runnable packs + 1 obsolete entry), bundle
`policy-packs/as2-all.zip`, robot spec `src/robots/as2.ts`, model bundle `public/assets/models/as2` (the official
unitree_mujoco `as2.xml`, visual meshes decimated to 4.6 MB; collision geoms untouched). Bench tools and captures:
`orange_pi_workbench/tools/ai_sport_probe` (section "AS2"), `captures_as2_202/`.

```
python3 scripts/import_as2_ai_sport.py --src <…/module/ai_sport/1.0.0.17/…/unitree/module/ai_sport>
node scripts/build-packs.mjs --robot as2 --public .          # -> policy-packs/as2-all.zip (23 packs, 76 MB)
```
(`mnn_graph_to_onnx.py` calls `mnnconvert`; it needs a Python with the `MNN` package.)

## Where the policies live
`/unitree/module/ai_sport/ai_sport` 1.0.0.17 is one executable (cyclonedds + statically linked MNN, Virbox
"headers" mode: fabricated section table and stripped locals, but plaintext code; decompiled with IDA). Configs are
plain YAML in `ai_sport/model/` (`rl_base.yaml` shared by every RL state, one yaml per net, `controller.yaml` foot
positions, `state_checker.yaml` nominal stance + torque limit). `model/pose/*.mnn` has no config and is never loaded.

## Policy table
| state | web pack | net | remote (ai_sport rule table) | notes |
|---|---|---|---|---|
| 3 NORMAL_WALK | `normal_walk/3_normal_walk` | normal_walk 2026-03-11 (51 + 6×42) | Start | gait clocks 1.8 Hz (high) / 1.5 Hz (low); Up / Down = speed level (1.5 / 0.8 m/s) |
| 4 SPORT_MODE | `sport_mode/4_sport_mode` | sport_mode 2026-05-14 (51 + 6×42) | L2+Start | 2.6 Hz, vx 3.5 × stick |
| 5 CLIMB_MODE | `climb_mode/5_climb_mode` | climb_mode 2026-01-30 (51 + 6×42) | R1+X | own default angle (front thighs 0.35 / calves −1.3), cmd limit 0.5 / 0.5 / 1.0 |
| 6 / 7 LEFT / RIGHT_SIDE_GAIT | `sided_gait/6_…`, `7_…` | sided_gait 2026-05-11 (47 + 4×42) | X ×2 / Y ×2 | roll command → ∓0.4 |
| 8 HANDSTAND / 9 BIPED_STAND | `sided_gait/8_handstand`, `9_biped_stand` | same net | R1 ×2 / R2 ×2 | pitch command → ±1.2 (front / hind legs) |
| 10 / 11 FRONT / BACK_FLIP | `flip/10_front_flip`, `11_back_flip` | flip f2026-06-07 / b2026-06-04 (46 + 4×42) | Up ×2 / Down ×2 — **refused** | no FSM edge into 10 / 11 (see below); the packs start in the flip |
| 12 RECOVERY | `recovery/12_recovery` | recovery 2026-01-07 (51 + 6×42) | L2+X (from PASSIVE / STAND_DOWN only) | relax 0.5 s → leg-IK fold 1 s → net 1.5 s → NORMAL_WALK; starts on the back |
| 13 BASE_HEIGHT | `base_height/13_base_height` | base_height 2026-06-09 (52 + 6×42) | L1+Up / L1+Down | height 0.35 m ± 0.005 per tick held, clamp [0.2, 0.38]; low speed level |
| 15 EULER | `euler/15_euler` | euler 2026-05-26 (45 + 6×42) | Select | stands; lx → roll 0.3, ly → pitch 0.5, rx → yaw 0.3 rad |
| 16 … 26 mimic | `mimic/16_bai_nian` … `26_up_jump` | one net per motion (obs 69 + time_step; outputs its own reference) | L1+A bai_nian, R2+Y bi_xin, R2+X content, L1+B dance1, L1+X dance2, R2+B handshake (wo_shou), R2+A stretch (shen_lan_yao), R1+B sit, R1+A front_jump, L1+Y push-up (fuwocheng), R1+Y up_jump | frames idx[0]..idx[1] of mimic.yaml, then NORMAL_WALK; SIT loops its last 50 frames until Start |
| 0 PASSIVE / 1 STAND_DOWN / 2 STAND_UP | (in every pack) | scripted (leg IK) | L2+B / L2+A toggles up ↔ down | see below |

Also on the remote: Y+B toggles auto-recovery (default on), Down+A+B = 102 HIP_TEST (motor test, not packed).

## Common to the RL states
- obs frame (42) = `[gyro·0.25, gravity, (q − default)·1, dq·0.05, last action]`, urdf order FL FR RL RR (= the web
  model order; ai_sport maps the SDK order FR FL RR RL with `[3,4,5,0,1,2,9,10,11,6,7,8]`).
- net input = `[current obs, history]`, history = the previous ticks' clipped obs[0:42], **oldest first**, filled
  with the entry frame on state entry; obs clipped ±10 (flip ±20), action clipped ±10 (flip ±20).
- target = default + 0.25·action; PD hip / thigh kp 40 kd 1, calf kp 60 kd 1.5; walk, sport, side gaits, biped,
  flips, base height and recovery then clamp each target so |kp(q* − q) − kd·dq| ≤ 60 / 60 / 90 N·m
  (`state_checker.yaml` torque_limit; climb, handstand, euler and the mimics don't).
- one inference per 20 ms tick (IDA shows two calls in some runners — decompiler artefacts; the bench captures show
  one session per tick and a single `bl` in the disassembly).
- raw command = (ly, −lx, −rx), dead zone 0.05; walk-like filters `f = (1−α)f + α·raw·limit` (α = vel_filter
  0.1); **vy is scaled by vx_limit** before being clamped to vy_limit (firmware quirk, visible in the captures);
  NORMAL_WALK: |vx| > 1 → wz limit 2.0, |wz| > 1.5 → vy limit 0.
- gait clocks: "moving" = |previous command| > 0.05 or a foot (leg FK) off the nominal stance by > 0.1 m in the 1 s
  state-checker window; moving: `[frac(φ), frac(φ+½), frac(φ+½), frac(φ)]`, φ = freq·t; standing: 0.75 ×4 and no
  foot-height term.
- fall: gravity z > 0.5 (climb −0.1, side gaits 0.0, handstand / biped 0.2) requests RECOVERY when auto-recovery
  is on, else PASSIVE — but the RECOVERY edge is only allowed from PASSIVE / STAND_DOWN (guard 106FE0), so per the
  decoded table a robot that falls while walking stays in the walking state (L2+B then L2+X recovers; this path was
  not exercised on the bench).

## Scripted states and transitions (sub_196C50 transition table)
- PASSIVE / STAND_DOWN: q* = q, kp 0, kd 3.
- PASSIVE / STAND_DOWN → STAND_UP: 1.5 s blending the live q into IK(foot_stand_down), then 1.2 s moving the feet
  stand_down → stand_up (kp 150/150/200, kd 10/10/12); STAND_UP holds IK(foot_stand_up).
- STAND_UP → STAND_DOWN: feet stand_up → lie_down in 1.2 s; walking → STAND_DOWN: same with kp 40/40/60 kd 5;
  walking → STAND_UP: 2 s blend of target and gains toward IK(foot_stand_up).
- exit routines: side gaits → walk = 1 s of the sided net with sticks released and posture ×0.95 per tick; euler →
  walk ≤ 0.8 s settle; base height → walk eases the height back to 0.35; SIT → walk = 0.5 s stand-up blend
  (kp 80/80/120, kd 4/4/6).
- leg IK / FK: hip origins (±0.2079, ±0.06, 0), hip offset 0.1044, thigh 0.212, calf 0.21348 (firmware geometry;
  the MJCF hips sit at ±0.2119).

## Mimic nets
obs = `[ref q (12), ref dq (12), anchor orientation 6D, gyro, q, dq, last action]` − obs_mean, joint-major order
(hips FL FR RL RR, thighs, calves). ref q / dq = the net's own previous `joint_pos` / `joint_vel` outputs;
orientation = first two columns of `R_robotᵀ · Rz(yaw₀) · R(body_quat_w[0])` (yaw₀ latched on entry; quaternion
wxyz; fitted against the captures, median error 6e-4). `time_step` = frame index, held at idx[0] during the 0.2 s
transition, then +1 per tick; target = act_mean + act·act_scale blended in from the entry pose with smoothstep(t/T).

## Flips: in the firmware, unreachable
The Up ×2 / Down ×2 rules request states 10 / 11, but the transition table has no edge into them and no sport-API
handler forces them, so 2.0.2.0 refuses them (bench: the state stays 3). Forced on the bench with
`as2_force_flips.py` (gdb: copies the flip states' enter / run / exit over the side-gait states): full 360° front
and back flips that land and walk on. The web packs start in the flip; "Flip again" (Up ×2 / Down ×2) re-runs it.

## Bench vs web (captures_as2_202, as2_benchsum.py vs as2_webrun.py)
| run | bench | web |
|---|---|---|
| standing z (walk / stand up) | 0.344 / 0.350 | 0.340 / 0.349 |
| euler pitch / roll tilt | 22.7° / 13.8° | 23.0° / 13.9° |
| handstand / biped z, tilt | 0.498, 71.1° / 0.508, 70.3° | 0.496, 70.1° / 0.504, 70.8° |
| base height (two L1+Down) | z 0.287 | z 0.287 |
| bai nian / dance1 / sit | 7.21 s, pitch −103.7° / 19.1 s / pitch −42.4° | 7.20 s, −103.9° / 19.06 s / −42.4° |
| front / up jump | z 0.373, 1.08 s / z 0.458, 1.27 s | z 0.371, 1.06 s / z 0.444, 1.30 s |
| forced front / back flip | +360°, z 0.749 / −361°, z 0.667 | +360°, z 0.737 / ≈ −370°, z 0.674 |
| walk vx at 0.9 cmd / sport vx at 1.75 cmd | 0.38 / 0.81 m/s | 0.87 / 1.62 m/s |

Postures, transitions, durations and the flips match. Walking speed doesn't: on the bench every gait reaches only
~45–50 % of its command (yaw ~60–75 %). The cause is not in the policy path — every net input ai_sport built during
the captures matches the decode, the ONNX conversions reproduce the bench MNN outputs (≤ 1e-4), the obs ai_sport sees
is consistent with the plant (q, dq, gyro, gravity, no lag), basic_service forwards the lowcmd targets unchanged, and
the same MJCF in MuJoCo 3.10 (the bench's version) with the decoded obs walks at the commanded 0.87 m/s offline. The
remaining suspect is the bench's command delivery (ai_sport → wire latency measured 4–36 ms, median 25 ms), still
open. The web runs the ideal plant (1 kHz, no command delay).
