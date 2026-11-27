/** Driver factory — builds the right driver for a policy config and loads its
 * networks (and clip / data files, when the family needs them). Every pack is a
 * firmware-faithful G1 / R1 / Go2 / AS2 policy: R1 → r1_fw.ts, Go2 → go2_fw.ts, AS2 → as2_fw.ts, G1 →
 * the g1*.ts drivers picked by `g1.obs_scheme`. */
import { OnnxPolicy, OnnxSession } from "../policy";
import type { EvalConfig, RobotSpec } from "../types";
import type { Engine } from "../../sim/engine";
import { assetUrl, loadTraj } from "../assets";
import { BaseDriver } from "./base";
import { G1LocoDriver, G1MimicDriver } from "./g1";
import { G1LegacyDriver } from "./g1_legacy";
import { G1VqDriver } from "./g1_vq";
import { G1PunchDriver } from "./g1_punch";
import { G1GetupDriver } from "./g1_getup";
import { G1FaceupDriver, G1Squat2StandDriver } from "./g1_faceup";
import { G1JumpDriver } from "./g1_jump";
import { makeR1FwDriver } from "./r1_fw";
import { makeGo2FwDriver } from "./go2_fw";
import { makeAs2FwDriver } from "./as2_fw";

export type { BaseDriver };

export async function makeDriver(cfg: EvalConfig, spec: RobotSpec, eng: Engine):
    Promise<BaseDriver> {
  const fam = cfg.family;
  const policyUrl = assetUrl(cfg.policy.path).replace(/\.mnn$/, ".onnx");

  // R1 ai_sport firmware policies (registry/r1, see r1_fw.ts)
  if (cfg.r1?.firmware) return makeR1FwDriver(cfg, spec, eng);
  // Go2 mcf firmware policies (registry/go2, see go2_fw.ts)
  if (cfg.go2?.firmware) return makeGo2FwDriver(cfg, spec, eng);
  // AS2 ai_sport firmware policies (registry/as2, see as2_fw.ts)
  if (cfg.as2?.firmware) return makeAs2FwDriver(cfg, spec, eng);

  if (fam.startsWith("g1_") || cfg.robot === "g1" || cfg.g1) {
    // 813 CpyWalkJump / 900 ZjxWalkJump: future-state / 201-dim mimics of the walk-jump clip (see g1_jump.ts)
    if (cfg.g1?.obs_scheme === "cpy_jump" || cfg.g1?.obs_scheme === "zjx_jump") {
      const sess = await OnnxSession.load(policyUrl);
      const traj = await loadTraj(cfg.g1.motion_clip!);
      return new G1JumpDriver(cfg, spec, eng, sess, traj.data, traj.nCols);
    }
    if (fam === "g1_mimic" || cfg.g1?.obs_scheme === "mimic") {
      const sess = await OnnxSession.load(policyUrl);
      const clipPath = cfg.g1?.motion_clip;
      if (!clipPath) throw new Error(`${cfg.name}: g1 mimic requires motion_clip`);
      const traj = await loadTraj(clipPath);
      return new G1MimicDriver(cfg, spec, eng, sess, traj.data, traj.nCols);
    }
    // ai_sport 1.5.4 families added from bench captures (see docs/G1_FIRMWARE_POLICIES.md):
    //   legacy = 2024 encoder/actor loco (200-203, 210); vq = VQ-latent decoders (503, 504, 706);
    //   bm_getup = BeyondMimic get-up with an embedded motion table (1001)
    //   faceup = 702 scripted face-up get-up + squat2stand VQ decoder; squat2stand = 703 phase-clock net
    //   punch = 505 PunchMode (traj walker + remote-triggered punch / get-down / get-up trajectories)
    const scheme = cfg.g1?.obs_scheme;
    if (scheme === "squat2stand")
      return new G1Squat2StandDriver(cfg, spec, eng, await OnnxSession.load(policyUrl));
    if (scheme === "legacy" || scheme === "vq" || scheme === "bm_getup" || scheme === "faceup" || scheme === "punch") {
      const sess = await OnnxSession.load(policyUrl);
      const aux: Record<string, OnnxSession> = {};
      for (const [role, p] of Object.entries(cfg.g1?.aux_models ?? {}))
        if (p) aux[role] = await OnnxSession.load(assetUrl(p).replace(/\.mnn$/, ".onnx"));
      const files = Object.fromEntries(Object.entries(cfg.g1?.aux_files ?? {}).map(([k, p]) => [k, assetUrl(p)]));
      if (scheme === "legacy") return new G1LegacyDriver(cfg, spec, eng, sess, aux, files);
      if (scheme === "vq") return new G1VqDriver(cfg, spec, eng, sess, aux, files);
      if (scheme === "faceup") return new G1FaceupDriver(cfg, spec, eng, sess, aux, files);
      if (scheme === "punch") return new G1PunchDriver(cfg, spec, eng, sess, aux, files);
      return new G1GetupDriver(cfg, spec, eng, sess, aux, files);
    }
    const isMultiInput = cfg.g1?.obs_scheme === "split" || cfg.g1?.obs_scheme === "human_loco" || (cfg.policy.onnx_inputs ?? 1) === 3;
    const sess = isMultiInput
      ? await OnnxSession.load(policyUrl)
      : await OnnxPolicy.load(policyUrl);
    // Cpy* locomotion: the stance / run networks the firmware swaps in
    const aux: Record<string, OnnxSession> = {};
    for (const [role, p] of Object.entries(cfg.g1?.aux_models ?? {}))
      if (p) aux[role] = await OnnxSession.load(assetUrl(p).replace(/\.mnn$/, ".onnx"));
    return new G1LocoDriver(cfg, spec, eng, sess, aux);
  }

  throw new Error(`no driver for family '${fam}' (${cfg.name}) — not a G1 / R1 / Go2 / AS2 firmware pack`);
}
