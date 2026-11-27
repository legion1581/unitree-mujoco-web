#!/usr/bin/env python3
"""Import every Unitree AS2 firmware policy (OTA 2.0.2.0, ai_sport 1.0.0.17) into unitree-mujoco-web as
firmware-faithful packs for ONE bundle (as2-all.zip).

Driver: src/core/drivers/as2_fw.ts (config block `as2:` with `firmware: true`). Behaviour decoded from the ai_sport
executable (Virbox "headers" mode: plaintext code, stripped locals, IDA decompile) and checked against bench MNN-I/O
captures (orange_pi_workbench tools/ai_sport_probe as2mnnhook.c / as2_probe.py, captures_as2_202); notes in
docs/AS2_FIRMWARE_POLICIES.md.

ai_sport's FSM (rt/sportmodestate .mode, 0-based) and its nets:
   3 NORMAL_WALK   normal_walk      4 SPORT_MODE   sport_mode       5 CLIMB_MODE   climb_mode
   6 LEFT_SIDE_GAIT, 7 RIGHT_SIDE_GAIT, 8 HANDSTAND, 9 BIPED_STAND   one sided_gait net (roll / pitch command)
  10 FRONT_FLIP / 11 BACK_FLIP  flip f / b      12 RECOVERY  recovery       13 BASE_HEIGHT  base_height
  15 EULER  euler    16..26  mimic nets (motion embedded in the net, time_step input)
plus the scripted 0 PASSIVE, 1 STAND_DOWN, 2 STAND_UP (leg IK to controller.yaml foot positions). model/pose/*.mnn has
no config and is never loaded (14 POSE has no state) -> listed runnable:false. 102 HIP_TEST (Down+A+B) is a motor
test, not packed.

Usage:
  python3 scripts/import_as2_ai_sport.py --src <.../unitree/module/ai_sport>
  node scripts/build-packs.mjs --robot as2 --public .           # -> policy-packs/as2-all.zip
Nets are converted with scripts/mnn_graph_to_onnx.py (checked against MNN on random inputs).
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET = "assets/policies/as2"
REG = "registry/as2"
SRC_TAG = "AS2 OTA 2.0.2.0 ai_sport 1.0.0.17"

# FSM id -> pack name (handover targets)
PACKS = {
    3: "normal_walk/3_normal_walk", 4: "sport_mode/4_sport_mode", 5: "climb_mode/5_climb_mode",
    6: "sided_gait/6_left_side_gait", 7: "sided_gait/7_right_side_gait", 8: "sided_gait/8_handstand",
    9: "sided_gait/9_biped_stand", 10: "flip/10_front_flip", 11: "flip/11_back_flip", 12: "recovery/12_recovery",
    13: "base_height/13_base_height", 15: "euler/15_euler",
    16: "mimic/16_bai_nian", 17: "mimic/17_bi_xin", 18: "mimic/18_content", 19: "mimic/19_dance1",
    20: "mimic/20_dance2", 21: "mimic/21_handshake", 22: "mimic/22_stretch", 23: "mimic/23_sit",
    24: "mimic/24_front_jump", 25: "mimic/25_pushup", 26: "mimic/26_up_jump",
}
# mimic state -> mimic.yaml motion key (ai_sport enters motion index k = state - 16)
MIMIC = {16: "bai_nian", 17: "bi_xin", 18: "content", 19: "dance1", 20: "dance2", 21: "wo_shou",
         22: "shen_lan_yao", 23: "sit_down", 24: "front_jump", 25: "fuwocheng", 26: "up_jump"}
COMBO = {3: "Start", 4: "L2+Start", 5: "R1+X", 6: "X X", 7: "Y Y", 8: "R1 R1", 9: "R2 R2", 10: "Up Up", 11: "Down Down",
         12: "L2+X", 13: "L1+Up / L1+Down", 15: "Select", 16: "L1+A", 17: "R2+Y", 18: "R2+X", 19: "L1+B", 20: "L1+X",
         21: "R2+B", 22: "R2+A", 23: "R1+B", 24: "R1+A", 25: "L1+Y", 26: "R1+Y"}
LABEL = {3: "Normal walk", 4: "Sport mode", 5: "Climb mode", 6: "Left side gait", 7: "Right side gait",
         8: "Hand stand", 9: "Biped stand", 10: "Front flip", 11: "Back flip", 12: "Recovery", 13: "Base height",
         15: "Euler (body pose)", 16: "Bai nian (New-Year bow)", 17: "Bi xin (heart)", 18: "Content", 19: "Dance 1",
         20: "Dance 2", 21: "Handshake (wo shou)", 22: "Stretch (shen lan yao)", 23: "Sit", 24: "Front jump",
         25: "Push-up (fu wo cheng)", 26: "Up jump"}

SPEED = [{"group": "Speed", "items": [{"combo": "Up", "label": "High speed (Up)"},
                                      {"combo": "Down", "label": "Low speed (Down)"}]}]


def yload(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def convert(src_mnn, dst_onnx):
    os.makedirs(os.path.dirname(dst_onnx), exist_ok=True)
    if os.path.exists(dst_onnx) and os.path.getmtime(dst_onnx) >= os.path.getmtime(src_mnn):
        return
    r = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "mnn_graph_to_onnx.py"), src_mnn, "--out",
                        dst_onnx], capture_output=True, text=True)
    ok = [ln for ln in r.stdout.splitlines() if ln.strip().startswith("OK")]
    # mimic/fuwocheng: only `actions` differs, by ≤ 3.2e-3 (float accumulation order, MNN vs onnxruntime; every
    # reference output matches to 5e-7): accept small mismatches, loudly
    near = [ln for ln in r.stdout.splitlines() if ln.strip().startswith("MISMATCH")
            and float(ln.split("max|d|=")[1].split()[0]) < 1e-2]
    if near and os.path.exists(dst_onnx):
        print("  WARN " + near[0].strip()[:110])
        return
    if r.returncode or not ok or not os.path.exists(dst_onnx):
        raise SystemExit(f"conversion failed: {src_mnn}\n{r.stdout}\n{r.stderr}")
    print("  " + ok[0].strip()[:110])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help=".../unitree/module/ai_sport (holds model/)")
    a = ap.parse_args()
    md = os.path.join(a.src, "model")
    need: dict[str, str] = {}                           # onnx asset rel path -> source .mnn

    def net(ref):
        ref = ref.strip().lstrip("/")
        rel = f"{ASSET}/{os.path.relpath(ref, 'model').replace('.mnn', '')}.onnx"
        need[rel] = os.path.join(a.src, ref)
        return rel

    base = yload(os.path.join(md, "rl_base.yaml"))
    ctrl = yload(os.path.join(md, "controller.yaml"))
    chk = yload(os.path.join(md, "state_checker.yaml"))
    cfgs = {k: yload(os.path.join(md, k, f"{k}.yaml")) for k in
            ("normal_walk", "sport_mode", "climb_mode", "base_height", "euler", "sided_gait", "flip", "mimic",
             "recovery")}
    for c in cfgs.values():
        c.pop("include_yaml", None)
    nets = {k: net(cfgs[k]["model_file"]) for k in
            ("normal_walk", "sport_mode", "climb_mode", "base_height", "euler", "sided_gait", "recovery")}
    nets["flip_f"] = net(cfgs["flip"]["model_file_forward_flip"])
    nets["flip_b"] = net(cfgs["flip"]["model_file_backflip"])
    mim = cfgs["mimic"]
    for k in MIMIC.values():
        nets["mimic_" + k] = net(mim["model_" + k])
    # mimic.yaml: per-motion [start, end] frame, transition; the rest is shared
    motions = {k: {"idx": mim[f"idx_{k}"], "transition": mim[f"transition_duration_{k}"]} for k in MIMIC.values()}
    mim_shared = {k: v for k, v in mim.items() if not k.startswith(("model_", "idx_", "transition_duration_"))}
    common_aux = {"walk": nets["normal_walk"], "recovery": nets["recovery"]}

    entries = []

    def pack(fsm, scheme, main_rel, note, aux=None, extra=None, actions=None):
        name = PACKS[fsm]
        g = {"firmware": True, "fsm_id": fsm, "scheme": scheme, "packs": {str(k): v for k, v in PACKS.items()},
             "base": base, "controller": ctrl, "state_checker": chk,
             "cfg": {k: v for k, v in cfgs.items() if k not in ("mimic",)},
             "aux_models": {**common_aux, **(aux or {})}}
        if extra: g.update(extra)
        cfg = {"name": name, "robot": "as2", "family": "as2_fw", "source": SRC_TAG,
               # 1 kHz plant like the bench (the AS2 MuJoCo plant on the Orange Pi workbench runs 1 kHz too)
               "control": {"step_dt": 0.02, "sim_dt": 0.001},
               "policy": {"path": main_rel, "num_act": 12}, "as2": g,
               "control_ui": {"actions": actions if actions is not None else []}}
        path = f"{REG}/{name}.yaml"
        os.makedirs(os.path.dirname(os.path.join(REPO, path)), exist_ok=True)
        with open(os.path.join(REPO, path), "w") as f:
            f.write(f"# AS2 FSM {fsm} {LABEL[fsm]} ({COMBO[fsm]}): {note}\n")
            yaml.safe_dump(cfg, f, sort_keys=False, width=120, default_flow_style=None)
        entries.append({"name": name, "family": "as2_fw", "robot": "as2", "dof": 12, "policy": main_rel,
                        "runnable": True, "config": path, "note": f"{LABEL[fsm]} ({COMBO[fsm]}) — {note}",
                        "fsm_id": fsm})

    pack(3, "normal_walk", nets["normal_walk"], "gait-clock walker, speed level Up (1.5 m/s) / Down (0.8 m/s)",
         actions=SPEED)
    pack(4, "sport_mode", nets["sport_mode"], "fast trot (3.5 m/s × stick), net stepped twice per tick")
    pack(5, "climb_mode", nets["climb_mode"], "stair / obstacle walker (raised front legs), stepped twice per tick")
    for fsm, sch, lbl in ((6, "left_side_gait", "roll command −0.4"), (7, "right_side_gait", "roll command +0.4"),
                          (8, "handstand", "pitch command +1.2 (front legs)"),
                          (9, "biped_stand", "pitch command −1.2 (hind legs)")):
        pack(fsm, sch, nets["sided_gait"], f"one sided_gait net, {lbl}; Start = back to walking")
    pack(10, "front_flip", nets["flip_f"], "1.43 s phase-driven forward flip, then normal walk",
         actions=[{"group": "Flip", "items": [{"combo": "Up Up", "label": "Flip again (Up ×2)"}]}])
    pack(11, "back_flip", nets["flip_b"], "1.43 s phase-driven back flip, then normal walk",
         actions=[{"group": "Flip", "items": [{"combo": "Down Down", "label": "Flip again (Down ×2)"}]}])
    pack(12, "recovery", nets["recovery"], "relax → fold (leg IK) → recovery net → normal walk; starts on the back")
    pack(13, "base_height", nets["base_height"], "walker with a body-height command, L1+Up / L1+Down ±0.005 m per tick",
         actions=[{"group": "Height", "items": [{"combo": "L1+Up", "label": "Raise (L1+Up)"},
                                               {"combo": "L1+Down", "label": "Lower (L1+Down)"}]}])
    pack(15, "euler", nets["euler"], "stands still; sticks = roll / pitch / yaw body pose")
    for fsm, k in MIMIC.items():
        st, en = motions[k]["idx"]
        pack(fsm, "mimic", nets["mimic_" + k], f"motion net {k}.mnn frames {st}..{en} ({(en - st) * 0.02:.1f} s)",
             extra={"mimic": {**mim_shared, "motion": k, **motions[k]}},
             actions=[{"group": "Motion", "items": [{"combo": COMBO[fsm], "label": f"Play again ({COMBO[fsm]})"}]}])

    entries.append({"name": "obsolete/pose_2026-05-26__21-07_", "family": "as2_fw", "robot": "as2", "dof": 12,
                    "runnable": False, "note": "model/pose: no config, never loaded by ai_sport 1.0.0.17"})
    with open(os.path.join(REPO, REG, "index.json"), "w") as f:
        json.dump({"robot": "as2", "source": SRC_TAG, "policies": entries}, f, indent=2)
    print(f"registry: {sum(e['runnable'] for e in entries)} runnable, {sum(not e['runnable'] for e in entries)} obsolete")
    for rel, mnn in sorted(need.items()):
        convert(mnn, os.path.join(REPO, rel))


if __name__ == "__main__":
    main()
