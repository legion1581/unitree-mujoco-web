/** UI shell — rendering, panel, keyboard and gamepad. All simulation runs in
 * the sim worker (see src/sim/protocol.ts for why); this side only exchanges
 * messages: commands in, model description + frames + status out. */
import type { Command } from "./core/types";
import { GamepadInput, type ComboAction } from "./input/gamepad";
import { loadPaint } from "./render/paint";
import { CanvasRecorder, screenshot } from "./render/recorder";
import { SceneView } from "./render/scene";
import type { FromWorker, InspectMsg, StatusMsg, ToWorker } from "./sim/protocol";
import type { FsmState } from "./sim/fsm";
import { TERRAINS, type TerrainName } from "./sim/terrain";
import { Panel } from "./ui/panel";

/** Robot family: r1 + r1_air share one model/UI slot; go2 is its own. */
const fam = (r: string) => (r.startsWith("r1") ? "r1" : r);

/** Session prefs (robot / terrain / policy) persisted across reloads. */
const PREFS_KEY = "ume-prefs";
interface Prefs { robot?: string; terrain?: TerrainName; policy?: string }
function loadPrefs(): Prefs {
  try { return JSON.parse(localStorage.getItem(PREFS_KEY) ?? "{}") as Prefs; }
  catch { return {}; }
}

/** Render the live obs/action inspector, mirroring the desktop tool's panel. */
function formatInspect(m: InspectMsg): string {
  if (m.playback) return "— kinematic playback (no policy obs) —";
  const vals = (v: number[], dim: number) => {
    const shown = v.slice(0, 4);
    return shown.map((x) => (x >= 0 ? "+" : "") + x.toFixed(2)).join(" ") +
      (dim > shown.length ? " …" : "");
  };
  const pad = (s: string, n: number) => s.padEnd(n).slice(0, n);
  const bar = (x: number, w = 10) => {
    const n = Math.round(Math.min(Math.abs(x) / 1.5, 1) * w);
    return x >= 0 ? "│" + "█".repeat(n) : " ".repeat(w - n) + "█".repeat(n) + "│";
  };
  const [vx, vy, wz] = m.command;
  const topo = m.onnxInputs === 1 ? "1 tensor" : `${m.onnxInputs} tensors`;
  const L: string[] = [];
  L.push(`${pad(m.policy, 30)} ${m.family}`);
  const xy = m.baseXY ? `   xy ${m.baseXY[0].toFixed(2)},${m.baseXY[1].toFixed(2)}` : "";
  L.push(`state ${m.state}   base_z ${m.baseZ.toFixed(2)}${xy}   drive ${m.drive}`);
  if (m.nObst !== undefined) L.push(`lidar ${m.nObst} occupied cells`);
  if (m.drive === "command")
    L.push(`cmd   vx ${vx.toFixed(2)} vy ${vy.toFixed(2)} wz ${wz.toFixed(2)}`);
  L.push("");
  L.push(`OBSERVATION  ${m.perFrame}/frame × ${m.stackDepth} hist, ${topo}`);
  if (m.terms.length)
    for (const t of m.terms)
      L.push(`  ${pad(t.name, 16)} [${String(t.dim).padStart(3)}] ${vals(t.vals, t.dim)}`);
  else L.push("  (warming up…)");
  L.push("");
  L.push(`ACTION → JOINT  (${m.actions.length} driven)`);
  const top = [...m.actions].sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 8);
  if (top.length)
    for (const a of top)
      L.push(`  ${pad(a.name.replace(/_joint$/, ""), 18)} ${(a.value >= 0 ? "+" : "") +
        a.value.toFixed(2)} ${bar(a.value)}`);
  else L.push("  (no action in this state)");
  return L.join("\n");
}

class App {
  view = new SceneView(document.getElementById("viewport")!);
  panel: Panel;
  pad = new GamepadInput();
  recorder = new CanvasRecorder();
  worker = new Worker(new URL("./worker/sim.worker.ts", import.meta.url),
                      { type: "module" });

  command: Command = [0, 0, 0];
  needsCommand = true;      // refreshed from status messages (family-dependent)
  private lastFrame: { xpos: Float32Array; xmat: Float32Array } | null = null;
  private baseXY: [number, number] = [0, 0];
  private status: StatusMsg | null = null;

  private overlay = document.getElementById("overlay")!;
  private overlayMsg = document.getElementById("overlay-msg")!;
  private inspectorBody = document.getElementById("inspector-body")!;
  private savedPrefs = "";
  private followZ = 0.7;    // camera look-at height, robot-size dependent

  constructor() {
    this.panel = new Panel(document.getElementById("panel")!,
                           document.getElementById("policy-manager")!, [], {
      onRobot: (r) => this.send({ t: "robot", robot: r }),
      onTerrain: (t) => this.send({ t: "terrain", terrain: t }),
      onPolicy: (name) => this.send({ t: "policy", name }),
      onState: (s) => this.setState(s),
      onReset: () => this.send({ t: "reset" }),
      onRecenter: () => this.send({ t: "recenter" }),
      onZeroCmd: () => this.stopAndRearm(),
      onScreenshot: () => screenshot(this.view.renderer.domElement),
      onRecordToggle: () => this.toggleRecording(),
      onLoadCsv: (file, fps) => this.loadCsv(file, fps),
      onStopPlayback: () => this.send({ t: "stopPlayback" }),
      onImportPack: (file) => this.importPack(file),
      onDeletePack: (id) => this.send({ t: "deletePack", id }),
    });
    this.worker.onmessage = (ev: MessageEvent<FromWorker>) => this.onWorker(ev.data);
    this.worker.onerror = (e) =>
      this.showOverlay(`sim worker crashed: ${e.message ?? e}`);
    window.addEventListener("keydown", (e) => this.onKey(e));
    // repaint live when the painter tab saves (storage fires cross-tab)
    window.addEventListener("storage", (e) => {
      if (e.key === "r1-paint") this.view.applyPaint(loadPaint());
    });
    // restore the previous session's robot / terrain / policy
    const prefs = loadPrefs();
    if (prefs.terrain && (TERRAINS as readonly string[]).includes(prefs.terrain))
      this.panel.setTerrain(prefs.terrain);
    this.send({ t: "init", robot: prefs.robot,
                terrain: this.panel.terrain, policy: prefs.policy });
    this.send({ t: "inspect", on: true });   // left column always streams
    this.renderLoop();
  }

  private send(m: ToWorker) { this.worker.postMessage(m); }

  private showOverlay(msg: string) {
    this.overlayMsg.textContent = msg;
    this.overlay.classList.remove("hidden");
  }

  private setState(s: FsmState) { this.send({ t: "state", state: s }); }

  private setCommand(cmd: Command) {
    this.command = cmd;
    this.send({ t: "command", cmd });
  }

  /** Space / Stop: zero the command AND re-sample the pad's rest position, so
   * a drifting or mis-neutraled pad can always be recovered hands-off. */
  private stopAndRearm() {
    this.pad.rearm();
    this.setCommand([0, 0, 0]);
  }

  private async loadCsv(file: File, fps: number) {
    try {
      const csv = await file.text();
      this.send({ t: "playback", csv, fps, name: file.name });
    } catch (e) {
      this.showOverlay(`failed to read ${file.name}: ${e}`);
    }
  }

  private importChain: Promise<void> = Promise.resolve();

  /** Reads are chained so multi-file imports reach the worker in the order
   * the user picked them (concurrent file reads finish smallest-first). */
  private importPack(file: File) {
    this.importChain = this.importChain.then(async () => {
      try {
        const zip = await file.arrayBuffer();
        this.worker.postMessage({ t: "importPack", zip }, [zip]);
      } catch (e) {
        this.showOverlay(`failed to read ${file.name}: ${e}`);
      }
    });
  }

  private toggleRecording() {
    if (this.recorder.recording) {
      void this.recorder.stop().then(() => this.panel.setRecording(false));
    } else {
      this.recorder.start(this.view.renderer.domElement);
      this.panel.setRecording(true);
    }
  }

  private nudge(i: 0 | 1 | 2, d: number) {
    const c: Command = [...this.command];
    c[i] += d;
    this.setCommand(c);
  }

  // ---- worker messages ------------------------------------------------
  private onWorker(m: FromWorker) {
    switch (m.t) {
      case "model":
        this.view.build(m.desc);
        this.lastFrame = null;
        // aim the camera at torso height: ~0.7 m for humanoids, lower for
        // the quadruped so its body doesn't sit at the bottom of the frame
        this.followZ = m.desc.robot.startsWith("go2") ? 0.35 : 0.7;
        return;
      case "frame":
        this.lastFrame = { xpos: m.xpos, xmat: m.xmat };
        this.baseXY = [m.baseX, m.baseY];
        return;
      case "status":
        this.applyStatus(m);
        return;
      case "inspect":
        this.inspectorBody.textContent = formatInspect(m);
        return;
      case "busy": return this.showOverlay(m.msg);
      case "ready": return this.overlay.classList.add("hidden");
      case "error": return this.showOverlay(m.msg);
    }
  }

  private updatePadDebug() {
    const s = this.pad.debugState();
    if (!s) { this.panel.setPadDebug("no gamepad (press any pad button)"); return; }
    const AXN = ["LX", "LY", "RX", "RY"];
    const BTN = ["A", "B", "X", "Y", "LB", "RB", "LT", "RT", "SEL", "STA",
                 "L3", "R3", "▲", "▼", "◀", "▶", "HOME"];
    const lines = [
      `${s.name.slice(0, 36)}`,
      `mapping ${s.mapping}   ${s.armed ? "ARMED" : "arming — release sticks"}`,
      ...s.axes.map((a, i) =>
        `ax${i} ${(AXN[i] ?? "  ").padEnd(2)} raw ${a.raw.toFixed(2).padStart(6)}` +
        `  n ${a.neutral.toFixed(2).padStart(6)}  → ${a.out.toFixed(2).padStart(6)}` +
        (Math.abs(a.out) > 0 ? "  ●" : "")),
      s.buttons.map((b, i) =>
        `${BTN[i] ?? i}${b.pressed ? "■" : b.value > 0.05 ? b.value.toFixed(1) : "·"}`)
        .join(" "),
    ];
    this.panel.setPadDebug(lines.join("\n"));
  }

  private applyStatus(s: StatusMsg) {
    const prev = this.status;
    this.status = s;
    if (!prev || prev.robots.join() !== s.robots.join() || prev.robot !== s.robot)
      this.panel.setRobots(s.robots, s.robot);
    const polKey = (m: StatusMsg) => m.policies.map((p) => p.name).join("|");
    if (!prev || prev.robot !== s.robot ||
        polKey(prev) !== polKey(s) || prev.policy !== s.policy)
      this.panel.setPolicies(s.policies, s.policy);
    // the manager lists only the ACTIVE family's packs (r1+r1_air | go2)
    const famPacks = s.packs.filter((p) => fam(p.robot) === s.robot);
    if (!prev || prev.robot !== s.robot ||
        prev.packs.map((p) => p.id).join("|") !== s.packs.map((p) => p.id).join("|"))
      this.panel.setPacks(famPacks);
    // persist session choices (robot / terrain / active policy)
    const pj = JSON.stringify({ robot: s.robot, terrain: this.panel.terrain,
      policy: s.policy !== "-" ? s.policy : undefined } satisfies Prefs);
    if (pj !== this.savedPrefs) { this.savedPrefs = pj; localStorage.setItem(PREFS_KEY, pj); }
    this.panel.setState(s.state);
    this.panel.setPlayback(s.playback ?? null);
    this.needsCommand = !s.playback && (s.family === "locomotion" || s.family === "mimic");
    const [vx, vy, wz] = s.command;
    const pad = this.pad.connected
      ? this.pad.name.slice(0, 24) + (this.pad.armed ? "" : " (release sticks to arm)")
      : "none (keys: WASD/QE)";
    this.panel.setStatus(
      (s.playback ? `PLAYBACK ${s.playback}\n` : "") +
      `policy  ${s.policy}\n` +
      `family  ${s.family}   dof ${s.dof}\n` +
      `state   ${s.playback ? "kinematic playback" : s.state}\n` +
      `cmd     vx ${vx.toFixed(2)}  vy ${vy.toFixed(2)}  wz ${wz.toFixed(2)}\n` +
      `load    ${Math.min(100 / Math.max(s.rtf, 0.01), 999).toFixed(0)}% of realtime budget\n` +
      `pad     ${pad}`);
  }

  // ---- input ----------------------------------------------------------
  private onKey(e: KeyboardEvent) {
    if (e.target instanceof HTMLSelectElement) return;
    const states: FsmState[] = ["zero_torque", "damping", "lock", "stance", "policy"];
    if (e.key >= "0" && e.key <= "4") return this.setState(states[Number(e.key)]);
    switch (e.key) {
      case "]": return this.send({ t: "cycle", step: 1 });
      case "[": return this.send({ t: "cycle", step: -1 });
      case "r": case "R": return this.send({ t: "reset" });
      case "c": case "C": return this.send({ t: "recenter" });
      case " ": e.preventDefault(); return this.stopAndRearm();
      case "w": case "ArrowUp": return this.nudge(0, 0.1);
      case "s": case "ArrowDown": return this.nudge(0, -0.1);
      case "a": case "ArrowLeft": return this.nudge(2, 0.1);
      case "d": case "ArrowRight": return this.nudge(2, -0.1);
      case "q": return this.nudge(1, 0.1);
      case "e": return this.nudge(1, -0.1);
    }
  }

  private onPadAction(a: ComboAction) {
    if (a.kind === "state") this.setState(a.state);
    else if (a.kind === "cycle") this.send({ t: "cycle", step: a.step });
    else if (a.kind === "reset") this.send({ t: "reset" });
  }

  // ---- render loop ----------------------------------------------------
  private renderLoop() {
    let lastPadCmd = "";
    let padWasActive = false;
    const tick = () => {
      const padCmd = this.pad.poll((a) => this.onPadAction(a));
      if (padCmd) {
        padWasActive = true;
        const key = padCmd.map((v) => v.toFixed(2)).join(",");
        if (key !== lastPadCmd) { lastPadCmd = key; this.setCommand(padCmd); }
      } else if (padWasActive) {
        // pad vanished or disarmed mid-drive — don't leave a stale command
        padWasActive = false;
        lastPadCmd = "";
        this.setCommand([0, 0, 0]);
      }
      if (this.lastFrame) {
        this.view.follow(this.baseXY[0], this.baseXY[1], this.followZ);
        this.view.sync(this.lastFrame.xpos, this.lastFrame.xmat);
      }
      this.view.render();
      if (this.panel.padDebugOpen()) this.updatePadDebug();
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
}

new App();
