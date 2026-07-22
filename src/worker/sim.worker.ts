/** Sim worker — owns MuJoCo wasm, onnxruntime, the FSM and the active driver.
 * See src/sim/protocol.ts for why this must run off the main thread. */
import { makeDriver, type BaseDriver } from "../core/drivers";
import { sampleTraj, wxyz } from "../core/math";
import type { PolicyEntry, Command, EvalConfig, RobotSpec } from "../core/types";
import { allPacks, deletePack, importAny, packEntries, resolvePackConfig,
         revokePack, type StoredPack } from "../packs";
import { Engine } from "../sim/engine";
import { SimFSM, type FsmState } from "../sim/fsm";
import { TERRAINS, terrainTransform, type TerrainName } from "../sim/terrain";
import { getSpec, SPECS } from "../robots";

/** Robot family: r1 + r1_air share one model (and one UI slot); go2 is its own. */
const fam = (r: string) => (r.startsWith("r1") ? "r1" : r);
import type { FromWorker, RenderModel, ToWorker } from "../sim/protocol";

/** Kinematic-playback state: a user-supplied reference clip is set into qpos
 * frame-by-frame (no physics), to preview retargeted mocap on the robot. */
interface Playback {
  rows: Float32Array; nRows: number; nCols: number;
  fps: number; njoints: number; name: string;
  bqp: number;                 // base qpos address
  jointAddr: Int32Array;       // qpos address per clip joint (model order)
  extraAddr: Int32Array;       // hinges not in the clip → held at 0
  t: number;                   // playback clock (s)
}

/** Parse a reference CSV: rows of [pos(3), quat_xyzw(4), joint_pos(n)].
 * Tolerates comma or whitespace delimiters and skips header/non-numeric rows. */
function parseTrajCsv(text: string):
    { rows: Float32Array; nRows: number; nCols: number } | null {
  const out: number[] = [];
  let nCols = 0;
  for (const line of text.split(/\r?\n/)) {
    const s = line.trim();
    if (!s) continue;
    const nums = s.split(/[,\s]+/).map(Number);
    if (nums.some((v) => !Number.isFinite(v))) continue;   // header / junk
    if (!nCols) nCols = nums.length;
    if (nums.length !== nCols) continue;
    for (const v of nums) out.push(v);
  }
  if (nCols < 8 || out.length < nCols) return null;
  return { rows: Float32Array.from(out), nRows: out.length / nCols, nCols };
}

const post = (m: FromWorker, transfer: Transferable[] = []) =>
  (self as unknown as Worker).postMessage(m, transfer);

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

class SimWorker {
  eng!: Engine;
  fsm!: SimFSM;
  driver: BaseDriver | null = null;

  robot = "r1";
  terrain: TerrainName = "flat";
  packs: StoredPack[] = [];            // imported policy packs — the ONLY policy source
  configs = new Map<string, EvalConfig>();
  policies: PolicyEntry[] = [];       // pack entries for the active robot
  polIdx = 0;
  command: Command = [0, 0, 0];
  busy = true;
  rtf = 0;
  playback: Playback | null = null;
  inspectOn = false;
  baseAddr = 0;                        // floating-base qpos addr (for driver-less frames)
  private lastInspect = 0;

  async init(m: { robot?: string; terrain?: TerrainName; policy?: string }) {
    post({ t: "busy", msg: "loading imported policy packs…" });
    this.packs = await allPacks().catch(() => []);
    if (m.terrain && (TERRAINS as readonly string[]).includes(m.terrain))
      this.terrain = m.terrain;
    const robots = [...new Set(this.allEntries().map((e) => e.robot))]
      .filter((r) => r in SPECS);
    this.robot = m.robot && m.robot in SPECS ? m.robot
      : robots.includes("r1") ? "r1" : (robots[0] ?? "r1");

    post({ t: "busy", msg: "compiling MuJoCo model (wasm)…" });
    await this.buildEngine();

    this.rebuildPolicies();
    if (this.policies.length) {
      const saved = m.policy ? this.policies.findIndex((e) => e.name === m.policy) : -1;
      await this.loadDriver(saved >= 0 ? saved : 0);
    } else this.driver = null;
    this.fsm.enter("policy");
    this.busy = false;
    post({ t: "ready" });
    this.sendStatus();
    void this.controlLoop();
  }

  /** Every imported pack entry, unfiltered. */
  private allEntries(): PolicyEntry[] {
    return this.packs.flatMap(packEntries);
  }

  /** Policy list for the active robot FAMILY (r1 covers r1_air — one model). */
  private rebuildPolicies() {
    this.policies = this.allEntries().filter((e) => fam(e.robot) === fam(this.robot));
  }

  private async buildEngine() {
    this.eng = await Engine.create(getSpec(this.robot).model_dir,
                                   terrainTransform(this.terrain));
    this.eng.timestep = getSpec(this.robot).sim_dt;
    this.baseAddr = this.eng.jointQposAdr(getSpec(this.robot).floating_base_joint);
    this.fsm = new SimFSM(this.eng);
    post({ t: "model", desc: this.renderModel() });
  }

  /** Copy the geom/mesh tables out of wasm memory for the renderer. */
  private renderModel(): RenderModel {
    const m = this.eng.model as any;
    const ngeom = this.eng.model.ngeom;
    const dataid = Int32Array.from(m.geom_dataid as Int32Array);
    const meshes: RenderModel["meshes"] = {};
    const vertadr: Int32Array = m.mesh_vertadr, vertnum: Int32Array = m.mesh_vertnum;
    const faceadr: Int32Array = m.mesh_faceadr, facenum: Int32Array = m.mesh_facenum;
    const vert: Float32Array = m.mesh_vert, face: Int32Array = m.mesh_face;
    for (let g = 0; g < ngeom; g++) {
      const mid = dataid[g];
      if (mid >= 0 && !(mid in meshes)) {
        const acc = m.mesh(mid);
        const name = acc.name as string;
        acc.delete?.();
        meshes[mid] = {
          name,
          vert: vert.slice(3 * vertadr[mid], 3 * (vertadr[mid] + vertnum[mid])),
          face: Uint32Array.from(
            face.subarray(3 * faceadr[mid], 3 * (faceadr[mid] + facenum[mid]))),
        };
      }
    }
    return {
      robot: this.robot,
      ngeom,
      type: Int32Array.from(m.geom_type as Int32Array),
      size: Float64Array.from(m.geom_size as Float64Array),
      rgba: Float32Array.from(m.geom_rgba as Float32Array),
      group: Int32Array.from(m.geom_group as Int32Array),
      contype: Int32Array.from(m.geom_contype as Int32Array),
      bodyid: Int32Array.from(m.geom_bodyid as Int32Array),
      dataid, meshes,
    };
  }

  private cfgFor(entry: PolicyEntry): EvalConfig {
    let c = this.configs.get(entry.name);
    if (!c) {
      const pack = this.packs.find((p) => p.id === entry.packId);
      if (!pack) throw new Error(`imported pack ${entry.packId} not found`);
      c = resolvePackConfig(pack, entry);
      this.configs.set(entry.name, c);
    }
    return c;
  }

  async loadDriver(idx: number) {
    this.busy = true;
    const entry = this.policies[idx];
    post({ t: "busy", msg: `loading ${entry.name}…` });
    // a family spans specs (r1 ↔ r1_air, same compiled model): adopt the
    // policy's own spec so joint subset / gains / fall height match
    if (entry.robot !== this.robot && entry.robot in SPECS &&
        getSpec(entry.robot).model_dir === getSpec(this.robot).model_dir)
      this.robot = entry.robot;
    try {
      const cfg = await this.cfgFor(entry);
      // launch the new policy where the robot currently stands, not at the
      // origin — Recenter (C) is the explicit way back to the middle
      const b = this.eng.jointQposAdr(getSpec(this.robot).floating_base_joint);
      const here: [number, number] = [this.eng.qpos[b], this.eng.qpos[b + 1]];
      this.driver = await makeDriver(cfg, getSpec(this.robot), this.eng);
      this.driver.homeXY = here;
      this.polIdx = idx;
      this.command = [0, 0, 0];
      this.driver.reset();
      post({ t: "ready" });
    } catch (e) {
      console.error(e);
      post({ t: "error", msg: `failed to load ${entry.name}: ${e}` });
    } finally {
      this.busy = false;
      this.sendStatus();
    }
  }

  /** Switch robot (family), recompiling the scene when the model differs, and
   * load its first policy (or idle-preview if it has none). Callers reenter(). */
  private async setRobot(robot: string) {
    const prevDir = getSpec(this.robot).model_dir;
    this.robot = robot;
    if (getSpec(robot).model_dir !== prevDir) {
      this.busy = true;
      post({ t: "busy", msg: "compiling MuJoCo model (wasm)…" });
      await this.buildEngine();
      this.busy = false;
    }
    this.rebuildPolicies();
    if (this.policies.length) await this.loadDriver(0);  // posts "ready"
    else { this.driver = null; this.polIdx = 0; post({ t: "ready" }); }  // idle preview
  }

  private reenter() {
    const st: FsmState = this.fsm.state === "policy" || this.fsm.state === "stance"
      ? this.fsm.state : "policy";
    this.fsm.enter(st);
    this.sendStatus();
  }

  /** Painter mode: compile the model and post its render description + one
   * frame of the default pose. No catalog, no policies, no control loop. */
  async initModel(robot?: string) {
    this.robot = robot ?? "r1";
    post({ t: "busy", msg: "compiling MuJoCo model (wasm)…" });
    await this.buildEngine();
    this.postFrame();
    post({ t: "ready" });
  }

  async onMessage(m: ToWorker) {
    switch (m.t) {
      case "init": return this.init(m);
      case "initModel": return this.initModel(m.robot);
      case "playback": return this.startPlayback(m.csv, m.fps, m.name);
      case "stopPlayback": return this.stopPlayback();
      case "importPack": return this.doImportPack(m.zip);
      case "deletePack": return this.doDeletePack(m.id);
      case "inspect": this.inspectOn = m.on; return;
      case "policy": {
        this.playback = null;
        const idx = this.policies.findIndex((e) => e.name === m.name);
        if (idx >= 0) { await this.loadDriver(idx); this.reenter(); }
        return;
      }
      case "cycle": {
        this.playback = null;
        if (!this.policies.length) return;
        const n = this.policies.length;
        await this.loadDriver((this.polIdx + m.step + n) % n);
        this.reenter();
        return;
      }
      case "state":
        this.playback = null;
        this.fsm.enter(m.state);
        return this.sendStatus();
      case "robot":
        this.playback = null;
        await this.setRobot(m.robot);
        return this.reenter();
      case "terrain":
        this.busy = true;
        this.terrain = m.terrain;
        post({ t: "busy", msg: `rebuilding scene (${m.terrain})…` });
        await this.buildEngine();
        if (this.playback) this.rebindPlayback();
        else if (this.policies.length) await this.loadDriver(this.polIdx);
        this.busy = false;
        return this.playback ? this.sendStatus() : this.reenter();
      case "command":
        this.command = m.cmd;
        return;
      case "reset":
        this.playback = null;
        this.driver?.reset();
        this.fsm.enter(this.fsm.state);
        return;
      case "recenter":
        if (this.driver) {
          this.driver.homeXY = [0, 0];   // recenter also re-homes future resets
          this.driver.recenter();
        }
        return;
    }
  }

  // ---- kinematic mocap playback --------------------------------------
  private specForJointCount(nj: number): RobotSpec | undefined {
    return Object.values(SPECS).find((s) => s.joint_names.length === nj);
  }

  /** Resolve base + joint qpos addresses for a spec against the live model. */
  private bindPlayback(spec: RobotSpec, base: Omit<Playback, "bqp" | "jointAddr" | "extraAddr">): Playback {
    const bqp = this.eng.jointQposAdr(spec.floating_base_joint);
    const jointAddr = Int32Array.from(spec.joint_names.map((n) => this.eng.jointQposAdr(n)));
    const driven = new Set(spec.joint_names);
    const h = this.eng.hinges();
    const extra: number[] = [];
    h.names.forEach((nm, i) => { if (!driven.has(nm)) extra.push(h.qadr[i]); });
    return { ...base, bqp, jointAddr, extraAddr: Int32Array.from(extra) };
  }

  private rebindPlayback() {
    if (!this.playback) return;
    const spec = this.specForJointCount(this.playback.njoints);
    if (spec) this.playback = this.bindPlayback(spec, this.playback);
  }

  async startPlayback(csv: string, fps: number, name: string) {
    const parsed = parseTrajCsv(csv);
    if (!parsed) {
      return post({ t: "error", msg: `${name}: could not parse a numeric CSV` });
    }
    const { nCols } = parsed;
    const njoints = nCols - 7;
    const spec = this.specForJointCount(njoints);
    if (!spec) {
      return post({ t: "error", msg:
        `${name}: ${nCols} columns → ${njoints} joints; expected ` +
        `[pos(3), quat_xyzw(4), joints] with ${Object.values(SPECS)
          .map((s) => s.joint_names.length).join(" or ")} joints` });
    }
    // ensure the compiled model matches the clip's robot (r1/r1_air share one)
    if (getSpec(this.robot).model_dir !== spec.model_dir) {
      this.busy = true;
      this.robot = spec.robot;
      post({ t: "busy", msg: "compiling MuJoCo model (wasm)…" });
      await this.buildEngine();
      this.busy = false;
    } else {
      this.robot = spec.robot;
    }
    this.playback = this.bindPlayback(spec,
      { ...parsed, fps: fps > 0 ? fps : 30, njoints, name, t: 0 });
    post({ t: "ready" });
    this.sendStatus();
  }

  private stopPlayback() {
    if (!this.playback) return;
    this.playback = null;
    this.driver?.reset();
    this.reenter();
  }

  // ---- policy pack import/delete --------------------------------------
  private async doImportPack(zip: ArrayBuffer) {
    try {
      const imported = await importAny(new Uint8Array(zip));   // single pack OR bundle
      const curName = this.policies[this.polIdx]?.name;
      for (const pack of imported) {
        this.packs = this.packs.filter((p) => p.id !== pack.id);   // replace by id
        this.packs.push(pack);
      }
      // if the current family has nothing to run, follow the import to its robot
      // (setRobot recompiles the scene if the model differs and loads policy 0)
      if (!this.allEntries().some((e) => fam(e.robot) === fam(this.robot)) &&
          imported[0].manifest.robot in SPECS) {
        await this.setRobot(imported[0].manifest.robot);
        return this.reenter();
      }
      this.rebuildPolicies();
      if (this.driver && curName) {
        // keep the running policy; entries may have reordered around it
        const i = this.policies.findIndex((e) => e.name === curName);
        if (i >= 0) this.polIdx = i;
        post({ t: "ready" });
        this.sendStatus();
      } else if (this.policies.length) {
        await this.loadDriver(0);    // first policy for this family just arrived
        this.reenter();
      } else {
        post({ t: "ready" });
        this.sendStatus();
      }
    } catch (e) {
      console.error(e);
      post({ t: "error", msg: `import failed: ${e instanceof Error ? e.message : e}` });
    }
  }

  private async doDeletePack(id: string) {
    await deletePack(id).catch(() => {});
    revokePack(id);
    const gone = new Set(this.packs.filter((p) => p.id === id).flatMap(packEntries).map((e) => e.name));
    for (const n of gone) this.configs.delete(n);
    this.packs = this.packs.filter((p) => p.id !== id);
    this.rebuildPolicies();
    // if the active policy vanished, fall back to the first available
    if (this.policies.length && !this.policies[this.polIdx]) {
      this.playback = null;
      await this.loadDriver(0);
      this.reenter();
    } else {
      this.sendStatus();
    }
  }

  private playbackStep() {
    const pb = this.playback!;
    const row = sampleTraj(pb.rows, pb.nRows, pb.nCols, pb.t, pb.fps);
    const q = this.eng.qpos;
    q[pb.bqp] = row[0]; q[pb.bqp + 1] = row[1]; q[pb.bqp + 2] = row[2];
    const wq = wxyz([row[3], row[4], row[5], row[6]]);
    q[pb.bqp + 3] = wq[0]; q[pb.bqp + 4] = wq[1]; q[pb.bqp + 5] = wq[2]; q[pb.bqp + 6] = wq[3];
    for (let i = 0; i < pb.njoints; i++) q[pb.jointAddr[i]] = row[7 + i];
    for (let i = 0; i < pb.extraAddr.length; i++) q[pb.extraAddr[i]] = 0;
    this.eng.qvel.fill(0);
    this.eng.forward();
    pb.t += 0.02;                          // 50 fps real-time playback
    if (pb.t >= pb.nRows / pb.fps) pb.t = 0;
  }

  private sendStatus() {
    const e = this.policies[this.polIdx];
    post({
      t: "status",
      // one selectable entry per FAMILY (r1 covers r1_air; the loaded policy
      // decides the exact spec)
      robots: [...new Set(Object.keys(SPECS).map(fam))],
      robot: fam(this.robot),
      policies: this.policies.map((p) => ({
        name: p.name, family: p.family, dof: p.dof,
        group: p.group, packId: p.packId })),
      packs: this.packs.map((p) => ({
        id: p.id, group: p.manifest.group, robot: p.manifest.robot,
        checkpoints: p.manifest.checkpoints.map((c) => c.id) })),
      policy: e?.name ?? "-",
      family: e?.family ?? "-",
      dof: e?.dof ?? 0,
      state: this.fsm.state,
      command: [...this.command] as Command,
      rtf: this.rtf,
      playback: this.playback?.name,
    });
  }

  /** Build + post the obs/action inspector snapshot (≤10 Hz, panel-gated). */
  private maybeInspect(now: number) {
    if (!this.inspectOn || now - this.lastInspect < 100) return;
    this.lastInspect = now;
    const drv = this.driver;
    if (this.playback || !drv) {
      post({ t: "inspect", policy: "—", family: "—",
             state: this.playback ? "kinematic playback" : "—",
             baseZ: 0, command: [0, 0, 0], drive: "—", onnxInputs: 1,
             stackDepth: 1, perFrame: 0, terms: [], actions: [],
             playback: !!this.playback });
      return;
    }
    const e = this.policies[this.polIdx];
    const cfg = this.configs.get(e?.name ?? "");
    const frame = drv.lastFrame;
    const perFrame = drv.frameLayout.reduce((s, [, d]) => s + d, 0);
    const terms: { name: string; dim: number; vals: number[] }[] = [];
    if (frame && frame.length >= perFrame && perFrame > 0) {
      let off = 0;
      for (const [name, dim] of drv.frameLayout) {
        terms.push({ name, dim, vals: Array.from(frame.subarray(off, off + Math.min(dim, 6))) });
        off += dim;
      }
    }
    const act = drv.lastAction;
    post({
      t: "inspect",
      policy: e?.name ?? "-",
      family: e?.family ?? "-",
      state: this.fsm.state,
      baseZ: this.eng.qpos[drv.bqp + 2],
      baseXY: [this.eng.qpos[drv.bqp], this.eng.qpos[drv.bqp + 1]],
      nObst: (drv as { nObst?: number }).nObst,
      command: [...this.command] as Command,
      drive: drv.clipDriven ? "clip" : drv.needsCommand ? "command" : "—",
      onnxInputs: (cfg?.policy?.onnx_inputs as number) ?? 1,
      stackDepth: drv.stackDepth,
      perFrame,
      terms,
      actions: drv.actJointNames.map((n, i) => ({ name: n, value: act?.[i] ?? 0 })),
      playback: false,
    });
  }

  private postFrame() {
    const xpos = Float32Array.from(this.eng.geomXpos);
    const xmat = Float32Array.from(this.eng.geomXmat);
    const b = this.playback?.bqp ?? this.driver?.bqp ?? this.baseAddr;
    post({ t: "frame", xpos, xmat,
           baseX: this.eng.qpos[b], baseY: this.eng.qpos[b + 1] },
         [xpos.buffer, xmat.buffer]);
  }

  private async controlLoop() {
    let lastStatus = 0;
    for (;;) {
      if (this.busy) { await sleep(50); continue; }
      if (this.playback) {
        const t0 = performance.now();
        this.playbackStep();
        this.postFrame();
        this.maybeInspect(t0);
        if (t0 - lastStatus > 400) { this.sendStatus(); lastStatus = t0; }
        await sleep(Math.max(20 - (performance.now() - t0), 0));
        continue;
      }
      if (!this.driver) {
        // robot selected but no policy loaded (e.g. go2 before importing a
        // pack): show it standing in its default pose, no physics stepping.
        const t0 = performance.now();
        this.postFrame();
        this.maybeInspect(t0);
        if (t0 - lastStatus > 400) { this.sendStatus(); lastStatus = t0; }
        await sleep(100);
        continue;
      }
      const cfg = this.configs.get(this.policies[this.polIdx]?.name);
      const stepDt = cfg?.control.step_dt ?? 0.02;
      const simDt = getSpec(this.robot).sim_dt;
      const nSub = Math.max(1, Math.round(stepDt / simDt));
      const t0 = performance.now();

      const cmd: Command = this.driver.needsCommand ? this.command : [0, 0, 0];
      try {
        const target = await this.fsm.infer(this.driver, cmd);
        for (let s = 0; s < nSub; s++) {
          this.fsm.substepTorque(this.driver, target);
          this.eng.step();
        }
      } catch (e) {
        console.error(e);
        post({ t: "error", msg: `sim step failed: ${e}` });
        this.fsm.enter("damping");
      }
      this.postFrame();
      this.maybeInspect(t0);
      const elapsed = performance.now() - t0;
      this.rtf = 0.9 * this.rtf + 0.1 * (stepDt * 1000 / Math.max(elapsed, 1e-3));
      if (t0 - lastStatus > 400) { this.sendStatus(); lastStatus = t0; }
      await sleep(Math.max(stepDt * 1000 - elapsed, 0));
    }
  }
}

const sim = new SimWorker();
// serialize message handling: handlers await (engine rebuilds, pack imports),
// and interleaving two of them corrupts robot/engine state — e.g. importing a
// bundle for another robot while a driver load is mid-flight
let chain: Promise<void> = Promise.resolve();
self.onmessage = (ev: MessageEvent<ToWorker>) => {
  chain = chain.then(() => sim.onMessage(ev.data)).catch((e) => {
    console.error(e);
    post({ t: "error", msg: String(e) });
  });
};
