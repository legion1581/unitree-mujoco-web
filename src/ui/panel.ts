/** Control panel — plain DOM, no framework. The app wires callbacks; the panel
 * owns only presentation. The slide-out policy-manager column (import / list /
 * delete packs) is part of this class too. */
import type { FsmState } from "../sim/fsm";
import { LABELS, STATES } from "../sim/fsm";
import type { PackInfo, PolicyInfo } from "../sim/protocol";
import { TERRAINS, type TerrainName } from "../sim/terrain";

export interface PanelCallbacks {
  onRobot(robot: string): void;
  onTerrain(t: TerrainName): void;
  onPolicy(name: string): void;
  onState(s: FsmState): void;
  onReset(): void;
  onRecenter(): void;
  onZeroCmd(): void;
  onScreenshot(): void;
  onRecordToggle(): void;
  onLoadCsv(file: File, fps: number): void;
  onStopPlayback(): void;
  onImportPack(file: File): void;
  onDeletePack(id: string): void;
}

export class Panel {
  private robotSel!: HTMLSelectElement;
  private terrainSel!: HTMLSelectElement;
  private policySel!: HTMLSelectElement;
  private stateBtns = new Map<FsmState, HTMLButtonElement>();
  private statusEl!: HTMLElement;
  private padDebugEl!: HTMLElement;
  private recBtn!: HTMLButtonElement;
  private pbBar!: HTMLElement;
  private pbName!: HTMLElement;
  private packListEl!: HTMLElement;
  private mgrEl!: HTMLElement;
  private mgrToggleBtn!: HTMLButtonElement;
  private packCount = 0;
  padDebugOpen: () => boolean = () => false;

  constructor(root: HTMLElement, mgrRoot: HTMLElement, robots: string[],
              readonly cb: PanelCallbacks) {
    root.innerHTML = "";
    this.mgrEl = mgrRoot;
    this.buildManager(mgrRoot);
    const h1 = document.createElement("h1");
    h1.textContent = "Unitree MuJoCo Web ";
    const paintLink = document.createElement("a");
    paintLink.href = "/painter.html";
    paintLink.target = "_blank";
    paintLink.textContent = "🎨";
    paintLink.title = "open the robot painter";
    paintLink.style.textDecoration = "none";
    h1.appendChild(paintLink);
    root.appendChild(h1);

    this.robotSel = this.select(root, "Robot", robots, (v) => cb.onRobot(v));
    this.terrainSel = this.select(root, "Terrain", TERRAINS, (v) => cb.onTerrain(v as TerrainName));
    this.policySel = this.select(root, "Policy", [], (v) => cb.onPolicy(v));

    // toggle for the slide-out policy manager (import/list/delete lives there)
    this.mgrToggleBtn = document.createElement("button");
    this.mgrToggleBtn.onclick = () => this.toggleManager();
    root.appendChild(this.mgrToggleBtn);
    this.updateMgrToggle();

    this.section(root, "FSM state");
    for (const s of STATES) {
      const b = document.createElement("button");
      b.textContent = LABELS[s];
      b.onclick = () => cb.onState(s);
      root.appendChild(b);
      this.stateBtns.set(s, b);
    }

    this.section(root, "Actions");
    const row = document.createElement("div");
    row.className = "row";
    for (const [label, fn] of [
      ["Reset (R)", cb.onReset], ["Recenter (C)", cb.onRecenter], ["Stop (Spc)", cb.onZeroCmd],
    ] as [string, () => void][]) {
      const b = document.createElement("button");
      b.textContent = label;
      b.onclick = fn;
      row.appendChild(b);
    }
    root.appendChild(row);

    const row2 = document.createElement("div");
    row2.className = "row";
    const shotBtn = document.createElement("button");
    shotBtn.textContent = "📷 Screenshot";
    shotBtn.onclick = cb.onScreenshot;
    row2.appendChild(shotBtn);
    this.recBtn = document.createElement("button");
    this.recBtn.textContent = "⏺ Record";
    this.recBtn.onclick = cb.onRecordToggle;
    row2.appendChild(this.recBtn);
    root.appendChild(row2);

    this.section(root, "Motion (mimic)");
    const inp = "background:#1a212b;color:#cdd6e0;border:1px solid #2b3542;" +
      "border-radius:6px;padding:6px 8px;font:inherit;box-sizing:border-box;";
    const mrow = document.createElement("div");
    mrow.className = "row";
    const fileBtn = document.createElement("button");
    fileBtn.textContent = "📁 Load reference CSV";
    fileBtn.style.flex = "1";
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = ".csv,.txt";
    fileInput.style.display = "none";
    const fpsInput = document.createElement("input");
    fpsInput.type = "number";
    fpsInput.value = "30";
    fpsInput.min = "1";
    fpsInput.title = "clip frame rate (fps)";
    fpsInput.style.cssText = inp + "width:62px;flex:0 0 auto;";
    fileBtn.onclick = () => fileInput.click();
    fileInput.onchange = () => {
      const f = fileInput.files?.[0];
      if (f) cb.onLoadCsv(f, Number(fpsInput.value) || 30);
      fileInput.value = "";     // allow re-loading the same file
    };
    mrow.append(fileBtn, fpsInput);
    root.append(mrow, fileInput);

    this.pbBar = document.createElement("div");
    this.pbBar.style.cssText =
      "display:none;align-items:center;gap:6px;margin-top:4px;" +
      "font-size:11.5px;color:#9fb0c0;";
    this.pbName = document.createElement("span");
    this.pbName.style.cssText = "flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
    const stopBtn = document.createElement("button");
    stopBtn.textContent = "⏹ Stop";
    stopBtn.style.cssText = "flex:0 0 auto;width:auto;padding:2px 10px;";
    stopBtn.onclick = cb.onStopPlayback;
    this.pbBar.append(this.pbName, stopBtn);
    root.appendChild(this.pbBar);

    const csvHint = document.createElement("div");
    csvHint.className = "hint";
    csvHint.textContent = "rows: pos(3) quat_xyzw(4) joints — 24 (R1) or 20 (Air). " +
      "Kinematic preview, no physics.";
    root.appendChild(csvHint);

    this.section(root, "Status");
    this.statusEl = document.createElement("div");
    this.statusEl.id = "status";
    root.appendChild(this.statusEl);

    const dbg = document.createElement("details");
    const sum = document.createElement("summary");
    sum.textContent = "Pad debug";
    sum.style.cssText = "cursor:pointer;color:#7d8794;font-size:11px;" +
      "text-transform:uppercase;letter-spacing:.08em;margin:14px 0 6px;";
    dbg.appendChild(sum);
    this.padDebugEl = document.createElement("div");
    this.padDebugEl.style.cssText = "white-space:pre;font-family:ui-monospace," +
      "monospace;font-size:11px;color:#9fb0c0;line-height:1.5;";
    dbg.appendChild(this.padDebugEl);
    root.appendChild(dbg);
    this.padDebugOpen = () => dbg.open;

    const help = document.createElement("div");
    help.innerHTML = `<h2>Keys</h2>
      <div style="color:#7d8794">
      <kbd>0</kbd>–<kbd>4</kbd> FSM state &nbsp; <kbd>[</kbd>/<kbd>]</kbd> policy<br>
      <kbd>W A S D</kbd>/<kbd>Q E</kbd> command &nbsp; <kbd>R</kbd> reset
      <kbd>C</kbd> recenter &nbsp; <kbd>Space</kbd> stop + re-zero pad<br>
      Gamepad: left stick vx/vy, right stick yaw.<br>
      LT+dpad-up Stance · LT+B Damping · RT+A Policy · RT+dpad ◀▶ switch policy
      </div>`;
    root.appendChild(help);
  }

  // ---- policy manager (slide-out column) -----------------------------
  private buildManager(root: HTMLElement) {
    root.innerHTML = "";
    const head = document.createElement("h2");
    const title = document.createElement("span");
    title.textContent = "Policy manager";
    const close = document.createElement("span");
    close.textContent = "✕";
    close.title = "close";
    close.style.cssText = "cursor:pointer;color:#7d8794;";
    close.onclick = () => this.toggleManager(false);
    head.append(title, close);
    root.appendChild(head);

    const importBtn = document.createElement("button");
    importBtn.className = "accent";
    importBtn.textContent = "📦 Import pack / bundle (.zip)";
    importBtn.title = "import a policy pack, or a whole per-model bundle (.zip of packs)";
    const zipInput = document.createElement("input");
    zipInput.type = "file";
    zipInput.accept = ".zip";
    zipInput.multiple = true;
    zipInput.style.display = "none";
    importBtn.onclick = () => zipInput.click();
    zipInput.onchange = () => {
      for (const f of Array.from(zipInput.files ?? [])) this.cb.onImportPack(f);
      zipInput.value = "";
    };
    root.append(importBtn, zipInput);

    const hint = document.createElement("div");
    hint.className = "hint";
    hint.textContent = "The app ships no policies — everything runs from packs " +
      "you import. They persist locally (IndexedDB) and appear in the Policy " +
      "list, grouped by obs scheme.";
    root.appendChild(hint);

    this.packListEl = document.createElement("div");
    this.packListEl.style.cssText = "margin-top:8px;";
    root.appendChild(this.packListEl);
  }

  private toggleManager(force?: boolean) {
    const open = force ?? this.mgrEl.classList.contains("hidden");
    this.mgrEl.classList.toggle("hidden", !open);
    this.updateMgrToggle();
  }

  private updateMgrToggle() {
    const open = !this.mgrEl.classList.contains("hidden");
    this.mgrToggleBtn.textContent = (open ? "▸ " : "📦 ") + "Policy manager" +
      (this.packCount ? `  (${this.packCount})` : "");
    this.mgrToggleBtn.classList.toggle("active", open);
  }

  // ---- shared helpers -------------------------------------------------
  private section(root: HTMLElement, title: string) {
    const h = document.createElement("h2");
    h.textContent = title;
    root.appendChild(h);
  }

  private select(root: HTMLElement, title: string, opts: string[],
                 onchange: (v: string) => void): HTMLSelectElement {
    this.section(root, title);
    const sel = document.createElement("select");
    for (const o of opts) sel.add(new Option(o, o));
    sel.onchange = () => onchange(sel.value);
    root.appendChild(sel);
    return sel;
  }

  // ---- state pushed from the app --------------------------------------
  setRobots(robots: string[], active: string) {
    this.robotSel.innerHTML = "";
    for (const r of robots) this.robotSel.add(new Option(r, r));
    this.robotSel.value = active;
  }

  setPolicies(entries: PolicyInfo[], active: string) {
    this.policySel.innerHTML = "";
    if (!entries.length) {
      const o = new Option("no policies — import a pack 📦", "");
      o.disabled = true;
      o.selected = true;
      this.policySel.add(o);
      return;
    }
    // group by obs-scheme label
    const byGroup = new Map<string, PolicyInfo[]>();
    for (const e of entries) {
      if (!byGroup.has(e.group)) byGroup.set(e.group, []);
      byGroup.get(e.group)!.push(e);
    }
    for (const [g, list] of byGroup) {
      const grp = document.createElement("optgroup");
      grp.label = g;
      for (const e of list) grp.appendChild(new Option(e.name, e.name));
      this.policySel.appendChild(grp);
    }
    this.policySel.value = active;
  }

  setPacks(packs: PackInfo[]) {
    this.packCount = packs.length;      // reflected on the manager toggle
    this.updateMgrToggle();

    this.packListEl.innerHTML = "";
    if (!packs.length) {
      const empty = document.createElement("div");
      empty.className = "hint";
      empty.textContent = "No imported packs yet.";
      this.packListEl.appendChild(empty);
      return;
    }
    // group cards by robot for scanability
    const byRobot = new Map<string, PackInfo[]>();
    for (const p of packs) {
      if (!byRobot.has(p.robot)) byRobot.set(p.robot, []);
      byRobot.get(p.robot)!.push(p);
    }
    for (const [robot, list] of [...byRobot].sort()) {
      const head = document.createElement("div");
      head.className = "pack-robot";
      head.textContent = `${robot} · ${list.length}`;
      this.packListEl.appendChild(head);
      for (const p of list.sort((a, b) => a.id.localeCompare(b.id))) {
        this.packListEl.appendChild(this.packCard(p));
      }
    }
  }

  private packCard(p: PackInfo): HTMLElement {
    const card = document.createElement("div");
    card.className = "pack-card";
    const info = document.createElement("div");
    info.className = "pack-info";
    const name = document.createElement("div");
    name.className = "pack-name";
    name.textContent = p.id;
    name.title = p.id;
    const meta = document.createElement("div");
    meta.className = "pack-meta";
    meta.textContent = p.group +
      (p.checkpoints.length > 1 ? ` · ${p.checkpoints.length} ckpts` : "");
    meta.title = p.checkpoints.join(", ");
    info.append(name, meta);
    const del = document.createElement("button");
    del.className = "pack-del";
    del.textContent = "🗑";
    del.title = `remove ${p.id}`;
    del.onclick = () => {
      const many = p.checkpoints.length > 1
        ? ` and its ${p.checkpoints.length} checkpoints (${p.checkpoints.join(", ")})` : "";
      if (confirm(`Remove "${p.id}"${many}?`)) this.cb.onDeletePack(p.id);
    };
    card.append(info, del);
    return card;
  }

  get terrain(): TerrainName { return this.terrainSel.value as TerrainName; }

  setTerrain(t: TerrainName) { this.terrainSel.value = t; }

  setState(s: FsmState) {
    for (const [k, b] of this.stateBtns) b.classList.toggle("active", k === s);
  }

  setStatus(text: string) { this.statusEl.textContent = text; }

  setPadDebug(text: string) { this.padDebugEl.textContent = text; }

  setRecording(on: boolean) {
    this.recBtn.textContent = on ? "⏹ Stop & save" : "⏺ Record";
    this.recBtn.classList.toggle("active", on);
  }

  setPlayback(name: string | null) {
    this.pbBar.style.display = name ? "flex" : "none";
    if (name) this.pbName.textContent = "▶ " + name;
  }
}
