/** Robot painter page — loads the model through the sim worker (model-only
 * mode), renders it at the default pose, and lets you color / hide / re-finish
 * each part. Overrides persist in localStorage; the sim applies them live. */
import { SceneView } from "./render/scene";
import type { FromWorker } from "./sim/protocol";
import { clearPaint, colorHex, loadPaint, partKey, partLabel, resolve,
         savePaint, type Finish, type PaintMap } from "./render/paint";

const view = new SceneView(document.getElementById("viewport")!);
(window as unknown as { __view: SceneView }).__view = view;   // debug/console hook
const overlay = document.getElementById("overlay")!;
const overlayMsg = document.getElementById("overlay-msg")!;
const partsEl = document.getElementById("parts")!;

let paint: PaintMap = loadPaint();
let frame: { xpos: Float32Array; xmat: Float32Array } | null = null;
let selected: string | null = null;

interface PartRow {
  key: string;
  meshName: string;               // representative mesh (for defaults)
  el: HTMLElement;
  colorIn: HTMLInputElement;
  hideIn: HTMLInputElement;
  finishSel: HTMLSelectElement;
}
let rows: PartRow[] = [];

const worker = new Worker(new URL("./worker/sim.worker.ts", import.meta.url),
                          { type: "module" });
worker.postMessage({ t: "initModel", robot: "r1" });
worker.onmessage = (ev: MessageEvent<FromWorker>) => {
  const m = ev.data;
  if (m.t === "model") {
    view.build(m.desc);
    buildRows();
  } else if (m.t === "frame") {
    frame = { xpos: m.xpos, xmat: m.xmat };
  } else if (m.t === "busy") {
    overlayMsg.textContent = m.msg;
    overlay.classList.remove("hidden");
  } else if (m.t === "ready") {
    overlay.classList.add("hidden");
  } else if (m.t === "error") {
    overlayMsg.textContent = m.msg;
    overlay.classList.remove("hidden");
  }
};

function apply() {
  savePaint(paint);
  view.applyPaint(paint);
}

function entry(key: string) {
  return (paint[key] ??= {});
}

function buildRows() {
  partsEl.innerHTML = "";
  rows = [];
  const byKey = new Map<string, string>();
  for (const name of view.partMeshNames()) {
    const k = partKey(name);
    if (!byKey.has(k)) byKey.set(k, name);
  }
  const keys = [...byKey.keys()].sort((a, b) => partLabel(a).localeCompare(partLabel(b)));
  for (const key of keys) {
    const meshName = byKey.get(key)!;
    const e = paint[key] ?? {};
    const d = resolve(meshName, {});          // baked default appearance
    const el = document.createElement("div");
    el.className = "part";
    el.dataset.key = key;

    const colorIn = document.createElement("input");
    colorIn.type = "color";
    colorIn.value = e.color ?? colorHex(d.color);
    colorIn.oninput = () => { entry(key).color = colorIn.value; apply(); };

    const name = document.createElement("span");
    name.className = "name";
    name.textContent = partLabel(key);
    name.onclick = () => select(key);

    const finishSel = document.createElement("select");
    for (const f of ["glossy", "matte"] as Finish[]) finishSel.add(new Option(f, f));
    finishSel.value = e.finish ?? d.finish;
    finishSel.onchange = () => { entry(key).finish = finishSel.value as Finish; apply(); };

    const hideLbl = document.createElement("label");
    hideLbl.title = "hidden";
    const hideIn = document.createElement("input");
    hideIn.type = "checkbox";
    hideIn.checked = e.hidden ?? d.hidden;
    hideIn.onchange = () => { entry(key).hidden = hideIn.checked; apply(); };
    hideLbl.append(hideIn, document.createTextNode("hide"));

    el.append(colorIn, name, finishSel, hideLbl);
    partsEl.appendChild(el);
    rows.push({ key, meshName, el, colorIn, hideIn, finishSel });
  }
}

function select(key: string | null) {
  selected = key;
  for (const r of rows) {
    r.el.classList.toggle("selected", r.key === key);
    if (r.key === key) r.el.scrollIntoView({ block: "nearest" });
  }
}

// click the robot to select its part (drag = orbit, so only fire on short clicks)
let downAt: [number, number] | null = null;
view.renderer.domElement.addEventListener("pointerdown", (ev) => {
  downAt = [ev.clientX, ev.clientY];
});
view.renderer.domElement.addEventListener("pointerup", (ev) => {
  if (!downAt) return;
  const moved = Math.hypot(ev.clientX - downAt[0], ev.clientY - downAt[1]);
  downAt = null;
  if (moved > 5) return;
  const meshName = view.pickPart(ev);
  select(meshName ? partKey(meshName) : null);
});

document.getElementById("reset")!.onclick = () => {
  clearPaint();
  paint = {};
  view.applyPaint(paint);
  buildRows();
  select(selected);
};

document.getElementById("copy")!.onclick = () => {
  void navigator.clipboard.writeText(JSON.stringify(paint, null, 2));
};

function tick() {
  if (frame) { view.sync(frame.xpos, frame.xmat); frame = null; }
  view.render();
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
