#!/usr/bin/env node
/** Build every runnable policy of a robot into individual packs, then wrap the
 * whole folder into ONE per-model bundle you can hand to a file host.
 *
 *   node scripts/build-packs.mjs [--robot g1|r1|go2|as2] [--public <dir>] [--no-build] [--no-bundle]
 *
 *   --robot  <name>   registry to read           (default: r1)
 *   --public <dir>    checkout holding registry/<robot>/ + assets/ (default: this repo; the
 *                     registries are written by scripts/import_*.py)
 *   --out    <dir>    output root                (default: policy-packs)
 *   --no-build        skip per-policy packs, only (re)bundle the folder
 *   --no-bundle       build packs but skip the per-model bundle
 *
 * Per-policy packs → <out>/<robot>/<flatName>.zip  (one checkpoint each; the
 * group label is the policy's top-level directory, so the UI clusters them).
 * Bundle          → <out>/<robot>-all.zip  (a .zip of those packs; the app's
 * import understands both a single pack and a bundle).
 *
 * The web runtime is ONNX-only: any .mnn policy is swapped for its .onnx
 * sibling (written by the importers via scripts/mnn_graph_to_onnx.py).
 */
import { readFileSync, writeFileSync, mkdirSync, existsSync, readdirSync, unlinkSync } from "node:fs";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { zipSync, strToU8 } from "fflate";
import { load as yamlLoad } from "js-yaml";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");

function parseArgs(argv) {
  const o = { public: ".", out: "policy-packs", robot: "r1",
              build: true, bundle: true };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--robot") o.robot = argv[++i];
    else if (a === "--public") o.public = argv[++i];
    else if (a === "--out") o.out = argv[++i];
    else if (a === "--no-build") o.build = false;
    else if (a === "--no-bundle") o.bundle = false;
    else { console.error(`unknown arg: ${a}`); process.exit(1); }
  }
  return o;
}

const opt = parseArgs(process.argv.slice(2));
const pub = opt.public.startsWith("/") ? opt.public : join(ROOT, opt.public);
const outRoot = resolve(ROOT, opt.out);      // absolute --out stays absolute
const outDir = join(outRoot, opt.robot);
const flat = (name) => name.replace(/[\/\\]/g, "_");

// prefer the .onnx sibling of any .mnn policy — the web build needs ONNX
function onnxOf(p) {
  if (p.endsWith(".onnx")) return p;
  const sib = p.replace(/\.mnn$/, ".onnx");
  if (p.endsWith(".mnn") && existsSync(join(pub, sib))) return sib;
  throw new Error(`no .onnx for ${p}`);
}

function buildPack(entry) {
  const cfg = yamlLoad(readFileSync(join(pub, entry.config), "utf8"));
  const dof = cfg.policy?.num_act ?? entry.dof ?? 0;
  const id = entry.name.split("/").pop();
  const group = entry.name.includes("/")
    ? entry.name.split("/")[0] : `${entry.family} · ${dof}dof`;
  const files = {};
  files["config.yaml"] = new Uint8Array(readFileSync(join(pub, entry.config)));
  const ckptFile = `checkpoints/${id}.onnx`;
  files[ckptFile] = new Uint8Array(readFileSync(join(pub, onnxOf(entry.policy))));
  const manifest = {
    schemaVersion: 1, group, robot: entry.robot, family: entry.family,
    baseName: entry.name, config: "config.yaml",
    checkpoints: [{ id, file: ckptFile }],
  };
  // extra networks the drivers run (G1 Cpy* stance / run, R1 turn / run / move nets, Go2 planner nets, AS2 walk /
  // recovery nets …)
  for (const [role, rel] of Object.entries({ ...(cfg.g1?.aux_models ?? {}), ...(cfg.r1?.aux_models ?? {}),
                                              ...(cfg.go2?.aux_models ?? {}), ...(cfg.as2?.aux_models ?? {}) })) {
    const f = `aux/${role}.onnx`;
    files[f] = new Uint8Array(readFileSync(join(pub, onnxOf(rel))));
    (manifest.aux ??= {})[role] = f;
  }
  // G1 / R1 drivers' extra data files (latent / motion csv, tables), copied as-is
  for (const [role, rel] of Object.entries({ ...(cfg.g1?.aux_files ?? {}), ...(cfg.r1?.aux_files ?? {}) })) {
    const f = `aux/${role}${rel.slice(rel.lastIndexOf("."))}`;
    files[f] = new Uint8Array(readFileSync(join(pub, rel)));
    (manifest.auxFiles ??= {})[role] = f;
  }
  // G1 mimic family: bundle motion_clip.csv
  const clipRel = cfg.g1?.motion_clip;
  if (clipRel && existsSync(join(pub, clipRel))) {
    files["motion_clip.csv"] = new Uint8Array(readFileSync(join(pub, clipRel)));
    manifest.clip = "motion_clip.csv";
  }
  files["pack.json"] = strToU8(JSON.stringify(manifest, null, 2));
  return zipSync(files, { level: 6 });
}

// ---- build per-policy packs -----------------------------------------
if (opt.build) {
  const catalogPath = join(pub, `registry/${opt.robot}/index.json`);
  if (!existsSync(catalogPath)) {
    console.error(`no registry at ${catalogPath} — run scripts/import_*.py first, or pass --public <checkout>`);
    process.exit(1);
  }
  const catalog = JSON.parse(readFileSync(catalogPath, "utf8")).policies
    .filter((p) => p.runnable);
  mkdirSync(outDir, { recursive: true });
  let ok = 0, skip = 0;
  const built = new Set();
  for (const entry of catalog) {
    try {
      const zip = buildPack(entry);
      writeFileSync(join(outDir, flat(entry.name) + ".zip"), zip);
      built.add(flat(entry.name) + ".zip");
      ok++;
    } catch (e) {
      console.warn(`  ⚠ skip ${entry.name}: ${e.message}`);
      skip++;
    }
  }
  // drop zips of policies that are no longer runnable (else the bundle keeps shipping them)
  const stale = readdirSync(outDir).filter((f) => f.endsWith(".zip") && !built.has(f));
  for (const f of stale) unlinkSync(join(outDir, f));
  console.log(`✓ built ${ok} packs → ${opt.out}/${opt.robot}/` +
    (skip ? `  (${skip} skipped)` : "") + (stale.length ? `  (${stale.length} stale removed)` : ""));
}

// ---- bundle the folder into one per-model zip -----------------------
if (opt.bundle) {
  if (!existsSync(outDir)) {
    console.error(`nothing to bundle: ${outDir} does not exist`);
    process.exit(1);
  }
  const packs = readdirSync(outDir).filter((f) => f.endsWith(".zip"));
  if (!packs.length) { console.error(`no packs in ${outDir}`); process.exit(1); }
  const bundle = {};
  for (const f of packs) bundle[f] = new Uint8Array(readFileSync(join(outDir, f)));
  // stored (level 0): inner zips are already deflated, so re-deflating wastes time
  const zip = zipSync(bundle, { level: 0 });
  const bundlePath = join(outRoot, `${opt.robot}-all.zip`);
  writeFileSync(bundlePath, zip);
  const mb = (zip.length / 1024 / 1024).toFixed(1);
  console.log(`✓ bundle ${packs.length} packs → ${opt.out}/${opt.robot}-all.zip (${mb} MB)`);
}
