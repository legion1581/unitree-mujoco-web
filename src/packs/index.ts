/** Policy-pack import + store (runs in the sim worker).
 *
 * A "policy pack" is a .zip a user imports; the repo ships none (weights are the
 * user's own — see README). Layout inside the zip:
 *
 *     pack.json                     manifest (below)
 *     config.yaml                   the policy config (obs scheme, gains, family)
 *     checkpoints/<id>.onnx …       one or more weight checkpoints (same obs)
 *     aux/<role>.onnx|csv|…         optional extra networks / data files (manifest aux / auxFiles)
 *     motion_clip.csv               optional reference clip (G1 mimic)
 *
 * Each checkpoint becomes one selectable policy; all checkpoints in a pack
 * share `config.yaml` (only the weights differ). `group` is the obs-scheme
 * label the UI groups by. Imported packs persist in IndexedDB and resolve to
 * blob: URLs so the existing driver/fetch code loads them unchanged.
 */
import { load as yamlLoad } from "js-yaml";
import { unzipSync, strFromU8 } from "fflate";
import type { EvalConfig, PolicyEntry } from "../core/types";

export interface PackManifest {
  schemaVersion: number;
  group: string;                 // obs-scheme label to group by in the UI
  robot: string;                 // g1 | r1 | r1_air | go2 | as2
  family: string;                // g1_* | r1_fw | go2_fw | as2_fw
  baseName: string;              // e.g. "cpy_walk_run_29dofs/actor"
  config: string;                // path in zip to the config yaml
  clip?: string;                 // path in zip to reference motion clip (G1 mimic)
  aux?: Record<string, string>;  // extra networks by role (G1 stance / run, R1 / Go2 aux nets) -> path in zip
  auxFiles?: Record<string, string>;  // extra data files (csv / tables) by role -> path in zip
  checkpoints: { id: string; file: string }[];
}

export interface StoredPack {
  id: string;                    // unique pack id (baseName)
  manifest: PackManifest;
  config: EvalConfig;            // parsed config.yaml
  files: Record<string, Uint8Array>;   // in-zip path -> bytes (onnx, csv, tables)
}

/** Metadata sent to the UI (no bytes). */
export interface PackInfo {
  id: string;
  group: string;
  robot: string;
  checkpoints: string[];         // checkpoint ids
}

const DB = "r1-policy-packs";
const STORE = "packs";

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE, { keyPath: "id" });
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function tx<T>(mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>):
    Promise<T> {
  return openDb().then((db) => new Promise<T>((resolve, reject) => {
    const req = fn(db.transaction(STORE, mode).objectStore(STORE));
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  }));
}

/** Parse + validate a .zip into a StoredPack (does not persist). */
export function parsePack(zip: Uint8Array): StoredPack {
  const files = unzipSync(zip);
  const manRaw = files["pack.json"];
  if (!manRaw) throw new Error("pack.json missing from zip");
  const manifest = JSON.parse(strFromU8(manRaw)) as PackManifest;
  for (const k of ["group", "robot", "family", "baseName", "config", "checkpoints"] as const)
    if (manifest[k] == null) throw new Error(`pack.json missing "${k}"`);
  const cfgRaw = files[manifest.config];
  if (!cfgRaw) throw new Error(`config "${manifest.config}" missing from zip`);
  const config = yamlLoad(strFromU8(cfgRaw)) as EvalConfig;
  if (!manifest.checkpoints.length) throw new Error("pack has no checkpoints");
  const kept: Record<string, Uint8Array> = {};
  for (const c of manifest.checkpoints) {
    if (!files[c.file]) throw new Error(`checkpoint "${c.file}" missing from zip`);
    kept[c.file] = files[c.file];
  }
  // networks and data files may share a role name, so keep both maps' files independently
  for (const [role, file] of [...Object.entries(manifest.aux ?? {}), ...Object.entries(manifest.auxFiles ?? {})]) {
    if (!files[file]) throw new Error(`aux "${role}" (${file}) missing from zip`);
    kept[file] = files[file];
  }
  if (manifest.clip) {
    if (!files[manifest.clip]) throw new Error(`clip "${manifest.clip}" missing from zip`);
    kept[manifest.clip] = files[manifest.clip];
  }
  return { id: manifest.baseName, manifest, config, files: kept };
}

export async function importPack(zip: Uint8Array): Promise<StoredPack> {
  const pack = parsePack(zip);
  await tx("readwrite", (s) => s.put(pack));
  return pack;
}

/** Import a .zip that is EITHER a single pack (has pack.json) OR a per-model
 * bundle (a .zip whose entries are themselves pack .zips — see
 * the pack build tooling). Persists every pack found and returns them. */
export async function importAny(zip: Uint8Array): Promise<StoredPack[]> {
  const files = unzipSync(zip);
  if (files["pack.json"]) return [await importPack(zip)];
  const inner = Object.keys(files).filter((n) => n.endsWith(".zip"));
  if (!inner.length)
    throw new Error("not a policy pack (no pack.json) or bundle (no inner .zip)");
  const out: StoredPack[] = [];
  for (const name of inner) {
    try {
      out.push(await importPack(files[name]));
    } catch (e) {
      console.warn(`bundle: skipping ${name}: ${e instanceof Error ? e.message : e}`);
    }
  }
  if (!out.length) throw new Error("bundle contained no valid packs");
  return out;
}

export async function allPacks(): Promise<StoredPack[]> {
  return (await tx<StoredPack[]>("readonly", (s) => s.getAll())) ?? [];
}

export async function deletePack(id: string): Promise<void> {
  await tx("readwrite", (s) => s.delete(id));
}

export async function deleteAllPacks(): Promise<void> {
  await tx("readwrite", (s) => s.clear());
}

// ---- resolution to catalog entries + configs -------------------------

/** One policy entry per checkpoint (name = baseName or baseName@id). */
export function packEntries(pack: StoredPack): PolicyEntry[] {
  const multi = pack.manifest.checkpoints.length > 1;
  const dof = pack.config.policy?.num_act ?? 0;
  return pack.manifest.checkpoints.map((c) => ({
    name: multi ? `${pack.manifest.baseName}@${c.id}` : pack.manifest.baseName,
    family: pack.manifest.family,
    robot: pack.manifest.robot,
    dof,
    group: pack.manifest.group,
    packId: pack.id,
    checkpointFile: c.file,
    fsmId: pack.config.g1?.fsm_id ?? (pack.config.r1?.fsm_id as number | undefined)
      ?? (pack.config.go2?.fsm_id as number | undefined) ?? (pack.config.as2?.fsm_id as number | undefined),
    models: pack.config.g1?.models ?? pack.config.r1?.models,
  }));
}

const blobUrls = new Map<string, string>();      // "packId::inzipPath" -> blob url

function blobUrl(packId: string, file: string, bytes: Uint8Array): string {
  const key = `${packId}::${file}`;
  let u = blobUrls.get(key);
  if (!u) {
    // copy into a fresh ArrayBuffer so the Blob owns contiguous bytes
    u = URL.createObjectURL(new Blob([bytes.slice()]));
    blobUrls.set(key, u);
  }
  return u;
}

export function revokePack(id: string) {
  for (const [key, url] of [...blobUrls]) {
    if (key.startsWith(`${id}::`)) { URL.revokeObjectURL(url); blobUrls.delete(key); }
  }
}

/** Resolve an imported entry to a ready-to-run EvalConfig: clone the pack's
 * config, point the policy / aux nets / data files at blob URLs for this checkpoint. */
export function resolvePackConfig(pack: StoredPack, entry: PolicyEntry): EvalConfig {
  const cfg: EvalConfig = structuredClone(pack.config);
  cfg.name = entry.name;
  cfg.robot = pack.manifest.robot;
  cfg.policy = { ...cfg.policy, path: blobUrl(pack.id, entry.checkpointFile,
                                              pack.files[entry.checkpointFile]) };
  if (pack.manifest.aux && cfg.g1) {
    const aux: Record<string, string> = { ...(cfg.g1.aux_models ?? {}) };
    for (const [role, file] of Object.entries(pack.manifest.aux))
      aux[role] = blobUrl(pack.id, file, pack.files[file]);
    cfg.g1.aux_models = aux;
  }
  if (pack.manifest.auxFiles && cfg.g1) {
    const files: Record<string, string> = { ...(cfg.g1.aux_files ?? {}) };
    for (const [role, file] of Object.entries(pack.manifest.auxFiles))
      files[role] = blobUrl(pack.id, file, pack.files[file]);
    cfg.g1.aux_files = files;
  }
  // R1 firmware packs keep their extra nets / motion CSVs under `r1:` (same manifest aux / auxFiles)
  if (pack.manifest.aux && cfg.r1) {
    const aux: Record<string, string> = { ...((cfg.r1.aux_models as Record<string, string>) ?? {}) };
    for (const [role, file] of Object.entries(pack.manifest.aux)) aux[role] = blobUrl(pack.id, file, pack.files[file]);
    cfg.r1.aux_models = aux;
  }
  if (pack.manifest.auxFiles && cfg.r1) {
    const files: Record<string, string> = { ...((cfg.r1.aux_files as Record<string, string>) ?? {}) };
    for (const [role, file] of Object.entries(pack.manifest.auxFiles)) files[role] = blobUrl(pack.id, file, pack.files[file]);
    cfg.r1.aux_files = files;
  }
  // Go2 firmware packs: extra nets under `go2.aux_models`
  if (pack.manifest.aux && cfg.go2) {
    const aux: Record<string, string> = { ...(cfg.go2.aux_models ?? {}) };
    for (const [role, file] of Object.entries(pack.manifest.aux)) aux[role] = blobUrl(pack.id, file, pack.files[file]);
    cfg.go2.aux_models = aux;
  }
  // AS2 firmware packs: extra nets under `as2.aux_models`
  if (pack.manifest.aux && cfg.as2) {
    const aux: Record<string, string> = { ...(cfg.as2.aux_models ?? {}) };
    for (const [role, file] of Object.entries(pack.manifest.aux)) aux[role] = blobUrl(pack.id, file, pack.files[file]);
    cfg.as2.aux_models = aux;
  }
  if (pack.manifest.clip && cfg.g1) {
    cfg.g1.motion_clip = blobUrl(pack.id, pack.manifest.clip,
                                 pack.files[pack.manifest.clip]);
  }
  return cfg;
}
