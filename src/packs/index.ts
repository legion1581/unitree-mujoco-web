/** Policy-pack import + store (runs in the sim worker).
 *
 * A "policy pack" is a .zip a user imports; the repo ships none (weights are the
 * user's own — see README). Layout inside the zip:
 *
 *     pack.json                     manifest (below)
 *     config.yaml                   the policy config (obs scheme, gains, family)
 *     checkpoints/<id>.onnx …       one or more weight checkpoints (same obs)
 *     traj.csv                      optional reference clip (motion-tracking)
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
  robot: string;                 // r1 | r1_air | …
  family: string;                // motion_tracking | locomotion | mimic | …
  baseName: string;              // e.g. "dances/gongfu"
  config: string;                // path in zip to the config yaml
  traj?: string;                 // path in zip to the reference csv (optional)
  checkpoints: { id: string; file: string }[];
}

export interface StoredPack {
  id: string;                    // unique pack id (baseName)
  manifest: PackManifest;
  config: EvalConfig;            // parsed config.yaml
  files: Record<string, Uint8Array>;   // in-zip path -> bytes (onnx, traj)
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
  if (manifest.traj) {
    if (!files[manifest.traj]) throw new Error(`traj "${manifest.traj}" missing from zip`);
    kept[manifest.traj] = files[manifest.traj];
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

// ---- resolution to catalog entries + configs -------------------------

/** One policy entry per checkpoint (name = baseName or baseName@id). */
export function packEntries(pack: StoredPack): PolicyEntry[] {
  const multi = pack.manifest.checkpoints.length > 1;
  const dof = pack.config.policy?.num_act ?? pack.config.gains?.default_joint_pos?.length ?? 0;
  return pack.manifest.checkpoints.map((c) => ({
    name: multi ? `${pack.manifest.baseName}@${c.id}` : pack.manifest.baseName,
    family: pack.manifest.family,
    robot: pack.manifest.robot,
    dof,
    group: pack.manifest.group,
    packId: pack.id,
    checkpointFile: c.file,
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
 * config, point policy/traj at blob URLs for this checkpoint. */
export function resolvePackConfig(pack: StoredPack, entry: PolicyEntry): EvalConfig {
  const cfg: EvalConfig = structuredClone(pack.config);
  cfg.name = entry.name;
  cfg.robot = pack.manifest.robot;
  cfg.policy = { ...cfg.policy, path: blobUrl(pack.id, entry.checkpointFile,
                                              pack.files[entry.checkpointFile]) };
  if (pack.manifest.traj) {
    cfg.traj = { ...(cfg.traj ?? {}), path: blobUrl(pack.id, pack.manifest.traj,
                                                    pack.files[pack.manifest.traj]) };
  }
  return cfg;
}
