/** Message protocol between the UI (main thread) and the sim worker.
 *
 * The whole simulation — MuJoCo wasm, onnxruntime, drivers, FSM — lives in a
 * Web Worker. This is not just a performance nicety: MuJoCo's model compiler
 * spawns pthreads (wasm workers) and *blocks* waiting for them, which
 * deadlocks the browser main thread; inside a worker, blocking is legal and
 * the pthread pool boots fine.
 */
import type { Command } from "../core/types";
import type { FsmState } from "./fsm";
import type { TerrainName } from "./terrain";

// ---- main -> worker --------------------------------------------------
export type ToWorker =
  | { t: "init"; robot?: string; terrain?: TerrainName; policy?: string }  // restore saved session
  | { t: "initModel"; robot?: string }   // painter: compile model only, no sim
  | { t: "policy"; name: string }
  | { t: "cycle"; step: 1 | -1 }
  | { t: "state"; state: FsmState }
  | { t: "robot"; robot: string }
  | { t: "terrain"; terrain: TerrainName }
  | { t: "command"; cmd: Command }
  | { t: "reset" }
  | { t: "recenter" }
  | { t: "playback"; csv: string; fps: number; name: string }  // kinematic mocap preview
  | { t: "stopPlayback" }
  | { t: "importPack"; zip: ArrayBuffer }
  | { t: "deletePack"; id: string }
  | { t: "inspect"; on: boolean };   // stream live obs/action inspector data

// ---- worker -> main --------------------------------------------------
/** Everything the renderer needs to build the scene, copied out of the model
 * (typed arrays are structured-clone copies, not wasm views). */
export interface RenderModel {
  robot: string;           // active robot (r1* → paint palette, else model rgba)
  ngeom: number;
  type: Int32Array;        // ngeom
  size: Float64Array;      // 3*ngeom
  rgba: Float32Array;      // 4*ngeom
  group: Int32Array;       // ngeom
  contype: Int32Array;     // ngeom (0 = visual-only geom)
  bodyid: Int32Array;      // ngeom (0 = worldbody/terrain)
  dataid: Int32Array;      // ngeom (mesh id or -1)
  meshes: Record<number, { name: string; vert: Float32Array; face: Uint32Array }>;
}

export interface PolicyInfo {
  name: string; family: string; dof: number; group: string;
  packId: string;            // owning pack (all policies come from packs)
}

export interface PackInfo {
  id: string; group: string; robot: string; checkpoints: string[];
}

export interface StatusMsg {
  t: "status";
  robots: string[];
  robot: string;
  policies: PolicyInfo[];
  packs: PackInfo[];        // imported packs (for the manager list)
  policy: string;
  family: string;
  dof: number;
  state: FsmState;
  command: Command;
  rtf: number;
  playback?: string;        // clip name when in kinematic-playback mode
}

/** Live obs/action inspector snapshot (streamed ~10 Hz while the panel is open). */
export interface InspectMsg {
  t: "inspect";
  policy: string; family: string; state: string;
  baseZ: number; baseXY?: [number, number]; command: Command; drive: string;
  nObst?: number;            // vision family: occupied LIDAR cells this step
  lidar?: { nx: number; ny: number; grid: number[] };  // occupancy minimap
  onnxInputs: number; stackDepth: number; perFrame: number;
  terms: { name: string; dim: number; vals: number[] }[];   // vals = first 6 of newest frame
  actions: { name: string; value: number }[];               // per driven joint
  playback: boolean;
}

export type FromWorker =
  | { t: "model"; desc: RenderModel }
  | { t: "frame"; xpos: Float32Array; xmat: Float32Array; baseX: number; baseY: number }
  | StatusMsg
  | InspectMsg
  | { t: "busy"; msg: string }
  | { t: "ready" }
  | { t: "error"; msg: string };
