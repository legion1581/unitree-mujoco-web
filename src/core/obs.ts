/** Observation-term builders, keyed by the on-robot observation term names
 * config. Motion-tracking drivers assemble the obs vector by concatenating, in
 * config order, the builder output for each `obs_terms` entry.
 *
 * Context `c` provided by the driver each control step (all plain arrays in
 * BFS/policy order where applicable — the driver does the index gymnastics).
 */
import { concatF32, matFromWxyz, matmul3, matTVec, rot6d, transpose } from "./math";
import type { ObsTerm } from "./types";

export interface ObsContext {
  refPos: Float64Array;             // reference joint pos, BFS order
  refVel: Float64Array;             // reference joint vel, BFS order
  refAnchorR: Float64Array;         // reference anchor rotation (row-major 9)
  anchorQuatWxyz: ArrayLike<number>; // current anchor body xquat
  baseQuatWxyz: ArrayLike<number>;   // floating-base quaternion
  baseAngVel: ArrayLike<number>;     // qvel[bqv+3 .. bqv+6]
  qBfs: Float64Array;               // current joint pos, BFS order
  qdBfs: Float64Array;              // current joint vel, BFS order
  defaultJointPos: ArrayLike<number>;
  lastAction: Float32Array;
}

type Builder = (c: ObsContext) => ArrayLike<number>;

const BUILDERS: Record<string, Builder> = {
  motion_command: (c) => concatF32([c.refPos, c.refVel]),
  motion_anchor_ori_b: (c) =>
    rot6d(matmul3(transpose(matFromWxyz(c.anchorQuatWxyz)), c.refAnchorR)),
  projected_gravity: (c) => matTVec(matFromWxyz(c.baseQuatWxyz), [0, 0, -1]),
  base_ang_vel: (c) => c.baseAngVel,
  joint_pos_rel: (c) => c.qBfs.map((v, i) => v - (c.defaultJointPos[i] as number)),
  joint_vel_rel: (c) => c.qdBfs,
  last_action: (c) => c.lastAction,
};

/** Assemble the float32 obs vector from config `obs_terms` and context `c`. */
export function buildObs(terms: ObsTerm[], c: ObsContext): Float32Array {
  const parts: ArrayLike<number>[] = [];
  for (const t of terms) {
    const b = BUILDERS[t.name];
    if (!b) throw new Error(`no obs builder for '${t.name}' — add it to core/obs.ts`);
    const v = b(c);
    if (v.length !== t.dim)
      throw new Error(`obs term '${t.name}' produced ${v.length} != dim ${t.dim}`);
    parts.push(v);
  }
  return concatF32(parts);
}
