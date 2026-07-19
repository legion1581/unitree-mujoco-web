/** Small math helpers shared by the obs builders and drivers.
 * Ported 1:1 from the reference Python implementation — the rot6d convention
 * and quat orders are load-bearing; a policy falls in ~0.5 s if they're wrong.
 * Matrices are number[9], row-major.
 */

export type Vec3 = [number, number, number];

/** 3x3 rotation (row-major, length 9) from an xyzw quaternion. */
export function quatToMatXyzw(q: ArrayLike<number>): Float64Array {
  const n = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  const x = q[0] / n, y = q[1] / n, z = q[2] / n, w = q[3] / n;
  return new Float64Array([
    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
  ]);
}

/** 3x3 rotation from a wxyz quaternion (MuJoCo order). */
export function matFromWxyz(q: ArrayLike<number>): Float64Array {
  return quatToMatXyzw([q[1], q[2], q[3], q[0]]);
}

/** xyzw (trajectory CSV order) -> wxyz (MuJoCo order). */
export function wxyz(qXyzw: ArrayLike<number>): [number, number, number, number] {
  return [qXyzw[3], qXyzw[0], qXyzw[1], qXyzw[2]];
}

/** R^T (row-major 9-vector). */
export function transpose(R: ArrayLike<number>): Float64Array {
  return new Float64Array([R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]]);
}

/** Row-major 3x3 product A@B. */
export function matmul3(A: ArrayLike<number>, B: ArrayLike<number>): Float64Array {
  const C = new Float64Array(9);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      C[3 * i + j] = A[3 * i] * B[j] + A[3 * i + 1] * B[3 + j] + A[3 * i + 2] * B[6 + j];
  return C;
}

/** R^T @ v for row-major R. */
export function matTVec(R: ArrayLike<number>, v: ArrayLike<number>): Vec3 {
  return [
    R[0] * v[0] + R[3] * v[1] + R[6] * v[2],
    R[1] * v[0] + R[4] * v[1] + R[7] * v[2],
    R[2] * v[0] + R[5] * v[1] + R[8] * v[2],
  ];
}

/** First two columns, ROW-interleaved [m00,m01,m10,m11,m20,m21] — the
 * IsaacLab / BeyondMimic convention (NOT column-major flattening). */
export function rot6d(R: ArrayLike<number>): Float64Array {
  return new Float64Array([R[0], R[1], R[3], R[4], R[6], R[7]]);
}

/** Linear-interpolate a reference-motion row at time t; renormalize the quat.
 * Row layout: [base_pos(3), base_quat_xyzw(4), joint_pos(nCols-7)].
 * traj is a flat Float32Array of shape [nRows, nCols]. */
export function sampleTraj(traj: Float32Array, nRows: number, nCols: number,
                           t: number, fps: number): Float64Array {
  const f = t * fps;
  const i0 = Math.min(Math.floor(f), nRows - 1);
  const i1 = Math.min(i0 + 1, nRows - 1);
  const a = f - i0;
  const row = new Float64Array(nCols);
  for (let c = 0; c < nCols; c++)
    row[c] = (1 - a) * traj[i0 * nCols + c] + a * traj[i1 * nCols + c];
  const qn = Math.hypot(row[3], row[4], row[5], row[6]) || 1;
  row[3] /= qn; row[4] /= qn; row[5] /= qn; row[6] /= qn;
  return row;
}

export function clip(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/** argsort of an integer permutation (inverse permutation). */
export function argsort(a: ArrayLike<number>): Int32Array {
  const inv = new Int32Array(a.length);
  for (let i = 0; i < a.length; i++) inv[a[i]] = i;
  return inv;
}

export interface GaitParams {
  dt: number;
  walk_cycle: number; run_cycle: number;
  walk_air_ratio: number; run_air_ratio: number;
  transition_speed_min: number; transition_speed_max: number;
}

/** The 6 gait-clock obs terms for the *_gait policies (disasm-confirmed):
 * two-foot antiphase clock whose cycle shortens walk->run with commanded speed.
 * Returns [newPhase, 6-vector [sin0,sin1,cos0,cos1,air,air]]. */
export function gaitClock(phase: number, cmd: ArrayLike<number>, g: GaitParams):
    [number, Float32Array] {
  const speed = Math.hypot(cmd[0], cmd[1]);
  const lo = g.transition_speed_min, hi = g.transition_speed_max;
  const t = clip((speed - lo) / Math.max(hi - lo, 1e-6), 0, 1);
  const air = g.walk_air_ratio + (g.run_air_ratio - g.walk_air_ratio) * t;
  const cycle = Math.max(g.walk_cycle + (g.run_cycle - g.walk_cycle) * t, 1e-6);
  const ph = (phase + g.dt / cycle) % 1.0;
  const phi0 = (ph + air) % 1.0;
  const phi1 = (ph + air + 0.5) % 1.0;         // legs antiphase
  const a0 = 2 * Math.PI * phi0, a1 = 2 * Math.PI * phi1;
  return [ph, new Float32Array([Math.sin(a0), Math.sin(a1), Math.cos(a0), Math.cos(a1), air, air])];
}

/** Concatenate float arrays into one Float32Array. */
export function concatF32(parts: ArrayLike<number>[]): Float32Array {
  let n = 0;
  for (const p of parts) n += p.length;
  const out = new Float32Array(n);
  let o = 0;
  for (const p of parts) { out.set(p as Float32Array, o); o += p.length; }
  return out;
}
