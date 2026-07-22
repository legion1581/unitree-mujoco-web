/** LIDAR height-map emulation for the Go2 vision (obstacle-avoidance) family —
 * TS port of the reference implementation's HeightMapSampler.
 *
 * The nav policies consume a flat 861-cell local terrain map, not a depth
 * image. The faithful emulation is a bank of *vertical* rays: for each grid
 * cell (in the robot's yaw-aligned horizontal frame) drop a ray straight down
 * from above the robot and read the terrain/obstacle elevation under it.
 * Vertical (not a single-apex fan) matters — a fan slants, so a raised
 * obstacle registers in the wrong cell. Robot geoms live in groups 2/3; the
 * mask keeps only the world (floor / terrain / obstacles, groups 0/1).
 *
 * `mode: "occupancy"` matches the on-robot obstacle grid: a cell is 1.0 iff
 * occ_lo < height < occ_hi (height relative to the ground under the base),
 * then inflated by `dilate` cells.
 */
import { matFromWxyz } from "./math";
import type { Engine } from "../sim/engine";

// world = groups 0/1; robot visual/collision geoms (2/3) are masked out
const WORLD_GROUPS = [1, 1, 0, 0, 0, 0];

// Go2 L1 LIDAR mount in the base frame (only needed for relative_to="lidar")
const BASELINK2LIDAR = [0.28216, 0.0, -0.02467];

export interface LidarCfg {
  nx?: number; ny?: number; res?: number;
  x0?: number; y0?: number;
  order?: "xy" | "yx";
  relative_to?: "base" | "lidar" | "base_foot" | "world";
  clip?: number; fill?: number; apex?: number;
  mode?: "height" | "occupancy";
  occ_lo?: number; occ_hi?: number; dilate?: number;
}

export class HeightMapSampler {
  readonly nx: number; readonly ny: number; readonly n: number;
  private readonly res: number;
  private readonly order: "xy" | "yx";
  private readonly relativeTo: string;
  private readonly clipH: number; private readonly fill: number;
  private readonly apex: number;
  private readonly mode: "height" | "occupancy";
  private readonly occLo: number; private readonly occHi: number;
  private readonly dilate: number;
  private readonly cells: Float64Array;       // n*2, yaw-aligned base frame
  private readonly centerIdx: number;         // cell nearest the base origin
  private readonly hitZ: Float64Array;        // scratch

  constructor(cfg: LidarCfg) {
    this.nx = cfg.nx ?? 41; this.ny = cfg.ny ?? 21; this.res = cfg.res ?? 0.05;
    const x0 = cfg.x0 ?? -(this.nx - 1) * this.res / 2;
    const y0 = cfg.y0 ?? -(this.ny - 1) * this.res / 2;
    this.order = cfg.order ?? "xy";
    this.relativeTo = cfg.relative_to ?? "base_foot";
    this.clipH = cfg.clip ?? 1.0; this.fill = cfg.fill ?? 0.0;
    this.apex = cfg.apex ?? 3.0;
    this.mode = cfg.mode ?? "occupancy";
    this.occLo = cfg.occ_lo ?? 0.15; this.occHi = cfg.occ_hi ?? 2.0;
    this.dilate = cfg.dilate ?? 1;
    this.n = this.nx * this.ny;
    this.cells = new Float64Array(this.n * 2);
    // order "xy": x (forward) outer, y inner — row-major (nx, ny) grid
    let k = 0;
    if (this.order === "xy") {
      for (let ix = 0; ix < this.nx; ix++)
        for (let iy = 0; iy < this.ny; iy++, k++) {
          this.cells[2 * k] = x0 + ix * this.res;
          this.cells[2 * k + 1] = y0 + iy * this.res;
        }
    } else {
      for (let iy = 0; iy < this.ny; iy++)
        for (let ix = 0; ix < this.nx; ix++, k++) {
          this.cells[2 * k] = x0 + ix * this.res;
          this.cells[2 * k + 1] = y0 + iy * this.res;
        }
    }
    let best = 0, bestD = Infinity;
    for (let i = 0; i < this.n; i++) {
      const d = this.cells[2 * i] ** 2 + this.cells[2 * i + 1] ** 2;
      if (d < bestD) { bestD = d; best = i; }
    }
    this.centerIdx = best;
    this.hitZ = new Float64Array(this.n);
  }

  /** Flat height/occupancy map for the current engine state; `bqp` is the
   * base free-joint qpos address. */
  sample(eng: Engine, bqp: number): Float32Array {
    const q = eng.qpos;
    const px = q[bqp], py = q[bqp + 1], pz = q[bqp + 2];
    const quat = [q[bqp + 3], q[bqp + 4], q[bqp + 5], q[bqp + 6]];
    const R = matFromWxyz(quat);                    // row-major 3x3
    const yaw = Math.atan2(R[3], R[0]);             // heading only
    const c = Math.cos(yaw), s = Math.sin(yaw);
    const z0 = pz + this.apex;
    const hitZ = this.hitZ;
    for (let i = 0; i < this.n; i++) {
      const cx = this.cells[2 * i], cy = this.cells[2 * i + 1];
      const wx = px + c * cx - s * cy;
      const wy = py + s * cx + c * cy;
      const { dist, geom } = eng.ray([wx, wy, z0], [0, 0, -1], WORLD_GROUPS);
      hitZ[i] = geom >= 0 && dist >= 0 ? z0 - dist : NaN;
    }
    let ref: number;
    if (this.relativeTo === "base") ref = pz;
    else if (this.relativeTo === "lidar")
      ref = pz + R[6] * BASELINK2LIDAR[0] + R[7] * BASELINK2LIDAR[1] + R[8] * BASELINK2LIDAR[2];
    else if (this.relativeTo === "base_foot") {
      const zc = hitZ[this.centerIdx];
      ref = Number.isNaN(zc) ? pz : zc;
    } else ref = 0;                                  // "world"

    const out = new Float32Array(this.n);
    if (this.mode === "height") {
      for (let i = 0; i < this.n; i++) {
        const h = Number.isNaN(hitZ[i]) ? this.fill : hitZ[i] - ref;
        out[i] = Math.min(Math.max(h, -this.clipH), this.clipH);
      }
      return out;
    }
    // occupancy: 1.0 where occ_lo < h < occ_hi, then `dilate`-cell inflation
    for (let i = 0; i < this.n; i++) {
      const h = Number.isNaN(hitZ[i]) ? this.fill : hitZ[i] - ref;
      out[i] = h > this.occLo && h < this.occHi ? 1 : 0;
    }
    // grid is (rows, cols) = (nx, ny) for "xy", (ny, nx) for "yx"
    const rows = this.order === "xy" ? this.nx : this.ny;
    const cols = this.order === "xy" ? this.ny : this.nx;
    let g = out;
    for (let pass = 0; pass < this.dilate; pass++) {
      const d = Float32Array.from(g);
      for (let r = 0; r < rows; r++)
        for (let cc = 0; cc < cols; cc++) {
          if (g[r * cols + cc] !== 1) continue;
          if (r > 0) d[(r - 1) * cols + cc] = 1;
          if (r < rows - 1) d[(r + 1) * cols + cc] = 1;
          if (cc > 0) d[r * cols + cc - 1] = 1;
          if (cc < cols - 1) d[r * cols + cc + 1] = 1;
        }
      g = d;
    }
    return g;
  }
}
