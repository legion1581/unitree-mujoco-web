/** Terrain presets — rewrite the scene XML before compiling. v1 ports the
 * cheap presets (flat / rough / slope); stairs & friends come later. All
 * randomness is seeded so a preset builds the same field every reload. */

export type TerrainName = "flat" | "rough" | "slope" | "stairs" | "park";
export const TERRAINS: TerrainName[] = ["flat", "rough", "slope", "stairs", "park"];

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function roughXml(amplitude = 0.03, tile = 0.35, extent = 4.0, seed = 7): string {
  const rnd = mulberry32(seed);
  const parts: string[] = [];
  for (let x = -extent; x < extent; x += tile)
    for (let y = -extent; y < extent; y += tile) {
      if (Math.hypot(x, y) < 0.6) continue;         // clear spawn area
      const h = amplitude * rnd();
      if (h < 0.004) continue;
      parts.push(`<geom type="box" size="${(tile / 2).toFixed(3)} ${(tile / 2).toFixed(3)} ${h.toFixed(4)}" ` +
        `pos="${(x + tile / 2).toFixed(3)} ${(y + tile / 2).toFixed(3)} ${h.toFixed(4)}" ` +
        `rgba="0.25 0.3 0.36 1" friction="1 0.005 0.0001"/>`);
    }
  return parts.join("\n    ");
}

/** Unitree-style slab stairs (terrain_tool/terrain_generator.py AddStairs):
 * each step is a slab of thickness `height` advancing +x by `width` and +z by
 * `height`; the first sits on the ground, later ones overhang. */
function stairsXml(width = 0.2, height = 0.15, length = 1.5, n = 10, startX = 1.0): string {
  const parts: string[] = [];
  for (let i = 0; i < n; i++) {
    const x = startX + (i + 1) * width;
    const z = height / 2 + i * height;
    parts.push(`<geom type="box" size="${(width / 2).toFixed(3)} ${(length / 2).toFixed(3)} ${(height / 2).toFixed(3)}" ` +
      `pos="${x.toFixed(3)} 0 ${z.toFixed(3)}" rgba="0.62 0.64 0.68 1" friction="1 0.005 0.0001"/>`);
  }
  return parts.join("\n    ");
}

function slopeXml(angleDeg = 10): string {
  const a = (angleDeg * Math.PI) / 180;
  return `<geom type="box" size="6 6 0.05" pos="3 0 ${(3 * Math.tan(a)).toFixed(3)}" ` +
    `euler="0 ${-angleDeg} 0" rgba="0.25 0.3 0.36 1" friction="1 0.005 0.0001"/>`;
}

/** zyx euler (rad) -> wxyz quaternion — same convention as the terrain tool. */
function eulerToQuat(roll: number, pitch: number, yaw: number): [number, number, number, number] {
  const cx = Math.cos(roll / 2), sx = Math.sin(roll / 2);
  const cy = Math.cos(pitch / 2), sy = Math.sin(pitch / 2);
  const cz = Math.cos(yaw / 2), sz = Math.sin(yaw / 2);
  return [
    cx * cy * cz + sx * sy * sz,
    sx * cy * cz - cx * sy * sz,
    cx * sy * cz + sx * cy * sz,
    cx * cy * sz - sx * sy * cz,
  ];
}

/** The unitree_mujoco terrain_tool playground (terrain_generator.py __main__),
 * rebuilt geom-for-geom: box + cylinder obstacle, big slope, stairs, suspended
 * stairs, and a field of randomly-tilted buried boxes as rough ground. The two
 * heightfield patches (perlin + image) are skipped — no hfield rendering yet. */
function parkXml(seed = 11): string {
  const rnd = mulberry32(seed);
  const u = (lo: number, hi: number) => lo + (hi - lo) * rnd();
  const parts: string[] = [];
  const RGBA = 'rgba="0.62 0.64 0.68 1"';
  const box = (pos: [number, number, number], euler: [number, number, number],
               full: [number, number, number]) => {
    const q = eulerToQuat(...euler);
    parts.push(`<geom type="box" pos="${pos.map((v) => v.toFixed(4)).join(" ")}" ` +
      `size="${full.map((v) => (v / 2).toFixed(4)).join(" ")}" ` +
      `quat="${q.map((v) => v.toFixed(6)).join(" ")}" ${RGBA} friction="1 0.005 0.0001"/>`);
  };
  // box obstacle + cylinder on top
  box([1.5, 0, 0.1], [0, 0, 0], [1, 1.5, 0.2]);
  parts.push(`<geom type="cylinder" pos="1.5 0 0.25" size="0.5 0.25" ${RGBA}/>`);
  // slope (pitch -0.5 rad)
  box([2.0, 2.0, 0.5], [0, -0.5, 0], [3, 1.5, 0.1]);
  // stairs at y=4 and suspended stairs at y=6 (0.2 deep, 0.15 rise, 10 steps)
  const stair = (y: number, thickness: number) => {
    let x = 1.0, z = -0.075;
    for (let i = 0; i < 10; i++) {
      x += 0.2; z += 0.15;
      box([x, y, z], [0, 0, 0], [0.2, 1.5, thickness]);
    }
  };
  stair(4.0, 0.15);          // solid-feeling staircase
  stair(6.0, 0.05);          // suspended slabs (height - gap)
  // rough ground: 10x8 randomly-tilted boxes buried to ~z=0
  let px = -2.5;
  for (let i = 0; i < 10; i++) {
    px += u(0.15, 0.25);
    let py = 5.0;
    for (let j = 0; j < 8; j++) {
      py += u(0.15, 0.25);
      box([px, py, -0.25],
          [u(-0.2, 0.2), u(-0.2, 0.2), u(-0.2, 0.2)],
          [u(0.45, 0.55), u(0.45, 0.55), u(0.45, 0.55)]);
    }
  }
  return parts.join("\n    ");
}

/** Returns a transform for Engine.create — injects preset geoms into scene.xml. */
export function terrainTransform(preset: TerrainName, args: Record<string, number> = {}) {
  return (name: string, xml: string): string => {
    if (!name.endsWith("scene.xml") || preset === "flat") return xml;
    const inject =
      preset === "rough" ? roughXml(args.amplitude, args.tile, args.extent, args.seed)
      : preset === "stairs" ? stairsXml(args.width, args.height, args.length, args.n, args.start_x)
      : preset === "park" ? parkXml(args.seed)
      : slopeXml(args.angle_deg);
    return xml.replace("</worldbody>", `  ${inject}\n  </worldbody>`);
  };
}
