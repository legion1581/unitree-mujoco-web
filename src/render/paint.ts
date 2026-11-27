/** Robot paint: a colour + surface finish per mesh PART, one palette per robot.
 *
 * A part is a mesh name with the left_/right_ prefix, the _link / _mirror
 * suffixes and the file extension stripped, so both sides of a limb (and the
 * Go2's mirrored leg meshes) paint together. Palettes model the shipping
 * colourways from the product photos: R1 = glossy white shell with navy
 * accents, mirrored orange visor lens and gunmetal hands; Go2 = light-grey matte body with
 * brushed-aluminium motor housings and black face / feet; AS2 = blue-grey anodised shell, polished aluminium
 * drums and calves, black LiDAR and rubber feet. Robots without a
 * palette (G1) render the model's own rgba.
 *
 * Where a real colour boundary falls inside a single mesh (the R1 head carries
 * visor + ear pods, a thigh carries the knee puck, ...), a part lists
 * `regions`: triangles whose centroid and face normal pass a test take another
 * paint. Tests are written in the mesh FILE's frame (STL/OBJ coordinates, not
 * MuJoCo's inertia-centred compiled frame) for the LEFT-side mesh; right-side
 * meshes (mirror images in y) are tested with y mirrored. A region can instead
 * select whole CAD surfaces (`scope`): "patch" = a smooth surface patch (faces
 * joined across edges bending < 20°), "shell" = a connected shell (e.g. the
 * visor lens); the test then sees the group's area-weighted centroid, mean
 * normal and bounding-box size, and edges follow the real surface seams.
 */

export type Finish =
  | "gloss"      // lacquered plastic shell: low roughness + clear coat
  | "satin"      // painted / anodised accent parts
  | "matte"      // matte plastic
  | "rubber"     // feet, soles
  | "metal"      // brushed aluminium motor housings
  | "gunmetal"   // dark metallic hands / linkages
  | "anodised"   // bead-blasted anodised aluminium shell (AS2)
  | "mirror";    // iridescent mirrored lens

export interface Paint {
  color: number;
  finish: Finish;
  hidden?: boolean;
}

/** Point (x, y, z) + unit normal (nx, ny, nz), mesh-file frame; for patch /
 * shell scope also the group's bounding-box size (sx, sy, sz), else 0. */
export type RegionTest = (x: number, y: number, z: number,
                          nx: number, ny: number, nz: number,
                          sx: number, sy: number, sz: number) => boolean;

/** A sub-mesh region: triangles (or whole patches / shells) passing `test`
 * take `paint`. */
export interface Region {
  test: RegionTest;
  paint: Paint;
  scope?: "tri" | "patch" | "shell";
}

interface Rule {
  re: RegExp;          // matched against the part key
  paint: Paint;
  regions?: Region[];
}

interface Palette {
  rules: Rule[];
  fallback: Paint;
}

/** MeshPhysicalMaterial parameters per finish. */
export const FINISH: Record<Finish, {
  roughness: number; metalness: number; clearcoat: number; clearcoatRoughness: number;
  iridescence?: number; iridescenceIOR?: number; iridescenceThicknessRange?: [number, number];
}> = {
  gloss:    { roughness: 0.25, metalness: 0.0,  clearcoat: 1.0,  clearcoatRoughness: 0.08 },
  satin:    { roughness: 0.4,  metalness: 0.0,  clearcoat: 0.45, clearcoatRoughness: 0.3 },
  matte:    { roughness: 0.62, metalness: 0.0,  clearcoat: 0.0,  clearcoatRoughness: 0.5 },
  rubber:   { roughness: 0.85, metalness: 0.0,  clearcoat: 0.0,  clearcoatRoughness: 0.5 },
  metal:    { roughness: 0.42, metalness: 0.9,  clearcoat: 0.0,  clearcoatRoughness: 0.3 },
  gunmetal: { roughness: 0.38, metalness: 0.6,  clearcoat: 0.3,  clearcoatRoughness: 0.3 },
  anodised: { roughness: 0.5,  metalness: 0.35, clearcoat: 0.15, clearcoatRoughness: 0.4 },
  mirror:   { roughness: 0.08, metalness: 0.6,  clearcoat: 1.0,  clearcoatRoughness: 0.03,
              iridescence: 1.0, iridescenceIOR: 1.8, iridescenceThicknessRange: [300, 520] },
};

/** Min signed distance of (u, v) to the edges of a convex CCW polygon:
 * > 0 inside (depth from the nearest edge), < 0 outside. */
const convex = (poly: [number, number][]) => (u: number, v: number) => {
  let d = Infinity;
  for (let i = 0; i < poly.length; i++) {
    const [au, av] = poly[i], [bu, bv] = poly[(i + 1) % poly.length];
    const eu = bu - au, ev = bv - av, l = Math.hypot(eu, ev);
    d = Math.min(d, (eu * (v - av) - ev * (u - au)) / l);
  }
  return d;
};

const disc = (cx: number, cz: number, r: number) =>
  (x: number, z: number) => (x - cx) ** 2 + (z - cz) ** 2 < r * r;

// ---- R1 (glossy white shell, navy accents) -----------------------------
const R1_WHITE: Paint = { color: 0xf1f1ef, finish: "gloss" };
const R1_NAVY: Paint = { color: 0x243e72, finish: "satin" };
const R1_LENS: Paint = { color: 0xc8602a, finish: "mirror" };     // mirrored orange lens
const R1_EAR_CAP: Paint = { color: 0x3c4e72, finish: "satin" };    // grey-blue ear centre cap
const R1_RED: Paint = { color: 0xd6283a, finish: "satin" };        // chest emblem outline
const R1_EMBLEM: Paint = { color: 0xdcdcd8, finish: "matte" };      // chest emblem inlay
const R1_PUCK: Paint = { color: 0x1c1d20, finish: "satin" };       // inner elbow / knee pucks
const R1_HAND: Paint = { color: 0x3b3e43, finish: "gunmetal" };    // fists
const R1_LINK: Paint = { color: 0x2b2d31, finish: "gunmetal" };    // ankle linkage, rods
const R1_CARBON: Paint = { color: 0x55575c, finish: "satin" };     // carbon ankle covers
const R1_CALF: Paint = { color: 0x232427, finish: "satin" };       // shin back (ankle actuators)
const R1_SHOE: Paint = { color: 0x1f2839, finish: "matte" };       // dark navy shoe upper
const R1_SOLE: Paint = { color: 0x141518, finish: "rubber" };

/** Head (head_yaw_link.STL frame). Visor: the goggle lens is its own shell
 * (white frame around it stays shell white). Ear pods: raised discs on the
 * y axis at (x, z) = (0.00546, -0.01655), rim r = 24.5 mm, centre cap r =
 * 13 mm, outer face from |y| = 46 mm (the helmet sits inside that). Forehead:
 * the navy "upside-down A" is the recessed front panel under the helmet's
 * top shell (the CAD strips within 34 mm of the centre line, so the legs run
 * back along the tongue of the top shell), with a white inverted triangle
 * in its middle under the tongue. */
const EAR_X = 0.00546, EAR_Z = -0.01655;
const earR = (x: number, z: number) => Math.hypot(x - EAR_X, z - EAR_Z);
const foreheadTri = convex([[-0.0105, 0.0185], [0, -0.0015], [0.0105, 0.0185]]);
const HEAD_REGIONS: Region[] = [
  { scope: "shell", paint: R1_LENS,
    test: (x, y, z, nx, ny, nz, sx, sy) => x > 0.075 && Math.abs(y) < 0.01 && sy > 0.08 },
  { paint: R1_EAR_CAP,
    test: (x, y, z) => Math.abs(y) > 0.058 && earR(x, z) < 0.0131 },
  { paint: R1_NAVY,
    test: (x, y, z) => Math.abs(y) > 0.046 && earR(x, z) < 0.0246 },
  { paint: R1_WHITE,
    test: (x, y, z, nx) => x > 0.05 && nx > 0.2 && foreheadTri(y, z) > 0 },
  { scope: "patch", paint: R1_NAVY,
    // (thin seam strips along the side-panel edges are not part of it)
    test: (x, y, z, nx, ny, nz, sx, sy) => x > 0.02 && z > -0.004 && sx < 0.07 && sy < 0.076 &&
      Math.abs(y) < 0.034 && !(Math.abs(y) > 0.028 && sy < 0.005) },
];

/** Chest emblem (waist_yaw_link.STL frame): a flush rounded inverted triangle
 * under the neck, red outline (~2.4 mm) around a light grey inlay. */
const CHEST_EMBLEM = convex([
  [0.0032, 0.1862], [0.0185, 0.2117], [0.0165, 0.2157],
  [-0.0165, 0.2157], [-0.0185, 0.2117], [-0.0032, 0.1862],
]);
const CHEST_REGIONS: Region[] = [
  { paint: R1_EMBLEM, test: (x, y, z, nx) => x > 0.04 && nx > 0.3 && CHEST_EMBLEM(y, z) > 0.0024 },
  { paint: R1_RED, test: (x, y, z, nx) => x > 0.04 && nx > 0.3 && CHEST_EMBLEM(y, z) > 0 },
];

/** Hip girdle (pelvis_link.STL frame): navy side pods over the hip-pitch
 * motors, white centre panel. */
const PELVIS_PODS: Region = {
  test: (x, y, z) => Math.abs(y) > 0.05 && z < 0.005, paint: R1_NAVY,
};

/** Shoulder ball (shoulder_roll): navy disc on the front/back clevis lobes. */
const SHOULDER_DISCS: Region[] = [{
  test: (x, y, z) => (x > 0.004 || x < -0.074) && y * y + z * z < 0.027 ** 2,
  paint: R1_NAVY,
}];

/** Elbow joint, left arm (+y = outer): black puck on the inner end of the
 * elbow motor (shoulder_yaw), navy disc on the forearm's outer clevis plate
 * (elbow link, joint axis at its origin). */
const ELBOW_PUCK: Region[] = [{
  test: (x, y, z, nx, ny) => y < -0.036 && Math.abs(ny) > 0.9 && disc(0.0165, -0.083, 0.036)(x, z),
  paint: R1_PUCK,
}];
const ELBOW_DISC: Region = {
  test: (x, y, z, nx, ny) => y > 0.0045 && Math.abs(ny) > 0.9 && disc(0, 0, 0.028)(x, z),
  paint: R1_NAVY,
};

const R1_RULES: Rule[] = [
  { re: /^imu_in_pelvis$/, paint: { ...R1_WHITE, hidden: true } },
  { re: /^head_yaw$/, paint: R1_WHITE, regions: HEAD_REGIONS },    // helmet
  { re: /^head_pitch$/, paint: R1_WHITE },                         // neck
  { re: /^waist_yaw$/, paint: R1_WHITE, regions: CHEST_REGIONS },  // torso
  { re: /^waist_roll$/, paint: R1_NAVY },                          // waist ring
  { re: /^pelvis$/, paint: R1_WHITE, regions: [PELVIS_PODS] },
  { re: /^shoulder_pitch$/, paint: R1_NAVY },                      // shoulder yoke
  { re: /^shoulder_roll$/, paint: R1_WHITE, regions: SHOULDER_DISCS },
  { re: /^shoulder_yaw$/, paint: R1_WHITE, regions: ELBOW_PUCK },
  { re: /^elbow$/, paint: R1_WHITE, regions: [ELBOW_DISC] },       // forearm
  // wrist_roll = forearm cuff + fist; the fist starts past the cuff seam
  { re: /^wrist_roll$/, paint: R1_WHITE, regions: [{ test: (x) => x > 0.075, paint: R1_HAND }] },
  // thigh: black knee-motor end cap on the inner side (-y for the left leg)
  { re: /^hip_yaw$/, paint: R1_WHITE, regions: [{
    test: (x, y, z, nx, ny) => y < -0.04 && Math.abs(ny) > 0.9 &&
      disc(-0.023, -0.159, 0.036)(x, z),
    paint: R1_PUCK,
  }] },
  // shin: navy knee flange (outer), carbon lower shin, dark actuator back
  { re: /^knee$/, paint: R1_WHITE, regions: [
    { test: (x, y, z, nx, ny) => y > 0.01 && Math.abs(ny) > 0.9 && disc(0, 0, 0.041)(x, z),
      paint: R1_NAVY },
    { test: (x, y, z) => z < -0.225, paint: R1_CARBON },
    { test: (x, y, z) => x < -0.035 && z < -0.05, paint: R1_CALF },
  ] },
  { re: /^ankle_(A|B)(_rod)?$|^ankle_constraint_(A|B)$|^ankle_pitch$/, paint: R1_LINK },
  { re: /^ankle_roll$/, paint: R1_SHOE, regions: [{ test: (x, y, z) => z < -0.043, paint: R1_SOLE }] },
];

/** R1 Air: torso + neck + head + waist + hip girdle are ONE pelvis mesh
 * (air_pelvis_link.stl, pelvis frame), and the fist is part of the forearm
 * mesh (air_left_elbow_link.stl). The EDU helmet / chest regions are reused,
 * shifted by the head's (0.0275, 0, 0.4205) and the torso's (0.0325, 0, 0.049)
 * offsets from the pelvis. */
const shifted = (regions: Region[], dx: number, dz: number, zmin: number): Region[] =>
  regions.map((r): Region => ({
    ...r,
    test: (x, y, z, nx, ny, nz, sx, sy, sz) => z > zmin &&
      r.test(x - dx, y, z - dz, nx, ny, nz, sx, sy, sz),
  }));
const AIR_PELVIS_REGIONS: Region[] = [
  ...shifted(HEAD_REGIONS, 0.0275, 0.4205, 0.33),
  ...shifted(CHEST_REGIONS, 0.0325, 0.049, 0.2),
  // waist column between hip girdle and torso
  { test: (x, y, z) => z > -0.002 && z < 0.046 && Math.abs(y) < 0.045, paint: R1_NAVY },
  PELVIS_PODS,
];

const R1_AIR_RULES: Rule[] = [
  { re: /^pelvis$/, paint: R1_WHITE, regions: AIR_PELVIS_REGIONS },
  // wrist_roll body sits at (0.11218, -0.03, -0.0117) in the elbow frame
  { re: /^elbow$/, paint: R1_WHITE, regions: [ELBOW_DISC, { test: (x) => x > 0.187, paint: R1_HAND }] },
  ...R1_RULES,
];

// ---- Go2 (light-grey matte, aluminium motors) ---------------------------
const GO2_BODY: Paint = { color: 0x8c9299, finish: "matte" };
const GO2_LEG: Paint = { color: 0x9ca1a7, finish: "matte" };
const GO2_ALU: Paint = { color: 0xc3c7cc, finish: "metal" };
const GO2_BLACK: Paint = { color: 0x1a1b1e, finish: "satin" };
const GO2_FOOT: Paint = { color: 0x19191b, finish: "rubber" };
const GO2_LOGO: Paint = { color: 0xeceef0, finish: "matte" };

const GO2_RULES: Rule[] = [
  { re: /^base_[012]$/, paint: GO2_BLACK },      // face plate, lidar cage, chin
  { re: /^base_3$/, paint: GO2_LOGO },           // UNITREE lettering, side marks
  { re: /^base_4$/, paint: GO2_BODY },           // body shell
  { re: /^hip_[01]$/, paint: GO2_ALU },          // hip + thigh motor housings (drums, trim rings)
  { re: /^thigh_1$/, paint: GO2_BODY },          // rounded outer motor caps
  { re: /^thigh_0$/, paint: GO2_LEG },           // thigh
  { re: /^calf_0$/, paint: GO2_LEG },
  { re: /^calf_1$|^foot$/, paint: GO2_FOOT },    // foot pads
];

// ---- AS2 (official unitree.com/As2 renders: blue-grey anodised shell and thigh covers, polished aluminium
// motor drums and calves, black LiDAR cage, dark rubber ball feet) --------------------------------------------
const AS2_SHELL: Paint = { color: 0x7c8598, finish: "anodised" };    // anodised blue-grey (#929bb1 lit / #686f81 shade)
const AS2_PANEL: Paint = { color: 0x666d7d, finish: "anodised" };    // side battery panel, body front / back faces
const AS2_BELLY: Paint = { color: 0x3a3d43, finish: "satin" };       // underside
const AS2_ALU: Paint = { color: 0xcfd2d6, finish: "metal" };         // motor drums, calves (#c0c2c5 … #dedfe0)
const AS2_LIDAR: Paint = { color: 0x1c1d1f, finish: "satin" };       // L2 LiDAR cage
const AS2_HANDLE: Paint = { color: 0xd6dae0, finish: "metal" };      // front handle around the LiDAR
const AS2_FOOT: Paint = { color: 0x2e2f33, finish: "rubber" };

/** base_link.STL frame (= body frame): hip-abduction drums at both ends (|x| > 0.104, below the neck), the head on
 * the neck with the LiDAR sphere at (0.283, 0, 0.078) and the U handle in front of it. */
const AS2_BASE_REGIONS: Region[] = [
  { paint: AS2_LIDAR, test: (x, y, z) => x > 0.235 && Math.hypot(x - 0.283, y, z - 0.078) < 0.05 },
  { paint: AS2_HANDLE, test: (x) => x > 0.29 },
  { paint: AS2_ALU, test: (x, y, z) => Math.abs(x) > 0.104 && x < 0.2 && z < 0.045 },
  { paint: AS2_BELLY, test: (x, y, z, nx, ny, nz) => z < -0.05 && nz < -0.5 },
  { paint: AS2_PANEL, test: (x, y, z, nx, ny) => Math.abs(y) > 0.106 && Math.abs(ny) > 0.7 && z < 0.06 },
  { paint: AS2_PANEL, test: (x, y, z, nx) => Math.abs(x) > 0.09 && Math.abs(x) < 0.106 && Math.abs(nx) > 0.8 && z < 0.05 },
];
/** *_thigh.STL: the thigh-motor drum runs inward (toward the body) from the outer cap; left legs inward = −y,
 * right legs (mirror images) +y. */
const as2Thigh = (side: number): Region[] => [{ paint: AS2_ALU, test: (x, y, z) => side * y < -0.045 && z > -0.045 }];

const AS2_RULES: Rule[] = [
  { re: /^base$/, paint: AS2_SHELL, regions: AS2_BASE_REGIONS },
  { re: /^(FL|FR|RL|RR)_hip$/, paint: AS2_ALU },
  { re: /^(FL|RL)_thigh$/, paint: AS2_SHELL, regions: as2Thigh(1) },
  { re: /^(FR|RR)_thigh$/, paint: AS2_SHELL, regions: as2Thigh(-1) },
  { re: /^(FL|FR|RL|RR)_calf$/, paint: AS2_ALU },
  { re: /^(FL|FR|RL|RR)_foot$/, paint: AS2_FOOT },
];

const PALETTES: Record<string, Palette> = {
  r1: { rules: R1_RULES, fallback: R1_WHITE },
  r1_air: { rules: R1_AIR_RULES, fallback: R1_WHITE },
  go2: { rules: GO2_RULES, fallback: GO2_BODY },
  as2: { rules: AS2_RULES, fallback: AS2_SHELL },
};

/** The palette for a robot (spec name), or null to keep the model rgba. */
function paletteFor(robot: string): Palette | null {
  if (robot === "r1_air") return PALETTES.r1_air;
  if (robot.startsWith("r1")) return PALETTES.r1;     // r1, r1_air_edu (EDU body)
  if (robot.startsWith("go2")) return PALETTES.go2;
  if (robot.startsWith("as2")) return PALETTES.as2;
  return null;
}

export function hasPalette(robot: string): boolean {
  return paletteFor(robot) !== null;
}

/** Both sides of a limb are one paint part: left_knee_link -> knee,
 * thigh_mirror_0 -> thigh_0. */
export function partKey(meshName: string): string {
  return meshName
    .replace(/\.(stl|obj)$/i, "")
    .replace(/^(left|right)_/, "")
    .replace(/_link$/, "")
    .replace(/_mirror/, "");
}

/** Resolved appearance of one mesh (+ sub-mesh regions, already mirrored
 * for right-side meshes). */
export function resolve(robot: string, meshName: string): { paint: Paint; regions: Region[] } {
  const pal = paletteFor(robot);
  if (!pal) throw new Error(`no palette for ${robot}`);
  const key = partKey(meshName);
  const rule = pal.rules.find((r) => r.re.test(key));
  if (!rule) return { paint: pal.fallback, regions: [] };
  const right = /^right_/.test(meshName);
  const regions = (rule.regions ?? []).map((r): Region => right
    ? { ...r, test: (x, y, z, nx, ny, nz, sx, sy, sz) => r.test(x, -y, z, nx, -ny, nz, sx, sy, sz) }
    : r);
  return { paint: rule.paint, regions };
}
