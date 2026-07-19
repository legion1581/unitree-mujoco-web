/** Robot paint: default palette + user overrides (color / hide / finish).
 *
 * Defaults model the shipping White colorway. Overrides are keyed by PART —
 * a mesh name with the left_/right_ prefix and _link suffix stripped, so both
 * sides paint together — and persist in localStorage. The painter page writes
 * them; the sim reads them (live, via the storage event, across tabs).
 */

export const SHELL = 0xf0eeea;      // white shell panels
export const DARK = 0x26282c;       // charcoal joint modules / structure
export const MID = 0x8d9096;        // mid-gray mechanism parts
export const VISOR = 0x101114;      // glossy black face / display

const PALETTE: [RegExp, number][] = [
  [/head_pitch/, VISOR],               // face / display
  [/head_yaw/, DARK],                  // head unit / neck
  [/pelvis|waist_roll/, DARK],         // hip girdle
  [/hip_(pitch|roll)/, DARK],          // hip actuator modules
  [/shoulder_(pitch|roll)/, DARK],     // shoulder actuator modules
  [/wrist_roll/, DARK],                // hands
  [/ankle_(A|B|constraint)/, MID],     // ankle 4-bar linkage rods
  [/ankle_pitch/, DARK],               // ankle actuators
  [/ankle_roll/, DARK],                // feet / soles
  [/imu|logo/, MID],
];

export type Finish = "glossy" | "matte";

export interface PaintEntry {
  color?: string;      // "#rrggbb"
  hidden?: boolean;
  finish?: Finish;
}

export type PaintMap = Record<string, PaintEntry>;

/** The default R1 skin (baked from a hand-tuned painter session): white head
 * and hip girdle, navy shoulder caps + waist ring, imu puck hidden. Layering:
 * user paint > DEFAULT_PAINT > base palette. */
export const DEFAULT_PAINT: PaintMap = {
  head_yaw: { color: "#f0eeea", finish: "glossy" },
  head_pitch: { color: "#f0eeea" },
  hip_pitch: { color: "#f0eeea", finish: "glossy" },
  hip_roll: { color: "#f0eeea", finish: "glossy" },
  pelvis: { color: "#f0eeea", finish: "glossy" },
  shoulder_roll: { color: "#f0eeea", finish: "glossy" },
  shoulder_pitch: { color: "#162f5f" },
  waist_roll: { color: "#162f5f" },
  imu_in_pelvis: { hidden: true },
};

const STORAGE_KEY = "r1-paint";

/** Both sides of a limb are one paint part: left_knee_link -> knee. */
export function partKey(meshName: string): string {
  return meshName
    .replace(/\.stl$/i, "")
    .replace(/^(left|right)_/, "")
    .replace(/_link$/, "");
}

/** Friendlier names for the part list. */
export const PART_LABELS: Record<string, string> = {
  waist_yaw: "torso", head_pitch: "face", head_yaw: "head",
  waist_roll: "waist", pelvis: "pelvis", imu_in_pelvis: "imu",
};

export function partLabel(key: string): string {
  return (PART_LABELS[key] ?? key).replace(/_/g, " ");
}

export function defaultColor(meshName: string): number {
  for (const [re, c] of PALETTE) if (re.test(meshName)) return c;
  return SHELL;
}

export function defaultFinish(color: number): Finish {
  return color === SHELL || color === VISOR ? "glossy" : "matte";
}

export function colorHex(c: number): string {
  return "#" + c.toString(16).padStart(6, "0");
}

export function loadPaint(): PaintMap {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as PaintMap) : {};
  } catch { return {}; }
}

export function savePaint(p: PaintMap) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(p)); } catch { /* private mode */ }
}

export function clearPaint() {
  try { localStorage.removeItem(STORAGE_KEY); } catch { /* private mode */ }
}

/** Resolved appearance of one mesh: user paint > DEFAULT_PAINT > palette. */
export function resolve(meshName: string, paint: PaintMap):
    { color: number; finish: Finish; hidden: boolean } {
  const key = partKey(meshName);
  const d = defaultColor(meshName);
  const base = DEFAULT_PAINT[key];
  const e = paint[key];
  const colorStr = e?.color ?? base?.color;
  return {
    color: colorStr ? parseInt(colorStr.slice(1), 16) : d,
    finish: e?.finish ?? base?.finish ?? defaultFinish(d),
    hidden: e?.hidden ?? base?.hidden ?? false,
  };
}
