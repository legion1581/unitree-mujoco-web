/** Robot registry. G1 later: add a spec + model bundle, list it here. */
import type { RobotSpec } from "../core/types";
import { R1 } from "./r1";
import { R1_AIR, R1_AIR_EDU } from "./r1_air";
import { GO2 } from "./go2";
import { G1 } from "./g1";
import { G1_AIR } from "./g1_air";
import { AS2 } from "./as2";

export const SPECS: Record<string, RobotSpec> = {
  r1: R1,
  r1_air: R1_AIR,       // 20-DOF Air model; auto-selected for Air-only packs (same UI slot as r1)
  r1_air_edu: R1_AIR_EDU,   // a 20-DOF pack on the 24-DOF EDU model (packs declaring models incl. "edu")
  go2: GO2,
  g1: G1,
  g1_air: G1_AIR,       // 23-DOF Air model; auto-selected for Air-only packs (same UI slot as g1)
  as2: AS2,
};

export function getSpec(robot: string): RobotSpec {
  const s = SPECS[robot];
  if (!s) throw new Error(`unknown robot '${robot}' — add it to src/robots/`);
  return s;
}
