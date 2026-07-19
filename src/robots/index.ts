/** Robot registry. G1 later: add a spec + model bundle, list it here. */
import type { RobotSpec } from "../core/types";
import { R1 } from "./r1";
import { R1_AIR } from "./r1_air";
import { GO2 } from "./go2";

export const SPECS: Record<string, RobotSpec> = {
  r1: R1,
  r1_air: R1_AIR,
  go2: GO2,
};

export function getSpec(robot: string): RobotSpec {
  const s = SPECS[robot];
  if (!s) throw new Error(`unknown robot '${robot}' — add it to src/robots/`);
  return s;
}
