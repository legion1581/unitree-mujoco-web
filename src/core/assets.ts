/** Asset helpers — URL resolution and reference-motion CSV loading.
 *
 * Paths inside pack configs are either repo-relative ("assets/models/…", for
 * the model bundles under public/) or absolute blob:/data:/http URLs (policy
 * and traj bytes from imported packs resolve to blob URLs — see src/packs). */

export function assetUrl(repoRelPath: string): string {
  // pass through already-absolute URLs (blob:/data:/http[s]:) so imported
  // policy packs can point policy/traj paths at in-memory blob URLs
  if (/^(blob:|data:|https?:)/.test(repoRelPath)) return repoRelPath;
  return "/" + repoRelPath.replace(/^\/+/, "");
}

export interface Traj { data: Float32Array; nRows: number; nCols: number }

/** Parse a reference-motion CSV into a flat Float32Array [nRows, nCols]. */
export async function loadTraj(repoRelPath: string): Promise<Traj> {
  const r = await fetch(assetUrl(repoRelPath));
  if (!r.ok) throw new Error(`failed to fetch traj ${repoRelPath}: ${r.status}`);
  const text = await r.text();
  const lines = text.split("\n").filter((l) => l.trim().length > 0);
  const first = lines[0].split(",").map(Number);
  const nCols = first.length;
  const nRows = lines.length;
  const data = new Float32Array(nRows * nCols);
  for (let i = 0; i < nRows; i++) {
    const parts = lines[i].split(",");
    for (let c = 0; c < nCols; c++) data[i * nCols + c] = Number(parts[c]);
  }
  return { data, nRows, nCols };
}
