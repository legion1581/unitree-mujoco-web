# policy-packs

Distributable **policy packs** (`.zip`) live here. Import them in the app via
**📦 Policy manager → Import pack / bundle (.zip)**. The repo ships none — the
actual `.zip` files are gitignored, because the weights are yours (extracted
from your own robot); this project is a neutral evaluator, not a policy
distributor.

## Pack format

A pack is a `.zip` containing:

```
pack.json                 manifest (see below)
config.yaml               the policy config (obs scheme, gains, family, robot)
checkpoints/<id>.onnx …   one or more weight checkpoints (SAME obs scheme)
traj.csv                  optional reference clip (motion-tracking families)
walker.onnx               optional aux low-level walker net (vision family)
```

All checkpoints in a pack share `config.yaml`; only the weights differ. Each
checkpoint shows up as its own selectable policy, and the whole pack is grouped
under `manifest.group` in the picker. The web runtime is ONNX-only — `.mnn`
weights must be converted to `.onnx` before packing.

`pack.json`:

```json
{
  "schemaVersion": 1,
  "group": "R1 motion-tracking · 24dof",
  "robot": "r1",
  "family": "motion_tracking",
  "baseName": "dances/gongfu",
  "config": "config.yaml",
  "traj": "traj.csv",
  "checkpoints": [
    { "id": "80k",  "file": "checkpoints/80k.onnx" },
    { "id": "120k", "file": "checkpoints/120k.onnx" }
  ]
}
```

- `robot` — `r1`, `r1_air` or `go2`. The r1 family shares one model; the app
  adopts the right spec from the selected policy automatically.
- `group` — the obs-scheme label the picker clusters policies under.
- `baseName` — the pack id and policy display name (`baseName@id` when a pack
  holds several checkpoints).

## Bundles (share a whole set)

A **bundle** is simply a `.zip` whose entries are pack `.zip`s — one file per
robot model (e.g. `r1-all.zip`, `go2-all.zip`). Upload it to any file host;
whoever downloads it imports the **one** file and gets every pack at once. The
importer auto-detects single pack vs bundle (a bundle has no top-level
`pack.json`).

Imported packs persist in the browser (IndexedDB); remove them any time from
the Policy manager.
