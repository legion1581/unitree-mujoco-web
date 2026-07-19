// Copies the mujoco wasm runtime out of node_modules into public/wasm/ so the
// emscripten glue can be dynamic-imported at a stable URL (vite must not
// transform it — its pthread worker bootstrap doesn't survive bundling).
// onnxruntime-web needs no copying: its bundle build is self-contained.
// Runs on postinstall; safe to re-run.
import { copyFileSync, mkdirSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const nm = join(root, "node_modules");

function copy(src, dst) {
  mkdirSync(dirname(dst), { recursive: true });
  copyFileSync(src, dst);
  console.log(`  ${src.replace(nm + "/", "")} -> ${dst.replace(root + "/", "")}`);
}

const mjDir = join(nm, "mujoco");
if (existsSync(mjDir)) {
  copy(join(mjDir, "mujoco.wasm"), join(root, "public", "wasm", "mujoco.wasm"));
  copy(join(mjDir, "mujoco.js"), join(root, "public", "wasm", "mujoco.js"));
} else console.warn("mujoco package not found — run npm install first");
