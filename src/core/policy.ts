/** ONNX policy loading via onnxruntime-web (wasm backend).
 * Canonical format is ONNX — the same files the robot runs. Single-input
 * policies go through OnnxPolicy; the split-input families (locomotion 3-input,
 * mimic p_obs+cmd) use OnnxSession directly with named feeds.
 */
import * as ort from "onnxruntime-web";

let ortConfigured = false;
function configureOrt() {
  if (ortConfigured) return;
  // onnxruntime-web resolves to its self-contained bundle build (embedded
  // loader; the .wasm is located via import.meta.url) — do NOT set wasmPaths,
  // it would reroute through a dynamic import vite refuses to serve.
  ort.env.wasm.numThreads = 1;      // no nested ort workers; deterministic
  ortConfigured = true;
}

async function createSession(url: string): Promise<ort.InferenceSession> {
  configureOrt();
  const buf = await fetch(url).then((r) => {
    if (!r.ok) throw new Error(`failed to fetch policy ${url}: ${r.status}`);
    return r.arrayBuffer();
  });
  return ort.InferenceSession.create(buf, { executionProviders: ["wasm"] });
}

/** The model's own static shape for input `name` when it holds exactly `len` elements, else `dims`.
 * Shape-faithful conversions (scripts/mnn_graph_to_onnx.py) keep inputs like p_obs [1, 10, 93] or
 * obs [1, 5, 76]; drivers that pass a flat [1, N] still work because the data is contiguous. */
function declaredDims(sess: ort.InferenceSession, name: string, len: number, dims: number[]): number[] {
  const meta = (sess as unknown as { inputMetadata?: { name: string; shape?: (number | string)[] }[] })
    .inputMetadata?.find((m) => m.name === name);
  const shp = meta?.shape;
  if (!shp || !shp.length || !shp.every((d) => typeof d === "number" && d > 0)) return dims;
  const n = (shp as number[]).reduce((a, b) => a * b, 1);
  return n === len ? (shp as number[]) : dims;
}

/** Single-input policy: obs vector in, action vector out. */
export class OnnxPolicy {
  private constructor(
    readonly sess: ort.InferenceSession,
    readonly inName: string,
    readonly inDim: number,
  ) {}

  static async load(url: string): Promise<OnnxPolicy> {
    const sess = await createSession(url);
    const inName = sess.inputNames[0];
    // input dims aren't exposed pre-run by ort-web; callers know num_obs from cfg
    return new OnnxPolicy(sess, inName, -1);
  }

  async run(obs: Float32Array): Promise<Float32Array> {
    const feeds: Record<string, ort.Tensor> = {
      [this.inName]: new ort.Tensor("float32", obs, declaredDims(this.sess, this.inName, obs.length, [1, obs.length])),
    };
    const out = await this.sess.run(feeds);
    const first = out[this.sess.outputNames[0]];
    return first.data as Float32Array;
  }
}

/** Multi-input session with named feeds (shapes supplied per call). */
export class OnnxSession {
  private constructor(readonly sess: ort.InferenceSession) {}

  static async load(url: string): Promise<OnnxSession> {
    return new OnnxSession(await createSession(url));
  }

  get inputNames(): readonly string[] { return this.sess.inputNames; }

  async run(feeds: Record<string, { data: Float32Array; dims: number[] }>):
      Promise<Float32Array> {
    const f: Record<string, ort.Tensor> = {};
    for (const name of this.sess.inputNames) {
      const t = feeds[name];
      if (!t) throw new Error(`missing ONNX input '${name}'`);
      f[name] = new ort.Tensor("float32", t.data, declaredDims(this.sess, name, t.data.length, t.dims));
    }
    const out = await this.sess.run(f);
    return out[this.sess.outputNames[0]].data as Float32Array;
  }

  /** Every output by name (multi-output nets, e.g. the AS2 mimic nets: actions + reference joint / body state). */
  async runAll(feeds: Record<string, { data: Float32Array; dims: number[] }>): Promise<Record<string, Float32Array>> {
    const f: Record<string, ort.Tensor> = {};
    for (const name of this.sess.inputNames) {
      const t = feeds[name];
      if (!t) throw new Error(`missing ONNX input '${name}'`);
      f[name] = new ort.Tensor("float32", t.data, declaredDims(this.sess, name, t.data.length, t.dims));
    }
    const out = await this.sess.run(f);
    return Object.fromEntries(this.sess.outputNames.map((n) => [n, out[n].data as Float32Array]));
  }
}
