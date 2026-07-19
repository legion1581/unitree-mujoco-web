/** Three.js renderer for a compiled MjModel: build one mesh per geom from the
 * model arrays (primitives + STL mesh buffers), then each frame copy
 * geom_xpos/geom_xmat straight into the mesh matrices. Z-up throughout. */
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { Reflector } from "three/examples/jsm/objects/Reflector.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { GEOM_BOX, GEOM_CAPSULE, GEOM_CYLINDER, GEOM_ELLIPSOID, GEOM_MESH,
         GEOM_PLANE, GEOM_SPHERE } from "../sim/engine";
import type { RenderModel } from "../sim/protocol";
import { loadPaint, resolve, type PaintMap } from "./paint";

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

// ---- MuJoCo groundplane (matches the bundle's scene.xml) --------------
// texture: builtin checker rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
// mark="edge" markrgb="0.8 0.8 0.8", texrepeat="5 5" texuniform → 0.2 m tiles;
// material reflectance="0.2" → blend 20% mirror under an 80%-opaque checker.
const FLOOR_REFLECTANCE = 0.2;

function checkerTexture(): THREE.CanvasTexture {
  // one MuJoCo texture tile = 2x2 checker cells; mark="edge" draws the light
  // border around the TILE (not each cell), exactly like the builtin checker
  const px = 128;                       // one checker cell
  const cv = document.createElement("canvas");
  cv.width = cv.height = px * 2;
  const g = cv.getContext("2d")!;
  const c1 = "rgb(51,77,102)";          // rgb1 0.2 0.3 0.4
  const c2 = "rgb(26,51,77)";           // rgb2 0.1 0.2 0.3
  for (let i = 0; i < 2; i++)
    for (let j = 0; j < 2; j++) {
      g.fillStyle = (i + j) % 2 ? c2 : c1;
      g.fillRect(i * px, j * px, px, px);
    }
  g.strokeStyle = "rgb(204,204,204)";   // markrgb 0.8
  g.lineWidth = 3;
  g.strokeRect(1.5, 1.5, 2 * px - 3, 2 * px - 3);
  const tex = new THREE.CanvasTexture(cv);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  return tex;
}

/** Reflector (mirror pass) + semi-transparent checker on top ≈ MuJoCo's
 * reflectance blend. Returned as one Object3D the frame sync can position.
 * `?plain` in the URL skips the mirror pass (low-end machines / capture). */
function buildFloor(size: number): THREE.Object3D {
  const plain = new URLSearchParams(location.search).has("plain");
  const group = new THREE.Group();
  if (!plain) {
    const mirror = new Reflector(new THREE.PlaneGeometry(size, size), {
      clipBias: 0.003,
      textureWidth: 1024,
      textureHeight: 1024,
      color: 0x9099a0,
    });
    group.add(mirror);
  }
  const tex = checkerTexture();
  const tile = 0.2;                     // texrepeat="5 5" texuniform → 0.2 m tiles
  tex.repeat.set(size / tile, size / tile);
  const checker = new THREE.Mesh(
    new THREE.PlaneGeometry(size, size),
    new THREE.MeshStandardMaterial({
      map: tex, transparent: !plain, opacity: plain ? 1 : 1 - FLOOR_REFLECTANCE,
      roughness: 0.9, metalness: 0.0,
    }));
  checker.position.z = 0.001;
  group.add(checker);
  return group;
}

/** Per-mesh triangle trims — cosmetic geometry to hide from the render.
 * A triangle is dropped when the predicate is true for ALL three vertices
 * (mesh-local coordinates). Example — hide the torso's backpack mounting rail
 * (shell back face ~x=-0.06; rail spans x∈[-0.098,-0.068] in the STL):
 *   waist_yaw_link: (x) => x < -0.068,
 */
const MESH_TRIM: Record<string, (x: number, y: number, z: number) => boolean> = {};

/** Drop trimmed triangles from an indexed geometry (returns the kept index). */
function trimIndex(name: string, vert: Float32Array, face: Uint32Array): Uint32Array {
  const pred = MESH_TRIM[name];
  if (!pred) return face;
  const kept: number[] = [];
  for (let t = 0; t < face.length; t += 3) {
    let drop = true;
    for (let k = 0; k < 3; k++) {
      const i = face[t + k];
      if (!pred(vert[3 * i], vert[3 * i + 1], vert[3 * i + 2])) { drop = false; break; }
    }
    if (!drop) kept.push(face[t], face[t + 1], face[t + 2]);
  }
  return Uint32Array.from(kept);
}

export class SceneView {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene: THREE.Scene;
  readonly camera: THREE.PerspectiveCamera;
  readonly controls: OrbitControls;
  private geomMeshes: (THREE.Object3D | null)[] = [];
  /** mesh name per rendered robot-mesh geom (null for primitives/floor) */
  private geomMeshNames: (string | null)[] = [];
  private readonly tmp = new THREE.Matrix4();

  constructor(readonly container: HTMLElement) {
    // preserveDrawingBuffer: screenshots/recording read the canvas anytime
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0b0e12);
    this.scene.fog = new THREE.Fog(0x0b0e12, 8, 30);

    this.camera = new THREE.PerspectiveCamera(45, 1, 0.05, 100);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(2.2, -2.2, 1.4);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0, 0.7);
    this.controls.enableDamping = true;

    const hemi = new THREE.HemisphereLight(0xdfeaf5, 0x1a2028, 0.8);
    this.scene.add(hemi);
    const dir = new THREE.DirectionalLight(0xffffff, 1.5);
    dir.position.set(2, 1, 4);
    this.scene.add(dir);
    const fill = new THREE.DirectionalLight(0xcfd8e6, 0.7);   // camera-side fill
    fill.position.set(-2, -3, 2.5);
    this.scene.add(fill);
    // neutral studio environment map — gives the plastic its highlights
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    this.scene.environmentIntensity = 0.55;

    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
  }

  private resize() {
    const w = this.container.clientWidth || 1;
    const h = this.container.clientHeight || 1;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  /** (Re)build the per-geom meshes from a RenderModel sent by the sim worker. */
  build(desc: RenderModel) {
    for (const m of this.geomMeshes) {
      if (!m) continue;
      this.scene.remove(m);
      m.traverse((o) => {
        if (o instanceof THREE.Mesh) {
          o.geometry.dispose();
          (o.material as THREE.Material).dispose();
        }
      });
    }
    this.geomMeshes = [];
    this.geomMeshNames = [];
    for (let g = 0; g < desc.ngeom; g++) {
      const a = desc.rgba[4 * g + 3];
      const isMesh = desc.type[g] === GEOM_MESH;
      // skip hidden geoms, high-group collision shadows, and — on robot bodies
      // — every colliding geom: those are physics-only shapes (mesh duplicates,
      // the torso/feet collision boxes) that would poke through the shell.
      // Worldbody geoms (terrain) always render.
      if (a <= 0 || desc.group[g] >= 3 ||
          (desc.bodyid[g] > 0 && desc.contype[g] !== 0)) {
        this.geomMeshes.push(null); this.geomMeshNames.push(null); continue;
      }
      if (desc.type[g] === GEOM_PLANE) {
        const floor = buildFloor(40);
        floor.matrixAutoUpdate = false;
        this.scene.add(floor);
        this.geomMeshes.push(floor);
        this.geomMeshNames.push(null);
        continue;
      }
      const geo = this.buildGeometry(desc, g);
      if (!geo) { this.geomMeshes.push(null); this.geomMeshNames.push(null); continue; }
      const meshName = isMesh ? desc.meshes[desc.dataid[g]]?.name ?? "" : "";
      // The paint palette / painter page is R1-specific (mesh-name keyed). Only
      // R1-family meshes go through it; every other robot (Go2, later G1) uses
      // the model's own rgba, which already carries its correct colors.
      const painted = isMesh && desc.robot.startsWith("r1");
      // DoubleSide: several R1 linkage STLs have inverted winding and vanish
      // under back-face culling from outside
      const mat = painted
        ? new THREE.MeshPhysicalMaterial({ metalness: 0.0, side: THREE.DoubleSide })
        : new THREE.MeshStandardMaterial({
            color: new THREE.Color(desc.rgba[4 * g], desc.rgba[4 * g + 1], desc.rgba[4 * g + 2]),
            transparent: a < 1, opacity: a,
            metalness: 0.05, roughness: isMesh ? 0.6 : 0.8,
          });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.matrixAutoUpdate = false;
      mesh.userData.meshName = painted ? meshName : null;
      this.scene.add(mesh);
      this.geomMeshes.push(mesh);
      this.geomMeshNames.push(painted ? meshName : null);
    }
    this.applyPaint(loadPaint());
  }

  /** Robot mesh names present in the current scene (rendered geoms only). */
  partMeshNames(): string[] {
    return [...new Set(this.geomMeshNames.filter((n): n is string => n !== null))];
  }

  /** Apply default palette + user paint overrides to every robot mesh:
   * color, glossy/matte finish, visibility. Cheap — call on every change. */
  applyPaint(paint: PaintMap) {
    for (let g = 0; g < this.geomMeshes.length; g++) {
      const name = this.geomMeshNames[g];
      const obj = this.geomMeshes[g];
      if (!name || !obj || !(obj instanceof THREE.Mesh)) continue;
      const { color, finish, hidden } = resolve(name, paint);
      const m = obj.material as THREE.MeshPhysicalMaterial;
      m.color.set(color);
      const glossy = finish === "glossy";
      m.roughness = glossy ? 0.18 : 0.5;
      m.clearcoat = glossy ? 1.0 : 0.35;
      m.clearcoatRoughness = glossy ? 0.1 : 0.4;
      m.needsUpdate = true;
      obj.visible = !hidden;
    }
  }

  /** Raycast a pointer event to the robot part under the cursor (or null). */
  pickPart(ev: PointerEvent | MouseEvent): string | null {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((ev.clientX - rect.left) / rect.width) * 2 - 1,
      -((ev.clientY - rect.top) / rect.height) * 2 + 1);
    const ray = new THREE.Raycaster();
    ray.setFromCamera(ndc, this.camera);
    const meshes = this.geomMeshes.filter(
      (o): o is THREE.Mesh => o instanceof THREE.Mesh && o.visible && !!o.userData.meshName);
    const hit = ray.intersectObjects(meshes, false)[0];
    return hit ? (hit.object.userData.meshName as string) : null;
  }

  private buildGeometry(desc: RenderModel, g: number): THREE.BufferGeometry | null {
    const s0 = desc.size[3 * g], s1 = desc.size[3 * g + 1], s2 = desc.size[3 * g + 2];
    switch (desc.type[g]) {
      case GEOM_SPHERE: return new THREE.SphereGeometry(s0, 24, 16);
      case GEOM_ELLIPSOID: {
        const geo = new THREE.SphereGeometry(1, 24, 16);
        geo.scale(s0, s1, s2);
        return geo;
      }
      case GEOM_CAPSULE: {
        const geo = new THREE.CapsuleGeometry(s0, 2 * s1, 8, 16);
        geo.rotateX(Math.PI / 2);       // three capsule axis Y -> mujoco Z
        return geo;
      }
      case GEOM_CYLINDER: {
        const geo = new THREE.CylinderGeometry(s0, s0, 2 * s1, 24);
        geo.rotateX(Math.PI / 2);
        return geo;
      }
      case GEOM_BOX: return new THREE.BoxGeometry(2 * s0, 2 * s1, 2 * s2);
      case GEOM_MESH: {
        const m = desc.meshes[desc.dataid[g]];
        if (!m) return null;
        const geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.BufferAttribute(m.vert, 3));
        geo.setIndex(new THREE.BufferAttribute(trimIndex(m.name, m.vert, m.face), 1));
        geo.computeVertexNormals();
        return geo;
      }
      default: return null;            // hfield/sdf: not rendered yet
    }
  }

  /** Apply a frame of geom world transforms from the sim worker. */
  sync(xpos: Float32Array, xmat: Float32Array) {
    for (let g = 0; g < this.geomMeshes.length; g++) {
      const mesh = this.geomMeshes[g];
      if (!mesh) continue;
      const p = 3 * g, r = 9 * g;
      this.tmp.set(
        xmat[r], xmat[r + 1], xmat[r + 2], xpos[p],
        xmat[r + 3], xmat[r + 4], xmat[r + 5], xpos[p + 1],
        xmat[r + 6], xmat[r + 7], xmat[r + 8], xpos[p + 2],
        0, 0, 0, 1);
      mesh.matrix.copy(this.tmp);
    }
  }

  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  /** Keep the camera orbiting around the robot base. */
  follow(x: number, y: number, z: number) {
    const t = this.controls.target;
    const dx = x - t.x, dy = y - t.y;
    t.set(x, y, z);
    this.camera.position.x += dx;
    this.camera.position.y += dy;
  }
}
