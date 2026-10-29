import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { Scene } from "./types";
import { api, media } from "./types";

export type Visual = {
  sha256: string;
  geoms: any[];
  meshes: Record<string, { vertices: number[]; faces: number[] }>;
};
export type Pose = {
  episode_id: string;
  tick: number;
  captured_at: number;
  sim_time_s: number;
  manifest_sha256: string;
  control_source: string;
  geoms: { id: number; position: number[]; rotation: number[] }[];
};
const colors: Record<string, number> = {
  red: 0xe50909,
  blue: 0x0a29f2,
  yellow: 0xf2cc0a,
  gray: 0x666e75,
};

function dispose(group: THREE.Object3D) {
  group.traverse((o) => {
    if (o instanceof THREE.Mesh || o instanceof THREE.LineSegments) {
      o.geometry.dispose();
      (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) =>
        m.dispose(),
      );
    }
  });
  group.clear();
}
function geometry(g: any, visual: Visual) {
  const s = g.size;
  switch (g.type) {
    case 0:
      return new THREE.PlaneGeometry(20, 20);
    case 2:
      return new THREE.SphereGeometry(s[0], 24, 16);
    case 3:
      return new THREE.CapsuleGeometry(s[0], s[1] * 2, 8, 20).rotateX(
        Math.PI / 2,
      );
    case 4:
      return new THREE.SphereGeometry(1, 24, 16).scale(
        ...(s as [number, number, number]),
      );
    case 5:
      return new THREE.CylinderGeometry(s[0], s[0], s[1] * 2, 24).rotateX(
        Math.PI / 2,
      );
    case 6:
      return new THREE.BoxGeometry(s[0] * 2, s[1] * 2, s[2] * 2);
    case 7: {
      const mesh = visual.meshes[String(g.mesh)],
        b = new THREE.BufferGeometry();
      b.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(mesh.vertices, 3),
      );
      b.setIndex(mesh.faces);
      b.computeVertexNormals();
      return b;
    }
    default:
      throw Error(`Unsupported MuJoCo geometry ${g.type}`);
  }
}
export function applyPose(group: THREE.Group, pose: Pose) {
  const byId = new Map(pose.geoms.map((g) => [g.id, g]));
  for (const child of group.children) {
    const p = byId.get(child.userData.geomId);
    if (!p) continue;
    const r = p.rotation,
      v = p.position;
    child.matrixAutoUpdate = false;
    child.matrix.set(
      r[0],
      r[1],
      r[2],
      v[0],
      r[3],
      r[4],
      r[5],
      v[1],
      r[6],
      r[7],
      r[8],
      v[2],
      0,
      0,
      0,
      1,
    );
    child.matrixWorldNeedsUpdate = true;
  }
}
function compiled(visual: Visual, wire = false) {
  const group = new THREE.Group();
  for (const g of visual.geoms) {
    const material = new THREE.MeshStandardMaterial({
      color: new THREE.Color(
        ...(g.rgba.slice(0, 3) as [number, number, number]),
      ),
      opacity: g.rgba[3],
      transparent: g.rgba[3] < 1,
      roughness: 0.72,
      metalness: 0.05,
      wireframe: wire && g.collision,
      side: THREE.DoubleSide,
    });
    const m = new THREE.Mesh(geometry(g, visual), material);
    m.userData.geomId = g.id;
    m.name = g.name;
    m.castShadow = g.type !== 0;
    m.receiveShadow = true;
    group.add(m);
  }
  return group;
}
type Props = {
  scene?: Scene;
  catalog?: any;
  robot?: { visual: Visual; state: Pose };
  visual?: Visual;
  pose?: Pose;
  selection?: string;
  top?: boolean;
  collision?: boolean;
  onSelect?: (id: string) => void;
  onDragStart?: () => void;
  onMove?: (id: string, x: number, y: number) => void;
};

export default function WorldViewport(props: Props) {
  const host = useRef<HTMLDivElement>(null),
    latest = useRef(props);
  latest.current = props;
  const runtime = useRef<any>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    if (!host.current) return;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    } catch {
      setError("当前浏览器无法启动 WebGL，请启用硬件加速。");
      return;
    }
    const gl = renderer.getContext();
    const info = gl.getExtension("WEBGL_debug_renderer_info");
    const software =
      !!info &&
      /swiftshader|llvmpipe|softpipe/i.test(
        String(gl.getParameter(info.UNMASKED_RENDERER_WEBGL)),
      );
    renderer.setPixelRatio(software ? 1 : Math.min(devicePixelRatio, 2));
    // Keep exact meshes/poses on software WebGL; omit costly shadow passes.
    renderer.shadowMap.enabled = !software;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    const world = new THREE.Scene();
    world.background = new THREE.Color("#edf0ec");
    const camera = new THREE.PerspectiveCamera(42, 1, 0.015, 80);
    camera.up.set(0, 0, 1);
    camera.position.set(3.4, -4.8, 4.4);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0.1);
    controls.enableDamping = true;
    controls.maxPolarAngle = Math.PI * 0.495;
    controls.minDistance = 0.4;
    controls.maxDistance = 14;
    world.add(new THREE.HemisphereLight(0xffffff, 0x77877e, 2.4));
    const sun = new THREE.DirectionalLight(0xfff4df, 3);
    sun.position.set(-2, -3, 7);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.camera.left = -4;
    sun.shadow.camera.right = 4;
    sun.shadow.camera.top = 4;
    sun.shadow.camera.bottom = -4;
    sun.shadow.normalBias = 0.015;
    world.add(sun);
    const objects = new THREE.Group();
    world.add(objects);
    const element = renderer.domElement;
    host.current.append(element);
    const resize = new ResizeObserver(() => {
      const { width, height } = host.current!.getBoundingClientRect();
      renderer.setSize(width, height);
      camera.aspect = width / Math.max(height, 1);
      camera.updateProjectionMatrix();
      if (runtime.current) runtime.current.needsRender = true;
    });
    resize.observe(host.current);
    const ray = new THREE.Raycaster(),
      plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
    let drag: string | null = null;
    const cursor = (e: PointerEvent) => {
      const b = element.getBoundingClientRect();
      ray.setFromCamera(
        new THREE.Vector2(
          ((e.clientX - b.left) / b.width) * 2 - 1,
          (-(e.clientY - b.top) / b.height) * 2 + 1,
        ),
        camera,
      );
    };
    const down = (e: PointerEvent) => {
      if (e.button !== 0 || !latest.current.scene || !latest.current.onMove)
        return;
      cursor(e);
      const hit = ray.intersectObjects(objects.children, true).find((h) => {
        let o: THREE.Object3D | null = h.object;
        while (o) {
          if (o.userData.objectId) return true;
          o = o.parent;
        }
        return false;
      });
      if (!hit) return;
      let o: THREE.Object3D | null = hit.object;
      while (o && !o.userData.objectId) o = o.parent;
      drag = o!.userData.objectId;
      latest.current.onSelect?.(drag!);
      latest.current.onDragStart?.();
      controls.enabled = false;
      element.setPointerCapture(e.pointerId);
    };
    const move = (e: PointerEvent) => {
      if (!drag) return;
      cursor(e);
      const p = new THREE.Vector3();
      if (ray.ray.intersectPlane(plane, p)) {
        const max = (latest.current.scene?.size || 3) / 2 - 0.08;
        latest.current.onMove?.(
          drag,
          Math.round(100 * Math.max(-max, Math.min(max, p.x))) / 100,
          Math.round(100 * Math.max(-max, Math.min(max, p.y))) / 100,
        );
      }
    };
    const up = () => {
      drag = null;
      controls.enabled = true;
    };
    element.addEventListener("pointerdown", down);
    element.addEventListener("pointermove", move);
    element.addEventListener("pointerup", up);
    element.addEventListener("pointercancel", up);
    runtime.current = {
      world,
      objects,
      camera,
      controls,
      renderer,
      needsRender: true,
    };
    let frame = 0;
    const draw = () => {
      const moved = controls.update();
      // Static editing and ended runs do not continuously redraw all official
      // meshes. New 10Hz poses, transforms, resizing and orbiting invalidate it.
      if (moved || runtime.current?.needsRender) {
        renderer.render(world, camera);
        runtime.current.needsRender = false;
      }
      frame = requestAnimationFrame(draw);
    };
    draw();
    return () => {
      cancelAnimationFrame(frame);
      resize.disconnect();
      controls.dispose();
      dispose(objects);
      renderer.dispose();
      element.remove();
      runtime.current = null;
    };
  }, []);
  useEffect(() => {
    const rt = runtime.current;
    if (!rt) return;
    const { camera, controls } = rt;
    rt.needsRender = true;
    if (props.top) {
      camera.position.set(0, -0.001, (props.scene?.size || 4) * 1.7);
      controls.enableRotate = false;
    } else {
      camera.position.set(3.4, -4.8, 4.4);
      controls.enableRotate = true;
    }
    controls.target.set(0, 0, 0.1);
    controls.update();
  }, [props.top]);
  useEffect(() => {
    const rt = runtime.current;
    if (!rt) return;
    const root: THREE.Group = rt.objects;
    rt.needsRender = true;
    const shapeKey = props.scene
      ? JSON.stringify({
          size: props.scene.size,
          target_id: props.scene.target_id,
          objects: props.scene.objects.map(({ x, y, yaw, ...o }) => o),
          selection: props.selection,
          collision: props.collision,
          robot: props.robot?.visual.sha256,
          assets: !!props.catalog,
        })
      : null;
    if (shapeKey && rt.shapeKey === shapeKey && props.scene) {
      for (const child of root.children) {
        const id = child.userData.objectId;
        const obj =
          id === "spawn"
            ? props.scene.spawn
            : props.scene.objects.find((o) => o.id === id);
        if (obj) {
          child.position.set(obj.x, obj.y, 0);
          child.rotation.z = (obj.yaw * Math.PI) / 180;
        }
        if (child.userData.targetId) {
          const t = props.scene.objects.find(
            (o) => o.id === child.userData.targetId,
          );
          if (t) child.position.set(t.x, t.y, 0.005);
        }
      }
      root.updateMatrixWorld(true);
      for (const child of root.children)
        if (child instanceof THREE.Box3Helper) {
          const obj = root.children.find(
            (c) => c.userData.objectId === child.userData.outlineId,
          );
          if (obj) child.box.setFromObject(obj);
        }
      return;
    }
    rt.shapeKey = shapeKey;
    dispose(root);
    if (props.visual) {
      const g = compiled(props.visual, props.collision);
      root.add(g);
      rt.compiled = g;
      if (props.pose) applyPose(g, props.pose);
      return;
    }
    const s = props.scene;
    if (!s) return;
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(s.size, s.size),
      new THREE.MeshStandardMaterial({ color: 0xf6f5ed, roughness: 1 }),
    );
    floor.receiveShadow = true;
    floor.position.z = -0.002;
    root.add(floor);
    const grid = new THREE.GridHelper(
      s.size,
      Math.round(s.size * 4),
      0xbac4b8,
      0xd9dfd3,
    );
    grid.rotation.x = Math.PI / 2;
    grid.position.z = 0.001;
    root.add(grid);
    const add = (
      group: THREE.Group,
      geo: THREE.BufferGeometry,
      pos: number[],
      color: THREE.Color | number,
    ) => {
      const m = new THREE.Mesh(
        geo,
        new THREE.MeshStandardMaterial({
          color,
          roughness: 0.75,
          wireframe: props.collision,
        }),
      );
      m.position.fromArray(pos);
      m.castShadow = true;
      m.receiveShadow = true;
      group.add(m);
    };
    for (const o of s.objects) {
      const group = new THREE.Group();
      group.userData.objectId = o.id;
      group.position.set(o.x, o.y, 0);
      group.rotation.z = (o.yaw * Math.PI) / 180;
      root.add(group);
      if (o.asset_id) {
        const a = props.catalog?.assets.find((a: any) => a.id === o.asset_id);
        const dims = [o.width, o.depth, o.height];
        const rgba = props.catalog?.materials[o.material || "wood"].rgba || [
          0.6, 0.45, 0.3,
        ];
        for (const [size, pos] of a?.parts || [])
          add(
            group,
            new THREE.BoxGeometry(
              size[0] * dims[0],
              size[1] * dims[1],
              size[2] * dims[2],
            ),
            pos.map((v: number, i: number) => v * dims[i]),
            new THREE.Color(...(rgba.slice(0, 3) as [number, number, number])),
          );
      } else if (o.kind === "ball") {
        const radius = Math.max(o.width, o.depth) / 2;
        add(
          group,
          new THREE.SphereGeometry(radius, 24, 16),
          [0, 0, radius],
          colors[o.color],
        );
      } else if (o.kind === "toy") {
        add(
          group,
          new THREE.SphereGeometry(1, 24, 16).scale(
            o.width / 2,
            o.depth / 2,
            o.height * 0.34,
          ),
          [0, 0, o.height * 0.34],
          colors[o.color],
        );
        add(
          group,
          new THREE.SphereGeometry(Math.min(o.width, o.depth) * 0.32, 24, 16),
          [0, 0, o.height * 0.78],
          colors[o.color],
        );
        for (const sign of [-1, 1])
          add(
            group,
            new THREE.SphereGeometry(o.width * 0.12, 16, 12),
            [sign * o.width * 0.25, 0, o.height * 0.98],
            colors[o.color],
          );
      } else
        add(
          group,
          new THREE.BoxGeometry(o.width, o.depth, o.height),
          [0, 0, o.height / 2],
          colors[o.color],
        );
      if (props.selection === o.id) {
        const edge = new THREE.Box3Helper(
          new THREE.Box3().setFromObject(group),
          0x167c82,
        );
        edge.userData.outlineId = o.id;
        root.add(edge);
      }
      if (o.id === s.target_id) {
        const ring = new THREE.Mesh(
          new THREE.RingGeometry(0.13, 0.145, 48),
          new THREE.MeshBasicMaterial({
            color: 0x167c82,
            side: THREE.DoubleSide,
          }),
        );
        ring.position.set(o.x, o.y, 0.005);
        ring.userData.targetId = o.id;
        root.add(ring);
      }
    }
    const spawn = new THREE.Group();
    spawn.userData.objectId = "spawn";
    spawn.position.set(s.spawn.x, s.spawn.y, 0);
    spawn.rotation.z = (s.spawn.yaw * Math.PI) / 180;
    if (props.robot) {
      const g = compiled(props.robot.visual);
      applyPose(g, props.robot.state);
      spawn.add(g);
    }
    const marker = new THREE.Mesh(
      new THREE.RingGeometry(0.16, 0.18, 48),
      new THREE.MeshBasicMaterial({ color: 0xd79360, side: THREE.DoubleSide }),
    );
    marker.position.z = 0.005;
    spawn.add(marker);
    const arrow = new THREE.ArrowHelper(
      new THREE.Vector3(1, 0, 0),
      new THREE.Vector3(0, 0, 0.01),
      0.33,
      0xd79360,
      0.06,
      0.04,
    );
    spawn.add(arrow);
    root.add(spawn);
  }, [
    props.scene,
    props.catalog,
    props.robot,
    props.visual,
    props.selection,
    props.collision,
  ]);
  useEffect(() => {
    if (props.pose && runtime.current?.compiled) {
      const group = runtime.current.compiled as THREE.Group;
      runtime.current.needsRender = true;
      applyPose(group, props.pose);
      group.updateMatrixWorld(true);
      const positions = new Map(
        props.pose.geoms.map((g) => [g.id, g.position]),
      );
      let error = 0;
      for (const child of group.children) {
        const expected = positions.get(child.userData.geomId);
        if (!expected) continue;
        const p = new THREE.Vector3();
        child.getWorldPosition(p);
        error = Math.max(
          error,
          ...p.toArray().map((x, i) => Math.abs(x - expected[i])),
        );
      }
      if (host.current) {
        host.current.dataset.positionErrorM = String(error);
        host.current.dataset.tick = String(props.pose.tick);
      }
    }
  }, [props.pose, props.visual]);
  return (
    <div
      className="world-viewport"
      ref={host}
      role="img"
      aria-label={
        props.scene ? "可拖动的三维场景编辑器" : "官方 MuJoCo 仿真位姿"
      }
    >
      {error && <div className="viewport-message">{error}</div>}
    </div>
  );
}

export function SensorCamera({
  runId,
  active,
  path,
}: {
  runId: string;
  active: boolean;
  path: string;
}) {
  const [attempt, setAttempt] = useState(0);
  const image = useRef<HTMLImageElement>(null);
  const retry = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  useEffect(() => {
    setAttempt(0);
    // Chromium can end a multipart image with zero naturalWidth without an
    // error event when its HTTP stream is interrupted during a Web restart.
    const watchdog = active
      ? setInterval(() => {
          if (image.current && image.current.naturalWidth === 0)
            setAttempt((n) => n + 1);
        }, 2000)
      : undefined;
    return () => {
      clearTimeout(retry.current);
      clearInterval(watchdog);
    };
  }, [runId, active, path]);
  return (
    <img
      ref={image}
      src={
        active
          ? `/api/runs/${runId}/stream/head?connection=${attempt}`
          : media(path)
      }
      alt="头部摄像头观察"
      onError={() => {
        if (!active) return;
        clearTimeout(retry.current);
        retry.current = setTimeout(() => setAttempt((n) => n + 1), 1500);
      }}
    />
  );
}

export function SimulationViewport({
  runId,
  path,
  active,
}: {
  runId: string;
  path: string;
  active: boolean;
}) {
  const [visual, setVisual] = useState<Visual>(),
    [pose, setPose] = useState<Pose>(),
    [connected, setConnected] = useState(false),
    [now, setNow] = useState(Date.now());
  const visualHash = useRef("");
  useEffect(() => {
    setPose(undefined);
    setVisual(undefined);
    visualHash.current = "";
    const events = new EventSource(`/api/runs/${runId}/state-stream`);
    let alive = true;
    events.addEventListener("state", (e) => {
      const p = JSON.parse((e as MessageEvent).data) as Pose;
      setPose(p);
      setConnected(true);
      if (visualHash.current !== p.manifest_sha256) {
        visualHash.current = p.manifest_sha256;
        api<Visual>(`/api/runs/${runId}/visual`)
          .then((v) => {
            if (alive && visualHash.current === v.sha256) setVisual(v);
          })
          .catch(() => {
            if (visualHash.current === p.manifest_sha256)
              visualHash.current = "";
          });
      }
    });
    events.onerror = () => setConnected(false);
    events.addEventListener("ended", () => {
      setConnected(false);
      events.close();
    });
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => {
      alive = false;
      events.close();
      clearInterval(t);
    };
  }, [runId, path, active]);
  const fresh =
    active && connected && pose && now - pose.captured_at * 1000 < 2500;
  return (
    <figure className="simulation-viewport">
      {visual && pose && visual.sha256 === pose.manifest_sha256 ? (
        <WorldViewport visual={visual} pose={pose} />
      ) : (
        <img
          src={
            active
              ? `/api/runs/${runId}/stream/observer`
              : media(path + "/observer.jpg")
          }
          alt="MuJoCo 原始观察者画面"
        />
      )}
      <figcaption>
        <span className={fresh ? "live-dot" : ""} />
        {fresh
          ? "仿真位姿正在更新"
          : active
            ? "等待位姿 / 连接中断 · 画面冻结"
            : "已结束 · 最后记录状态"}
        {pose &&
          ` · tick ${pose.tick} · ${new Date(pose.captured_at * 1000).toLocaleTimeString()}`}
        <span>拖动旋转 · 滚轮缩放</span>
      </figcaption>
    </figure>
  );
}
