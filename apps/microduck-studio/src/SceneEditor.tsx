import { useRef, useState } from "react";
import {
  ArrowCounterClockwise,
  Copy,
  Trash,
  Play,
  Cube,
  Sphere,
  Robot,
} from "@phosphor-icons/react";
import type { Scene, Obj } from "./types";
import { tr } from "./types";

export default function SceneEditor({
  scene,
  onChange,
  onPreview,
}: {
  scene: Scene;
  onChange: (s: Scene) => void;
  onPreview: () => void;
}) {
  const [selection, setSelection] = useState(scene.target_id);
  const [undo, setUndo] = useState<Scene[]>([]);
  const drag = useRef<string | null>(null);
  const svg = useRef<SVGSVGElement>(null);
  const selected =
    selection === "spawn"
      ? null
      : scene.objects.find((o) => o.id === selection);
  const save = (s: Scene, remember = true) => {
    if (remember) setUndo((a) => [...a.slice(-30), structuredClone(scene)]);
    onChange(s);
  };
  const patch = (values: Record<string, any>) =>
    save({
      ...scene,
      objects: scene.objects.map((o) =>
        o.id === selection ? { ...o, ...values } : o,
      ),
    });
  const add = (kind: string) => {
    const id = kind + "_" + Date.now().toString(36);
    save({
      ...scene,
      objects: [
        ...scene.objects,
        {
          id,
          kind,
          color: kind === "obstacle" ? "gray" : "blue",
          x: -0.7,
          y: -0.6,
          yaw: 0,
          width: kind === "obstacle" ? 0.22 : 0.08,
          depth: kind === "obstacle" ? 0.26 : 0.08,
          height: kind === "obstacle" ? 0.25 : 0.1,
          jitter: 0.12,
        },
      ],
    });
    setSelection(id);
  };
  const move = (event: React.PointerEvent<SVGSVGElement>) => {
    if (!drag.current || !svg.current) return;
    const pt = svg.current.createSVGPoint();
    pt.x = event.clientX;
    pt.y = event.clientY;
    const pos = pt.matrixTransform(svg.current.getScreenCTM()!.inverse());
    const values = {
      x: Math.round(pos.x * 100) / 100,
      y: -Math.round(pos.y * 100) / 100,
    };
    if (
      Math.abs(values.x) > scene.size / 2 - 0.1 ||
      Math.abs(values.y) > scene.size / 2 - 0.1
    )
      return;
    if (drag.current === "spawn")
      onChange({ ...scene, spawn: { ...scene.spawn, ...values } });
    else
      onChange({
        ...scene,
        objects: scene.objects.map((o) =>
          o.id === drag.current ? { ...o, ...values } : o,
        ),
      });
  };
  const start = (event: React.PointerEvent, id: string) => {
    event.stopPropagation();
    setSelection(id);
    drag.current = id;
    setUndo((a) => [...a.slice(-30), structuredClone(scene)]);
    svg.current?.setPointerCapture(event.pointerId);
  };
  const half = scene.size / 2;
  return (
    <div className="editor">
      <div className="editor-main">
        <div className="editor-toolbar">
          <div>
            <button onClick={() => add("ball")}>
              <Sphere />球
            </button>
            <button onClick={() => add("box")}>
              <Cube />
              盒子
            </button>
            <button onClick={() => add("toy")}>
              <Robot />
              玩偶
            </button>
            <button onClick={() => add("obstacle")}>
              <Cube weight="fill" />
              障碍
            </button>
          </div>
          <button
            aria-label="撤销"
            disabled={!undo.length}
            onClick={() => {
              onChange(undo.at(-1)!);
              setUndo((a) => a.slice(0, -1));
            }}
          >
            <ArrowCounterClockwise />
          </button>
        </div>
        <svg
          ref={svg}
          className="scene-canvas"
          aria-label="俯视场景编辑器"
          viewBox={`${-half - 0.15} ${-half - 0.15} ${scene.size + 0.3} ${scene.size + 0.3}`}
          onPointerMove={move}
          onPointerUp={() => (drag.current = null)}
          onPointerCancel={() => (drag.current = null)}
        >
          <defs>
            <pattern
              id="grid"
              width=".25"
              height=".25"
              patternUnits="userSpaceOnUse"
            >
              <path
                d="M .25 0 L 0 0 0 .25"
                fill="none"
                stroke="#dce3e7"
                strokeWidth=".004"
              />
            </pattern>
          </defs>
          <rect
            x={-half}
            y={-half}
            width={scene.size}
            height={scene.size}
            fill="#f7fafb"
            stroke="#95a9b1"
            strokeWidth=".025"
          />
          <rect
            x={-half}
            y={-half}
            width={scene.size}
            height={scene.size}
            fill="url(#grid)"
          />
          {scene.objects.map((o) => (
            <g
              key={o.id}
              transform={`translate(${o.x},${-o.y}) rotate(${-o.yaw})`}
              onPointerDown={(e) => start(e, o.id)}
              style={{ cursor: "grab" }}
            >
              {o.kind === "ball" || o.kind === "toy" ? (
                <circle
                  r={o.width / 2}
                  fill={
                    {
                      red: "#df635d",
                      blue: "#648ed2",
                      yellow: "#e0b446",
                      gray: "#7d8f9b",
                    }[o.color]
                  }
                  stroke={selection === o.id ? "#083c47" : "white"}
                  strokeWidth=".012"
                />
              ) : (
                <rect
                  x={-o.width / 2}
                  y={-o.depth / 2}
                  width={o.width}
                  height={o.depth}
                  rx=".01"
                  fill={
                    o.kind === "obstacle"
                      ? "#8999a7"
                      : { red: "#df635d", blue: "#648ed2", yellow: "#e0b446" }[
                          o.color
                        ]
                  }
                  stroke={selection === o.id ? "#083c47" : "white"}
                  strokeWidth=".012"
                />
              )}
              <text
                y={-o.depth / 2 - 0.07}
                textAnchor="middle"
                fontSize=".065"
                fill="#516670"
                transform={`rotate(${o.yaw})`}
              >
                {o.id === scene.target_id ? "目标 · " : ""}
                {tr(o.kind)}
              </text>
            </g>
          ))}
          <g
            transform={`translate(${scene.spawn.x},${-scene.spawn.y}) rotate(${-scene.spawn.yaw})`}
            onPointerDown={(e) => start(e, "spawn")}
            style={{ cursor: "grab" }}
          >
            <circle
              r=".09"
              fill="#0b4650"
              stroke={selection === "spawn" ? "#eea46f" : "white"}
              strokeWidth=".012"
            />
            <path d="M .16 0 L .075 -.045 L .075 .045 Z" fill="#0b4650" />
            <text
              y="-.15"
              textAnchor="middle"
              fontSize=".065"
              fill="#516670"
              transform={`rotate(${scene.spawn.yaw})`}
            >
              出生点
            </text>
          </g>
        </svg>
        <div className="canvas-footer">
          <span>俯视图 · 单位 m · 拖动物体调整布局</span>
          <button className="primary" onClick={onPreview}>
            <Play />
            三维预览
          </button>
        </div>
      </div>
      <aside className="properties">
        <div className="eyebrow">OBJECT PROPERTIES</div>
        <h3>
          {selection === "spawn"
            ? "机器人出生点"
            : selected
              ? tr(selected.kind)
              : "选择一个物体"}
        </h3>
        {(selected || selection === "spawn") && (
          <>
            {(["x", "y", "yaw"] as const).map((key) => (
              <label key={key}>
                {key === "yaw" ? "朝向 °" : key.toUpperCase() + " · m"}
                <input
                  type="number"
                  step={key === "yaw" ? 5 : 0.05}
                  value={
                    selection === "spawn" ? scene.spawn[key] : selected![key]
                  }
                  onChange={(e) =>
                    selection === "spawn"
                      ? save({
                          ...scene,
                          spawn: {
                            ...scene.spawn,
                            [key]: Number(e.target.value),
                          },
                        })
                      : patch({ [key]: Number(e.target.value) })
                  }
                />
              </label>
            ))}
          </>
        )}
        {selected && (
          <>
            {selected.kind !== "obstacle" && (
              <button
                className="target-select"
                disabled={
                  selected.id === scene.target_id || selected.color === "gray"
                }
                onClick={() => save({ ...scene, target_id: selected.id })}
              >
                {selected.id === scene.target_id
                  ? "当前任务目标"
                  : "设为任务目标"}
              </button>
            )}
            <div className="two-col">
              {(["width", "depth", "height"] as const).map((key, i) => (
                <label key={key}>
                  {["宽", "深", "高"][i]} · m
                  <input
                    type="number"
                    min=".04"
                    step=".01"
                    value={selected[key]}
                    onChange={(e) => patch({ [key]: Number(e.target.value) })}
                  />
                </label>
              ))}
            </div>
            <label>
              颜色
              <select
                value={selected.color}
                onChange={(e) => patch({ color: e.target.value })}
              >
                {["red", "blue", "yellow", "gray"].map((c) => (
                  <option key={c} value={c}>
                    {tr(c)}
                  </option>
                ))}
              </select>
            </label>
            <label>
              位置随机范围 ±m
              <input
                type="number"
                min="0"
                max="1"
                step=".01"
                value={selected.jitter}
                onChange={(e) => patch({ jitter: Number(e.target.value) })}
              />
            </label>
            <div className="button-row">
              <button
                onClick={() => {
                  const id =
                    selected.id + "_" + Math.random().toString(36).slice(2, 7);
                  save({
                    ...scene,
                    objects: [
                      ...scene.objects,
                      { ...selected, id, x: selected.x - 0.3 },
                    ],
                  });
                  setSelection(id);
                }}
              >
                <Copy />
                复制
              </button>
              <button
                disabled={selected.id === scene.target_id}
                onClick={() =>
                  save({
                    ...scene,
                    objects: scene.objects.filter((o) => o.id !== selection),
                  })
                }
              >
                <Trash />
                删除
              </button>
            </div>
          </>
        )}
        {selection === "spawn" && (
          <label>
            位置随机范围 ±m
            <input
              type="number"
              min="0"
              max=".5"
              step=".01"
              value={scene.spawn.jitter}
              onChange={(e) =>
                save({
                  ...scene,
                  spawn: { ...scene.spawn, jitter: Number(e.target.value) },
                })
              }
            />
          </label>
        )}
        <p className="hint">
          保存时检查物体重叠、出生区域、目标唯一性与可通行路径。随机范围单独保存。
        </p>
      </aside>
    </div>
  );
}
