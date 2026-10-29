import { useEffect, useRef, useState } from "react";
import {
  ArrowCounterClockwise,
  ArrowClockwise,
  Copy,
  Trash,
  Play,
  Cube,
  Sphere,
  Robot,
  Armchair,
  Table,
  Sparkle,
  DownloadSimple,
  UploadSimple,
  SquaresFour,
  Compass,
} from "@phosphor-icons/react";
import type { Scene, Skill, Run } from "./types";
import { api, tr, requestId } from "./types";
import WorldViewport from "./WorldViewport";

export default function SceneEditor({
  scene,
  onChange,
  onPreview,
  onSave,
}: {
  scene: Scene;
  onChange: (s: Scene) => void;
  onPreview: () => void;
  onSave: () => Promise<Skill | undefined>;
}) {
  const [selection, setSelection] = useState(scene.target_id),
    [undo, setUndo] = useState<Scene[]>([]),
    [redo, setRedo] = useState<Scene[]>([]);
  const [catalog, setCatalog] = useState<any>(),
    [robot, setRobot] = useState<any>(),
    [top, setTop] = useState(false),
    [collision, setCollision] = useState(false);
  const [error, setError] = useState(""),
    [robotError, setRobotError] = useState(""),
    [status, setStatus] = useState<any>(),
    [prompt, setPrompt] = useState("");
  const [draft, setDraft] = useState<Run>(),
    [previewDraft, setPreviewDraft] = useState(false),
    [busy, setBusy] = useState(false);
  const draftBase = useRef(""),
    file = useRef<HTMLInputElement>(null);
  useEffect(() => {
    api("/api/assets")
      .then(setCatalog)
      .catch((e) => setError(e.message));
    api("/api/assets/robot")
      .then(setRobot)
      .catch((e) => setRobotError(e.message));
    api("/api/authoring/status")
      .then(setStatus)
      .catch(() => {});
  }, []);
  useEffect(() => {
    if (!draft || !["queued", "running"].includes(draft.status)) return;
    const timer = setInterval(
      () =>
        api<Run>(`/api/runs/${draft.id}`)
          .then(setDraft)
          .catch((e) => setError(e.message)),
      1800,
    );
    return () => clearInterval(timer);
  }, [draft?.id, draft?.status]);
  const selected = scene.objects.find((o) => o.id === selection);
  const remember = () => {
    setUndo((a) => [...a.slice(-49), structuredClone(scene)]);
    setRedo([]);
    setPreviewDraft(false);
  };
  const save = (s: Scene) => {
    remember();
    onChange({ ...s, version: 2 });
  };
  const patch = (v: Record<string, any>) =>
    save(
      selection === "spawn"
        ? { ...scene, spawn: { ...scene.spawn, ...v } }
        : {
            ...scene,
            objects: scene.objects.map((o) =>
              o.id === selection ? { ...o, ...v } : o,
            ),
          },
    );
  const attempt = async (fn: () => Promise<any>) => {
    setError("");
    setBusy(true);
    try {
      await fn();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const add = (kind: string, asset?: any) => {
    const id = (asset?.id || kind) + "_" + Date.now().toString(36);
    const [w, d, h] =
      asset?.dimensions ||
      (kind === "obstacle" ? [0.22, 0.26, 0.25] : [0.08, 0.08, 0.1]);
    save({
      ...scene,
      objects: [
        ...scene.objects,
        {
          id,
          kind,
          color: kind === "obstacle" ? "gray" : "blue",
          x: -0.65,
          y: -0.65,
          yaw: 0,
          width: w,
          depth: d,
          height: h,
          jitter: asset ? 0 : 0.12,
          ...(asset
            ? {
                asset_id: asset.id,
                asset_version: asset.version,
                material: asset.material,
              }
            : {}),
        },
      ],
    });
    setSelection(id);
  };
  const selectionValue = selection === "spawn" ? scene.spawn : selected;
  const draftChanged = !!draft && draftBase.current !== JSON.stringify(scene);
  return (
    <div className="scene-workshop">
      <div className="scene-heading">
        <div>
          <div className="eyebrow">WORLD BUILDER</div>
          <h2>给技能一个可以探索的世界</h2>
          <p>摆放物体，调整空间，再让鸭子走进去。</p>
        </div>
        <div className="button-row">
          <button
            onClick={() =>
              attempt(async () => {
                const data = await api("/api/environments/export", scene);
                const u = URL.createObjectURL(
                  new Blob([JSON.stringify(data, null, 2)], {
                    type: "application/json",
                  }),
                );
                const a = document.createElement("a");
                a.href = u;
                a.download = "microduck-environment.json";
                a.click();
                URL.revokeObjectURL(u);
              })
            }
          >
            <DownloadSimple />
            导出环境
          </button>
          <button onClick={() => file.current?.click()}>
            <UploadSimple />
            导入
          </button>
        </div>
        <input
          ref={file}
          hidden
          type="file"
          accept=".json"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f)
              attempt(async () => {
                if (f.size > 1024 * 1024) throw Error("环境文件超过 1MB");
                const result = await api(
                  "/api/environments/import",
                  JSON.parse(await f.text()),
                );
                save(result.scene);
              });
            e.target.value = "";
          }}
        />
      </div>
      <div className="world-editor">
        <aside className="asset-library">
          <div className="eyebrow">物体目录</div>
          <h3>构建空间</h3>
          <div className="asset-grid">
            {[
              ["ball", Sphere],
              ["box", Cube],
              ["toy", Robot],
              ["obstacle", SquaresFour],
            ].map(([kind, Icon]: any) => (
              <button key={kind} onClick={() => add(kind)}>
                <Icon size={25} />
                <span>{tr(kind)}</span>
              </button>
            ))}
          </div>
          <div className="catalog-caption">家具 · 静态碰撞</div>
          <div className="asset-grid">
            {catalog?.assets.map((a: any) => (
              <button key={a.id} onClick={() => add("obstacle", a)}>
                {a.id === "table" ? (
                  <Table size={25} />
                ) : (
                  <Armchair size={25} />
                )}
                <span>{a.name}</span>
              </button>
            ))}
          </div>
          <div className="object-list">
            <div className="catalog-caption">场景中的物体</div>
            <button
              className={selection === "spawn" ? "chosen" : ""}
              onClick={() => setSelection("spawn")}
            >
              <Robot />
              MicroDuck 出生点
            </button>
            {scene.objects.map((o) => (
              <button
                className={selection === o.id ? "chosen" : ""}
                key={o.id}
                onClick={() => setSelection(o.id)}
              >
                <span className={`color-dot ${o.color}`} />
                {tr(o.asset_id || o.kind)}
                {o.id === scene.target_id && <small>目标</small>}
              </button>
            ))}
          </div>
        </aside>
        <div className="world-center">
          <div className="world-toolbar">
            <div className="segmented">
              <button
                className={!top ? "selected" : ""}
                onClick={() => setTop(false)}
              >
                <Cube />
                三维
              </button>
              <button
                className={top ? "selected" : ""}
                onClick={() => setTop(true)}
              >
                <SquaresFour />
                俯视
              </button>
            </div>
            <div className="button-row">
              <button
                aria-label="撤销"
                disabled={!undo.length}
                onClick={() => {
                  setRedo((a) => [...a, scene]);
                  onChange(undo.at(-1)!);
                  setUndo((a) => a.slice(0, -1));
                }}
              >
                <ArrowCounterClockwise />
              </button>
              <button
                aria-label="恢复"
                disabled={!redo.length}
                onClick={() => {
                  setUndo((a) => [...a, scene]);
                  onChange(redo.at(-1)!);
                  setRedo((a) => a.slice(0, -1));
                }}
              >
                <ArrowClockwise />
              </button>
              <button
                aria-pressed={collision}
                onClick={() => setCollision(!collision)}
              >
                碰撞线框
              </button>
            </div>
          </div>
          <WorldViewport
            scene={
              previewDraft && draft?.result.scene ? draft.result.scene : scene
            }
            catalog={catalog}
            robot={robot}
            top={top}
            collision={collision}
            selection={selection}
            onSelect={previewDraft ? undefined : setSelection}
            onDragStart={previewDraft ? undefined : remember}
            onMove={
              previewDraft
                ? undefined
                : (id, x, y) =>
                    onChange({
                      ...scene,
                      version: 2,
                      ...(id === "spawn"
                        ? { spawn: { ...scene.spawn, x, y } }
                        : {
                            objects: scene.objects.map((o) =>
                              o.id === id ? { ...o, x, y } : o,
                            ),
                          }),
                    })
            }
          />
          <div className="canvas-footer">
            <span>
              <Compass />米 · Z 轴向上 · 拖动物体，空白处旋转视角
            </span>
            <button
              className="primary"
              onClick={onPreview}
              disabled={previewDraft}
            >
              <Play weight="fill" />
              MuJoCo 运行预览
            </button>
          </div>
          {robotError && (
            <p className="hint">{robotError}。橙色环仅表示出生点。</p>
          )}
          {previewDraft && (
            <p className="draft-indicator">
              正在预览草案，尚未应用；应用后可编辑与运行
            </p>
          )}
        </div>
        <aside className="properties">
          <div className="eyebrow">属性</div>
          <h3>
            {selection === "spawn"
              ? "MicroDuck 出生点"
              : selected
                ? tr(selected.asset_id || selected.kind)
                : "房间"}
          </h3>
          {selectionValue && (
            <div className="transform-fields">
              {(["x", "y", "yaw"] as const).map((k) => (
                <label key={k}>
                  {k === "yaw" ? "朝向 °" : k.toUpperCase() + " · m"}
                  <input
                    type="number"
                    step={k === "yaw" ? 5 : 0.05}
                    value={selectionValue[k]}
                    onChange={(e) => patch({ [k]: Number(e.target.value) })}
                  />
                </label>
              ))}
            </div>
          )}
          {selected && (
            <>
              <div className="transform-fields">
                {(["width", "depth", "height"] as const).map((k, i) => (
                  <label key={k}>
                    {["宽", "深", "高"][i]} · m
                    <input
                      type="number"
                      min=".04"
                      max={selected.asset_id ? 3 : 0.8}
                      step=".01"
                      value={selected[k]}
                      onChange={(e) => patch({ [k]: Number(e.target.value) })}
                    />
                  </label>
                ))}
              </div>
              {selected.asset_id ? (
                <label>
                  材质
                  <select
                    value={selected.material}
                    onChange={(e) => patch({ material: e.target.value })}
                  >
                    {Object.entries(catalog?.materials || {}).map(
                      ([k, v]: any) => (
                        <option key={k} value={k}>
                          {v.name}
                        </option>
                      ),
                    )}
                  </select>
                </label>
              ) : (
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
              )}
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
              <div className="button-row">
                <button
                  onClick={() => {
                    const id =
                      selected.kind +
                      "_" +
                      requestId().replaceAll("-", "").slice(0, 12);
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
                  onClick={() => {
                    save({
                      ...scene,
                      objects: scene.objects.filter((o) => o.id !== selection),
                    });
                    setSelection("spawn");
                  }}
                >
                  <Trash />
                  删除
                </button>
              </div>
            </>
          )}
          {selectionValue && (
            <details>
              <summary>随机范围</summary>
              <label>
                位置偏移 ±m
                <input
                  type="number"
                  min="0"
                  max={selected ? 1 : 0.5}
                  step=".01"
                  value={selectionValue.jitter}
                  onChange={(e) => patch({ jitter: Number(e.target.value) })}
                />
              </label>
              {!selected && (
                <label>
                  朝向偏移 ±°
                  <input
                    type="number"
                    min="0"
                    max="180"
                    value={scene.spawn.yaw_jitter}
                    onChange={(e) =>
                      patch({ yaw_jitter: Number(e.target.value) })
                    }
                  />
                </label>
              )}
            </details>
          )}
          <div className="room-settings">
            <label>
              房间名称
              <input
                value={scene.name}
                onChange={(e) => save({ ...scene, name: e.target.value })}
              />
            </label>
            <label>
              房间边长 · m
              <input
                type="number"
                min="2"
                max="6"
                step=".25"
                value={scene.size}
                onChange={(e) =>
                  save({ ...scene, size: Number(e.target.value) })
                }
              />
            </label>
          </div>
          <p className="hint">
            保存时检查边界、重叠和通路。家具可编辑，鸭子保持官方本体。
          </p>
        </aside>
      </div>
      <div className="layout-assistant">
        <div>
          <Sparkle size={24} />
          <div>
            <h3>用一句话调整布局</h3>
            <p>
              {status?.scene_model_configured
                ? "生成草案后预览，确认后才应用。"
                : "尚未配置布局模型，手工编辑可正常使用。"}
            </p>
          </div>
        </div>
        <div className="draft-input">
          <input
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="例如：在球与鸭子之间放一个柜子，留出右侧通道"
          />
          <button
            disabled={
              busy ||
              !status?.scene_model_configured ||
              !prompt.trim() ||
              (!!draft && ["queued", "running"].includes(draft.status))
            }
            onClick={() =>
              attempt(async () => {
                const s = await onSave();
                if (!s) return;
                draftBase.current = JSON.stringify(s.scene);
                setDraft(
                  await api(`/api/skills/${s.id}/runs`, {
                    kind: "scene_draft",
                    prompt,
                    idempotency_key: requestId(),
                  }),
                );
                setPreviewDraft(false);
              })
            }
          >
            生成草案
          </button>
        </div>
        {draft && (
          <div className="draft-result">
            <span>
              {tr(draft.status)} {draft.result.reason || ""}
            </span>
            {["queued", "running"].includes(draft.status) && (
              <button
                onClick={() =>
                  attempt(async () =>
                    setDraft(await api(`/api/runs/${draft.id}/cancel`, {})),
                  )
                }
              >
                取消
              </button>
            )}
            {draft.result.scene && (
              <>
                <button onClick={() => setPreviewDraft(!previewDraft)}>
                  {previewDraft ? "查看原布局" : "预览草案"}
                </button>
                <button
                  className="primary"
                  disabled={busy || draftChanged}
                  onClick={() =>
                    attempt(async () => {
                      const result = await api<Skill>(
                        `/api/runs/${draft.id}/apply-scene`,
                        {},
                      );
                      save(result.scene);
                      setDraft(undefined);
                      setPreviewDraft(false);
                    })
                  }
                >
                  应用布局
                </button>
                {draftChanged && <small>场景已修改，请重新生成草案</small>}
              </>
            )}
          </div>
        )}
      </div>
      {error && (
        <div className="inline-error" role="alert">
          {error}
        </div>
      )}
    </div>
  );
}
