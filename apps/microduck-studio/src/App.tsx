import { useEffect, useRef, useState } from "react";
import {
  House,
  Books,
  Flask,
  Cube,
  GearSix,
  Bird,
  Plus,
  ArrowRight,
  ArrowLeft,
  Play,
  UploadSimple,
  DownloadSimple,
  Stop,
  CheckCircle,
  WarningCircle,
  Clock,
  Monitor,
  HardDrives,
  ArrowClockwise,
  CaretRight,
  Code,
  Copy,
  VideoCamera,
  ChartLine,
} from "@phosphor-icons/react";
import SceneEditor from "./SceneEditor";
import { api, media, tr, requestId } from "./types";
import type { Skill, Run, Version, Recipe } from "./types";

const STAGES = [
  "定义任务",
  "场景编辑",
  "准备数据",
  "训练",
  "验证与优化",
  "发布与运行",
];
const fmt = (n: number) =>
  new Date(n * 1000).toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
const active = (r: Run) => ["running", "queued"].includes(r.status);
function Badge({ status }: { status: string }) {
  return (
    <span
      className={
        "badge " +
        (status.startsWith("accepted")
          ? "good"
          : status === "experimental" || status === "blocked"
            ? "amber"
            : status)
      }
    >
      {tr(status)}
    </span>
  );
}
function Empty({ children }: { children: React.ReactNode }) {
  return (
    <div className="empty">
      <Flask size={28} />
      <p>{children}</p>
    </div>
  );
}
function Info({ children }: { children: React.ReactNode }) {
  return (
    <div className="note">
      <WarningCircle size={18} />
      <span>{children}</span>
    </div>
  );
}

export default function App() {
  const [boot, setBoot] = useState<any>(null),
    [runs, setRuns] = useState<Run[]>([]),
    [page, setPage] = useState("library"),
    [skill, setSkill] = useState<Skill | null>(null),
    [stage, setStage] = useState(0);
  const [error, setError] = useState(""),
    [toast, setToast] = useState(""),
    [busy, setBusy] = useState(false),
    [viewRun, setViewRun] = useState<string | null>(null),
    [history, setHistory] = useState(false);
  const importInput = useRef<HTMLInputElement>(null);
  const [login, setLogin] = useState(false),
    [token, setToken] = useState("");
  const refresh = async () => {
    const [b, r] = await Promise.all([
      api("/api/bootstrap"),
      api<Run[]>("/api/runs"),
    ]);
    setBoot(b);
    setRuns(r);
  };
  useEffect(() => {
    refresh().catch((e) => {
      setError(e.message);
      if (e.message === "authentication required") setLogin(true);
    });
    const t = setInterval(() => {
      api<Run[]>("/api/runs")
        .then(setRuns)
        .catch(() => {});
    }, 2500);
    const workers = setInterval(
      () =>
        api("/api/bootstrap")
          .then(setBoot)
          .catch(() => {}),
      30000,
    );
    return () => {
      clearInterval(t);
      clearInterval(workers);
    };
  }, []);
  const perform = async (fn: () => Promise<any>, message = "") => {
    setError("");
    setBusy(true);
    try {
      const result = await fn();
      if (message) {
        setToast(message);
        setTimeout(() => setToast(""), 3500);
      }
      return result;
    } catch (e) {
      setError((e as Error).message);
      return null;
    } finally {
      setBusy(false);
    }
  };
  const open = (s: Skill, n = 0) => {
    setSkill(structuredClone(s));
    setPage("skill");
    setStage(n);
    setHistory(false);
    window.location.hash = `skill/${s.id}/${n}`;
  };
  const create = () =>
    perform(async () => {
      const s = await api<Skill>("/api/skills", boot.template);
      await refresh();
      open(s);
    }, "技能已创建");
  const save = async () => {
    if (!skill) return;
    const { id, ...body } = skill;
    const s = await api<Skill>("/api/skills/" + id, body, "PUT");
    setSkill(s);
    await refresh();
    return s;
  };
  const submit = async (options: Record<string, any>) =>
    perform(async () => {
      const s = await save();
      if (!s) return;
      const r = await api<Run>(`/api/skills/${s.id}/runs`, {
        ...options,
        idempotency_key: requestId(),
      });
      setViewRun(r.id);
      await refresh();
      return r;
    }, "已加入作业队列");
  const navigate = (p: string) => {
    setPage(p);
    setHistory(false);
    window.location.hash = p;
  };
  const selectedRuns = runs.filter((r) => r.skill_id === skill?.id);
  const versions: Version[] =
    boot?.versions.filter((v: Version) => v.skill_id === skill?.id) || [];
  const displayedRun = runs.find((r) => r.id === viewRun);
  return (
    <div className="app-shell">
      <aside className="rail">
        <button
          className="brand"
          aria-label="MicroDuck Studio 首页"
          onClick={() => navigate("library")}
        >
          <Bird size={37} weight="fill" />
          <span>
            MicroDuck
            <br />
            Studio
          </span>
        </button>
        <nav>
          {[
            { id: "library", icon: House, label: "技能库" },
            { id: "experiments", icon: Flask, label: "实验记录" },
            { id: "scenes", icon: Cube, label: "场景编辑" },
            { id: "workspace", icon: GearSix, label: "工作区" },
          ].map(({ id, icon: Icon, label }) => (
            <button
              key={id}
              className={
                page === id || (id === "library" && page === "skill")
                  ? "selected"
                  : ""
              }
              onClick={() => navigate(id)}
              aria-label={label}
              title={label}
            >
              <Icon size={25} weight={page === id ? "fill" : "regular"} />
              <span>{label}</span>
            </button>
          ))}
        </nav>
        <div className="rail-bottom">
          <Monitor size={22} />
          <small>仿真</small>
        </div>
      </aside>
      <main>
        <header className="topbar">
          <div>
            <House size={19} />
            <span className="muted">工作区</span>
            <CaretRight size={14} />
            <span>
              {page === "skill"
                ? skill?.name
                : page === "experiments"
                  ? "实验记录"
                  : page === "scenes"
                    ? "场景编辑"
                    : page === "workspace"
                      ? "运行环境"
                      : "技能库"}
            </span>
          </div>
          <span>{new Date().toLocaleDateString("sv-SE")}</span>
        </header>
        {error && (
          <div role="alert" className="error-banner">
            <WarningCircle />
            {error}
            <button onClick={() => setError("")}>关闭</button>
          </div>
        )}
        {toast && (
          <div className="toast" role="status">
            <CheckCircle />
            {toast}
          </div>
        )}
        {login ? (
          <section className="form-main login">
            <h2>连接你的工作区</h2>
            <p>输入管理员设置的访问令牌。</p>
            <form
              onSubmit={(e) => {
                e.preventDefault();
                perform(async () => {
                  await api("/api/session", { token });
                  setLogin(false);
                  setToken("");
                  await refresh();
                });
              }}
            >
              <label>
                访问令牌
                <input
                  type="password"
                  value={token}
                  onChange={(e) => setToken(e.target.value)}
                  autoComplete="current-password"
                />
              </label>
              <button className="primary">
                进入工作区
                <ArrowRight />
              </button>
            </form>
          </section>
        ) : !boot ? (
          <Empty>正在连接工作区…</Empty>
        ) : (
          <>
            {page === "library" && !history && (
              <>
                <div className="page-heading">
                  <div>
                    <h1>高级能力，持续生长。</h1>
                    <p>
                      在真实的仿真任务中训练与验证，让机器人掌握更多有用的技能。
                    </p>
                  </div>
                  <button
                    className="primary large"
                    onClick={create}
                    disabled={busy}
                  >
                    <Plus />
                    创建新技能
                  </button>
                </div>
                <div className="feature-grid">
                  <article className="feature">
                    <div className="feature-image recorded">
                      <img
                        src="/media/red-ball.jpg"
                        alt="MicroDuck 寻球的真实 MuJoCo 仿真录屏"
                      />
                      <span className="record-label">
                        <VideoCamera weight="fill" />
                        录屏
                      </span>
                      <button
                        className="image-play"
                        aria-label="播放历史技能录像"
                        onClick={() => setHistory(true)}
                      >
                        <Play weight="fill" />
                      </button>
                    </div>
                    <div className="feature-body">
                      <h2>寻找并接近 · 彩色球</h2>
                      <div className="feature-bottom">
                        <p>历史开发版本 · 官方运动控制</p>
                        <button
                          onClick={() =>
                            perform(async () => {
                              const r = await api<Run>("/api/history/runs", {
                                seed: 2000,
                              });
                              setViewRun(r.id);
                              await refresh();
                            })
                          }
                        >
                          <Play />
                          运行
                        </button>
                        <button
                          className="primary"
                          onClick={() => setHistory(true)}
                        >
                          查看效果
                          <ArrowRight />
                        </button>
                      </div>
                    </div>
                  </article>
                  <article className="feature template">
                    <div className="feature-image">
                      <img
                        src="/skill-template.png"
                        alt="红球、蓝色盒子与黄色玩偶的技能模板示意"
                      />
                    </div>
                    <div className="feature-body">
                      <h2>训练你的第一个技能</h2>
                      <p>从任务与场景开始，逐步完成采集、训练与验证。</p>
                      <button
                        className="primary"
                        onClick={create}
                        disabled={busy}
                      >
                        <Plus />
                        从模板创建
                      </button>
                    </div>
                  </article>
                </div>
                {boot.skills.length > 0 && (
                  <section className="skill-list">
                    <div className="section-heading">
                      <h2>
                        我的技能 <span>{boot.skills.length}</span>
                      </h2>
                      <button onClick={() => importInput.current?.click()}>
                        <UploadSimple />
                        导入技能包
                      </button>
                    </div>
                    {boot.skills.map((s: Skill) => (
                      <button
                        className="skill-row"
                        key={s.id}
                        onClick={() => open(s)}
                      >
                        <div className="skill-icon">
                          <Books size={25} />
                        </div>
                        <div>
                          <h3>{s.name}</h3>
                          <p>{s.description}</p>
                        </div>
                        <span className="muted">
                          {s.task.categories.map(tr).join(" · ")}
                        </span>
                        <Badge
                          status={
                            boot.versions.find(
                              (v: Version) => v.skill_id === s.id,
                            )?.status || "experimental"
                          }
                        />
                        <ArrowRight />
                      </button>
                    ))}
                  </section>
                )}
                <section className="activity">
                  <div className="section-heading">
                    <h2>工作区动态</h2>
                    <button
                      className="text-button"
                      onClick={() => navigate("experiments")}
                    >
                      查看全部
                      <ArrowRight />
                    </button>
                  </div>
                  {runs.length ? (
                    runs.slice(0, 3).map((r) => (
                      <button
                        className="activity-row"
                        key={r.id}
                        onClick={() => setViewRun(r.id)}
                      >
                        <span className={"dot " + r.status} />
                        <Flask size={21} />
                        <span>
                          {r.snapshot.skill.name} ·{" "}
                          {tr(r.snapshot.request.kind)}
                        </span>
                        <Badge status={r.status} />
                        <time>{fmt(r.created)}</time>
                      </button>
                    ))
                  ) : (
                    <>
                      <div className="activity-row">
                        <HardDrives size={21} />
                        <span>FastWAM / 轻量视觉策略</span>
                        <span className="muted">
                          选择同一份数据，比较两种后端
                        </span>
                      </div>
                      <div className="activity-row">
                        <Cube size={21} />
                        <span>新的高级物体能力</span>
                        <Badge status="experimental" />
                        <span className="muted">等待教师与模型验证</span>
                      </div>
                    </>
                  )}
                </section>
              </>
            )}
            {page === "library" && history && (
              <History
                boot={boot}
                onBack={() => setHistory(false)}
                onCreate={create}
                onLive={(color, group, variant) =>
                  perform(async () => {
                    const r = await api<Run>("/api/history/runs", {
                      color,
                      group,
                      variant,
                      seed: 2000,
                    });
                    setViewRun(r.id);
                    await refresh();
                  })
                }
              />
            )}
            {page === "skill" && skill && (
              <>
                <div className="page-heading compact">
                  <div>
                    <button
                      className="back"
                      onClick={() => navigate("library")}
                    >
                      <ArrowLeft />
                      技能库
                    </button>
                    <h1>{skill.name}</h1>
                    <p>{skill.description}</p>
                  </div>
                  <button
                    onClick={() => perform(save, "新配置已保存")}
                    disabled={busy}
                  >
                    <CheckCircle />
                    保存配置
                  </button>
                </div>
                <nav className="steps" aria-label="技能训练流程">
                  {STAGES.map((s, i) => (
                    <button
                      key={s}
                      className={stage === i ? "current" : ""}
                      onClick={() => setStage(i)}
                    >
                      <span>{String(i + 1).padStart(2, "0")}</span>
                      {s}
                    </button>
                  ))}
                </nav>
                {stage === 0 && (
                  <TaskEditor
                    skill={skill}
                    setSkill={setSkill}
                    onNext={() =>
                      perform(async () => {
                        await save();
                        setStage(1);
                      })
                    }
                  />
                )}
                {stage === 1 && (
                  <>
                    <SceneEditor
                      scene={skill.scene}
                      onChange={(scene) => setSkill({ ...skill, scene })}
                      onPreview={() => submit({ kind: "preview" })}
                    />
                    <div className="stage-footer">
                      <button onClick={() => perform(save, "场景已校验并保存")}>
                        保存场景
                      </button>
                      <button
                        className="primary"
                        onClick={() =>
                          perform(async () => {
                            await save();
                            setStage(2);
                          })
                        }
                      >
                        准备数据
                        <ArrowRight />
                      </button>
                    </div>
                  </>
                )}
                {stage >= 2 && stage <= 5 && (
                  <Workflow
                    stage={stage}
                    skill={skill}
                    runs={selectedRuns}
                    versions={versions}
                    boot={boot}
                    submit={submit}
                    onRun={setViewRun}
                    perform={perform}
                    refresh={refresh}
                    busy={busy}
                  />
                )}
              </>
            )}
            {page === "scenes" && (
              <>
                <div className="page-heading">
                  <div>
                    <h1>把环境，变成训练场。</h1>
                    <p>编辑布局，预览真实物理，并让同一技能面对不同条件。</p>
                  </div>
                  <button className="primary" onClick={create}>
                    <Plus />
                    新建场景任务
                  </button>
                </div>
                {boot.skills.length ? (
                  <div className="scene-list">
                    {boot.skills.map((s: Skill) => (
                      <button
                        key={s.id}
                        className="skill-row"
                        onClick={() => open(s, 1)}
                      >
                        <Cube size={32} />
                        <div>
                          <h3>{s.scene.name}</h3>
                          <p>
                            {s.name} · {s.scene.objects.length} 个物体 ·{" "}
                            {s.scene.size} × {s.scene.size} m
                          </p>
                        </div>
                        <ArrowRight />
                      </button>
                    ))}
                  </div>
                ) : (
                  <Empty>从技能模板创建第一间房间。</Empty>
                )}
              </>
            )}
            {page === "experiments" && (
              <>
                <div className="page-heading">
                  <div>
                    <h1>每次尝试，都有迹可循。</h1>
                    <p>按技能与阶段查看记录，保留失败与成功的完整证据。</p>
                  </div>
                </div>
                <ComparePanel runs={runs} />
                <RunList runs={runs} onSelect={setViewRun} />
              </>
            )}
            {page === "workspace" && (
              <Workspace
                boot={boot}
                onImport={() => importInput.current?.click()}
              />
            )}
          </>
        )}
        <footer>
          <span>
            <Monitor size={16} />
            MuJoCo 仿真 · 尚未进行真机验证
          </span>
          <span>
            MicroDuck Studio <i /> LightX2V
          </span>
        </footer>
      </main>
      <input
        type="file"
        accept=".tar"
        hidden
        ref={importInput}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f)
            perform(async () => {
              const form = new FormData();
              form.append("file", f);
              await api("/api/import", form);
              await refresh();
            }, "技能包已导入；外部证据需重新验证");
          e.target.value = "";
        }}
      />
      {displayedRun && (
        <RunDialog
          run={displayedRun}
          onClose={() => setViewRun(null)}
          action={(fn) =>
            perform(async () => {
              await fn();
              await refresh();
            })
          }
        />
      )}
    </div>
  );
}

function History({
  boot,
  onBack,
  onCreate,
  onLive,
}: {
  boot: any;
  onBack: () => void;
  onCreate: () => void;
  onLive: (color: string, group: string, variant: string) => void;
}) {
  const [group, setGroup] = useState(boot.history.groups[0].id),
    [clipId, setClipId] = useState(""),
    [color, setColor] = useState("red"),
    [variant, setVariant] = useState("optimized");
  const data = boot.history.groups.find((g: any) => g.id === group);
  const clip = data.clips.find((c: any) => c.id === clipId) || data.clips[0];
  return (
    <>
      <button className="back" onClick={onBack}>
        <ArrowLeft />
        返回技能库
      </button>
      <div className="page-heading">
        <div>
          <div className="eyebrow">RECORDED EVIDENCE</div>
          <h1>看看它，已经做到了什么。</h1>
          <p>
            当前发布模型的历史开发录像。新任务、新传感器与新运行条件需要分别验证。
          </p>
        </div>
        <button onClick={onCreate}>
          <Copy />
          创建新的技能版本
        </button>
      </div>
      <div className="segmented">
        {boot.history.groups.map((g: any) => (
          <button
            key={g.id}
            className={g.id === group ? "active" : ""}
            onClick={() => {
              setGroup(g.id);
              setClipId("");
            }}
          >
            {g.id === "clear-room"
              ? "开阔房间"
              : g.id === "single-obstacle"
                ? "单障碍"
                : "其他条件"}
            <span>
              {g.passed}/{g.total}
            </span>
          </button>
        ))}
      </div>
      <div className="evidence-layout">
        <div>
          <video
            className="evidence-video"
            key={clip.video}
            src={"/media/" + clip.video}
            poster={"/media/" + clip.poster}
            controls
            playsInline
          />
          <div className="video-caption">
            <Badge status={clip.passed ? "complete" : "failed"} />
            <span>
              {tr(clip.task)} · seed {clip.seed} ·{" "}
              {clip.mode === "realtime" ? "连续实时" : "离线"}录屏
            </span>
          </div>
        </div>
        <aside>
          <h3>运行片段</h3>
          {data.clips.map((c: any) => (
            <button
              key={c.id}
              className={"clip-row " + (c.id === clip.id ? "selected" : "")}
              onClick={() => setClipId(c.id)}
            >
              <img src={"/media/" + c.poster} alt="仿真片段" />
              <div>
                <strong>{tr(c.task)}</strong>
                <p>
                  {c.passed ? "本集通过" : "失败样例"} · seed {c.seed}
                </p>
              </div>
              <Play />
            </button>
          ))}
          <div className="history-run">
            <h3>运行这个历史模型</h3>
            <label>
              目标
              <select value={color} onChange={(e) => setColor(e.target.value)}>
                {["red", "blue", "yellow"].map((c) => (
                  <option key={c} value={c}>
                    {tr(c)}球
                  </option>
                ))}
              </select>
            </label>
            <label>
              部署配置
              <select
                value={variant}
                onChange={(e) => setVariant(e.target.value)}
              >
                <option value="optimized">LightX2V 加速</option>
                <option value="plain">关闭缓存与 CUDA Graph</option>
              </select>
            </label>
            <button
              className="primary"
              onClick={() =>
                onLive(
                  color,
                  group === "clear-room" ? "clear" : "obstacle",
                  variant,
                )
              }
            >
              <Play />
              运行实时仿真
            </button>
            <p className="hint">
              需要连接 GPU 运行器和发布权重。48D
              历史模型保持原来的输入与执行契约。
            </p>
          </div>
          <Info>
            这些是历史开发结果，不是新平台的正式验收。机器人摄像头与观察者视角并排展示。
          </Info>
        </aside>
      </div>
    </>
  );
}

function TaskEditor({
  skill,
  setSkill,
  onNext,
}: {
  skill: Skill;
  setSkill: (s: Skill) => void;
  onNext: () => void;
}) {
  const update = (key: string, value: any) =>
    setSkill({ ...skill, task: { ...skill.task, [key]: value } });
  const choices = (key: "categories" | "colors" | "groups", values: string[]) =>
    values.map((v) => (
      <label key={v} className="choice">
        <input
          type="checkbox"
          checked={skill.task[key].includes(v)}
          onChange={(e) =>
            update(
              key,
              e.target.checked
                ? [...skill.task[key], v]
                : skill.task[key].filter((k) => k !== v),
            )
          }
        />
        <span>{v === "obstacle" ? "静态绕障" : tr(v)}</span>
      </label>
    ));
  return (
    <div className="form-layout">
      <section className="form-main">
        <div className="eyebrow">01 / DEFINE THE SKILL</div>
        <h2>先确定，怎样才算完成。</h2>
        <label>
          技能名称
          <input
            value={skill.name}
            onChange={(e) => setSkill({ ...skill, name: e.target.value })}
          />
        </label>
        <label>
          一句话说明
          <input
            value={skill.description}
            onChange={(e) =>
              setSkill({ ...skill, description: e.target.value })
            }
          />
        </label>
        <fieldset>
          <legend>目标类别</legend>
          <div className="choices">
            {choices("categories", ["ball", "box", "toy"])}
          </div>
        </fieldset>
        <fieldset>
          <legend>目标颜色</legend>
          <div className="choices">
            {choices("colors", ["red", "blue", "yellow"])}
          </div>
        </fieldset>
        <fieldset>
          <legend>需要覆盖的环境</legend>
          <div className="choices">
            {choices("groups", [
              "clear",
              "outside_view",
              "obstacle",
              "occlusion",
            ])}
          </div>
        </fieldset>
        <div className="three-col">
          {[
            ["stop_min", "最近停距 · m"],
            ["stop_max", "最远停距 · m"],
            ["timeout_s", "任务时限 · 秒"],
          ].map(([key, title]) => (
            <label key={key}>
              {title}
              <input
                type="number"
                step={key === "timeout_s" ? 5 : 0.01}
                value={(skill.task as any)[key]}
                onChange={(e) => update(key, Number(e.target.value))}
              />
            </label>
          ))}
        </div>
        <button className="primary" onClick={onNext}>
          进入场景编辑
          <ArrowRight />
        </button>
      </section>
      <aside className="form-aside">
        <Cube size={32} />
        <h3>让技能可以复用</h3>
        <p>
          同一个冻结版本，面对新的布局和同类物体实例。新增目标类别时，创建新的数据与模型版本。
        </p>
        <div className="rule-row">
          <span>可用输入</span>
          <strong>头部 RGB · ToF · 自身状态</strong>
        </div>
        <div className="rule-row">
          <span>底层运动</span>
          <strong>官方 ONNX · 50 Hz</strong>
        </div>
        <div className="rule-row">
          <span>完成条件</span>
          <strong>进入停距后，稳定保持至少 2 秒</strong>
        </div>
        <p className="hint">碰撞、跌倒、超时与控制接管均计入失败。</p>
      </aside>
    </div>
  );
}

function Workflow({
  stage,
  skill,
  runs,
  versions,
  boot,
  submit,
  onRun,
  perform,
  refresh,
  busy,
}: {
  stage: number;
  skill: Skill;
  runs: Run[];
  versions: Version[];
  boot: any;
  submit: (b: any) => any;
  onRun: (id: string) => void;
  perform: (f: () => Promise<any>, m?: string) => any;
  refresh: () => Promise<void>;
  busy: boolean;
}) {
  const [backend, setBackend] = useState("lightweight"),
    [recipe, setRecipe] = useState<Recipe>(boot.recipes.lightweight),
    [dataRun, setDataRun] = useState(""),
    [teacherRun, setTeacherRun] = useState(""),
    [version, setVersion] = useState(""),
    [policy, setPolicy] = useState("rule"),
    [mode, setMode] = useState("offline"),
    [episodes, setEpisodes] = useState(1000),
    [cases, setCases] = useState(200),
    [teacherCases, setTeacherCases] = useState(100),
    [color, setColor] = useState("red"),
    [category, setCategory] = useState("ball"),
    [group, setGroup] = useState("clear");
  const eligibleData = runs.filter(
    (r) =>
      r.status === "complete" &&
      ["collect", "pipeline"].includes(r.snapshot.request.kind),
  );
  const teachers = runs.filter(
    (r) => r.status === "complete" && r.result.teacher_qualified,
  );
  const candidates = runs.filter(
    (r) => r.status === "complete" && r.result.candidate,
  );
  const needingModel = ["fastwam", "lightweight"].includes(policy);
  const options = {
    policy,
    version: version || undefined,
    mode,
    category,
    color,
    group,
  };
  const recent = runs.filter((r) =>
    (stage === 2
      ? ["teacher", "collect", "correction"]
      : stage === 3
        ? ["train", "pipeline"]
        : stage === 4
          ? ["evaluate", "optimize"]
          : ["live"]
    ).includes(r.snapshot.request.kind),
  );
  return (
    <>
      <div className="workflow-heading">
        <div className="eyebrow">
          {
            [
              "",
              "",
              "02 / DEMONSTRATE",
              "03 / TRAIN",
              "04 / VERIFY",
              "05 / SHARE & RUN",
            ][stage]
          }
        </div>
        <h2>
          {
            [
              "",
              "",
              "好的示范，是学习的起点。",
              "让示范，变成可执行的技能。",
              "先看证据，再做选择。",
              "冻结一个版本，随时调用。",
            ][stage]
          }
        </h2>
      </div>
      <div className="form-layout">
        <section className="form-main">
          {stage === 2 && (
            <>
              <h3>教师检查</h3>
              <p className="muted">
                按难度组独立检查。每组至少 100 个条件、95%
                严格成功率才开放正式采集。
              </p>
              <div className="inline-fields">
                <label>
                  每组案例
                  <input
                    type="number"
                    min="1"
                    value={teacherCases}
                    onChange={(e) => setTeacherCases(+e.target.value)}
                  />
                </label>
                <button
                  className="primary"
                  disabled={busy}
                  onClick={() =>
                    submit({ kind: "teacher", teacher_cases: teacherCases })
                  }
                >
                  <Play />
                  检查教师
                </button>
              </div>
              <div className="divider" />
              <h3>采集示范</h3>
              <label>
                已通过检查的教师
                <select
                  value={teacherRun}
                  onChange={(e) => setTeacherRun(e.target.value)}
                >
                  <option value="">
                    {teachers.length ? "选择教师检查记录" : "暂无通过记录"}
                  </option>
                  {teachers.map((r) => (
                    <option key={r.id} value={r.id}>
                      {fmt(r.created)} · {r.id.slice(0, 8)}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                示范数量
                <input
                  type="number"
                  min="1"
                  value={episodes}
                  onChange={(e) => setEpisodes(+e.target.value)}
                />
              </label>
              <button
                className="primary"
                disabled={!teacherRun || busy}
                onClick={() =>
                  submit({ kind: "collect", teacher_run: teacherRun, episodes })
                }
              >
                <Plus />
                开始采集
              </button>
              <Info>
                失败轨迹单独保留，归一化只使用训练集。纠错可在「发布与运行」中开启键盘控制。
              </Info>
            </>
          )}
          {stage === 3 && (
            <>
              <label>
                训练后端
                <div className="backend-choices">
                  {["lightweight", "fastwam"].map((b) => (
                    <button
                      key={b}
                      className={backend === b ? "chosen" : ""}
                      onClick={() => {
                        setBackend(b);
                        setRecipe(boot.recipes[b]);
                      }}
                    >
                      <strong>{tr(b)}</strong>
                      <span>
                        {b === "lightweight"
                          ? "ResNet18 + GRU · 176D 状态"
                          : "世界动作模型 · LightX2V 部署"}
                      </span>
                    </button>
                  ))}
                </div>
              </label>
              <label>
                数据版本
                <select
                  value={dataRun}
                  onChange={(e) => setDataRun(e.target.value)}
                >
                  <option value="">选择已审计的数据</option>
                  {eligibleData.map((r) => (
                    <option key={r.id} value={r.id}>
                      {fmt(r.created)} · {r.id.slice(0, 8)}
                    </option>
                  ))}
                </select>
              </label>
              <div className="recipe-summary">
                <span>{recipe.max_steps.toLocaleString()} 步上限</span>
                <span>batch {recipe.batch_size}</span>
                <span>每 {recipe.validate_every} 步验证</span>
              </div>
              <details>
                <summary>高级训练设置</summary>
                <div className="two-col">
                  {[
                    ["learning_rate", "学习率"],
                    ["batch_size", "有效 batch"],
                    ["micro_batch", "Micro batch"],
                    ["max_steps", "最大步数"],
                    ["validate_every", "验证间隔"],
                    ["max_hours", "本轮最长小时"],
                  ].map(([key, title]) => (
                    <label key={key}>
                      {title}
                      <input
                        type="number"
                        step="any"
                        value={(recipe as any)[key]}
                        onChange={(e) =>
                          setRecipe({ ...recipe, [key]: +e.target.value })
                        }
                      />
                    </label>
                  ))}
                </div>
                <label>
                  精度
                  <select
                    value={recipe.precision}
                    onChange={(e) =>
                      setRecipe({ ...recipe, precision: e.target.value })
                    }
                  >
                    <option value="bf16">BF16</option>
                    <option value="fp32">FP32</option>
                  </select>
                </label>
              </details>
              <div className="button-row">
                <button
                  className="primary"
                  disabled={!dataRun || busy}
                  onClick={() =>
                    submit({ kind: "train", data_run: dataRun, recipe })
                  }
                >
                  <Play />
                  训练这一轮
                </button>
                <button
                  disabled={busy}
                  onClick={() =>
                    submit({
                      kind: "pipeline",
                      recipe,
                      episodes,
                      teacher_cases: 100,
                      evaluation_cases: 200,
                    })
                  }
                >
                  从教师检查开始完整一轮
                  <ArrowRight />
                </button>
              </div>
              <Info>
                每轮独立运行。未过门槛会停止，不自动增加训练轮次，也不会自动发布模型。
              </Info>
            </>
          )}
          {stage >= 4 && (
            <>
              <div className="two-col">
                <label>
                  策略
                  <select
                    value={policy}
                    onChange={(e) => setPolicy(e.target.value)}
                  >
                    {[
                      "rule",
                      "teacher",
                      "fastwam",
                      "lightweight",
                      "vlm_local",
                      "vlm_cloud",
                    ].map((p) => (
                      <option key={p} value={p}>
                        {p === "teacher" ? "特权教师对照" : tr(p)}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  运行方式
                  <select
                    value={mode}
                    onChange={(e) => setMode(e.target.value)}
                  >
                    <option value="offline">离线评测</option>
                    <option value="realtime">连续实时</option>
                  </select>
                </label>
              </div>
              {needingModel && (
                <label>
                  冻结模型版本
                  <select
                    value={version}
                    onChange={(e) => setVersion(e.target.value)}
                  >
                    <option value="">选择匹配的模型</option>
                    {versions
                      .filter((v) => v.backend === policy)
                      .map((v) => (
                        <option key={v.id} value={v.id}>
                          {v.name} · {tr(v.status)}
                        </option>
                      ))}
                  </select>
                </label>
              )}
              {stage === 4 ? (
                <>
                  <label>
                    独立测试条件
                    <input
                      type="number"
                      min="1"
                      value={cases}
                      onChange={(e) => setCases(+e.target.value)}
                    />
                  </label>
                  <div className="button-row">
                    <button
                      className="primary"
                      disabled={busy || (needingModel && !version)}
                      onClick={() =>
                        submit({
                          kind: "evaluate",
                          ...options,
                          evaluation_cases: cases,
                        })
                      }
                    >
                      <Play />
                      运行评测
                    </button>
                    <button
                      disabled={busy || policy !== "fastwam" || !version}
                      onClick={() =>
                        submit({
                          kind: "optimize",
                          ...options,
                          evaluation_cases: cases,
                        })
                      }
                    >
                      <ChartLine />
                      LightX2V 加速对照
                    </button>
                  </div>
                  <Info>
                    每个版本至少 200 个未见条件，严格成功率目标
                    80%。较少案例只作诊断，离线与实时分别标记。
                  </Info>
                </>
              ) : (
                <>
                  <div className="three-col">
                    <label>
                      目标类别
                      <select
                        value={category}
                        onChange={(e) => setCategory(e.target.value)}
                      >
                        {skill.task.categories.map((c) => (
                          <option key={c} value={c}>
                            {tr(c)}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label>
                      颜色
                      <select
                        value={color}
                        onChange={(e) => setColor(e.target.value)}
                      >
                        {skill.task.colors.map((c) => (
                          <option key={c} value={c}>
                            {tr(c)}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label>
                      环境
                      <select
                        value={group}
                        onChange={(e) => setGroup(e.target.value)}
                      >
                        {skill.task.groups.map((c) => (
                          <option key={c} value={c}>
                            {c === "obstacle" ? "静态绕障" : tr(c)}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>
                  <div className="button-row">
                    <button
                      className="primary"
                      disabled={busy || (needingModel && !version)}
                      onClick={() =>
                        submit({ kind: "live", ...options, mode: "realtime" })
                      }
                    >
                      <Play />
                      启动实时仿真
                    </button>
                    <button
                      disabled={busy || (needingModel && !version)}
                      onClick={() =>
                        submit({
                          kind: "correction",
                          ...options,
                          mode: "realtime",
                        })
                      }
                    >
                      键盘纠错
                    </button>
                  </div>
                </>
              )}
            </>
          )}
        </section>
        <aside className="form-aside">
          <div className="eyebrow">THIS SKILL</div>
          <h3>{skill.name}</h3>
          <p>
            {skill.task.categories.map(tr).join(" / ")}
            <br />
            {skill.task.colors.map(tr).join(" / ")}
          </p>
          <div className="rule-row">
            <span>教师检查</span>
            <strong>{teachers.length ? "有通过记录" : "待验证"}</strong>
          </div>
          <div className="rule-row">
            <span>数据版本</span>
            <strong>{eligibleData.length} 个</strong>
          </div>
          <div className="rule-row">
            <span>已发布版本</span>
            <strong>{versions.length} 个</strong>
          </div>
          <Info>
            新 176D 观测包含 ToF。已有 48D FastWAM 的成绩不能直接继承到这里。
          </Info>
        </aside>
      </div>
      {stage === 5 && (
        <section>
          <div className="section-heading">
            <h2>发布与导出</h2>
          </div>
          {candidates.length ? (
            candidates.map((r) => (
              <div className="skill-row" key={r.id}>
                <CheckCircle size={25} />
                <div>
                  <h3>
                    {tr(r.snapshot.request.recipe.backend)} · 候选{" "}
                    {r.id.slice(0, 8)}
                  </h3>
                  <p>
                    开发验证{" "}
                    {r.result.best_validation_rate >= 0
                      ? Math.round(r.result.best_validation_rate * 100) + "%"
                      : "待查看"}{" "}
                    · 尚未自动发布
                  </p>
                </div>
                <button
                  onClick={() =>
                    perform(async () => {
                      await api(`/api/skills/${skill.id}/publish`, {
                        run_id: r.id,
                      });
                      await refresh();
                    }, "模型已登记，状态以独立评测为准")
                  }
                >
                  发布为本地版本
                  <ArrowRight />
                </button>
              </div>
            ))
          ) : (
            <Empty>训练完成后，在这里审查并发布候选模型。</Empty>
          )}
          {versions.map((v) => (
            <div key={v.id} className="skill-row">
              <Books />
              <div>
                <h3>{v.name}</h3>
                <p>{v.id}</p>
              </div>
              <Badge status={v.status} />
              {runs
                .filter(
                  (r) =>
                    r.status === "complete" &&
                    r.snapshot.request.kind === "evaluate" &&
                    r.snapshot.request.version === v.id &&
                    (r.result.evaluation || r.result).accepted,
                )
                .slice(0, 1)
                .map((r) => (
                  <button
                    key={r.id}
                    onClick={() =>
                      perform(async () => {
                        await api(`/api/versions/${v.id}/qualify`, {
                          run_id: r.id,
                        });
                        await refresh();
                      }, "已核对独立评测并登记验收版本")
                    }
                  >
                    <CheckCircle />
                    登记验收版本
                  </button>
                ))}
              <a className="button" href={`/api/versions/${v.id}/export`}>
                <DownloadSimple />
                导出
              </a>
            </div>
          ))}
          <details>
            <summary>
              <Code />
              通过 Python / VLM 调用
            </summary>
            <pre>{`from lightx2v_microduck.studio.sdk import Studio\nclient = Studio("${window.location.origin}")\nrun = client.run("${skill.id}", "${version || "VERSION_ID"}",\n                 category="${category}", color="${color}", backend="${needingModel ? policy : "lightweight"}")\nclient.status(run["run_id"])\n# client.cancel(run["run_id"])`}</pre>
            <p className="hint">
              SDK 仅向规划器返回执行状态，不返回仿真真值或成功答案。
            </p>
          </details>
        </section>
      )}
      <section className="recent">
        <div className="section-heading">
          <h2>这一阶段的记录</h2>
          <span>{recent.length} 次</span>
        </div>
        <RunList runs={recent} onSelect={onRun} />
      </section>
    </>
  );
}

function RunList({
  runs,
  onSelect,
}: {
  runs: Run[];
  onSelect: (id: string) => void;
}) {
  const [filter, setFilter] = useState("all"),
    [q, setQ] = useState("");
  const shown = runs.filter(
    (r) =>
      (filter === "all" || r.status === filter) &&
      (r.snapshot.skill.name + r.snapshot.request.kind + r.id).includes(q),
  );
  return (
    <>
      <div className="list-controls">
        <input
          aria-label="搜索实验"
          placeholder="搜索技能、任务或记录编号"
          value={q}
          onChange={(e) => setQ(e.target.value)}
        />
        <select
          aria-label="按状态筛选"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        >
          <option value="all">全部状态</option>
          {[
            "running",
            "queued",
            "complete",
            "blocked",
            "failed",
            "cancelled",
          ].map((s) => (
            <option value={s} key={s}>
              {tr(s)}
            </option>
          ))}
        </select>
      </div>
      {shown.length ? (
        <div className="run-table">
          {shown.map((r) => (
            <button
              key={r.id}
              className="run-row"
              onClick={() => onSelect(r.id)}
            >
              <div className="run-symbol">
                <Flask size={21} />
              </div>
              <div>
                <strong>{r.snapshot.skill.name}</strong>
                <p>
                  {tr(r.snapshot.request.kind)} · {r.id.slice(0, 8)}
                </p>
              </div>
              <span className="muted">
                {r.progress.step
                  ? `${r.progress.step} / ${r.progress.max_steps} 步`
                  : r.progress.completed
                    ? `${r.progress.completed} / ${r.progress.total} 集`
                    : r.result.rate != null
                      ? `${r.result.passed}/${r.result.total} 通过`
                      : "—"}
              </span>
              <Badge status={r.status} />
              <time>{fmt(r.created)}</time>
              <CaretRight />
            </button>
          ))}
        </div>
      ) : (
        <Empty>这里还没有实验记录。开始一轮检查、训练或评测后即可查看。</Empty>
      )}
    </>
  );
}

function RunDialog({
  run,
  onClose,
  action,
}: {
  run: Run;
  onClose: () => void;
  action: (fn: () => Promise<any>) => any;
}) {
  const [dataset, setDataset] = useState<any>(null);
  const [metrics, setMetrics] = useState<any[]>([]),
    [report, setReport] = useState<any>(null),
    [tab, setTab] = useState("view");
  const path = run.progress.media || run.result.media || `runs/${run.id}/live`;
  useEffect(() => {
    let alive = true;
    const update = () => {
      if (["collect", "pipeline"].includes(run.snapshot.request.kind))
        api(`/api/runs/${run.id}/dataset`)
          .then((d) => {
            if (alive) setDataset(d);
          })
          .catch(() => {});
      api(`/api/runs/${run.id}/metrics`)
        .then((points) => {
          if (alive) setMetrics(points);
        })
        .catch(() => {});
      const p = run.result.evaluation || run.result;
      if (alive) setReport(p);
    };
    update();
    const t = setInterval(update, 4000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [run.id, run.status, run.result]);
  useEffect(() => {
    const before = document.activeElement as HTMLElement | null;
    const dialog = document.querySelector<HTMLElement>(".run-dialog");
    dialog?.querySelector<HTMLElement>("button")?.focus();
    const keyboard = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
      if (e.key !== "Tab" || !dialog) return;
      const nodes = Array.from(
        dialog.querySelectorAll<HTMLElement>(
          "button:not([disabled]), a[href], input, select, textarea, video[controls]",
        ),
      ).filter((n) => n.getClientRects().length);
      if (!nodes.length) return;
      const first = nodes[0],
        last = nodes[nodes.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      }
      if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", keyboard);
    return () => {
      window.removeEventListener("keydown", keyboard);
      before?.focus();
    };
  }, [run.id]);
  useEffect(() => {
    if (run.snapshot.request.kind !== "correction" || run.status !== "running")
      return;
    let held: string | null = null;
    const send = (key: string | null) => {
      const value =
        key === "w"
          ? [0.3, 0, 0, 0, 0]
          : key === "s"
            ? [-0.1, 0, 0, 0, 0]
            : key === "a"
              ? [0, 0, 0.7, 0, 0]
              : key === "d"
                ? [0, 0, -0.7, 0, 0]
                : [0, 0, 0, 0, 1];
      api(`/api/runs/${run.id}/control`, { action: value }).catch(() => {});
    };
    const down = (e: KeyboardEvent) => {
      if (["w", "a", "s", "d", " "].includes(e.key.toLowerCase())) {
        e.preventDefault();
        held = e.key.toLowerCase();
        send(held);
      }
    };
    const up = () => {
      held = null;
      send(null);
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    window.addEventListener("blur", up);
    const timer = setInterval(() => {
      if (held) send(held);
    }, 100);
    return () => {
      clearInterval(timer);
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
      window.removeEventListener("blur", up);
      send(null);
    };
  }, [run.id, run.status]);
  const chart = metrics.slice(-200);
  const max = Math.max(...chart.map((p) => p.loss), 0.001);
  const points = chart
    .map(
      (p, i) =>
        `${(i / Math.max(chart.length - 1, 1)) * 700},${140 - (p.loss / max) * 125}`,
    )
    .join(" ");
  return (
    <div className="dialog-backdrop" onClick={onClose}>
      <section
        className="run-dialog"
        role="dialog"
        aria-modal="true"
        aria-label="实验详情"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="dialog-heading">
          <div>
            <div className="eyebrow">{run.id}</div>
            <h2>
              {tr(run.snapshot.request.kind)}
              <Badge status={run.status} />
            </h2>
          </div>
          <button onClick={onClose}>关闭</button>
        </div>
        <div className="dialog-tools">
          <div className="segmented">
            <button
              className={tab === "view" ? "active" : ""}
              onClick={() => setTab("view")}
            >
              画面与结果
            </button>
            <button
              className={tab === "details" ? "active" : ""}
              onClick={() => setTab("details")}
            >
              版本与日志
            </button>
          </div>
          <div>
            {active(run) && (
              <button
                onClick={() => action(() => api(`/api/runs/${run.id}/cancel`))}
              >
                <Stop />
                取消作业
              </button>
            )}
            {["cancelled", "failed", "interrupted"].includes(run.status) && (
              <button
                onClick={() => action(() => api(`/api/runs/${run.id}/resume`))}
              >
                <ArrowClockwise />
                恢复训练
              </button>
            )}
          </div>
        </div>
        {tab === "details" ? (
          <>
            <pre>
              {JSON.stringify(
                {
                  request: run.snapshot.request,
                  progress: run.progress,
                  result: run.result,
                },
                null,
                2,
              )}
            </pre>
            <a
              href={media(`runs/${run.id}/worker.log`)}
              target="_blank"
              rel="noreferrer"
            >
              查看完整日志
            </a>
          </>
        ) : (
          <>
            {run.status === "queued" ? (
              <Empty>
                <Clock size={28} />
                等待匹配运行器。GPU 作业串行执行；工作区中可查看运行器是否在线。
              </Empty>
            ) : run.progress.media ||
              run.snapshot.request.kind === "preview" ? (
              <>
                <div className="live-grid">
                  <figure>
                    <img
                      src={
                        active(run)
                          ? `/api/runs/${run.id}/stream/observer`
                          : media(path + "/observer.jpg") +
                            "?t=" +
                            Math.floor(Date.now() / 2000)
                      }
                      alt="MuJoCo 观察者视角"
                    />
                    <figcaption>观察者视角 · 用于展示</figcaption>
                  </figure>
                  <figure>
                    <img
                      src={
                        active(run)
                          ? `/api/runs/${run.id}/stream/head`
                          : media(
                              path +
                                "/" +
                                (run.progress.head_file || "head.jpg"),
                            ) +
                            "?t=" +
                            Math.floor(Date.now() / 2000)
                      }
                      alt="头部摄像头观察"
                    />
                    <figcaption>机器人头部 RGB</figcaption>
                  </figure>
                </div>
                <div className="runtime-strip">
                  <span>
                    {["live", "correction", "historical"].includes(
                      run.snapshot.request.kind,
                    )
                      ? "连续实时运行"
                      : "验证 / 预览画面"}
                  </span>
                  <span>{run.progress.sim_s?.toFixed(1) || 0} 秒</span>
                  <span>{run.progress.rtf?.toFixed(2) || "—"} × 实时速度</span>
                  <span>控制：{run.progress.control_source || "—"}</span>
                </div>
              </>
            ) : null}
            {run.snapshot.request.kind === "correction" &&
              run.status === "running" && (
                <Info>
                  W / S 前后，A / D
                  转向，空格停止。松键或窗口失焦停止；纠错另行记录。
                </Info>
              )}
            {chart.length > 0 && (
              <div className="loss-chart">
                <h3>
                  训练损失 <span>{chart.at(-1)?.loss.toFixed(5)}</span>
                </h3>
                <svg
                  viewBox="0 0 700 160"
                  role="img"
                  aria-label="真实训练损失曲线"
                >
                  <line x1="0" y1="145" x2="700" y2="145" stroke="#d6e1e5" />
                  <polyline
                    points={points}
                    fill="none"
                    stroke="#167c82"
                    strokeWidth="2.5"
                  />
                </svg>
                <p className="hint">
                  损失用于诊断，模型选择以闭环验证表现为准。
                </p>
              </div>
            )}
            {run.progress.latest_validation_media && !run.progress.media && (
              <section className="validation-preview">
                <h3>最近闭环验证 · {run.progress.latest_validation_model}</h3>
                <video
                  controls
                  preload="metadata"
                  poster={media(
                    run.progress.latest_validation_media + "/observer.jpg",
                  )}
                  src={media(
                    run.progress.latest_validation_media + "/recording.mp4",
                  )}
                />
                <p className="hint">
                  已完成的验证录像，与当前训练步骤分开记录。
                </p>
              </section>
            )}
            {dataset && (
              <section className="dataset-preview">
                <h3>
                  示范覆盖 · {dataset.eligible} / {dataset.total} 条合格
                </h3>
                <details>
                  <summary>查看类别、颜色、难度与集合分布</summary>
                  {Object.entries(dataset.coverage).map(([key, count]) => (
                    <div className="coverage-row" key={key}>
                      <span>{key.split(" / ").map(tr).join(" · ")}</span>
                      <strong>{String(count)}</strong>
                    </div>
                  ))}
                </details>
                <div className="episode-grid">
                  {dataset.examples.map((e: any) => (
                    <div key={e.id}>
                      <video
                        controls
                        preload="none"
                        poster={media(e.media + "/observer.jpg")}
                        src={media(e.media + "/recording.mp4")}
                      />
                      <p>
                        {tr(e.category)} · {tr(e.color)} ·{" "}
                        {e.eligible ? "合格示范" : "失败记录"}
                      </p>
                      <small>
                        {e.split} · {tr(e.group)}
                      </small>
                    </div>
                  ))}
                </div>
              </section>
            )}
            {run.result.reason && <Info>{run.result.reason}</Info>}
            {report?.total != null && (
              <div className="report">
                <div className="result-number">
                  {report.passed}
                  <span> / {report.total}</span>
                  <p>严格成功</p>
                </div>
                <div>
                  <Badge
                    status={
                      report.accepted
                        ? "accepted_" + report.mode
                        : "experimental"
                    }
                  />
                  <p>
                    95% 置信区间{" "}
                    {report.confidence95
                      ?.map((n: number) => (n * 100).toFixed(1) + "%")
                      .join(" – ")}
                  </p>
                  <p>
                    失败：
                    {Object.entries(report.failures || {})
                      .map(([k, v]) => `${k} ${v}`)
                      .join(" · ") || "无"}
                  </p>
                </div>
              </div>
            )}
            {report?.episodes?.length > 0 && (
              <div className="episode-grid">
                {report.episodes.slice(0, 18).map((e: any, i: number) => (
                  <div key={i}>
                    {e.media && (
                      <video
                        controls
                        preload="none"
                        poster={media(e.media + "/observer.jpg")}
                        src={media(e.media + "/recording.mp4")}
                      />
                    )}
                    <p>
                      {tr(e.category)} · {tr(e.color)} ·{" "}
                      {e.success ? "通过" : e.reason}
                    </p>
                    <small>
                      {tr(e.group)} · seed {e.seed}
                    </small>
                  </div>
                ))}
              </div>
            )}
            {run.status === "complete" && report?.total == null && (
              <Info>
                作业已完成。能力状态需查看独立评测，完成进程本身不代表技能达标。
              </Info>
            )}
          </>
        )}
      </section>
    </div>
  );
}

function Workspace({ boot, onImport }: { boot: any; onImport: () => void }) {
  return (
    <>
      <div className="page-heading">
        <div>
          <h1>你的工作区，独立运行。</h1>
          <p>Web 管理任务与证据，Docker 运行器负责仿真和计算。</p>
        </div>
        <button onClick={onImport}>
          <UploadSimple />
          导入技能包
        </button>
      </div>
      <section className="workspace-section">
        <h2>运行器</h2>
        {boot.workers.length ? (
          boot.workers.map((w: any) => (
            <div className="skill-row" key={w.id}>
              <HardDrives size={32} />
              <div>
                <h3>{w.id}</h3>
                <p>
                  {w.capabilities.join(" · ")} · {w.image}
                </p>
              </div>
              <span className={"badge " + (w.online ? "good" : "amber")}>
                {w.online ? "在线" : "离线"}
              </span>
            </div>
          ))
        ) : (
          <Empty>尚未连接运行器。连接后才能执行仿真、采集和训练。</Empty>
        )}
        <p className="hint">
          同一 GPU
          运行器一次处理一个任务。中断的作业需要显式恢复，不会重复启动。
        </p>
      </section>
      <section className="workspace-section">
        <h2>部署与扩展</h2>
        <p>
          设置共享工作区、模型目录和运行器令牌，然后按部署文档启动独立容器。
        </p>
        <pre>
          docker compose -f deploy/microduck/studio/compose.yaml up --build
        </pre>
        <div className="two-col">
          <div>
            <h3>教师与任务插件</h3>
            <p>通过 Python entry point 注册任务、教师与独立评测器。</p>
          </div>
          <div>
            <h3>VLM 与工具接口</h3>
            <p>配置本地或云端 VLM，或让规划器调用发布后的高级技能。</p>
          </div>
        </div>
      </section>
    </>
  );
}

function ComparePanel({ runs }: { runs: Run[] }) {
  const [selected, setSelected] = useState<string[]>([]),
    [result, setResult] = useState<any>(null),
    [error, setError] = useState(""),
    [price, setPrice] = useState(""),
    [upfront, setUpfront] = useState("");
  const choices = runs.filter(
    (r) => r.status === "complete" && r.snapshot.request.kind === "evaluate",
  );
  if (choices.length < 2) return null;
  return (
    <details className="compare-panel">
      <summary>
        <ChartLine />
        选择两次评测，比较效果与成本
      </summary>
      <div className="comparison-choices">
        {choices.map((r) => (
          <label className="choice" key={r.id}>
            <input
              type="checkbox"
              checked={selected.includes(r.id)}
              onChange={(e) =>
                setSelected(
                  e.target.checked
                    ? [...selected.slice(-1), r.id]
                    : selected.filter((id) => id !== r.id),
                )
              }
            />
            <span>
              {r.snapshot.skill.name} · {tr(r.snapshot.request.policy)} ·{" "}
              {fmt(r.created)} · {r.result.passed}/{r.result.total}
            </span>
          </label>
        ))}
      </div>
      <div className="two-col">
        <label>
          GPU 每小时单价（可不填）
          <input
            type="number"
            min="0"
            value={price}
            onChange={(e) => setPrice(e.target.value)}
          />
        </label>
        <label>
          采集、训练与调试前期总成本（同币种）
          <input
            type="number"
            min="0"
            value={upfront}
            onChange={(e) => setUpfront(e.target.value)}
          />
        </label>
      </div>
      <button
        className="primary"
        disabled={selected.length !== 2}
        onClick={async () => {
          try {
            setError("");
            setResult(
              await api("/api/compare", {
                runs: selected,
                prices: {
                  ...(price ? { gpu_hour: +price } : {}),
                  ...(upfront ? { upfront_cost: +upfront } : {}),
                },
              }),
            );
          } catch (e) {
            setError((e as Error).message);
          }
        }}
      >
        比较这两次评测
        <ArrowRight />
      </button>
      {error && <Info>{error}</Info>}
      {result && (
        <>
          <Info>
            {result.comparable
              ? "任务条件和执行配置一致，可比较实际结果。"
              : "场景、案例或执行配置不一致；只展示结果，不给出优劣结论。"}
          </Info>
          <div className="comparison-results">
            {result.rows.map((r: any, i: number) => (
              <div key={i}>
                <h3>{i === 0 ? "对照" : "候选"}</h3>
                <p>成功率 {(r.success_rate * 100).toFixed(1)}%</p>
                <p>推理 p95 {r.p95_ms?.toFixed(1)} ms</p>
                <p>GPU {r.gpu_hours.toFixed(3)} 小时</p>
                <p>
                  每次成功成本 {r.cost_per_success?.toFixed(3) ?? "未填写单价"}
                </p>
              </div>
            ))}
          </div>
          {result.break_even_successes != null && (
            <p>
              计入上述前期成本后，约需成功复用 {result.break_even_successes}{" "}
              次回本。
            </p>
          )}
        </>
      )}
    </details>
  );
}
