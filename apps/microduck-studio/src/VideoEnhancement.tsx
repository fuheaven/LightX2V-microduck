import { useEffect, useRef, useState } from "react";
import {
  Sparkle,
  Play,
  Pause,
  CaretLeft,
  CaretRight,
  DownloadSimple,
  Plus,
  Trash,
} from "@phosphor-icons/react";
import { api, media, tr, requestId } from "./types";
import type { Run } from "./types";

export function VideoReview({ run }: { run: Run }) {
  const left = useRef<HTMLVideoElement>(null),
    right = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false),
    [frame, setFrame] = useState(0),
    [note, setNote] = useState(""),
    [error, setError] = useState(""),
    [saved, setSaved] = useState("");
  const [values, setValues] = useState<Record<string, string>>({
    target: "uncertain",
    layout: "uncertain",
    passage: "uncertain",
    occlusion: "uncertain",
    continuity: "uncertain",
  });
  const [issues, setIssues] = useState<
    { start_s: number; end_s: number; note: string }[]
  >([]);
  const ready = run.status === "complete",
    p = run.result.media || `runs/${run.id}/video`,
    fps = run.result.effective?.fps || 24;
  const count = run.result.effective?.frames || 0;
  useEffect(() => {
    let alive = true;
    setPlaying(false);
    setFrame(0);
    setSaved("");
    setNote("");
    setIssues([]);
    setError("");
    setValues(
      Object.fromEntries(
        ["target", "layout", "passage", "occlusion", "continuity"].map((k) => [
          k,
          "uncertain",
        ]),
      ),
    );
    if (!ready) return;
    api<any[]>(`/api/runs/${run.id}/video-reviews`)
      .then((rows) => {
        if (!alive) return;
        const last = rows.at(-1);
        if (last) {
          setSaved(last.status);
          setNote(last.note);
          setIssues(last.issues);
          setValues(
            Object.fromEntries(
              ["target", "layout", "passage", "occlusion", "continuity"].map(
                (k) => [k, last[k]],
              ),
            ),
          );
        }
      })
      .catch((e) => {
        if (alive) setError(e.message);
      });
    return () => {
      alive = false;
    };
  }, [run.id, ready]);
  useEffect(() => {
    if (!playing) return;
    let id = 0;
    const loop = () => {
      if (left.current && right.current) {
        const t = left.current.currentTime;
        if (Math.abs(right.current.currentTime - t) > 0.06)
          right.current.currentTime = t;
        setFrame(Math.round(t * fps));
      }
      id = requestAnimationFrame(loop);
    };
    loop();
    return () => cancelAnimationFrame(id);
  }, [playing, fps]);
  const seek = (n: number) => {
    setPlaying(false);
    const f = Math.max(0, Math.min(count - 1, n));
    setFrame(f);
    for (const v of [left.current, right.current])
      if (v) {
        v.pause();
        v.currentTime = f / fps;
      }
  };
  const toggle = async () => {
    if (playing) {
      left.current?.pause();
      right.current?.pause();
      setPlaying(false);
    } else {
      try {
        if (left.current?.ended) seek(0);
        await Promise.all([left.current?.play(), right.current?.play()]);
        setPlaying(true);
      } catch {
        setError("录像尚未准备好");
      }
    }
  };
  const save = async (status: string) => {
    try {
      setError("");
      await api(`/api/runs/${run.id}/video-reviews`, {
        ...values,
        status,
        note,
        issues,
      });
      setSaved(status);
    } catch (e) {
      setError((e as Error).message);
    }
  };
  if (!ready)
    return (
      <section className="video-stage">
        <Sparkle size={28} />
        <h3>{tr(run.status)} · 视频增强</h3>
        <p>
          {run.result.reason ||
            run.progress.message ||
            run.progress.phase ||
            "等待独立视频运行器"}
        </p>
        {run.progress.step && (
          <progress value={run.progress.step} max={run.progress.total} />
        )}
        <p className="hint">原始录像和任务成绩保留不变。</p>
      </section>
    );
  return (
    <section className="video-review">
      <div className="review-heading">
        <div>
          <div className="eyebrow">VISUAL REVIEW</div>
          <h3>对照原始证据，逐项检查</h3>
        </div>
        <span className="ai-label">
          AI 生成 ·{" "}
          {
            (
              {
                pending: "待审核",
                passed: "视觉审核通过",
                failed: "不通过",
                uncertain: "不确定",
              } as any
            )[saved || "pending"]
          }
        </span>
      </div>
      <div className="video-pair">
        <figure>
          <video
            ref={left}
            src={media(p + "/reference.mp4")}
            muted
            playsInline
            preload="metadata"
            onEnded={() => {
              right.current?.pause();
              setPlaying(false);
            }}
          />
          <figcaption>原始帧 · 按记录的时间映射重采样</figcaption>
        </figure>
        <figure>
          <div className="ai-overlay">AI 生成</div>
          <video
            ref={right}
            src={media(p + "/generated.mp4")}
            muted
            playsInline
            preload="metadata"
          />
          <figcaption>H3 视觉增强 · 仅供视觉审核</figcaption>
        </figure>
      </div>
      <div className="video-transport">
        <button aria-label="前一帧" onClick={() => seek(frame - 1)}>
          <CaretLeft />
        </button>
        <button aria-label={playing ? "暂停对照" : "播放对照"} onClick={toggle}>
          {playing ? <Pause /> : <Play />}
        </button>
        <button aria-label="后一帧" onClick={() => seek(frame + 1)}>
          <CaretRight />
        </button>
        <input
          type="range"
          aria-label="逐帧时间轴"
          min="0"
          max={Math.max(0, count - 1)}
          value={frame}
          onChange={(e) => seek(+e.target.value)}
        />
        <span>
          {(frame / fps).toFixed(2)}s · 帧 {frame}
        </span>
      </div>
      <div className="review-checks">
        {[
          ["target", "目标类别与颜色"],
          ["layout", "物体布局"],
          ["passage", "障碍与通道"],
          ["occlusion", "遮挡关系"],
          ["continuity", "运动连续性"],
        ].map(([key, label]) => (
          <label key={key}>
            {label}
            <select
              value={values[key]}
              onChange={(e) => setValues({ ...values, [key]: e.target.value })}
            >
              <option value="uncertain">不确定 / 未检查</option>
              <option value="passed">通过</option>
              <option value="failed">不通过</option>
            </select>
          </label>
        ))}
      </div>
      <div className="issue-list">
        <button
          onClick={() =>
            setIssues([
              ...issues,
              {
                start_s: frame / fps,
                end_s: Math.min((frame + 24) / fps, count / fps),
                note: "",
              },
            ])
          }
        >
          <Plus />
          标注当前时间段
        </button>
        {issues.map((issue, i) => (
          <div key={i}>
            <input
              aria-label="问题开始秒"
              type="number"
              min="0"
              step=".04"
              value={issue.start_s}
              onChange={(e) =>
                setIssues(
                  issues.map((v, j) =>
                    i === j ? { ...v, start_s: +e.target.value } : v,
                  ),
                )
              }
            />
            <span>至</span>
            <input
              aria-label="问题结束秒"
              type="number"
              min="0"
              step=".04"
              value={issue.end_s}
              onChange={(e) =>
                setIssues(
                  issues.map((v, j) =>
                    i === j ? { ...v, end_s: +e.target.value } : v,
                  ),
                )
              }
            />
            <input
              aria-label="问题描述"
              placeholder="描述看到的不一致"
              value={issue.note}
              onChange={(e) =>
                setIssues(
                  issues.map((v, j) =>
                    i === j ? { ...v, note: e.target.value } : v,
                  ),
                )
              }
            />
            <button
              aria-label="删除问题"
              onClick={() => setIssues(issues.filter((_, j) => i !== j))}
            >
              <Trash />
            </button>
          </div>
        ))}
      </div>
      <label>
        审核备注
        <textarea
          value={note}
          onChange={(e) => setNote(e.target.value)}
          placeholder="记录观察结果和不确定之处"
        />
      </label>
      <div className="button-row">
        <button className="primary" onClick={() => save("passed")}>
          视觉审核通过
        </button>
        <button onClick={() => save("failed")}>不通过</button>
        <button onClick={() => save("uncertain")}>不确定</button>
        <a className="button" href={`/api/runs/${run.id}/enhanced-export`}>
          <DownloadSimple />
          导出录像与报告
        </a>
      </div>
      {error && <p className="inline-error">{error}</p>}
      <p className="hint">
        审核通过只代表视觉检查通过。生成过程可能利用未来参考帧；增强录像不作为策略输入、训练数据或技能验收依据。
      </p>
      <details>
        <summary>生效参数与来源</summary>
        <pre>
          {JSON.stringify(
            {
              source: run.result.source,
              effective: run.result.effective,
              model: run.result.model,
              automatic_checks: run.result.automatic_checks,
              wall_seconds: run.result.wall_seconds,
              gpu_seconds: run.result.gpu_seconds,
            },
            null,
            2,
          )}
        </pre>
      </details>
    </section>
  );
}

export default function VideoEnhancement({ run }: { run: Run }) {
  const [recordings, setRecordings] = useState<any[]>([]),
    [status, setStatus] = useState<any>(),
    [selected, setSelected] = useState(""),
    [page, setPage] = useState(0);
  const [prompt, setPrompt] = useState(
      "将房间表现为自然光下的真实室内空间，保持鸭子、目标颜色、家具位置和运动轨迹一致。",
    ),
    [start, setStart] = useState(0),
    [duration, setDuration] = useState(5),
    [steps, setSteps] = useState(30),
    [seed, setSeed] = useState(42),
    [error, setError] = useState(""),
    [job, setJob] = useState<Run>(),
    [busy, setBusy] = useState(false);
  const source = recordings.find((r) => r.file === selected);
  useEffect(() => {
    api<any[]>(`/api/runs/${run.id}/recordings`)
      .then((rows) => {
        setRecordings(rows);
        setSelected(
          rows.find((r) => r.view === "observer")?.file || rows[0]?.file || "",
        );
      })
      .catch(() => {});
    api("/api/authoring/status")
      .then(setStatus)
      .catch(() => {});
  }, [run.id, run.status]);
  useEffect(() => {
    if (!job || !["queued", "running"].includes(job.status)) return;
    const t = setInterval(
      () =>
        api<Run>(`/api/runs/${job.id}`)
          .then(setJob)
          .catch((e) => setError(e.message)),
      2000,
    );
    return () => clearInterval(t);
  }, [job?.id, job?.status]);
  if (run.snapshot.request.kind === "video_enhance")
    return <VideoReview run={run} />;
  if (!recordings.length) return null;
  return (
    <section className="enhancement-panel">
      <div className="review-heading">
        <div>
          <div className="eyebrow">VIDEO LAB</div>
          <h3>
            <Sparkle />
            让仿真录像拥有新的质感
          </h3>
          <p className="hint">选择原始片段，生成后对照审核。</p>
        </div>
        <span className="ai-label">离线视觉增强</span>
      </div>
      <div className="recording-picker">
        {recordings.slice(page * 6, (page + 1) * 6).map((r) => (
          <button
            key={r.file}
            className={selected === r.file ? "selected" : ""}
            onClick={() => setSelected(r.file)}
          >
            {r.view === "head" ? "头部摄像头" : "观察者"}
            <small>{r.file.replace(/\/(head|observer).mp4$/, "")}</small>
          </button>
        ))}
      </div>
      {recordings.length > 6 && (
        <div className="button-row">
          <button disabled={!page} onClick={() => setPage(page - 1)}>
            上一页
          </button>
          <span>
            {page + 1} / {Math.ceil(recordings.length / 6)}
          </span>
          <button
            disabled={(page + 1) * 6 >= recordings.length}
            onClick={() => setPage(page + 1)}
          >
            下一页
          </button>
        </div>
      )}
      {source && (
        <video
          className="source-clip"
          src={media(source.path)}
          controls
          preload="metadata"
        />
      )}
      <div className="two-col">
        <label>
          从第几秒开始
          <input
            type="number"
            min="0"
            step=".1"
            value={start}
            onChange={(e) => setStart(+e.target.value)}
          />
        </label>
        <label>
          片段长度 · 秒
          <input
            type="number"
            min="5"
            max="15"
            value={duration}
            onChange={(e) => setDuration(+e.target.value)}
          />
        </label>
      </div>
      <label>
        材质、光线或装修风格
        <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} />
      </label>
      <details>
        <summary>生成参数</summary>
        <div className="two-col">
          <label>
            采样步数
            <input
              type="number"
              min="1"
              max="100"
              value={steps}
              onChange={(e) => setSteps(+e.target.value)}
            />
          </label>
          <label>
            种子
            <input
              type="number"
              min="0"
              value={seed}
              onChange={(e) => setSeed(+e.target.value)}
            />
          </label>
        </div>
        <p className="hint">
          960 × 544，24fps。帧数按模型规则调整，时间映射随结果保存。
        </p>
      </details>
      <div className="button-row">
        <button
          className="primary"
          disabled={
            !status?.video_ready ||
            busy ||
            !source ||
            (!!job && ["queued", "running"].includes(job.status))
          }
          onClick={async () => {
            setError("");
            setBusy(true);
            try {
              setJob(
                await api(`/api/skills/${run.skill_id}/runs`, {
                  kind: "video_enhance",
                  prompt,
                  video: {
                    source_run: run.id,
                    source_file: selected,
                    start_s: start,
                    duration_s: duration,
                    steps,
                    seed,
                  },
                  idempotency_key: requestId(),
                }),
              );
            } catch (e) {
              setError((e as Error).message);
            } finally {
              setBusy(false);
            }
          }}
        >
          <Sparkle />
          生成增强录像
        </button>
        {job && ["queued", "running"].includes(job.status) && (
          <button
            onClick={() => api(`/api/runs/${job.id}/cancel`, {}).then(setJob)}
          >
            取消生成
          </button>
        )}
      </div>
      {!status?.video_ready && (
        <p className="hint">{status?.video_reason || "正在检查模型运行器"}</p>
      )}
      {error && <p className="inline-error">{error}</p>}
      {job && <VideoReview run={job} />}
    </section>
  );
}
