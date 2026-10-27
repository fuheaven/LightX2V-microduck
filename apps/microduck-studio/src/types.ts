export type Obj = {
  id: string;
  kind: string;
  color: string;
  x: number;
  y: number;
  yaw: number;
  width: number;
  depth: number;
  height: number;
  jitter: number;
};
export type Scene = {
  name: string;
  size: number;
  spawn: {
    x: number;
    y: number;
    yaw: number;
    jitter: number;
    yaw_jitter: number;
  };
  target_id: string;
  objects: Obj[];
};
export type Task = {
  plugin: string;
  categories: string[];
  colors: string[];
  groups: string[];
  stop_min: number;
  stop_max: number;
  hold_s: number;
  timeout_s: number;
};
export type Skill = {
  id?: string;
  name: string;
  description: string;
  task: Task;
  scene: Scene;
};
export type Recipe = {
  backend: string;
  learning_rate: number;
  batch_size: number;
  micro_batch: number;
  max_steps: number;
  validate_every: number;
  max_hours: number;
  seed: number;
  precision: string;
};
export type Run = {
  id: string;
  skill_id: string;
  status: string;
  created: number;
  progress: Record<string, any>;
  result: Record<string, any>;
  snapshot: { skill: Skill; request: Record<string, any> };
};
export type Version = {
  id: string;
  skill_id: string;
  name: string;
  backend: string;
  status: string;
  report: Record<string, any>;
};
export const labels: Record<string, string> = {
  historical: "历史 FastWAM 实时运行",
  ball: "球",
  box: "盒子",
  toy: "玩偶",
  obstacle: "障碍",
  red: "红色",
  blue: "蓝色",
  yellow: "黄色",
  gray: "灰色",
  clear: "开阔区域",
  outside_view: "视野外搜索",
  occlusion: "遮挡后重新寻找",
  fastwam: "FastWAM",
  lightweight: "轻量视觉策略",
  teacher: "检查教师",
  collect: "采集示范",
  train: "训练模型",
  pipeline: "完整单轮流程",
  evaluate: "验证技能",
  optimize: "加速对照",
  live: "实时仿真",
  preview: "三维预览",
  correction: "人工纠错",
  queued: "排队中",
  running: "正在运行",
  complete: "作业完成",
  blocked: "门槛未通过",
  failed: "作业失败",
  cancelled: "已取消",
  interrupted: "已中断",
  experimental: "实验版本",
  accepted_offline: "离线已验收",
  accepted_realtime: "实时已验收",
  rule: "图像规则基线",
  vlm_local: "本地 VLM",
  vlm_cloud: "云端 VLM",
  training: "正在训练",
  collection: "正在采集",
  simulation: "仿真验证",
};
export const tr = (s: string) => labels[s] || s;
export const requestId = () =>
  globalThis.crypto?.randomUUID?.() ||
  `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
export async function api<T = any>(
  url: string,
  body?: any,
  method?: string,
): Promise<T> {
  const response = await fetch(url, {
    method: method || (body === undefined ? "GET" : "POST"),
    headers:
      body instanceof FormData ? {} : { "Content-Type": "application/json" },
    body:
      body === undefined
        ? undefined
        : body instanceof FormData
          ? body
          : JSON.stringify(body),
  });
  const result = await response.json();
  if (!response.ok)
    throw Error(
      typeof result.detail === "string"
        ? result.detail
        : JSON.stringify(result.detail),
    );
  return result;
}
export const media = (p: string) =>
  "/api/files/" + p.split("/").map(encodeURIComponent).join("/");
