"use strict";
const $ = id => document.getElementById(id);
const colors = ["#a7e7bc", "#8dbaf5", "#e9be82", "#e5a4d5", "#8bd6df", "#f2a69d"];
const stageNames = {cpu_replay:"重放与核对已保存动作",recovery_validation:"恢复前固定条件验证",resume_validation:"复核已有数据，准备恢复采集",preflight:"检查采集条件",collect:"采集教师示范",collection:"采集教师示范",merge_audit:"数据合并与审计",dagger_merge_audit:"纠偏数据合并与审计",data_audit:"检查数据",constant_mean_baseline:"评估常量对照",training:"模仿训练",development_evaluation:"开发集闭环评测",dagger:"DAgger 改进",dagger_collection:"采集学生纠偏示范",acceptance:"正式验收",evaluation:"评测",final_evaluation:"最终评测",complete:"流程结束"};
const statusNames = {running:"进行中",complete:"已完成",finished:"已结束",failed:"失败",partial_failed:"部分任务失败",cancelled:"已取消",allocation_finished:"分配已结束",unknown:"未知",unavailable:"尚无任务记录"};
let data = null, loading = false;
let queuedRefresh = false, queuedRun = "";
const comparisonEvidenceCache = new Map();
const outcomeSelections = new Map();
let replayVideosKey = "";
let pairedReplayKey = "";
const number = (value, digits=3) => typeof value === "number" && Number.isFinite(value) ? value.toLocaleString("zh-CN", {maximumFractionDigits:digits}) : "—";
const percent = value => typeof value === "number" && Number.isFinite(value) ? `${number(value*100,1)}%` : "未评测";
const date = value => value ? new Date(typeof value === "number" ? value*1000 : value).toLocaleString("zh-CN", {hour12:false}) : "尚无记录";
function element(tag, text, className) { const node=document.createElement(tag); if(text!==undefined)node.textContent=text; if(className)node.className=className; return node; }
function evidenceLink(id, label="报告") { const a=(data?.artifacts||[]).find(item=>item.id===id); if(!a?.available)return element("small","证据尚未同步"); const node=element("a",label); node.href=a.url;node.target="_blank";node.rel="noopener";return node; }
function counts(score) { return score ? `${number(score.successes,0)} / ${number(score.episodes,0)} 条 · ${number(score.distinct_conditions,0)} 个不同条件` : "暂无记录"; }
const failedCheckLabels = {
 correct_foot_contact:"指定脚未接触球",no_wrong_foot_contact:"另一只脚碰到球",ball_moved:"球移动距离不足",
 rolled:"未完成翻滚",circle_after_roll:"翻滚阶段后转向不足",travelled_after_roll:"翻滚阶段后行进不足",
 sat_down:"未达到下蹲要求",walked_after_sit:"下蹲阶段后行进不足",rise_complete_before_walk:"行走开始时未完成起身",
 walk_phase_started:"未进入行走阶段",forward_not_path_only:"有效前进距离不足",no_backward_excursion:"后退距离超限",
 terminal_stable:"结束时姿态不稳定",terminal_stopped:"结束时未停稳",no_unintended_fall:"发生意外跌倒",
 no_unintended_tip:"发生意外倾倒",no_collision:"发生碰撞",no_recovery_takeover:"发生恢复接管",
 initialization_valid:"初始化未达要求",no_fault:"运行发生故障",forward_progress:"前进距离不足",
 limited_lateral_drift:"横向偏移超限",left_turn:"左转幅度不足",right_turn:"右转幅度不足",travelled:"行进距离不足",
 left_arc:"左侧弧线不足",right_arc:"右侧弧线不足",left_leg:"首段左转不足",right_leg:"第二段右转不足",
 final_forward:"最后阶段行进不足",sway:"身体摇摆不足",circle:"转圈幅度不足",wheels_rolled:"滚轮转动不足",
 glided:"滑行距离不足",crouched:"蹲伏幅度不足",glided_after_crouch:"蹲伏后滑行不足"
};
function checkpointOutcomes(run, plan){
 if(!plan)return [];
 return (run?.checkpoints||[]).filter(row=>row.complete===true&&row.protocol==="project_qualified_evaluation_v2"&&Number.isSafeInteger(row.step)&&row.step>0).sort((a,b)=>b.step-a.step);
}
function validOutcomeScores(report, plan){
 const tasks=report?.tasks, integer=Number.isSafeInteger;
 if(!tasks||Object.keys(tasks).length!==plan.tasks.length)return false;
 return plan.tasks.every(task=>{const s=tasks[task],fail=s?.failed_checks;
  return s&&integer(s.episodes)&&s.episodes>0&&integer(s.successes)&&s.successes>=0&&s.successes<=s.episodes&&
   integer(s.distinct_conditions)&&s.distinct_conditions===s.episodes&&Number.isFinite(s.strict_rate)&&Math.abs(s.strict_rate-s.successes/s.episodes)<1e-9&&
   fail&&typeof fail==="object"&&!Array.isArray(fail)&&Object.values(fail).every(n=>integer(n)&&n>=0&&n<=s.episodes-s.successes);
 });
}
function renderCheckpointOutcomes(run){
 const section=$("checkpoint-outcomes"),select=$("outcome-checkpoint"),cards=$("checkpoint-outcome-cards"),note=$("checkpoint-outcomes-note"),evidence=$("checkpoint-outcomes-evidence");
 const plan=qualifiedPlan(run),reports=checkpointOutcomes(run,plan);
 section.hidden=!reports.length;select.replaceChildren();cards.replaceChildren();note.textContent="";evidence.replaceChildren();
 if(!reports.length)return;
 let selection=outcomeSelections.get(run.id)||"latest";
 if(selection!=="latest"&&!reports.some(r=>String(r.step)===selection))selection="latest";
 const latest=element("option",`最近完成 · step ${number(reports[0].step,0)}`);latest.value="latest";select.append(latest);
 for(const report of reports){const best=typeof report.checkpoint==="string"&&typeof report.checkpoint_sha256==="string"&&run.best_validation?.checkpoint===report.checkpoint&&run.best_validation?.checkpoint_sha256===report.checkpoint_sha256;
  const option=element("option",`step ${number(report.step,0)}${best?" · 本轮当前最佳":""}`);option.value=String(report.step);select.append(option);}
 select.value=selection;outcomeSelections.set(run.id,selection);
 const report=selection==="latest"?reports[0]:reports.find(r=>String(r.step)===selection);
 evidence.append(evidenceLink(report.artifact_id,"打开此 checkpoint 的完整验证报告"));
 if(!validOutcomeScores(report,plan)){note.textContent="这份报告的任务范围或计数缺失、不一致，暂不绘制结果。请查看原始报告。";return;}
 const scores=plan.tasks.map(t=>report.tasks[t]),successes=scores.reduce((n,s)=>n+s.successes,0),episodes=scores.reduce((n,s)=>n+s.episodes,0);
 note.textContent=`step ${number(report.step,0)} · ${number(plan.tasks.length,0)} 项任务，共 ${number(successes,0)} / ${number(episodes,0)} 条严格通过。这里是训练期间的小样本验证，不计入正式验收。`;
 const progress=run.checkpoint_validation;
 if(progress&&progress.step>reports[0].step&&progress.completed_episodes<progress.total_episodes)note.textContent+=` 第 ${number(progress.step,0)} 步仍在评测，已记录 ${number(progress.completed_episodes,0)} / ${number(progress.total_episodes,0)} 条；完成后可查看该 checkpoint 的完整结果。`;
 for(const task of plan.tasks){
  const score=report.tasks[task],card=element("article",undefined,"collection-task"+(score.successes===0?" failed":score.successes<score.episodes?" partial":""));
  card.dataset.task=task;
  const bar=element("progress");bar.max=score.episodes;bar.value=score.successes;bar.setAttribute("aria-label",`${taskLabel(task)}：${score.successes} / ${score.episodes} 条严格通过`);
  card.append(element("strong",taskLabel(task)),element("small",task),element("div",`${number(score.successes,0)} / ${number(score.episodes,0)} 条 · ${percent(score.strict_rate)}`,"outcome-rate"),bar);
  const failures=Object.entries(score.failed_checks).filter(([,n])=>n>0).sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0]));
  if(!failures.length)card.append(element("small",score.successes===score.episodes?"本次验证条件均通过；仍需独立验收。":"报告未提供失败分类。"));
  else {const list=element("ul");for(const [key,n] of failures){const item=element("li",`${failedCheckLabels[key]||key}：${number(n,0)} 条`);item.title=key;item.dataset.check=key;list.append(item);}card.append(element("small",`未通过 ${number(score.episodes-score.successes,0)} 条；涉及以下检查：`),list);}
  cards.append(card);
 }
}
function qualifiedPlan(run){
 const plan=run?.qualified_plan;
 if(!plan||!["legs","rollers"].includes(plan.body)||!Array.isArray(plan.tasks)||!plan.tasks.length||!plan.tasks.every(task=>typeof task==="string"&&/^[a-z][a-z0-9_]*$/.test(task))||new Set(plan.tasks).size!==plan.tasks.length)return null;
 if(!Array.isArray(plan.development_seeds)||!plan.development_seeds.length||!plan.development_seeds.every(seed=>Number.isSafeInteger(seed)&&seed>=0)||new Set(plan.development_seeds).size!==plan.development_seeds.length)return null;
 if(!Number.isSafeInteger(plan.total_development_episodes)||plan.total_development_episodes!==plan.tasks.length*plan.development_seeds.length||![plan.request_artifact_id,plan.plan_artifact_id].every(id=>typeof id==="string"&&id)||typeof plan.plan_sha256!=="string"||!/^[a-f0-9]{64}$/.test(plan.plan_sha256))return null;
 return plan;
}
function qualifiedDevelopment(run,plan){
 const report=run?.development_report;
 if(!plan||report?.protocol!=="project_qualified_evaluation_v2"||report.complete!==true||report.purpose!=="development"||report.evaluator_role!=="student"||report.ablation||!report.tasks||!sameKeys(Object.keys(report.tasks).sort(),[...plan.tasks].sort()))return null;
 const episodes=plan.development_seeds.length;
 if(plan.tasks.some(task=>{const score=report.tasks[task];return !score||score.episodes!==episodes||score.distinct_conditions!==episodes||!Number.isInteger(score.successes)||score.successes<0||score.successes>episodes||!Number.isFinite(score.strict_rate)||Math.abs(score.strict_rate-score.successes/episodes)>1e-12;}))return null;
 return report;
}
function taskLabel(task){return (data?.matrix||[]).find(row=>row.task===task)?.label||task;}
function renderDevelopmentProgress(run){
 const section=$("development-progress-section"),cards=$("development-progress-tasks"),note=$("development-progress-note"),evidence=$("development-progress-evidence");
 cards.replaceChildren();evidence.replaceChildren();note.textContent="";
 const plan=qualifiedPlan(run),p=run?.development_progress;
 section.hidden=!plan||!p||Boolean(qualifiedDevelopment(run,plan));if(section.hidden)return;
 const invalid=()=>{note.textContent="开发评测记录正在更新或尚未核实，暂不展示分任务计数。";};
 if(p.protocol!=="qualified_development_progress_v1"||p.counts_as_skill_success!==false||!["partial_records","records_complete"].includes(p.status)){invalid();return;}
 const integer=value=>Number.isSafeInteger(value)&&value>=0;
 if(!Array.isArray(p.tasks)||!sameKeys(p.tasks.map(row=>row?.task).sort(),[...plan.tasks].sort())||!integer(p.completed_episodes)||p.completed_episodes>plan.total_development_episodes||p.total_episodes!==plan.total_development_episodes||typeof p.progress_artifact_id!=="string"){invalid();return;}
 if(p.tasks.some(row=>!integer(row.episodes)||row.planned_episodes!==plan.development_seeds.length||row.episodes>row.planned_episodes||!integer(row.successes)||row.successes>row.episodes||!row.failed_checks||Array.isArray(row.failed_checks)||typeof row.failed_checks!=="object"||Object.values(row.failed_checks).some(count=>!integer(count)||count>row.episodes))||p.tasks.reduce((sum,row)=>sum+row.episodes,0)!==p.completed_episodes||p.status==="records_complete"&&p.completed_episodes!==p.total_episodes){invalid();return;}
 note.textContent=`当前作业已记录 ${number(p.completed_episodes,0)} / ${number(p.total_episodes,0)} 集。${p.status==="records_complete"?"逐集记录已齐，等待完整评测报告。":"尚未完成全部评测。"}下方通过数仅对应已完成记录，不计入正式验收；一集可能有多个失败检查项。`;
 for(const task of plan.tasks){const row=p.tasks.find(item=>item.task===task),card=element("article",undefined,"collection-task"),bar=element("progress");bar.max=row.planned_episodes;bar.value=row.episodes;bar.setAttribute("aria-label",`${taskLabel(task)}开发评测进度`);card.append(element("strong",taskLabel(task)),element("div",`已记录 ${number(row.episodes,0)} / ${number(row.planned_episodes,0)} 集`),bar,element("div",row.episodes?`已记录样本中严格通过 ${number(row.successes,0)} / ${number(row.episodes,0)} 集`:"尚未评测"));const failures=Object.entries(row.failed_checks).sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0]));if(failures.length)card.append(element("small",`失败检查项：${failures.map(([key,count])=>`${key} ${number(count,0)}`).join("；")}`));cards.append(card);}
 evidence.append(evidenceLink(p.progress_artifact_id,"查看当前逐集记录"));
}
function checkpointValidationText(run) {
  const p=run?.checkpoint_validation;
  if(p?.protocol!=="qualified_checkpoint_progress_v1"||p.counts_as_skill_success!==false||!Number.isInteger(p.step)||p.step<=0||p.step!==run.step)return null;
  const prefix=`step ${number(p.step,0)}`;
  if(p.status==="awaiting_report")return `${prefix} · 模型已保存，等待完整闭环报告`;
  if(p.status==="incomplete")return `${prefix} · 该模型的闭环检查未完成`;
  if(p.status==="invalid_progress")return `${prefix} · 闭环进度记录不一致，等待核查`;
  const valid=Number.isInteger(p.total_episodes)&&p.total_episodes>0&&Number.isInteger(p.completed_episodes)&&p.completed_episodes>=0&&p.completed_episodes<=p.total_episodes;
  if(!valid)return `${prefix} · 等待可核实的闭环进度`;
  if(p.status==="partial_report")return `${prefix} · 固定闭环已记录 ${number(p.completed_episodes,0)} / ${number(p.total_episodes,0)} 集`;
  if(p.status==="finalizing_report"&&p.completed_episodes===p.total_episodes)return `${prefix} · ${number(p.total_episodes,0)} 集已记录，等待完整报告`;
  return null;
}
function paddingComparison(run){
 const p=run?.padding_comparison,integer=value=>Number.isSafeInteger(value)&&value>=0;
 if(p?.protocol!=="padding_comparison_snapshot_v1"||p.counts_as_skill_success!==false||!["running","complete"].includes(p.status)||
    !((p.device==="CPU"&&p.precision==="fp32")||(p.device==="H200"&&p.precision==="bf16"))||
    !Array.isArray(p.seeds)||!p.seeds.length||!p.seeds.every(integer)||new Set(p.seeds).size!==p.seeds.length||
    !Array.isArray(p.bodies)||!p.bodies.length||new Set(p.bodies.map(b=>b?.body)).size!==p.bodies.length)return null;
 for(const body of p.bodies){
  if(!["legs","rollers"].includes(body?.body)||!["waiting","running","complete"].includes(body.status)||
     !Array.isArray(body.tasks)||!body.tasks.length||!body.tasks.every(t=>typeof t==="string")||new Set(body.tasks).size!==body.tasks.length||
     body.expected_episodes!==body.tasks.length*p.seeds.length||!body.arms||!Array.isArray(body.manifest_artifact_ids))return null;
  if(body.status==="waiting"){if(Object.keys(body.arms).length)return null;continue;}
  if(!/^[a-f0-9]{64}$/.test(body.weights_sha256)||body.manifest_artifact_ids.length!==2||Object.keys(body.arms).length!==2)return null;
  for(const key of ["legacy","batch_consistent"]){
   const arm=body.arms[key];
   if(!arm||!["waiting","partial","finalizing","complete"].includes(arm.status)||!integer(arm.episodes)||arm.episodes>body.expected_episodes||
      !integer(arm.successes)||arm.successes>arm.episodes||!Array.isArray(arm.tasks)||
      new Set(arm.tasks.map(row=>row?.task)).size!==arm.tasks.length)return null;
   if(arm.status==="waiting"){if(arm.episodes||arm.successes||arm.tasks.length)return null;continue;}
   if(arm.tasks.length!==body.tasks.length||arm.tasks.some(row=>!body.tasks.includes(row?.task)||!integer(row.episodes)||row.episodes>p.seeds.length||!integer(row.successes)||row.successes>row.episodes)||
      arm.tasks.reduce((sum,row)=>sum+row.episodes,0)!==arm.episodes||arm.tasks.reduce((sum,row)=>sum+row.successes,0)!==arm.successes||
      (["finalizing","complete"].includes(arm.status)&&arm.episodes!==body.expected_episodes))return null;
  }
  if(body.status==="complete"&&!Object.values(body.arms).every(a=>a.status==="complete"))return null;
 }
 if(p.status==="complete"&&!p.bodies.every(b=>b.status==="complete"))return null;
 return p;
}
function renderPaddingComparison(run){
 const section=$("padding-comparison-section"),note=$("padding-comparison-note"),container=$("padding-comparison-bodies");
 section.hidden=!run?.padding_comparison;container.replaceChildren();note.textContent="";
 if(section.hidden)return;
 const p=paddingComparison(run);
 if(!p){note.textContent="对照工件或逐集记录尚未匹配，暂不显示统计。请查看作业原始报告。";return;}
 const total=p.bodies.reduce((sum,b)=>sum+b.expected_episodes*2,0),done=p.bodies.reduce((sum,b)=>sum+Object.values(b.arms).reduce((n,a)=>n+a.episodes,0),0);
 note.textContent=`${p.device} · ${p.precision.toUpperCase()} · 每项 ${number(p.seeds.length,0)} 个固定条件 · 两组共已记录 ${number(done,0)} / ${number(total,0)} 条${p.status==="complete"?"，完整对照已生成。":"，其余条件仍待评测。"} CPU 与 H200 的结果分别保留，不合并为一个成绩。`;
 const labels={legacy:"原填充处理",batch_consistent:"统一填充处理"};
 for(const b of p.bodies){
  const article=element("article",undefined,"padding-body");article.dataset.body=b.body;
  article.append(element("h4",`${b.body==="legs"?"腿式鸭子":"滚轮鸭子"} · ${b.tasks.length} 项任务${b.checkpoint?` · ${b.checkpoint}`:""}`));
  if(b.status==="waiting"){article.append(element("p","等待此本体的模型工件与评测记录；尚无可比较的结果。"));container.append(article);continue;}
  const arms=element("div",undefined,"collection-grid");
  for(const key of ["legacy","batch_consistent"]){
   const a=b.arms[key],card=element("div",undefined,"collection-task"),bar=element("progress");card.dataset.arm=key;
   bar.max=b.expected_episodes;bar.value=a.episodes;bar.setAttribute("aria-label",`${labels[key]}：已记录 ${a.episodes} / ${b.expected_episodes} 条`);
   card.append(element("strong",labels[key]),element("div",`已记录 ${number(a.episodes,0)} / ${number(b.expected_episodes,0)} 条`),bar,
    element("small",a.episodes?`其中严格通过 ${number(a.successes,0)} / ${number(a.episodes,0)} 条${a.status==="complete"?" · 完整报告":" · 当前部分记录"}`:"尚未记录结果"));
   if(a.status==="finalizing")card.append(element("small","所有条件已记录，等待完整报告。"));
   arms.append(card);
  }
  article.append(arms);
  const wrap=element("div",undefined,"table-wrap"),table=element("table",undefined,"padding-table"),head=element("thead"),titles=element("tr"),tbody=element("tbody");
  for(const title of ["任务",labels.legacy,labels.batch_consistent,"通过条数变化"])titles.append(element("th",title));head.append(titles);table.append(head,tbody);
  for(const task of b.tasks){
   const tr=element("tr");tr.dataset.task=task;tr.append(element("th",taskLabel(task)));
   const values={};
   for(const key of ["legacy","batch_consistent"]){
    const a=b.arms[key],score=a.tasks.find(row=>row.task===task);values[key]=score;
    const td=element("td");
    if(!score?.episodes)td.append(element("span","待评测","pending"));
    else td.append(element("strong",`${score.successes} / ${score.episodes} 通过`),element("small",`已记录 ${score.episodes} / ${p.seeds.length}`));
    tr.append(td);
   }
   const delta=b.status==="complete"?values.batch_consistent.successes-values.legacy.successes:null;
   tr.append(element("td",delta===null?"待完整对照":delta===0?"不变":`${delta>0?"+":""}${delta} 条`,delta===null?"pending":""));tbody.append(tr);
  }
  wrap.append(table);article.append(wrap);
  const details=element("details",undefined,"padding-evidence");details.append(element("summary","模型与原始记录"),element("p",`权重 SHA256：${b.weights_sha256}`));
  b.manifest_artifact_ids.forEach((id,i)=>details.append(evidenceLink(id,i?"统一填充处理 · 模型清单":"原填充处理 · 模型清单")));
  for(const key of ["legacy","batch_consistent"]){const a=b.arms[key];if(a.artifact_id)details.append(evidenceLink(a.artifact_id,`${labels[key]} · ${a.status==="complete"?"完整逐集报告":"当前逐集记录"}`));}
  article.append(details);container.append(article);
 }
}
function reselection(run){
 const p=run?.padding_reselection,integer=v=>Number.isSafeInteger(v)&&v>=0;
 if(p?.protocol!=="padding_reselection_snapshot_v1"||p.counts_as_skill_success!==false||!["running","interrupted","complete"].includes(p.status)||p.precision!=="bf16"||p.device!=="H200"||p.total_candidates!==19||p.expected_episodes!==600||
    !Array.isArray(p.bodies)||p.bodies.length!==2||p.bodies[0]?.body!=="legs"||p.bodies[1]?.body!=="rollers")return null;
 let done=0,episodes=0;
 for(const b of p.bodies){
  if(!Array.isArray(b.tasks)||b.tasks.length!==(b.body==="legs"?10:3)||!Array.isArray(b.candidates)||b.candidates.length!==(b.body==="legs"?9:10))return null;
  for(const [i,c]of b.candidates.entries()){
   if(c.step!==(i+1)*500||!["waiting","partial","finalizing","complete"].includes(c.status)||!integer(c.episodes)||c.expected_episodes!==b.tasks.length*5||c.episodes>c.expected_episodes||!integer(c.successes)||c.successes>c.episodes||!Array.isArray(c.tasks))return null;
   if(c.status==="waiting"){if(c.episodes||c.successes||c.tasks.length)return null;continue;}
   if(c.tasks.length!==b.tasks.length||new Set(c.tasks.map(t=>t.task)).size!==b.tasks.length||c.tasks.some(t=>!b.tasks.includes(t.task)||!integer(t.episodes)||t.episodes>5||!integer(t.successes)||t.successes>t.episodes)||
      c.tasks.reduce((n,t)=>n+t.episodes,0)!==c.episodes||c.tasks.reduce((n,t)=>n+t.successes,0)!==c.successes||
      (["complete","finalizing"].includes(c.status)&&c.episodes!==c.expected_episodes))return null;
   episodes+=c.episodes;done+=Number(c.status==="complete");
  }
  if(b.selected_step!==null&&(!b.candidates.every(c=>c.status==="complete")||!b.candidates.some(c=>c.step===b.selected_step)||!b.selection_artifact_id))return null;
 }
 if(done!==p.completed_candidates||episodes!==p.recorded_episodes||p.status==="complete"&&(done!==19||!p.selected_for_development||p.bodies.some(b=>b.selected_step===null)))return null;
 return p;
}
function renderReselection(run){
 const section=$("reselection-section"),container=$("reselection-bodies"),note=$("reselection-note");
 section.hidden=!run?.padding_reselection;container.replaceChildren();note.textContent="";if(section.hidden)return;
 const p=reselection(run);if(!p){note.textContent="模型或评测记录尚未匹配，暂不显示比较结果。";return;}
 note.textContent=`H200 · BF16 · 完整评测 ${p.completed_candidates} / 19 个模型 · 已记录 ${p.recorded_episodes} / 600 条${p.status==="interrupted"?"。作业已中断，保留已完成记录；尚未完成模型选择。":p.status==="complete"?"。两种本体的模型选择已完成，仍需开发评测。":"。未完成的模型不参与最终选择。"}`;
 for(const b of p.bodies){
  const article=element("article",undefined,"padding-body");article.dataset.body=b.body;
  article.append(element("h4",`${b.body==="legs"?"腿式鸭子":"滚轮鸭子"} · ${b.candidates.length} 个 checkpoint`));
  const plot=element("div",undefined,"chart"),legend=element("div",undefined,"legend");plot.id=`reselection-${b.body}-chart`;legend.id=`reselection-${b.body}-legend`;article.append(plot,legend);
  plot.hidden=legend.hidden=!b.candidates.some(c=>c.status==="complete");
  if(plot.hidden)article.append(element("p","首个模型完成全部条件后显示成功率曲线；下表持续更新逐集进度。"));
  const wrap=element("div",undefined,"table-wrap"),table=element("table",undefined,"diagnostic-table reselection-table"),head=element("thead"),tr=element("tr"),tbody=element("tbody");
  for(const title of ["模型 / 评测进度",...b.tasks.map(taskLabel)])tr.append(element("th",title));head.append(tr);table.append(head,tbody);
  for(const c of b.candidates){
   const row=element("tr");row.dataset.step=c.step;const th=element("th",`step ${number(c.step,0)}`);
   th.append(element("small",c.status==="waiting"?"待评测":`${c.episodes} / ${c.expected_episodes} 条 · ${c.status==="complete"?"完整报告":c.status==="finalizing"?"整理报告":"部分记录"}`));
   if(c.step===b.selected_step)th.append(element("small","选入开发评测","accepted"));
   if(c.artifact_id)th.append(evidenceLink(c.artifact_id,"逐集记录"));row.append(th);
   for(const task of b.tasks){const score=c.tasks.find(t=>t.task===task),td=element("td",score?.episodes?`${score.successes} / ${score.episodes}`:"待评测");if(score?.episodes){td.className=score.episodes<5?"partial":score.successes>=4?"good":"low";td.title=score.episodes<5?`部分记录，已评测 ${score.episodes} / 5 条`:"固定验证条件，非正式验收";}row.append(td);}
   tbody.append(row);
  }
  wrap.append(table);article.append(wrap);
  if(b.selection_artifact_id)article.append(evidenceLink(b.selection_artifact_id,"模型选择报告"));
  container.append(article);
  const rows=b.candidates.map(c=>Object.assign({step:c.step},Object.fromEntries(b.tasks.map(t=>{const s=c.tasks.find(v=>v.task===t);return [t,c.status==="complete"?s.successes/5:null];}))));
  chart(plot.id,legend.id,rows,b.tasks.map(t=>({key:t,label:taskLabel(t)})),{rate:true});
 }
}
function paddingDevelopment(run){
 const p=run?.padding_development,integer=n=>Number.isSafeInteger(n)&&n>=0;
 const tasks=["walk_forward","circle_left","figure_eight","zigzag","sway_circle","kick_left","kick_right","ground_lower","sit_walk","roll_circle","roller_glide","roller_crouch","roller_slalom"];
 if(p?.protocol!=="padding_development_snapshot_v1"||p.counts_as_skill_success!==false||p.precision!=="bf16"||p.padding_policy!=="batch_consistent_v1"||p.expected_episodes!==260||p.readiness_successes_per_task!==18||
    !["running","interrupted","records_complete","complete"].includes(p.status)||!Array.isArray(p.seeds)||!sameKeys(p.seeds,Array.from({length:20},(_,i)=>610000+i))||!Array.isArray(p.bodies)||p.bodies.length!==2)return null;
 for(const [i,b]of p.bodies.entries()){
  const body=["legs","rollers"][i],expected=tasks.filter(t=>t.startsWith("roller_")===(body==="rollers")).sort();
  if(b.body!==body||!integer(b.step)||b.step<500||b.step%500||b.step>(i?5000:4500)||!["waiting","partial","records_complete","complete"].includes(b.status)||b.expected_episodes!==expected.length*20||
     !Array.isArray(b.tasks)||!sameKeys(b.tasks.map(t=>t?.task).sort(),expected)||!integer(b.episodes)||!integer(b.successes)||b.episodes>b.expected_episodes||b.successes>b.episodes||
     ![true,false,null].includes(b.ready_for_final_test))return null;
  for(const t of b.tasks){
   if(!integer(t.episodes)||t.episodes>20||t.planned_episodes!==20||!integer(t.successes)||t.successes>t.episodes||![true,false,null].includes(t.gate)||
      !t.failed_checks||typeof t.failed_checks!=="object"||Array.isArray(t.failed_checks)||Object.values(t.failed_checks).some(v=>!integer(v)||v>t.episodes)||
      (b.status==="complete"?t.episodes!==20||t.gate!==(t.successes>=18):t.gate!==null))return null;
  }
  if(b.episodes!==b.tasks.reduce((n,t)=>n+t.episodes,0)||b.successes!==b.tasks.reduce((n,t)=>n+t.successes,0)||
     b.status==="waiting"&&b.episodes||["records_complete","complete"].includes(b.status)&&b.episodes!==b.expected_episodes||
     (b.status==="complete"?b.ready_for_final_test!==b.tasks.every(t=>t.gate)||!b.result_artifact_id:b.ready_for_final_test!==null))return null;
 }
 if(p.recorded_episodes!==p.bodies.reduce((n,b)=>n+b.episodes,0)||
    (["complete","records_complete"].includes(p.status)&&(!p.bodies.every(b=>b.status==="complete")||p.recorded_episodes!==260||!p.result_artifact_id))||
    (p.status==="complete"?p.ready_for_final_test!==p.bodies.every(b=>b.ready_for_final_test):p.ready_for_final_test!==null))return null;
 return p;
}
function renderPaddingDevelopment(run){
 const section=$("padding-development-section"),content=$("padding-development-bodies"),note=$("padding-development-note"),links=$("padding-development-evidence");
 section.hidden=!run?.padding_development;content.replaceChildren();links.replaceChildren();note.textContent="";if(section.hidden)return;
 const p=paddingDevelopment(run);if(!p){note.textContent="开发评测记录尚未匹配，暂不展示计数和门槛结论。";return;}
 const ending=p.status==="complete"?(p.ready_for_final_test?"全部技能达到开发门槛，待独立最终测试。":"部分技能未达到开发门槛，需要继续改进。"):
  p.status==="records_complete"?"报告已齐，等待作业正常结束及最终核对。":p.status==="interrupted"?"作业已中断，保留可核实的记录。":"正在评测；未完成部分不填零分。";
 note.textContent=`已记录 ${p.recorded_episodes} / 260 集 · ${ending}这些成绩不计入正式验收。`;
 for(const b of p.bodies){
  const article=element("article",undefined,"padding-body");article.dataset.body=b.body;
  const title=b.body==="legs"?"腿式鸭子 · 10 项任务":"滚轮鸭子 · 3 项任务",bar=element("progress");bar.max=b.expected_episodes;bar.value=b.episodes;bar.setAttribute("aria-label",`${title}开发评测进度`);bar.style.width="100%";
  article.append(element("h4",`${title} · checkpoint ${number(b.step,0)}`),element("p",`已记录 ${b.episodes} / ${b.expected_episodes} 集`),bar);
  const wrap=element("div",undefined,"table-wrap"),table=element("table",undefined,"padding-table"),head=element("thead"),hr=element("tr"),rows=element("tbody");
  for(const label of ["技能","已评测 / 计划","严格通过","开发门槛"])hr.append(element("th",label));head.append(hr);table.append(head,rows);
  for(const t of b.tasks){const row=element("tr");row.dataset.task=t.task;const verdict=element("td",t.gate===null?(t.episodes?"待完整报告":"待评测"):t.gate?"达到 18/20":"未达到 18/20",t.gate===null?"":t.gate?"accepted":"failed");
   row.append(element("th",taskLabel(t.task)),element("td",`${t.episodes} / 20`),element("td",t.episodes?`${t.successes} / ${t.episodes}`:"待评测"),verdict);
   const failures=Object.entries(t.failed_checks).filter(([,n])=>n>0).sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0]));
   if(failures.length){const details=element("details"),list=element("ul");details.append(element("summary","失败原因"));for(const [key,n]of failures)list.append(element("li",`${failedCheckLabels[key]||key}：${n} 集`));details.append(list,element("small","一集可能触发多项失败检查。"));verdict.append(details);}
   rows.append(row);
  }
  wrap.append(table);article.append(wrap);const evidence=element("p",undefined,"padding-evidence");
  if(b.report_artifact_id)evidence.append(evidenceLink(b.report_artifact_id,b.status==="partial"?"当前逐集记录":"开发评测报告"));
  if(b.result_artifact_id)evidence.append(evidenceLink(b.result_artifact_id,"本体开发门槛报告"));article.append(evidence);content.append(article);
 }
 for(const [id,label]of [[p.plan_artifact_id,"本轮开发评测计划"],[p.selection_artifact_id,"候选模型选择报告"],[p.result_artifact_id,"完整开发集结果"]])if(id)links.append(evidenceLink(id,label));
}
function precisionProgress(run){
 const p=run?.precision_progress,integer=n=>Number.isSafeInteger(n)&&n>=0;
 if(p?.protocol!=="padding_precision_progress_v1"||p.counts_as_skill_success!==false||!["partial","records_complete"].includes(p.status)||p.expected_episodes!==65||!Array.isArray(p.bodies)||p.bodies.length!==2)return null;
 for(const [i,b]of p.bodies.entries())if(b.body!==["legs","rollers"][i]||b.expected_episodes!==[50,15][i]||!integer(b.episodes)||b.episodes>b.expected_episodes||!["waiting","partial","records_complete"].includes(b.status)||b.status==="waiting"&&b.episodes||b.status==="records_complete"&&b.episodes!==b.expected_episodes)return null;
 if(p.recorded_episodes!==p.bodies.reduce((n,b)=>n+b.episodes,0)||p.status==="records_complete"&&p.recorded_episodes!==65)return null;
 return p;
}
function precisionComparison(run){
 const p=run?.precision_comparison,integer=n=>Number.isSafeInteger(n)&&n>=0;
 if(p?.protocol!=="padding_precision_comparison_v1"||p.status!=="complete"||p.device!=="H200"||p.counts_as_skill_success!==false||p.conditions_per_arm!==65||!Array.isArray(p.seeds)||!sameKeys(p.seeds,[500000,500001,500002,500003,500004])||!Array.isArray(p.bodies)||p.bodies.length!==2||!p.report_artifact_id)return null;
 for(const [i,b]of p.bodies.entries()){
  if(b.body!==["legs","rollers"][i]||b.episodes!==[50,15][i]||!Array.isArray(b.tasks)||b.tasks.length!==[10,3][i]||!integer(b.bf16_successes)||!integer(b.fp32_successes)||!b.report_artifact_ids?.bf16||!b.report_artifact_ids?.fp32)return null;
  if(new Set(b.tasks.map(t=>t.task)).size!==b.tasks.length)return null;
  for(const t of b.tasks)if(typeof t.task!=="string"||t.task.startsWith("roller_")!==(b.body==="rollers")||t.episodes!==5||!integer(t.bf16_successes)||!integer(t.fp32_successes)||t.bf16_successes>5||t.fp32_successes>5)return null;
  for(const key of ["bf16_successes","fp32_successes"])if(b[key]!==b.tasks.reduce((n,t)=>n+t[key],0))return null;
 }
 return p;
}
function renderPrecisionComparison(run){
 const box=$("precision-comparison-bodies");box.replaceChildren();
 if(!run?.precision_comparison||run.precision_comparison.status==="waiting")return;
 const p=precisionComparison(run);if(!p){box.append(element("p","完整精度对照证据尚未匹配，暂不显示统计。"));return;}
 box.append(element("p","完整验证对照已生成：每项使用相同的 5 个条件。通过条数不能替代最终验收，也不代表实时部署效果。"));
 for(const b of p.bodies){
  const article=element("article",undefined,"padding-body");article.dataset.body=b.body;
  article.append(element("h4",`${b.body==="legs"?"腿式鸭子":"滚轮鸭子"} · BF16 ${b.bf16_successes}/${b.episodes} · FP32 ${b.fp32_successes}/${b.episodes}`));
  const wrap=element("div",undefined,"table-wrap"),table=element("table",undefined,"padding-table"),head=element("thead"),titles=element("tr"),tbody=element("tbody");
  for(const title of ["任务","BF16 通过","FP32 通过","FP32 条数变化"])titles.append(element("th",title));head.append(titles);table.append(head,tbody);
  for(const t of b.tasks){const tr=element("tr"),delta=t.fp32_successes-t.bf16_successes;tr.dataset.task=t.task;tr.append(element("th",taskLabel(t.task)),element("td",`${t.bf16_successes} / 5`),element("td",`${t.fp32_successes} / 5`),element("td",delta===0?"不变":`${delta>0?"+":""}${delta} 条`));tbody.append(tr);}
  wrap.append(table);article.append(wrap,evidenceLink(b.report_artifact_ids.bf16,"BF16 完整逐集报告"),document.createTextNode(" · "),evidenceLink(b.report_artifact_ids.fp32,"FP32 完整逐集报告"));box.append(article);
 }
 box.append(evidenceLink(p.report_artifact_id,"完整配对报告"),document.createTextNode(" · "),evidenceLink(p.request_artifact_id,"固定精度实验配置"));
}
function renderPrecisionProgress(run){
 const section=$("precision-progress-section"),note=$("precision-progress-note"),content=$("precision-progress-bodies");section.hidden=!run?.precision_progress;content.replaceChildren();$("precision-comparison-bodies").replaceChildren();note.textContent="";if(section.hidden)return;
 const p=precisionProgress(run);if(!p){note.textContent="精度评测进度记录尚未匹配，等待核查。";return;}
 note.textContent=`新增 FP32 记录 ${p.recorded_episodes} / 65 条；BF16 对照复用此前完整记录，不重复计入本轮进度。`;
 for(const b of p.bodies){const card=element("article",undefined,"collection-task"),bar=element("progress");bar.max=b.expected_episodes;bar.value=b.episodes;bar.setAttribute("aria-label",`${b.body}：${b.episodes} / ${b.expected_episodes}`);card.append(element("strong",b.body==="legs"?"腿式鸭子 · 10 项任务":"滚轮鸭子 · 3 项任务"),element("span",`${b.episodes} / ${b.expected_episodes} 条`),bar,element("small",b.status==="waiting"?"待评测":b.status==="records_complete"?(precisionComparison(run)?"完整逐集与配对报告已核对":"逐集记录已完成，结果仍需完整配对核查"):"正在评测，尚无完整对照结果"));if(b.artifact_id)card.append(evidenceLink(b.artifact_id,"逐集记录"));content.append(card);}
 renderPrecisionComparison(run);
}
function jobStage(run) {
 if(paddingDevelopment(run))return "候选模型开发集评测";
 if(precisionProgress(run))return "推理精度对照（FP32）";
 if(reselection(run))return "历史模型重新比较";
 if(paddingComparison(run))return "填充处理对照评测";
 const p=run?.checkpoint_validation;
 if(run?.stage==="training"&&run.status==="running"&&run.allocation?.status==="running"&&
    p?.protocol==="qualified_checkpoint_progress_v1"&&p.counts_as_skill_success===false&&
    Number.isInteger(p.step)&&p.step>0&&p.step===run.step&&Number.isInteger(p.total_episodes)&&p.total_episodes>0&&
    Number.isInteger(p.completed_episodes)&&p.completed_episodes>=0&&p.completed_episodes<=p.total_episodes){
  if(p.status==="partial_report"&&p.completed_episodes<p.total_episodes)return "固定验证集闭环评测";
  if(p.status==="finalizing_report"&&p.completed_episodes===p.total_episodes)return "整理闭环评测报告";
 }
 if(run?.prefix_collection&&run.stage==="collection")return "采集纠偏数据";
 return stageNames[run?.stage]||run?.stage||statusNames[run?.status]||"未知";
}
function jobProgress(run) {
 const development=paddingDevelopment(run);if(development)return `开发集已记录 ${development.recorded_episodes} / 260 集`;
 const precision=precisionProgress(run);if(precision)return `新增 FP32 记录 ${precision.recorded_episodes} / 65 条`;
 const selection=reselection(run);if(selection)return `${selection.completed_candidates} / 19 个模型完成评测 · ${selection.recorded_episodes} / 600 条验证记录`;
 const padding=paddingComparison(run);
 if(padding){const total=padding.bodies.reduce((n,b)=>n+b.expected_episodes*2,0),done=padding.bodies.reduce((n,b)=>n+Object.values(b.arms).reduce((s,a)=>s+a.episodes,0),0);return `两组已记录 ${number(done,0)} / ${number(total,0)} 条 · 诊断不计入验收`;}
 if(run?.stage==="cpu_replay"){
  const p=run.replay_progress;
  return p?`已重放 ${number(p.completed,0)} / ${number(p.total,0)} 条记录；诊断不计入技能验收`:"正在核对已保存动作，等待进度记录";
 }
 if(run?.stage==="dagger_collection"&&qualifiedPlan(run)){
  const done=run.completed_tasks,total=run.total_tasks;
  return Number.isSafeInteger(done)&&Number.isSafeInteger(total)&&total>0&&done>=0&&done<=total
   ? `已完成 ${number(done,0)} / ${number(total,0)} 项任务的采集；数据仍需审计`
   : "正在采集多任务纠偏数据";
 }
 if(run?.stage==="recovery_validation")return `已验证 ${number(run.recovery_validation_episodes,0)} 条固定条件`;if(run?.stage==="development_evaluation"){const count=run.development_episodes,total=qualifiedPlan(run)?.total_development_episodes??run.development_plan?.total_episodes;return Number.isInteger(count)?`已评测 ${number(count,0)}${Number.isInteger(total)&&total>0&&count>=0&&count<=total?` / ${number(total,0)}`:""} 条开发条件`:"等待开发评测记录";}const checkpointNotice=checkpointValidationText(run);if(checkpointNotice)return checkpointNotice;return ["collect","collection","dagger_collection"].includes(run?.stage) ? `${number(run.count,0)} / ${number(run.requested_episodes,0)} 集` : run?.step!=null ? `step ${number(run.step,0)}` : "等待阶段记录"; }
function renderRunning(){const runs=(data?.runs||[]).filter(r=>r.allocation?.status==="running");$("running-section").hidden=!runs.length;$("running-jobs").replaceChildren();for(const run of runs){const card=element("article",undefined,"job-card");card.append(element("strong",`${run.allocation.hardware==="cpu"?"CPU":"GPU"} · ${run.id}`),element("div",`${jobStage(run)} · ${jobProgress(run)}`));if(run.error)card.append(element("small",run.error));const button=element("button","查看此任务");button.type="button";button.addEventListener("click",()=>{$("run-select").value=run.id;refresh();});card.append(button);$("running-jobs").append(card);}}
function renderCorrectionTasks(labels){
 const container=$("correction-by-task");container.replaceChildren();container.hidden=true;
 const rows=labels?.by_task,keys=["episodes","branches","teacher_completable","rejected","unverified"],integer=value=>Number.isSafeInteger(value)&&value>=0;
 if(labels?.counts_as_student_success!==false||!Array.isArray(rows)||!rows.length)return;
 if(rows.some(row=>typeof row?.task!=="string"||!keys.every(key=>integer(row[key]))||row.branches!==row.teacher_completable+row.rejected+row.unverified)||new Set(rows.map(row=>row.task)).size!==rows.length)return;
 if(!keys.every(key=>integer(labels[key])&&rows.reduce((sum,row)=>sum+row[key],0)===labels[key]))return;
 const table=element("table",undefined,"diagnostic-table"),head=element("thead"),titles=element("tr"),body=element("tbody");
 table.append(element("caption","逐任务纠偏数据 · 尚未筛选训练集"));
 for(const label of ["任务","已记录集数","教师可完成","已拒绝","未核实"])titles.append(element("th",label));
 head.append(titles);table.append(head,body);
 for(const row of rows){const tr=element("tr");tr.append(element("td",taskLabel(row.task)));for(const key of ["episodes","teacher_completable","rejected","unverified"])tr.append(element("td",number(row[key],0)));body.append(tr);}
 container.append(table);container.hidden=false;
}
function renderCollection(run){renderCorrectionTasks(run?.correction_labels);const tasks=run?.collection_tasks||[];const summary=run?.collection_summary;$("collection-summary").textContent=summary?`已记录 ${number(summary.recorded_episodes,0)} 集；任务文件校验通过 ${number(summary.file_verified_episodes,0)} 集；失败批次中有 ${number(summary.failed_task_episodes,0)} 集暂不可用。完整数据集仍需合并与审计。`:"记录数量不等于可训练数据量；还需文件校验、合并与数据审计。";$("collection-section").hidden=!tasks.length;$("collection-tasks").replaceChildren();for(const task of tasks){const card=element("article",undefined,"collection-task"),label=(data.matrix||[]).find(r=>r.task===task.task)?.label||task.task;card.append(element("strong",label));const progress=element("progress");progress.max=task.requested_episodes||1;progress.value=task.count||0;progress.setAttribute("aria-label",`${label}示范采集`);card.append(element("div",`已记录 ${number(task.count,0)} / ${number(task.requested_episodes,0)} 集`),progress,element("small",`${statusNames[task.status]||task.status||"未知"} · 已尝试 ${number(task.attempted_episodes,0)} 集`));if(task.error){card.classList.add("failed");card.append(element("p",`校验或采集失败，暂不可用于训练：${task.error}`));}else if(task.file_integrity_verified)card.append(element("small","任务文件校验通过；完整数据集审计另行执行。"));$("collection-tasks").append(card);}
 const labels=run?.correction_labels;$("correction-section").hidden=!labels;$("correction-counts").replaceChildren();$("correction-track").replaceChildren();if(!labels)return;const groups=[["teacher_completable","教师可完成","valid"],["rejected","已拒绝","rejected"],["unverified","未核实","unknown"]];for(const [key,title,style]of groups){const item=element("div");item.append(element("strong",number(labels[key],0)),element("span",title));$("correction-counts").append(item);const segment=element("div",undefined,style);segment.style.width=`${labels.branches?labels[key]/labels.branches*100:0}%`;$("correction-track").append(segment);}$("correction-track").setAttribute("aria-label",groups.map(([key,title])=>`${title} ${labels[key]||0}`).join("，"));$("correction-note").textContent=`来自 ${number(labels.episodes,0)} 集已完成记录，共 ${number(labels.branches,0)} 个教师分支。这里尚未筛选训练集；有效训练行数以数据审计报告为准。这些是教师纠偏结果，不代表学生成功率。`;}
function svgNode(name, attrs={}, text) { const node=document.createElementNS("http://www.w3.org/2000/svg",name); for(const [key,value] of Object.entries(attrs))node.setAttribute(key,String(value));if(text!==undefined)node.textContent=text;return node; }
function chart(id, legendId, rows, series, {rate=false,label}={}) {
  const container=$(id), legend=$(legendId);container.replaceChildren();legend.replaceChildren();
  series=series.filter(s=>rows.some(r=>Number.isFinite(r.step)&&typeof r[s.key]==="number"&&Number.isFinite(r[s.key])));
  if(!series.length){container.append(element("div","暂无对应指标记录","empty"));return;}
  const all=series.flatMap(s=>rows.filter(r=>Number.isFinite(r.step)&&typeof r[s.key]==="number"&&Number.isFinite(r[s.key])).map(r=>({x:r.step,y:r[s.key]})));
  const width=Math.max(300,container.clientWidth||540),height=255,left=65,right=20,top=16,bottom=38;
  let xmin=Math.min(...all.map(p=>p.x)),xmax=Math.max(...all.map(p=>p.x));const singleStep=xmin===xmax?xmin:null;if(singleStep!==null){xmin=Math.max(0,xmin-1);xmax+=1;}
  const ymin=0,ymax=rate?1:Math.max(...all.map(p=>p.y),.000001)*1.08;
  const x=v=>left+(v-xmin)/(xmax-xmin)*(width-left-right),y=v=>height-bottom-(v-ymin)/(ymax-ymin)*(height-top-bottom);
  const svg=svgNode("svg",{viewBox:`0 0 ${width} ${height}`,role:"img","aria-label":label||(id==="loss-chart"?"训练及验证损失曲线":id==="rmse-chart"?"动作均方根误差曲线":"各任务闭环成功率曲线")});
  for(let i=0;i<=4;i++){const value=ymax*i/4,yy=y(value);svg.append(svgNode("line",{x1:left,x2:width-right,y1:yy,y2:yy,stroke:"#385044","stroke-width":1}));svg.append(svgNode("text",{x:left-9,y:yy+4,"text-anchor":"end"},rate?`${Math.round(value*100)}%`:number(value, ymax<.01?5:3)));}
  const stepTicks=singleStep!==null?[singleStep]:[...new Set(Array.from({length:5},(_,i)=>Math.round(xmin+(xmax-xmin)*i/4)))];
  for(const value of stepTicks){svg.append(svgNode("text",{x:x(value),y:height-12,"text-anchor":"middle"},number(value,0)));}
  series.forEach((s,i)=>{const points=rows.filter(r=>Number.isFinite(r.step)&&typeof r[s.key]==="number"&&Number.isFinite(r[s.key])).sort((a,b)=>a.step-b.step);const color=s.color||colors[i%colors.length];
    svg.append(svgNode("polyline",{points:points.map(p=>`${x(p.step)},${y(p[s.key])}`).join(" "),fill:"none",stroke:color,"stroke-width":2}));
    for(const point of points){const dot=svgNode("circle",{cx:x(point.step),cy:y(point[s.key]),r:points.length>500?1.5:3,fill:color});dot.append(svgNode("title",{},`${s.label} · step ${point.step}: ${rate?percent(point[s.key]):number(point[s.key],6)}`));svg.append(dot);}
    const key=element("span"),swatch=element("i");swatch.style.borderColor=color;key.append(swatch,document.createTextNode(s.label));legend.append(key);
  });container.append(svg);
}
function renderPrefixDiagnostic(){
 const reports=(data?.other_reports||[]).filter(r=>r.protocol==="walk_state_prefix_cpu_diagnostic_v1"&&r.status==="diagnosed"&&r.counts_as_skill_success===false).sort((a,b)=>b.modified_at-a.modified_at),r=reports[0];
 $("prefix-diagnostic-section").hidden=!r;if(!r)return;
 $("prefix-diagnostic-rate").textContent=`${number(r.summary?.successes,0)} / ${number(r.summary?.episodes,0)} 条通过`;
 $("prefix-diagnostic-detail").textContent=`${number(r.training_samples,0)} 个训练样本；以 ${number(r.validation_samples,0)} 个验证样本的动作误差选择第 ${number(r.selected_step,0)} 步模型。CPU 时间 ${number(r.cpu_seconds,1)} 秒。未使用 GPU。`;
 $("prefix-diagnostic-evidence").replaceChildren(evidenceLink(r.artifact_id,"查看诊断报告"));
}
function renderMixedCollection(run){
 const prefix=run?.prefix_collection;
 const preflight=run?.prefix_preflight;
 const replayed=preflight?.protocol==="qualified_prefix_additions_preflight_v1"&&preflight.status==="reported_complete"&&preflight.counts_as_skill_success===false;
 const retraining=qualifiedPlan(run)?.prefix_retraining;
 const retrain=retraining?.protocol==="qualified_prefix_retraining_v1"&&retraining.counts_as_skill_success===false;
 const fixed=prefix?.protocol==="qualified_prefix_collection_v1"&&prefix.counts_as_skill_success===false;
 const probability=run?.collection_recipe?.teacher_rollin_probability;
 const qualified=run?.collection_recipe?.protocol==="qualified_dagger_collection_v1"&&qualifiedPlan(run)?.stage_kind==="dagger"&&run.collection_recipe.plan_sha256===run.qualified_plan.plan_sha256;
 const studentOnly=probability===0&&(run?.scope==="walk_dagger_v1"||qualified);
 const active=replayed||retrain||fixed||studentOnly||(typeof probability==="number"&&Number.isFinite(probability)&&probability>0&&probability<1);
 $("mixed-collection-section").hidden=!active;
 $("mixed-coverage-rows").replaceChildren();$("mixed-coverage-evidence").replaceChildren();
 $("mixed-coverage-table").hidden=true;
 if(!active)return;
 if(replayed){
  $("mixed-collection-title").textContent="新增纠偏数据 · 物理重放预检";
  $("mixed-collection-recipe").textContent=`共 ${number(preflight.phases.length,0)} 组交接阶段，${number(preflight.episodes,0)} 条不同初始条件，${number(preflight.eligible_queries,0)} 个有效训练查询。`;
  $("mixed-collection-status").textContent=`已从学生执行后的状态重放 ${number(preflight.branch_steps,0)} 步教师标签；报告中的最大绝对误差 ${number(preflight.maximum_abs_error,8)}。`;
  $("mixed-collection-note").textContent="此 CPU 作业检查数据加载与标签重放，没有更新模型。预检通过不代表学生学会任务，也不计入正式验收。";
  $("mixed-coverage-evidence").append(evidenceLink(preflight.report_artifact_id,"查看联合预检报告"));
  for(const phase of preflight.phases)$("mixed-coverage-evidence").append(document.createTextNode(" · "),evidenceLink(phase.report_artifact_id,`交接后 tick ${phase.source_ticks.join("/")} 的重放记录`));
  return;
 }
 if(retrain){
  $("mixed-collection-title").textContent="分阶段纠偏训练 · 保留完整任务范围";
  $("mixed-collection-recipe").textContent=`在原累计数据上加入 ${retraining.phases.map(phase=>`${number(phase.teacher_prefix_ticks/50,1)} 秒交接：${number(phase.episodes,0)} 集 / ${number(phase.eligible_queries,0)} 个有效查询`).join("；")}。`;
  $("mixed-collection-status").textContent="模型从原序列选定的最佳权重继续，沿用原归一化。训练及闭环结果见下方曲线。";
  $("mixed-collection-note").textContent="教师前缀、首次交接和失败分支不进入新增监督。保留全部腿式任务和原验证条件，最终能力仍需独立验收。";
  $("mixed-coverage-evidence").append(evidenceLink(retraining.parent_plan_artifact_id,"查看原训练计划"));
  for(const phase of retraining.phases)$("mixed-coverage-evidence").append(document.createTextNode(" · "),evidenceLink(phase.report_artifact_id,`${number(phase.teacher_prefix_ticks/50,1)} 秒交接数据报告`));
  return;
 }
 if(fixed){
  $("mixed-collection-title").textContent="分阶段交接 · 新增纠偏数据";
  $("mixed-collection-recipe").textContent=`每集先由官方教师执行 ${number(prefix.teacher_prefix_ticks/50,1)} 秒，再交给 TinyVLA。使用新的训练条件，保留原有验证集。`;
  $("mixed-collection-status").textContent=`已记录 ${number(prefix.recorded_episodes,0)} / ${number(prefix.requested_episodes,0)} 集；采集器记录的有效学生状态标签 ${number(prefix.reported_eligible_queries,0)} 条。`;
  const audited=prefix.completed_audit?.status==="reported_complete";
  $("mixed-collection-note").textContent="教师前缀、首次交接状态和失败教师分支不进入新增纠偏训练。"+(audited?"采集报告已完成数据合并与审计；":"数量来自已保存的采集记录，完整数据仍须合并与审计；")+"不代表学生能从起点独立完成任务。";
  $("mixed-coverage-evidence").append(evidenceLink(prefix.request_artifact_id,"查看采集配置"));
  if(prefix.records_artifact_id)$("mixed-coverage-evidence").append(document.createTextNode(" · "),evidenceLink(prefix.records_artifact_id,"查看逐集记录"));
  if(audited)$("mixed-coverage-evidence").append(document.createTextNode(" · "),evidenceLink(prefix.completed_audit.audit_artifact_id,"查看数据审计"));
  return;
 }
 $("mixed-collection-title").textContent=studentOnly?"学生自主采集 · 训练前数据门槛":"教师与学生混合采集 · 训练前数据门槛";
 $("mixed-collection-recipe").textContent=studentOnly?"采集时，每个动作块都由学生执行，物理状态持续推进。教师只在复制状态的分支中生成纠偏标签，不接管实际学生轨迹。这里描述的是采集配置。":`采集时，每个动作块以 ${percent(probability)} 的概率由教师执行，其余由学生执行；物理状态持续推进。这里的比例是采集配置。`;
 if(qualified){
  $("mixed-collection-note").textContent="在实际访问的状态上复制仿真，由教师生成未来 50 步标签并继续完成原任务。教师无法完成的标签会被排除。教师帮助下的完成不算学生成功；正式评测关闭混合执行。";
  $("mixed-collection-status").textContent=`当前阶段：${stageNames[run.stage]||run.stage||"等待阶段记录"}。所有采集数据须通过合并与审计后才能进入本轮优化。上方标签数量尚未筛选训练集。`;
  $("mixed-coverage-evidence").append(evidenceLink(run.qualified_plan.plan_artifact_id,"查看本轮采集与训练计划"));
  return;
 }
 $("mixed-collection-note").textContent=studentOnly?"仅统计学生动作之后、教师能够从该状态独立完成任务的训练标签。教师分支完成不代表实际学生轨迹成功，也不构成能力验收。":"仅统计学生动作之后、教师能够从该状态独立完成任务的训练标签。教师帮助下的完成不算学生成功；正式评测关闭混合执行。";
 const audit=run?.data_audit,coverage=audit?.mixed_rollin_coverage;
 const verified=coverage?.protocol==="mixed_walk_coverage_v1"&&typeof coverage.eligible_for_training==="boolean";
 $("mixed-coverage-table").hidden=!verified;
 $("mixed-collection-status").textContent=!verified?"覆盖审计尚未完成；数据通过门槛后才能进入本轮优化。":coverage.eligible_for_training===true?"数据覆盖门槛通过；是否开始优化以上方作业阶段为准。这不是能力验收。":"数据覆盖门槛未通过：本轮不启动优化，需要先改善采集覆盖。";
 if(!verified)return;
 for(const [key,label]of [["early_walk","早期行走"],["later_walk","后期行走"],["stop","停止"]]){
   const phase=coverage.phases?.[key],row=element("tr");
   row.append(element("td",label),element("td",`${number(phase?.eligible_rows_after_student,0)} / 至少 ${number(coverage.minimum_rows_per_phase,0)} 行`),element("td",`${number(phase?.distinct_training_episodes,0)} / 至少 ${number(coverage.minimum_episodes_per_phase,0)} 集`));
   $("mixed-coverage-rows").append(row);
 }
 $("mixed-coverage-evidence").append(evidenceLink(audit.artifact_id,"查看本轮数据审计"));
}
function renderLossTailDiagnostic(run){
 const note=$("loss-tail-note");note.replaceChildren();note.hidden=true;
 const report=(data?.other_reports||[]).filter(r=>r.protocol==="walk_mixed_batch_tail_association_v1"&&r.counts_as_skill_success===false&&r.summary?.identity?.run_id===run?.id&&r.summary?.identity?.checkpoint_rng_checked===true).sort((a,b)=>b.summary.identity.steps-a.summary.identity.steps)[0];
 if(!report)return;
 const summary=report.summary,steps=summary.identity.steps,tail=summary.all?.contains_tail,other=summary.all?.no_tail;
 if(!Number.isInteger(steps)||steps<1||![tail,other].every(g=>Number.isInteger(g?.steps)&&g.steps>0&&Number.isFinite(g.loss_mean)&&g.loss_mean>=0)||tail.steps+other.steps!==steps)return;
 note.hidden=false;
 note.append(element("p",`前 ${number(steps,0)} 步采样器状态已与 checkpoint 核对。含大幅度动作标签的 ${number(tail.steps,0)} 个批次，平均 loss ${number(tail.loss_mean,3)}；其余 ${number(other.steps,0)} 个批次为 ${number(other.loss_mean,3)}。`),element("small","这是批次与 loss 的关联，不代表这些标签导致闭环跌倒。"),evidenceLink(report.artifact_id,"查看标签与批次诊断"));
}
function renderRefitComparison(run){
 const report=(data?.other_reports||[]).filter(r=>r.protocol==="walk_refit_pair_interim_v1"&&r.counts_as_skill_success===false&&r.summary?.same_initialization_data_normalization_recipe_and_sampling===true&&Object.hasOwn(r.summary?.latest_steps||{},run?.id)).sort((a,b)=>b.modified_at-a.modified_at)[0];
 $("refit-comparison-section").hidden=!report;
 if(!report)return;
 const matched=(report.summary.matched_validation||[]).filter(r=>Number.isFinite(r.step)).sort((a,b)=>a.step-b.step);
 $("refit-comparison-context").textContent=`阶段报告 · 更新于 ${date(report.modified_at)} · ${matched.length?`包含第 ${number(matched[0].step,0)}–${number(matched.at(-1).step,0)} 步的 ${matched.length} 组配对验证点`:"尚无配对验证点"}。这是报告快照；两组各自的最新进度见上方批次曲线。`;
 const series=[{key:"uniform",label:"完整动作块等权",color:colors[1]},{key:"prefix",label:"前 5 步权重 ×10",color:colors[0]}];
 for(const [name,metric,label]of [["prefix","prefix5_rmse_rad","等权与前缀加权的前五步动作误差"],["full","full_rmse_rad","等权与前缀加权的完整动作块误差"]]){
   chart(`refit-${name}-chart`,`refit-${name}-legend`,matched.map(r=>({step:r.step,uniform:r.uniform?.[metric],prefix:r.prefix_weight10?.[metric]})),series,{label});
 }
 const result=$("refit-comparison-result");result.replaceChildren();
 const byStep=report.summary.closed_loop_by_step||{},steps=Object.keys(byStep).map(Number).filter(Number.isFinite);
 const loopStep=steps.length?Math.max(...steps):500;
 const loops=byStep[String(loopStep)]||report.summary.step500_closed_loop||{};
 for(const [id,label]of [["walk-refit-uniform-pilot-v1","等权"],["walk-refit-prefix-pilot-v1","前缀加权"]]){
   const score=loops[id];if(!score)continue;
   result.append(element("div",`第 ${number(loopStep,0)} 步闭环 · ${label}：${number(score.successes,0)} / ${number(score.episodes,0)} 条严格通过；跌倒 ${number(score.failed_checks?.no_fall,0)} 条，结束不稳定 ${number(score.failed_checks?.terminal_stable,0)} 条。`));
   const development=report.summary.completed_development?.[id]?.grade;
   if(development)result.append(element("div",`选定模型的开发集 · ${label}：${number(development.successes,0)} / ${number(development.episodes,0)} 条严格通过；不属于正式技能测试。`));
 }
 result.append(element("div","动作误差下降不代表闭环成功。此处是训练期间的小样本对照，不计入正式行为验收；不同权重的 loss 不直接横比。"));
 $("refit-comparison-evidence").replaceChildren(evidenceLink(report.artifact_id,"查看配对报告与来源校验信息"));
}
function sameEvidence(a,b){
 if(a===b)return true;
 if(!a||!b||typeof a!=="object"||typeof b!=="object"||Array.isArray(a)!==Array.isArray(b))return false;
 const keys=Object.keys(a).sort();return sameKeys(keys,Object.keys(b).sort())&&keys.every(key=>sameEvidence(a[key],b[key]));
}
function sameKeys(a,b){return a.length===b.length&&a.every((value,index)=>value===b[index]);}
async function readComparisonArtifact(artifact){
 if(!artifact?.available||!artifact.url||!artifact.sha256)throw new Error("配对证据尚未同步");
 const key=`${artifact.id}:${artifact.sha256}`;
 if(comparisonEvidenceCache.has(key))return comparisonEvidenceCache.get(key);
 const response=await fetch(artifact.url,{cache:"no-store"});if(!response.ok)throw new Error("配对证据正在同步");
 const bytes=await response.arrayBuffer(),digest=await crypto.subtle.digest("SHA-256",bytes);
 const hash=Array.from(new Uint8Array(digest),value=>value.toString(16).padStart(2,"0")).join("");
 if(hash!==artifact.sha256)throw new Error("配对证据版本已变化，等待下次刷新");
  const value=JSON.parse(new TextDecoder().decode(bytes));comparisonEvidenceCache.set(key,value);return value;
}
function relativeEvidencePath(path,snapshot){
 const root=snapshot.source?.root;
 if(typeof root!=="string"||typeof path!=="string"||!path.startsWith(root.replace(/\/$/,"")+"/"))return null;
 const relative=path.slice(root.replace(/\/$/,"").length+1);
 return relative.split("/").some(part=>!part||part===".."||part===".")?null:relative;
}
function qualifiedValidationIdentity(report,tasks,seeds){
 if(!Array.isArray(seeds)||!seeds.length||seeds.some(seed=>!Number.isSafeInteger(seed)||seed<0)||new Set(seeds).size!==seeds.length)throw Error("固定验证 seed 清单无效");
 if(report?.protocol!=="project_qualified_evaluation_v2"||report.complete!==true||report.purpose!=="validation"||report.evaluator_role!=="student"||report.mode!=="offline"||report.ablation!==null||report.counts_as_skill_success!==false||report.independent_final_test!==false||report.offline_accepted!==false||report.realtime_accepted!==false||report.execute_steps!==5||report.semantic_refresh_steps!==5||report.action_padding!=="training_noise_path"||report.uses_prescribed_command13!==true||!report.evaluator_source_sha256||!report.task_contracts_sha256||!sameKeys(Object.keys(report.tasks||{}).sort(),[...tasks].sort())||!Array.isArray(report.episodes)||report.episodes.length!==tasks.length*seeds.length)throw Error("固定验证范围或执行契约不一致");
 const conditions=[],outcomes={};
 for(const task of tasks){
  const rows=report.episodes.filter(row=>row.task===task).sort((a,b)=>a.seed-b.seed),score=report.tasks[task];
  if(!sameKeys(rows.map(row=>row.seed),[...seeds].sort((a,b)=>a-b))||new Set(rows.map(row=>row.initial_condition_sha256)).size!==seeds.length||score.episodes!==seeds.length||score.distinct_conditions!==seeds.length||score.counts_as_skill_success!==false||score.offline_accepted!==false||score.realtime_accepted!==false)throw Error("固定条件存在遗漏、重复或验收状态不一致");
  for(const row of rows){
   if(typeof row.success!=="boolean"||row.counts_as_skill_success!==false||!row.initial_condition_sha256||!row.command_program_sha256||!row.teacher_version||!row.success_protocol||!row.initial_condition||row.task_result?.success!==row.success||(row.success&&(row.fault||row.collision||row.recovery_events)))throw Error("逐集结果缺少可核对证据");
   conditions.push([task,row.seed,row.initial_condition_sha256,row.initial_condition,row.command_program_sha256,row.teacher_version,row.success_protocol]);outcomes[`${task}:${row.seed}`]=row.success;
  }
  const successes=rows.filter(row=>row.success).length;
  if(score.successes!==successes||score.strict_rate!==successes/seeds.length)throw Error("汇总分数与逐集记录不一致");
 }
 return {identity:{conditions,evaluator:report.evaluator_source_sha256,contracts:report.task_contracts_sha256,task_contracts:report.task_contracts},outcomes};
}
async function loadInitializationComparison(snapshot,read=readComparisonArtifact){
 const run=(snapshot.runs||[]).find(row=>row.id===snapshot.selected_run_id),plan=qualifiedPlan(run);
 const checkpoints=(run?.checkpoints||[]).filter(row=>row.complete===true).sort((a,b)=>b.step-a.step);
 if(plan?.stage_kind!=="dagger"||!checkpoints.length)return null;
 const result={run_id:run.id,verified:false};
 try{
  const artifacts=snapshot.artifacts||[],requestArtifact=artifacts.find(a=>a.id===plan.request_artifact_id&&a.run_id===run.id&&a.path===`${run.source}/request.json`),recipeArtifact=artifacts.find(a=>a.run_id===run.id&&a.path===`${run.source}/training-recipe.json`);
  const [request,recipe]=await Promise.all([read(requestArtifact),read(recipeArtifact)]),selected=request.previous?.selected;
  if(request.protocol!=="project_qualified_training_v2"||request.plan?.sha256!==plan.plan_sha256||request.stage?.kind!=="dagger"||request.stage.body!==plan.body||request.previous?.body!==plan.body||request.previous?.status!=="complete"||!selected?.checkpoint||recipe.model_path!==selected.checkpoint||run.training_recipe?.model_path!==selected.checkpoint||recipe.normalization_path!==selected.normalization?.path)throw Error("实际初始化模型与登记来源不一致");
  const referenceArtifact=artifacts.find(a=>a.path===relativeEvidencePath(selected.validation?.path,snapshot)&&a.sha256===selected.validation?.sha256),projected=checkpoints[0],candidateArtifact=artifacts.find(a=>a.id===projected.artifact_id&&a.run_id===run.id&&a.path===projected.source);
  const [reference,candidate]=await Promise.all([read(referenceArtifact),read(candidateArtifact)]);
  if(reference.checkpoint!==selected.checkpoint||reference.checkpoint_sha256!==selected.weights?.sha256||reference.artifact_manifest_sha256!==selected.manifest?.sha256||candidate.checkpoint!==projected.checkpoint||candidate.checkpoint_sha256!==projected.checkpoint_sha256||!sameEvidence(candidate.tasks,projected.tasks))throw Error("模型或报告版本未匹配");
  const before=qualifiedValidationIdentity(reference,plan.tasks,plan.validation_seeds),after=qualifiedValidationIdentity(candidate,plan.tasks,plan.validation_seeds);
  if(!sameEvidence(before.identity,after.identity)||!sameEvidence(candidate.task_contracts,plan.task_contracts))throw Error("初始条件、命令或评测器发生变化，不能直接配对");
  const changes=plan.tasks.map(task=>{const keys=plan.validation_seeds.map(seed=>`${task}:${seed}`);return {task,before:reference.tasks[task].successes,after:candidate.tasks[task].successes,total:keys.length,gained:keys.filter(k=>!before.outcomes[k]&&after.outcomes[k]).length,lost:keys.filter(k=>before.outcomes[k]&&!after.outcomes[k]).length};});
  return {...result,verified:true,step:projected.step,reference_checkpoint:selected.checkpoint,changes,reference_artifact_id:referenceArtifact.id,candidate_artifact_id:candidateArtifact.id};
 }catch(error){return {...result,error:`初始化对比尚未核验：${error.message}`};}
}
function renderInitializationComparison(run){
 const comparison=data?.initialization_comparison,box=$("initialization-comparison");box.replaceChildren();box.hidden=!comparison||comparison.run_id!==run?.id;
 if(box.hidden)return;if(!comparison.verified){box.textContent=comparison.error;return;}
 const sum=key=>comparison.changes.reduce((total,row)=>total+row[key],0);
 box.append(element("strong",`同条件固定验证：本轮初始化模型 ${sum("before")} / ${sum("total")} → 最新完成 step ${comparison.step}：${sum("after")} / ${sum("total")}。`),element("div",`新增 ${sum("gained")} 条成功，丢失 ${sum("lost")} 条原成功；这是训练中验证，不计入正式验收。`));
 for(const row of comparison.changes.filter(row=>row.gained||row.lost))box.append(element("div",`${taskLabel(row.task)}：${row.before}/${row.total} → ${row.after}/${row.total}（新增 ${row.gained}，丢失 ${row.lost}）。`));
 box.append(evidenceLink(comparison.reference_artifact_id,"初始化模型的原始验证报告"),document.createTextNode(" · "),evidenceLink(comparison.candidate_artifact_id,"当前 checkpoint 验证报告"));
}
async function loadOnsetFits(snapshot,read=readComparisonArtifact){
 const runs=snapshot.runs||[],artifacts=snapshot.artifacts||[],protocol="qualified_onset_fit_cpu_v1";
 const candidates=runs.flatMap(run=>(run.evaluation_reports||[]).filter(r=>r.protocol===protocol).map(projected=>({run,projected})));
 return Promise.all(candidates.map(async({run,projected})=>{
  const result={run_id:run.id,verified:false,modified_at:projected.modified_at};
  try{
   const artifact=artifacts.find(a=>a.id===projected.artifact_id&&a.run_id===run.id&&a.path===`${run.source}/report.json`),report=await read(artifact);
   const requestArtifact=artifacts.find(a=>a.run_id===run.id&&a.path===`${run.source}/request.json`),request=await read(requestArtifact);
   if(report.protocol!==protocol||request.protocol!==protocol||report.status!=="diagnosed"||report.source_files_unchanged!==true||report.request?.sha256!==requestArtifact.sha256||relativeEvidencePath(report.request?.path,snapshot)!==requestArtifact.path)throw Error("诊断来源不一致");
   for(const value of [request,report]){
    if(value.device!=="cpu"||value.precision!=="fp32"||value.population!=="training split only"||value.action_padding!=="training_noise_path"||value.active_action_dim!==14||value.padded_action_dim!==64||[value.gpu_used,value.closed_loop_evaluated,value.independent_final_test,value.counts_as_skill_success,value.model_promoted,value.training_data_produced].some(v=>v!==false))throw Error("诊断范围与记录不一致");
   }
   const owner=runs.find(r=>(r.checkpoints||[]).some(c=>`${c.checkpoint}/model.safetensors`===request.checkpoint?.path&&c.checkpoint_sha256===request.checkpoint?.sha256)),ownerPlan=qualifiedPlan(owner);
   if(!ownerPlan||ownerPlan.body!=="legs"||request.plan?.sha256!==ownerPlan.plan_sha256)throw Error("诊断未绑定原腿式模型");
   const planArtifact=artifacts.find(a=>a.id===ownerPlan.plan_artifact_id&&a.sha256===request.plan.sha256&&a.path===relativeEvidencePath(request.plan.path,snapshot)),plan=await read(planArtifact);
   if(!sameEvidence(request.dataset,plan.datasets?.legs?.index)||![plan.validation_seeds,plan.development_seeds,plan.reserved_final_seeds].every(Array.isArray))throw Error("训练数据或隔离清单不一致");
   const forbidden=new Set([...plan.validation_seeds,...plan.development_seeds,...plan.reserved_final_seeds,...Array.from({length:100},(_,i)=>910000+i)]),samples=request.samples;
   if(!Array.isArray(samples)||samples.length!==3||!sameKeys(samples.map(s=>s.tick),[20,25,30])||samples.some(s=>s.task!=="kick_left"||s.split!=="train"||!Number.isSafeInteger(s.seed)||s.seed<0||s.seed!==samples[0].seed||forbidden.has(s.seed)||s.instruction!==samples[0].instruction))throw Error("采样状态或保留条件不一致");
   for(const sample of samples)for(const key of ["image_rgb_sha256","state61_sha256","action_sha256","action_is_pad_sha256"])if(!/^[a-f0-9]{64}$/.test(sample[key]||""))throw Error("缺少输入或标签版本");
   const steps=request.evaluation_steps,noises=request.evaluation_noise_seeds,arms=request.arms;
   if(![160,1000].includes(request.optimizer_steps_per_arm)||!Array.isArray(steps)||steps[0]!==0||steps.at(-1)!==request.optimizer_steps_per_arm||steps.some((s,i)=>!Number.isSafeInteger(s)||s<0||(i>0&&s<=steps[i-1]))||!Array.isArray(noises)||noises.length!==3||new Set(noises).size!==3||noises.some(s=>!Number.isSafeInteger(s)||s<0)||request.batch_size!==3||request.flow_loss!=="huber"||request.flow_huber_delta!==1||request.executed_prefix_weight!==10||request.freeze_vision!==true||request.unfreeze_lm_last_n!==0||request.learning_rate!==3e-5||request.warmup_steps!==20)throw Error("拟合配方或测量步骤不一致");
   if(!arms||!Object.keys(arms).length||!sameKeys(Object.keys(arms).sort(),Object.keys(report.evaluations||{}).sort()))throw Error("拟合分组不一致");
   const curves={};
   for(const [arm,indices]of Object.entries(arms)){
    const expected=arm==="onset_only"?[1,1,1]:arm==="adjacent_states"?[0,1,2]:null,measurements=report.evaluations[arm];
    if(!expected||!sameKeys(indices,expected)||!Array.isArray(measurements)||!sameKeys(measurements.map(m=>m.step),steps))throw Error("拟合分组或测量缺失");
    curves[arm]=measurements.map(m=>{
     if(!Array.isArray(m.rows)||m.rows.length!==9)throw Error("重复或缺失噪声测量");
     const point={step:m.step,ranges:{}};
     for(const [i,tick]of [20,25,30].entries()){
      const rows=m.rows.filter(r=>r.tick===tick);
      if(rows.length!==3||!sameKeys(rows.map(r=>r.noise_seed).sort((a,b)=>a-b),[...noises].sort((a,b)=>a-b)))throw Error("噪声条件未配对");
      if(rows.some(r=>r.used_in_this_arm!==indices.includes(i)||![r.prefix5_rmse_rad,r.first_rmse_rad].every(v=>typeof v==="number"&&Number.isFinite(v)&&v>=0)))throw Error("动作误差或训练状态标记错误");
      const values=rows.map(r=>r.prefix5_rmse_rad);point[`tick${tick}`]=values.reduce((a,b)=>a+b,0)/3;point.ranges[tick]=[Math.min(...values),Math.max(...values)];
     }
     return point;
    });
   }
   return {...result,verified:true,owner_id:owner.id,request,report,curves,artifact,requestArtifact};
  }catch(error){return {...result,error:`收敛诊断尚未核对：${error.message}`};}
 }));
}
async function loadPairedReplays(snapshot,read=readComparisonArtifact){
 const artifacts=snapshot.artifacts||[],runs=snapshot.runs||[];
 return Promise.all((snapshot.other_reports||[]).filter(p=>p.protocol==="qualified_paired_validation_replay_v1").map(async projected=>{
  const result={owner_id:projected.owner_run_id,verified:false,modified_at:projected.modified_at};
  try{
   const artifact=artifacts.find(a=>a.id===projected.artifact_id&&a.path===projected.source&&a.kind==="report"),report=await read(artifact);
   const owner=runs.find(r=>r.id===report.owner_run_id),plan=qualifiedPlan(owner);
   if(!plan||report.owner_run_id!==projected.owner_run_id||report.protocol!==projected.protocol||report.status!=="diagnosed"||report.source_purpose!=="validation"||[report.gpu_used,report.student_inference_performed,report.training_data_produced,report.counts_as_skill_success,report.independent_final_test].some(v=>v!==false))throw Error("重放范围或所属训练批次不一致");
   const steps=Object.keys(report.models||{}).map(Number).sort((a,b)=>a-b);
   if(steps.length!==2||steps.some(s=>!Number.isSafeInteger(s)||s<=0)||steps[0]===steps[1]||!Number.isFinite(report.atol)||report.atol<0||report.atol>2e-5||!Number.isFinite(report.max_abs_error)||report.max_abs_error<0||report.max_abs_error>report.atol)throw Error("配对模型或物理误差无效");
   const originals=[],sourceArtifacts=[],identities=[];
   for(const step of steps){
    const m=report.models[step],checkpoint=owner.checkpoints.find(c=>c.step===step&&c.checkpoint===m.checkpoint&&c.checkpoint_sha256===m.checkpoint_sha256);
    const source=artifacts.find(a=>a.id===checkpoint?.artifact_id&&a.run_id===owner.id&&a.sha256===m.report_sha256&&a.path===relativeEvidencePath(`${m.checkpoint}/closed-loop-validation.json`,snapshot));
    const original=await read(source);
    if(original.checkpoint!==m.checkpoint||original.checkpoint_sha256!==m.checkpoint_sha256||!sameEvidence(original.task_contracts,plan.task_contracts))throw Error("原始评测模型身份不一致");
    identities.push(qualifiedValidationIdentity(original,plan.tasks,plan.validation_seeds));originals.push(original);sourceArtifacts.push(source);
   }
   if(!sameEvidence(identities[0].identity,identities[1].identity))throw Error("两个模型的条件、命令或评测器未配对");
   const expected=plan.tasks.length*plan.validation_seeds.length,seen=new Set(),videoSeeds=[...plan.validation_seeds].sort((a,b)=>a-b).slice(0,2);
   if(!Array.isArray(report.pairs)||report.pairs.length!==expected||report.episodes!==expected*2||report.distinct_paired_conditions!==expected)throw Error("重放没有覆盖全部配对条件");
   const pairs=report.pairs.map(pair=>{
    const key=`${pair.task}:${pair.seed}`;
    if(seen.has(key)||!plan.tasks.includes(pair.task)||!plan.validation_seeds.includes(pair.seed))throw Error("重复或越界的配对条件");seen.add(key);
    for(const [i,name]of ["before","after"].entries()){
     const row=pair[name],original=originals[i].episodes.find(e=>e.task===pair.task&&e.seed===pair.seed);
     if(!row||row.task!==pair.task||row.seed!==pair.seed||row.step!==steps[i]||row.success!==original.success||row.initial_condition_sha256!==original.initial_condition_sha256||row.trajectory_sha256!==original.trajectory_sha256||!sameEvidence(row.failed_checks,original.task_result.failed_checks)||!row.max_abs_error||Object.values(row.max_abs_error).some(e=>!Number.isFinite(e)||e<0||e>report.atol))throw Error("重放结果与原始轨迹不一致");
     const fall=row.first_unintended_fall_s;
     if((fall!==null&&(!Number.isFinite(fall)||fall<0||fall>5))||(fall===null)!==original.task_result.checks.no_unintended_fall)throw Error("跌倒时间与原任务判定不一致");
    }
    const video=pair.paired_video;
    if(Boolean(video)!==videoSeeds.includes(pair.seed)||(video&&(video.policy_input!==false||video.frames!==50||video.fps!==10)))throw Error("视频选择或时钟与固定规则不一致");
    const media=video?artifacts.find(a=>a.kind==="video"&&a.run_id===owner.id&&a.path===relativeEvidencePath(video.path,snapshot)&&a.sha256===video.sha256):null;
    return {...pair,key,media};
   });
   return {...result,verified:true,report,pairs,steps,artifact,evidence_ids:[artifact.id,...sourceArtifacts.map(a=>a.id)]};
  }catch(error){return {...result,error:`配对回放尚未核验：${error.message}`};}
 }));
}
function renderPairedReplay(run){
 const entry=(data?.paired_replays||[]).filter(r=>r.owner_id===run?.id).sort((a,b)=>b.modified_at-a.modified_at)[0];
 $("paired-replay-section").hidden=!entry;$("paired-replay-content").hidden=!entry?.verified;
 if(!entry?.verified){$("paired-replay-context").textContent=entry?.error||"";$("paired-replay-video").replaceChildren();pairedReplayKey="";return;}
 const {steps,pairs}=entry,select=$("paired-replay-select"),old=select.value,clips=pairs.filter(p=>p.paired_video);
 $("paired-replay-context").textContent=`${run.id} · 第 ${steps[0]} 与 ${steps[1]} 步 · ${pairs.length} 个相同固定验证条件全部重放核对。历史结果不计入正式验收。`;
 select.replaceChildren();for(const p of clips){const o=element("option",`${taskLabel(p.task)} · seed ${p.seed} · ${p.before.success?"通过":"失败"} → ${p.after.success?"通过":"失败"}`);o.value=p.key;select.append(o);}select.value=clips.some(p=>p.key===old)?old:clips[0]?.key||"";
 const chosen=clips.find(p=>p.key===select.value),key=JSON.stringify([entry.artifact.sha256,chosen?.key,chosen?.media?.available,chosen?.media?.url]);
 if(key!==pairedReplayKey){const box=$("paired-replay-video");box.replaceChildren();if(chosen?.media?.available){const video=element("video");video.controls=true;video.playsInline=true;video.preload="metadata";video.src=chosen.media.url;video.style.width="100%";video.style.maxWidth="960px";video.setAttribute("aria-label",`第 ${steps[0]} 与 ${steps[1]} 步 · ${chosen.task} · seed ${chosen.seed}`);box.append(video);}else box.append(element("p","此视频尚未完成来源匹配或同步"));pairedReplayKey=key;}
 $("paired-replay-before").textContent=`第 ${steps[0]} 步`;$("paired-replay-after").textContent=`第 ${steps[1]} 步`;$("paired-replay-rows").replaceChildren();
 for(const p of pairs){const tr=element("tr");tr.append(element("td",`${taskLabel(p.task)} / ${p.seed}`));for(const row of [p.before,p.after])tr.append(element("td",`${row.success?"通过":"失败"} · ${row.first_unintended_fall_s===null?"未检测到意外跌倒":`${number(row.first_unintended_fall_s,2)} 秒首次意外跌倒`}`));$("paired-replay-rows").append(tr);}
 $("paired-replay-evidence").replaceChildren(...entry.evidence_ids.flatMap((id,i)=>[...(i?[document.createTextNode(" · ")]:[]),evidenceLink(id,i?`第 ${steps[i-1]} 步原始验证报告`:"完整重放报告")]));
}
function renderOnsetFits(run){
 const entries=(data?.onset_fits||[]).filter(f=>f.run_id===run?.id||f.owner_id===run?.id),section=$("onset-fit-section");
 section.hidden=!entries.length;$("onset-fit-content").hidden=true;if(section.hidden)return;
 const valid=entries.filter(f=>f.verified).sort((a,b)=>b.request.optimizer_steps_per_arm-a.request.optimizer_steps_per_arm||b.modified_at-a.modified_at);
 if(!valid.length){$("onset-fit-context").textContent=entries[0].error;return;}
 const options=valid.flatMap(f=>Object.keys(f.curves).map(arm=>({fit:f,arm,value:`${f.run_id}:${arm}`}))),select=$("onset-fit-arm"),previous=select.value;select.replaceChildren();
 for(const item of options){const option=element("option",`${item.arm==="onset_only"?"只拟合启动状态":"同时拟合三个相邻状态"} · ${item.fit.request.optimizer_steps_per_arm} 步`);option.value=item.value;select.append(option);}
 select.value=options.some(o=>o.value===previous)?previous:options[0].value;
 const {fit,arm}=options.find(o=>o.value===select.value),points=fit.curves[arm],last=points.at(-1),names=["启动前 · 0.4 秒","启动时 · 0.5 秒","启动后 · 0.6 秒"];
 $("onset-fit-context").textContent=`${fit.run_id} · 已完成 ${fit.request.optimizer_steps_per_arm} 步 · kick_left / seed ${fit.request.samples[0].seed} · CPU FP32。曲线来自通过来源校验的报告；原始数组重算见诊断记录。`;
 $("onset-fit-content").hidden=false;
 chart("onset-fit-chart","onset-fit-legend",points,[20,25,30].map((tick,i)=>({key:`tick${tick}`,label:names[i]})),{label:"踢球启动前、启动时、启动后的动作误差曲线"});
 $("onset-fit-rows").replaceChildren();
 for(const [i,tick]of [20,25,30].entries()){const row=element("tr"),[lo,hi]=last.ranges[tick];for(const value of [names[i],fit.request.arms[arm].includes(i)?"参与":"未参与",number(last[`tick${tick}`],4),`${number(lo,4)}—${number(hi,4)}`])row.append(element("td",value));$("onset-fit-rows").append(row);}
 $("onset-fit-evidence").replaceChildren(evidenceLink(fit.artifact.id,"查看完整测量报告"),document.createTextNode(" · "),evidenceLink(fit.requestArtifact.id,"查看固定配方与输入版本"));
}
async function loadCriticalFit(snapshot,read=readComparisonArtifact){
 const protocols={qualified_critical_fit_cpu_v1:"huber",qualified_critical_fit_mse_cpu_v1:"mse",qualified_critical_fit_lm2_cpu_v1:"lm2"},runs=snapshot.runs||[],artifacts=snapshot.artifacts||[];
 const candidates=runs.flatMap(run=>(run.evaluation_reports||[]).filter(report=>protocols[report.protocol]).map(report=>({run,report})));
 const fits=await Promise.all(candidates.map(async({run,report:projected})=>{
  const result={run_id:run.id,verified:false,modified_at:projected.modified_at};
  try{
   const artifact=artifacts.find(a=>a.id===projected.artifact_id&&a.run_id===run.id&&a.path===projected.source&&a.kind==="report"),report=await read(artifact);
   const requestArtifact=artifacts.find(a=>a.run_id===run.id&&a.path===`${run.source}/request.json`&&a.kind==="report"),request=await read(requestArtifact),objective=protocols[report.protocol];
   if(!objective||projected.protocol!==report.protocol||request.protocol!==report.protocol||request.flow_loss!==(objective==="lm2"?"huber":objective)||report.status!=="diagnosed"||report.population!=="training split only"||report.device!=="cpu"||report.precision!=="fp32"||report.action_padding!=="training_noise_path"||report.source_files_unchanged!==true||[report.gpu_used,report.model_promoted,report.independent_final_test,report.closed_loop_evaluated,report.counts_as_skill_success,report.training_data_produced,report.normalization_recomputed].some(v=>v!==false))throw Error("拟合诊断的用途或执行范围不一致");
   const owner=runs.find(r=>(r.checkpoints||[]).some(c=>c.checkpoint===report.checkpoint?.path&&c.checkpoint_sha256===report.checkpoint?.sha256)),plan=qualifiedPlan(owner);
   if(!plan||plan.body!=="legs")throw Error("拟合诊断未绑定原始腿式模型");
   const planArtifact=artifacts.find(a=>a.id===plan.plan_artifact_id&&a.sha256===plan.plan_sha256&&a.kind==="report"),registered=await read(planArtifact);
   if(!sameEvidence(report.dataset,registered.datasets?.legs?.index)||![registered.validation_seeds,registered.development_seeds,registered.reserved_final_seeds].every(Array.isArray))throw Error("拟合数据或隔离清单不一致");
   const forbidden=new Set([...registered.validation_seeds,...registered.development_seeds,...registered.reserved_final_seeds,...Array.from({length:100},(_,i)=>910000+i)]),metadata=report.sample_metadata;
   const integer=v=>Number.isSafeInteger(v)&&v>=0,positive=v=>Number.isFinite(v)&&v>0;
   if(!Array.isArray(metadata)||metadata.length!==60||request.fit_samples!==48||report.fit_samples!==48||report.train_regression_samples!==12||report.optimizer_steps!==160||request.optimizer_steps!==160||request.batch_size!==4||request.freeze_vision!==true||request.unfreeze_lm_last_n!==(objective==="lm2"?2:0)||!integer(request.training_seed)||!integer(request.evaluation_noise_seed)||![request.learning_rate,request.executed_prefix_weight,request.grad_clip_norm,request.weight_decay].every(positive)||request.warmup_steps!==20||!sameKeys(request.eval_steps||[],[0,40,80,120,160]))throw Error("拟合范围或配方无效");
   const seen=new Set(),fitTasks=new Set(),probeTasks=new Set();
   for(const m of metadata){
    const key=JSON.stringify([m.task,m.seed,m.tick]);
    if(!plan.tasks.includes(m.task)||!integer(m.seed)||forbidden.has(m.seed)||!integer(m.tick)||m.split!=="train"||!['fit','train_regression_probe'].includes(m.group)||typeof m.instruction!=="string"||!m.archive?.sha256||seen.has(key))throw Error("拟合样本包含重复、保留条件或无效来源");
    seen.add(key);(m.group==="fit"?fitTasks:probeTasks).add(m.task);
   }
   if(fitTasks.size!==4||probeTasks.size!==6||[...fitTasks].some(t=>probeTasks.has(t)))throw Error("拟合样本与旁路检查分组不一致");
   const stateKeys=[];
   for(const task of fitTasks){
    const selected=metadata.filter(m=>m.group==="fit"&&m.task===task),ticks=[...new Set(selected.map(m=>m.tick))].sort((a,b)=>a-b),seeds=[...new Set(selected.map(m=>m.seed))];
    if(selected.length!==12||ticks.length!==3||seeds.length!==4||ticks.some(t=>selected.filter(m=>m.tick===t).length!==4))throw Error("拟合任务条件不完整");
    for(const tick of ticks)stateKeys.push({task,tick,key:`${task}:${tick}`});
   }
   for(const task of probeTasks){const ms=metadata.filter(m=>m.task===task);if(ms.length!==2||ms[0].seed!==ms[1].seed||!sameKeys(ms.map(m=>m.tick).sort((a,b)=>a-b),[25,50]))throw Error("旁路检查不是同一轨迹的两个训练状态");}
   if(!Array.isArray(report.evaluations)||!sameKeys(report.evaluations.map(e=>e.optimizer_step),request.eval_steps))throw Error("拟合测量点不完整");
   const curves=[],probes={};
   for(const evaluation of report.evaluations){
    if(evaluation.samples!==60||evaluation.counts_as_skill_success!==false||evaluation.closed_loop_evaluated!==false||evaluation.action_padding!=="training_noise_path"||!Array.isArray(evaluation.rows)||evaluation.rows.length!==60)throw Error("拟合测量范围不一致");
    for(let i=0;i<60;i++){const row=evaluation.rows[i],m=metadata[i];if(Object.keys(m).some(k=>!sameEvidence(row[k],m[k]))||[row.prefix5_rmse_rad,row.first_rmse_rad].some(v=>!Number.isFinite(v)||v<0))throw Error("动作误差或训练状态不匹配");}
    const mean=rows=>rows.reduce((sum,row)=>sum+row.prefix5_rmse_rad,0)/rows.length;
    curves.push({step:evaluation.optimizer_step,...Object.fromEntries(stateKeys.map(s=>[s.key,mean(evaluation.rows.filter(r=>r.task===s.task&&r.tick===s.tick))]))});
    probes[evaluation.optimizer_step]=Object.fromEntries([...probeTasks].map(task=>[task,mean(evaluation.rows.filter(r=>r.task===task))]));
   }
   return {...result,verified:true,owner_id:owner.id,objective,artifact,requestArtifact,request,report,stateKeys,curves,probes};
  }catch(error){return {...result,error:`拟合诊断证据尚未核验：${error.message}`};}
 }));
 for(const fit of fits.filter(f=>f.verified&&["mse","lm2"].includes(f.objective))){
  const binding=fit.objective==="lm2"?fit.request.paired_frozen_report:fit.request.paired_huber_report;
  const reference=fits.find(f=>f.verified&&f.objective==="huber"&&f.artifact.sha256===binding?.sha256&&f.artifact.path===relativeEvidencePath(binding?.path,snapshot));
  const fields=['optimizer_steps','batch_size','learning_rate','warmup_steps','executed_prefix_weight','grad_clip_norm','weight_decay','freeze_vision','training_seed','evaluation_noise_seed','eval_steps','fit_samples','baseline',...(fit.objective==="lm2"?['flow_loss','flow_huber_delta']:['unfreeze_lm_last_n'])];
  if(!reference||fields.some(k=>!sameEvidence(reference.request[k],fit.request[k]))||['checkpoint','normalization','dataset','sample_metadata'].some(k=>!sameEvidence(reference.report[k],fit.report[k]))||!sameEvidence(reference.curves[0],fit.curves[0])||!sameEvidence(reference.probes[0],fit.probes[0])){fit.verified=false;fit.error="拟合对照的初始化、样本或配方尚未匹配";}
  else fit.reference_run_id=reference.run_id;
 }
 return fits;
}
function renderCriticalFit(run){
 const all=data?.critical_fits||[],selected=all.filter(f=>f.run_id===run?.id||f.owner_id===run?.id),section=$("critical-fit-section");
 section.hidden=!selected.length;$("critical-fit-content").hidden=true;if(section.hidden)return;
 const invalid=selected.find(f=>!f.verified),chosen=selected.find(f=>f.verified&&f.objective==="huber")||all.find(f=>f.verified&&f.run_id===selected.find(s=>s.verified)?.reference_run_id);
 if(!chosen){$("critical-fit-context").textContent=invalid?.error||"拟合诊断正在同步";return;}
 const partner=all.find(f=>f.verified&&f.objective==="mse"&&f.reference_run_id===chosen.run_id),language=all.find(f=>f.verified&&f.objective==="lm2"&&f.reference_run_id===chosen.run_id),fits=[chosen,...partner?[partner]:[],...language?[language]:[]],states=$("critical-fit-state"),previous=states.value;
 states.replaceChildren();for(const state of chosen.stateKeys){const option=element("option",`${taskLabel(state.task)} · ${number(state.tick/50,2)} 秒`);option.value=state.key;states.append(option);}states.value=chosen.stateKeys.some(s=>s.key===previous)?previous:chosen.stateKeys.find(s=>s.key==="kick_left:25")?.key||chosen.stateKeys[0].key;
 $("critical-fit-context").textContent=`原始模型：${chosen.owner_id} · CPU FP32 · 每批 4 个状态，优化 160 步。${partner?"Huber 与 MSE 使用相同初始化、数据、随机种子和其他训练设置。":"Huber 已完成；MSE 配对报告尚未完成或尚未核验。"}${language?" 解冻对照与冻结 Huber 同条件，仅额外训练语言网络最后两层。":" 解冻对照尚未完成或尚未核验。"}`;
 $("critical-fit-content").hidden=false;
 const rows=chosen.curves.map(row=>({step:row.step,huber:row[states.value],mse:partner?.curves.find(r=>r.step===row.step)?.[states.value],lm2:language?.curves.find(r=>r.step===row.step)?.[states.value]}));
 chart("critical-fit-chart","critical-fit-legend",rows,[{key:"huber",label:"冻结语言 · Huber",color:colors[0]},{key:"mse",label:"冻结语言 · MSE",color:colors[1]},{key:"lm2",label:"解冻语言末两层 · Huber",color:colors[3]}],{label:"训练状态拟合诊断，横轴优化步数，纵轴动作 RMSE，单位 rad"});
 const table=$("critical-fit-probes");table.replaceChildren();
 for(const task of Object.keys(chosen.probes[0]).sort()){const row=element("tr");row.append(element("td",taskLabel(task)),element("td",number(chosen.probes[0][task],4)),element("td",number(chosen.probes[160][task],4)),element("td",partner?number(partner.probes[160][task],4):"待完成"),element("td",language?number(language.probes[160][task],4):"待完成"));table.append(row);}
 $("critical-fit-evidence").replaceChildren(...fits.flatMap(f=>{const label=f.objective==="lm2"?"解冻语言末两层":f.objective.toUpperCase();return [evidenceLink(f.artifact.id,`${label} 原始动作误差报告`),evidenceLink(f.requestArtifact.id,`${label} 实验配方`)];}));
}
async function loadTrainingCoverage(snapshot,read=readComparisonArtifact){
 const artifacts=snapshot.artifacts||[],runs=snapshot.runs||[],protocol="qualified_training_coverage_cpu_v1";
 const reports=runs.flatMap(run=>(run.evaluation_reports||[]).filter(report=>report.protocol===protocol).map(report=>({run,report})));
 return Promise.all(reports.map(async({run,report:projected})=>{
  const owners=runs.filter(candidate=>(candidate.checkpoints||[]).some(checkpoint=>checkpoint.checkpoint===projected.checkpoint?.path));
  const result={run_ids:[run.id,...owners.map(owner=>owner.id)],modified_at:projected.modified_at,verified:false};
  try{
   const artifact=artifacts.find(item=>item.id===projected.artifact_id&&item.run_id===run.id&&item.kind==="report"&&item.path===projected.source);
   const report=await read(artifact),owner=owners.find(candidate=>(candidate.checkpoints||[]).some(checkpoint=>checkpoint.checkpoint===report.checkpoint?.path&&checkpoint.checkpoint_sha256===report.checkpoint?.sha256)),plan=qualifiedPlan(owner);
   if(!owner||!plan||plan.body!=="legs"||!sameEvidence(report.checkpoint,projected.checkpoint))throw Error("训练诊断未绑定同一腿式模型");
   if(report.protocol!==protocol||report.status!=="diagnosed"||report.population!=="training split only"||report.device!=="cpu"||report.precision!=="fp32"||report.optimizer_steps!==0||report.model_files_unchanged!==true||[report.counts_as_skill_success,report.independent_final_test,report.closed_loop_evaluated,report.training_data_produced,report.gpu_used,report.normalization_recomputed,report.model_files_changed].some(value=>value!==false))throw Error("训练诊断的数据用途或执行范围不一致");
   const requestArtifact=artifacts.find(item=>item.run_id===run.id&&item.path===`${run.source}/request.json`&&item.kind==="report"),request=await read(requestArtifact);
   const planArtifact=artifacts.find(item=>item.path===relativeEvidencePath(request.plan?.path,snapshot)&&item.sha256===request.plan?.sha256&&item.kind==="report"),registered=await read(planArtifact);
   const ownerPlanArtifact=artifacts.find(item=>item.id===plan.plan_artifact_id&&item.sha256===plan.plan_sha256&&item.kind==="report"),ownerPlan=await read(ownerPlanArtifact);
   const ticks=request.sample_ticks,body=report.bodies?.legs,integer=value=>Number.isSafeInteger(value)&&value>=0;
   if(request.protocol!==protocol||request.population!=="training split only"||!integer(request.noise_seed)||!Array.isArray(ticks)||!ticks.length||ticks.some((tick,i)=>!integer(tick)||(i>0&&tick<=ticks[i-1]))||!body||!sameEvidence(body.index,registered.datasets?.legs?.index)||!sameEvidence(body.index,ownerPlan.datasets?.legs?.index)||!sameKeys(Object.keys(body.tasks||{}).sort(),[...plan.tasks].sort()))throw Error("训练数据、任务或采样时刻与登记配置不一致");
   const forbidden=[...registered.validation_seeds||[],...registered.development_seeds||[],...registered.reserved_final_seeds||[],...Array.from({length:100},(_,i)=>910000+i)];
   if(![registered.validation_seeds,registered.development_seeds,registered.reserved_final_seeds].every(Array.isArray))throw Error("训练诊断缺少隔离数据配置");
   let episodes=0,eligible=0,sampled=0;
   for(const task of plan.tasks){
    const row=body.tasks[task],samples=row.sampled_episodes;
    if(!integer(row.episodes)||row.episodes<1||!integer(row.eligible_rows)||row.eligible_rows<1||!Array.isArray(samples)||samples.length!==4||new Set(samples.map(sample=>sample.seed)).size!==4||samples.some(sample=>!integer(sample.seed)||forbidden.includes(sample.seed)||sample.split!=="train"||!Number.isFinite(sample.command_max_abs_error)||sample.command_max_abs_error<0||sample.command_max_abs_error>1e-6)||!Number.isFinite(row.uniform_sample_probability)||Math.abs(row.uniform_sample_probability-row.eligible_rows/body.eligible_rows)>1e-12)throw Error("训练覆盖计数或抽查条件无效");
    for(const [field,total]of [["decision_ticks",row.eligible_rows],["instructions",row.episodes],["instruction_templates",row.episodes]]){const counts=Object.values(row[field]||{});if(!counts.length||counts.some(value=>!integer(value))||counts.reduce((a,b)=>a+b,0)!==total)throw Error("训练覆盖汇总不一致");}
    episodes+=row.episodes;eligible+=row.eligible_rows;sampled+=samples.length;
   }
   if(episodes!==body.train_episodes||eligible!==body.eligible_rows||sampled!==body.sampled_episodes||!Array.isArray(report.predictions)||report.predictions.length!==plan.tasks.length*ticks.length)throw Error("训练样本或动作预测条数不完整");
   const seen=new Set();
   for(const row of report.predictions){
    const sample=body.tasks[row.task]?.sampled_episodes[0],key=JSON.stringify([row.task,row.tick]);
    if(!sample||row.seed!==sample.seed||row.split!=="train"||!sameEvidence(row.archive,sample.archive)||!ticks.includes(row.tick)||seen.has(key)||[row.first_rmse_rad,row.prefix5_rmse_rad,row.valid_chunk_rmse_rad].some(value=>!Number.isFinite(value)||value<0))throw Error("训练状态动作误差存在重复或不匹配样本");seen.add(key);
   }
   const checkpoint=owner.checkpoints.find(item=>item.checkpoint===report.checkpoint.path),followups=[];
   for(const candidate of runs.flatMap(item=>(item.evaluation_reports||[]).filter(row=>row.protocol==="qualified_phase_fit_cpu_v2").map(row=>({run:item,projected:row})))){
    try{
     const linked=artifacts.find(item=>item.id===candidate.projected.artifact_id&&item.run_id===candidate.run.id&&item.path===candidate.projected.source&&item.kind==="report"),detail=await read(linked);
     if(detail.protocol==="qualified_phase_fit_cpu_v2"&&detail.status==="diagnosed"&&detail.action_padding==="training_noise_path"&&detail.population==="training split only"&&detail.counts_as_skill_success===false&&detail.closed_loop_evaluated===false&&detail.model_promoted===false&&sameEvidence(detail.dataset,body.index)&&Array.isArray(detail.models)&&detail.models.some(model=>sameEvidence(model.checkpoint,report.checkpoint)))followups.push(linked.id);
    }catch(error){/* Optional follow-up evidence may still be synchronizing. */}
   }
   return {...result,verified:true,run_ids:[run.id,owner.id],owner_id:owner.id,step:checkpoint.step,report,body,ticks,tasks:plan.tasks,evidence_ids:[artifact.id,requestArtifact.id],followup_ids:followups};
  }catch(error){return {...result,error:`训练诊断证据尚未核验：${error.message}`};}
 }));
}
function renderTrainingCoverage(run){
 const diagnostic=(data?.training_coverage||[]).filter(item=>item.run_ids.includes(run?.id)).sort((a,b)=>b.modified_at-a.modified_at)[0],section=$("training-coverage-section");
 section.hidden=!diagnostic;$("training-coverage-content").hidden=!diagnostic?.verified;
 if(!diagnostic?.verified){$("training-coverage-context").textContent=diagnostic?.error||"";return;}
 const {body,report,ticks,tasks}=diagnostic;
 $("training-coverage-context").textContent=`${diagnostic.owner_id} · 历史模型第 ${number(diagnostic.step,0)} 步 · 仅训练集。每项取 1 条教师轨迹，在 ${ticks.length} 个时刻用同一固定噪声进行独立 FP32 CPU 推理。`;
 $("training-coverage-summary").textContent=`训练集共 ${number(body.train_episodes,0)} 条轨迹、${number(body.eligible_rows,0)} 个动作块；完整核查其中 ${number(body.sampled_episodes,0)} 条轨迹的阶段命令和动作标签。下表覆盖率由全量训练索引计算，误差仅来自每项 1 条轨迹。`;
 const head=element("tr");for(const label of ["任务 / 训练 seed","训练动作块 / 占比",...ticks.map(tick=>`${number(tick/50,2)} 秒`)])head.append(element("th",label));$("training-coverage-head").replaceChildren(head);
 const rows=$("training-coverage-rows");rows.replaceChildren();
 for(const task of tasks){const values=body.tasks[task],predictions=report.predictions.filter(row=>row.task===task),tr=element("tr"),name=element("td",taskLabel(task));name.append(element("small",`${task} · seed ${predictions[0].seed}`));tr.append(name,element("td",`${number(values.eligible_rows,0)} / ${percent(values.uniform_sample_probability)}`));for(const tick of ticks)tr.append(element("td",number(predictions.find(row=>row.tick===tick).prefix5_rmse_rad,3)));rows.append(tr);}
 $("training-coverage-evidence").replaceChildren(evidenceLink(diagnostic.evidence_ids[0],"旧版诊断原始报告"),evidenceLink(diagnostic.evidence_ids[1],"采样规则与来源"),...(diagnostic.followup_ids||[]).map(id=>evidenceLink(id,"后续正式采样配置复核报告")));
}
async function loadReplayDiagnostics(snapshot,read=readComparisonArtifact){
 const purposes={qualified_saved_actions_cpu_diagnosis_v1:"validation",qualified_saved_development_cpu_diagnosis_v1:"development"},artifacts=snapshot.artifacts||[],runs=snapshot.runs||[];
 const reports=runs.flatMap(run=>(run.evaluation_reports||[]).filter(report=>Object.hasOwn(purposes,report.protocol)).map(report=>({run,report})));
 return Promise.all(reports.map(async({run,report:projected})=>{
  const protocol=projected.protocol,purpose=purposes[protocol];
  const owners=runs.filter(candidate=>(candidate.checkpoints||[]).some(checkpoint=>checkpoint.checkpoint===projected.checkpoint));
  const result={run_ids:[run.id,...owners.map(owner=>owner.id)],diagnostic_id:run.id,modified_at:projected.modified_at,verified:false};
  try{
   const artifact=artifacts.find(item=>item.id===projected.artifact_id&&item.run_id===run.id&&item.kind==="report"&&item.path===projected.source);
   const report=await read(artifact);
   if(report.protocol!==protocol||report.status!=="diagnosed"||report.checkpoint!==projected.checkpoint||report.weight_bytes_read!==0||[report.counts_as_skill_success,report.independent_final_test,report.gpu_used,report.student_inference_performed,report.training_data_produced].some(value=>value!==false))throw Error("诊断范围与登记记录不一致");
   if(purpose==="development"&&report.source_purpose!==purpose)throw Error("开发集诊断的数据用途不一致");
   const binding=report[purpose],validationPath=relativeEvidencePath(binding?.path,snapshot);
   const validationArtifact=artifacts.find(item=>item.kind==="report"&&item.path===validationPath&&item.sha256===binding?.sha256);
   const owner=owners.find(candidate=>(candidate.checkpoints||[]).some(checkpoint=>(purpose==="development"?candidate.development_report?.artifact_id:checkpoint.artifact_id)===validationArtifact?.id&&checkpoint.checkpoint===report.checkpoint&&checkpoint.checkpoint_sha256===report.declared_checkpoint_sha256&&Number.isSafeInteger(checkpoint.step)&&checkpoint.step>0));
   const expectedPath=purpose==="development"?`${snapshot.source?.root}/${owner?.source}/development-evaluation.json`:`${report.checkpoint}/closed-loop-validation.json`;
   if(!owner||validationArtifact?.run_id!==owner.id||binding?.path!==expectedPath)throw Error("诊断未绑定到同一 checkpoint 的原始评测报告");
   const validation=await read(validationArtifact);
   if(validation.protocol!=="project_qualified_evaluation_v2"||validation.purpose!==purpose||validation.evaluator_role!=="student"||validation.complete!==true||validation.independent_final_test!==false||validation.counts_as_skill_success!==false||validation.checkpoint!==report.checkpoint||validation.checkpoint_sha256!==report.declared_checkpoint_sha256)throw Error("原始学生评测报告身份不一致");
   if(purpose==="development"){
    const plan=qualifiedPlan(owner),rows=validation.episodes;
    if(validation.ablation||validation.offline_accepted!==false||validation.realtime_accepted!==false||!plan||!sameKeys(Object.keys(validation.tasks||{}).sort(),[...plan.tasks].sort())||!Array.isArray(rows)||rows.length!==plan.total_development_episodes)throw Error("开发集任务或样本量与已登记计划不一致");
    const seen=new Set();
    for(const row of rows){const key=JSON.stringify([row.task,row.seed]);if(!plan.tasks.includes(row.task)||!plan.development_seeds.includes(row.seed)||seen.has(key)||typeof row.success!=="boolean"||row.counts_as_skill_success!==false)throw Error("开发集存在重复、越界或无效记录");seen.add(key);}
    for(const task of plan.tasks){const selected=rows.filter(row=>row.task===task),score=validation.tasks[task];if(selected.length!==plan.development_seeds.length||score.episodes!==selected.length||score.successes!==selected.filter(row=>row.success).length)throw Error("开发集汇总与逐集记录不一致");}
   }
   const ticks=report.source_ticks,tasks=Object.entries(report.tasks||{}),integer=value=>Number.isSafeInteger(value)&&value>=0;
   if(!Array.isArray(ticks)||!ticks.length||ticks.some((tick,index)=>!integer(tick)||(index>0&&tick<=ticks[index-1]))||!tasks.length||!integer(report.episodes)||report.episodes<1||!sameKeys(tasks.map(([task])=>task).sort(),Object.keys(validation.tasks||{}).sort()))throw Error("诊断任务或时刻记录不完整");
   let episodes=0,successes=0,branches=0,maxError=0;
   for(const [task,row]of tasks){
    const original=validation.tasks[task];
    if(!integer(row.episodes)||!row.episodes||!integer(row.student_successes)||row.student_successes>row.episodes||original.episodes!==row.episodes||original.successes!==row.student_successes||!Number.isFinite(row.replay_max_abs_error)||row.replay_max_abs_error<0)throw Error("诊断中的学生成绩与原始验证不一致");
    let taskBranches=0,taskSuccesses=0;
    for(const tick of ticks){
     const cell=row.by_source_tick?.[tick];
     if(!cell||!integer(cell.branches)||cell.branches<1||cell.branches>row.episodes||!integer(cell.teacher_successes)||cell.teacher_successes>cell.branches||!Number.isFinite(cell.prefix5_rmse_rad_mean)||cell.prefix5_rmse_rad_mean<0)throw Error("教师分支计数或动作误差无效");
     taskBranches+=cell.branches;taskSuccesses+=cell.teacher_successes;
    }
    if(taskBranches!==row.teacher_branches||taskSuccesses!==row.teacher_completed_branches)throw Error("教师分支汇总与逐时刻记录不一致");
    episodes+=row.episodes;successes+=row.student_successes;branches+=taskBranches;maxError=Math.max(maxError,row.replay_max_abs_error);
   }
   if(episodes!==report.episodes||!Array.isArray(validation.episodes)||validation.episodes.length!==episodes)throw Error("重放条数与原始验证不一致");
   const checkpoint=owner.checkpoints.find(item=>item.checkpoint===report.checkpoint&&item.checkpoint_sha256===report.declared_checkpoint_sha256);
   const videos=(report.observer_videos||[]).filter(video=>video.policy_input===false).map(video=>({declared:video,artifact:artifacts.find(item=>item.run_id===run.id&&item.kind==="video"&&item.path===relativeEvidencePath(video.path,snapshot)&&item.sha256===video.sha256)}));
   return {...result,verified:true,run_ids:[run.id,owner.id],owner_id:owner.id,purpose,step:checkpoint.step,report,episodes,successes,branches,max_error:maxError,videos,evidence_ids:[artifact.id,validationArtifact.id]};
  }catch(error){return {...result,error:`诊断证据尚未核验：${error.message}`};}
 }));
}
function selectedReplayDiagnostic(run){
 return (data?.replay_diagnostics||[]).filter(report=>report.run_ids.includes(run?.id)).sort((a,b)=>b.modified_at-a.modified_at)[0];
}
function renderReplayDiagnostic(run){
 const diagnostic=selectedReplayDiagnostic(run),section=$("replay-diagnostic-section"),content=$("replay-diagnostic-content");
 section.hidden=!diagnostic;content.hidden=!diagnostic?.verified;
 if(!diagnostic?.verified){$("replay-diagnostic-context").textContent=diagnostic?.error||"";$("replay-diagnostic-videos").replaceChildren();replayVideosKey="";return;}
 const owner=(data.runs||[]).find(item=>item.id===diagnostic.owner_id),report=diagnostic.report;
 const population=diagnostic.purpose==="development"?"开发集":"固定验证集";
 $("replay-diagnostic-context").textContent=`${population} · 历史 checkpoint · 第 ${number(diagnostic.step,0)} 步 · ${diagnostic.owner_id}。${owner?.step>diagnostic.step?`该批次已训练到第 ${number(owner.step,0)} 步；下方诊断仅对应第 ${number(diagnostic.step,0)} 步。`:"此处展示已保存动作的 CPU 诊断。"}这些重放和教师分支不进入训练集，也不计入正式验收。`;
 const summary=$("replay-diagnostic-summary");summary.replaceChildren();
 for(const [value,label]of [[`${diagnostic.successes} / ${diagnostic.episodes}`,"原学生严格通过 / 重放条数"],[number(diagnostic.branches,0),"教师接手分支"],[number(diagnostic.max_error,8),"重放状态、动作与物理轨迹的最大绝对差"]]){const item=element("div");item.append(element("strong",value),element("span",label));summary.append(item);}
 const errorView=$("replay-diagnostic-metric").value==="error",head=element("tr"),title=element("th","任务");title.scope="col";head.append(title);
 for(const tick of report.source_ticks){const th=element("th",`${number(tick/50,2)} 秒`);th.scope="col";th.append(element("small",`tick ${tick}`));head.append(th);}$("replay-diagnostic-head").replaceChildren(head);
 $("replay-diagnostic-caption").textContent=errorView?"学生与教师前 5 步动作差异 · 各条件 RMSE 的均值（rad，越低越接近）。从同一状态分别继续运行后比较动作，后续状态可能不同。":"教师接手后仍能完成原任务的分支 / 全部分支；每一列分别统计，不合并成学生成功率。";
 const rows=$("replay-diagnostic-rows");rows.replaceChildren();
 for(const [task,row]of Object.entries(report.tasks)){
  const tr=element("tr"),label=(data.matrix||[]).find(item=>item.task===task)?.label||task,th=element("th",label);th.scope="row";th.append(element("small",task));tr.append(th);
  for(const tick of report.source_ticks){const cell=row.by_source_tick[tick],rate=cell.teacher_successes/cell.branches,td=element("td",errorView?number(cell.prefix5_rmse_rad_mean,3):`${cell.teacher_successes} / ${cell.branches}`,errorView?"":rate===1?"good":rate===0?"low":"partial");td.title=`${label} · tick ${tick} · 教师完成 ${cell.teacher_successes}/${cell.branches} · 动作 RMSE ${number(cell.prefix5_rmse_rad_mean,4)} rad`;tr.append(td);}rows.append(tr);
 }
 const videosKey=JSON.stringify([diagnostic.diagnostic_id,diagnostic.videos.map(({artifact})=>[artifact?.id,artifact?.sha256,artifact?.available,artifact?.url])]);
 if(videosKey!==replayVideosKey){
  const videos=$("replay-diagnostic-videos");videos.replaceChildren();
  for(const {artifact,declared}of diagnostic.videos){const figure=element("figure"),caption=element("figcaption",`${videoCaption(declared.path)} · 第 ${diagnostic.step} 步`);
   if(artifact?.available){const video=element("video");video.controls=true;video.playsInline=true;video.preload="metadata";video.src=artifact.url;video.setAttribute("aria-label",caption.textContent);figure.append(video);}else figure.append(element("p","重放视频尚未同步或来源未匹配"));
   figure.append(caption);videos.append(figure);
  }
  if(!diagnostic.videos.length)videos.append(element("p","此诊断未登记视频"));replayVideosKey=videosKey;
 }
 $("replay-diagnostic-evidence").replaceChildren(evidenceLink(diagnostic.evidence_ids[0],"完整重放与教师分支报告"),evidenceLink(diagnostic.evidence_ids[1],`该 checkpoint 的原始${population}报告`));
}
async function loadDevelopmentPlans(snapshot,read=readComparisonArtifact){
 await Promise.all((snapshot.runs||[]).map(async run=>{
  delete run.development_plan;
  if(run.stage!=="development_evaluation"||qualifiedPlan(run))return;
  const artifact=(snapshot.artifacts||[]).find(item=>item.run_id===run.id&&item.path===`${run.source}/pilot-config.json`);
  try{
   const config=await read(artifact),seeds=config?.development_seeds;
   if(typeof config?.root!=="string"||!config.root||config.output!==`${config.root.replace(/\/$/,"")}/${run.source}`||!Array.isArray(seeds)||!seeds.length||!seeds.every(seed=>Number.isSafeInteger(seed)&&seed>=0)||new Set(seeds).size!==seeds.length)return;
   run.development_plan={total_episodes:seeds.length,config_artifact_id:artifact.id,config_sha256:artifact.sha256};
  }catch{/* Missing or still-synchronizing evidence keeps the recorded-count wording. */}
 }));
}
async function loadReplayProgress(snapshot,read=readComparisonArtifact){
 const artifacts=snapshot.artifacts||[],purposes={qualified_saved_actions_cpu_diagnosis_v1:"validation",qualified_saved_development_cpu_diagnosis_v1:"development"};
 await Promise.all((snapshot.runs||[]).map(async run=>{
  delete run.replay_progress;
  if(run.stage!=="cpu_replay")return;
  const own=name=>artifacts.find(a=>a.run_id===run.id&&a.kind==="report"&&a.path===`${run.source}/${name}.json`);
  try{
   const request=await read(own("request")),progress=run.cpu_replay_progress,purpose=purposes[request.protocol];
   if(!purpose||[request.gpu_used,request.training_data_produced,request.counts_as_skill_success,request.student_inference_performed,request.independent_final_test].some(v=>v!==false)||request.weight_bytes_read!==0)return;
   const binding=request[purpose],path=relativeEvidencePath(binding?.path,snapshot),artifact=artifacts.find(a=>a.kind==="report"&&a.path===path&&a.sha256===binding?.sha256);
   const original=await read(artifact);
   if(original.protocol!=="project_qualified_evaluation_v2"||original.purpose!==purpose||original.evaluator_role!=="student"||original.complete!==true||original.independent_final_test!==false||original.counts_as_skill_success!==false||original.checkpoint!==request.checkpoint||original.checkpoint_sha256!==request.declared_checkpoint_sha256||!Array.isArray(original.episodes))return;
   if(progress?.stage!=="cpu_replay"||progress.status!=="running"||progress.gpu_used!==false||!Number.isSafeInteger(progress.episodes)||progress.episodes<0||!Number.isSafeInteger(progress.expected_episodes)||progress.expected_episodes!==original.episodes.length||progress.expected_episodes<1||progress.episodes>progress.expected_episodes)return;
   run.replay_progress={completed:progress.episodes,total:progress.expected_episodes,counts_as_skill_success:false};
  }catch{/* Updating or unavailable evidence keeps the neutral progress wording. */}
 }));
}
function pairedObjectiveMetrics(huber,reference){
 const measured=run=>new Map((run.metrics||[]).filter(row=>Number.isInteger(row.step)&&row.validation_measured===true&&(!row.validation_loss_objective||row.validation_loss_objective==="mse")&&[row.validation_loss,row.prefix5_rmse_rad].every(value=>Number.isFinite(value)&&value>=0)).map(row=>[row.step,row]));
 const h=measured(huber),m=measured(reference);
 return [...h.keys()].filter(step=>m.has(step)).sort((a,b)=>a-b).map(step=>({step,huber_mse:h.get(step).validation_loss,reference_mse:m.get(step).validation_loss,huber_prefix:h.get(step).prefix5_rmse_rad,reference_prefix:m.get(step).prefix5_rmse_rad}));
}
function fixedValidationIdentity(report,seeds){
 const score=report?.tasks?.walk_forward_stop,episodes=report?.episodes;
 if(report?.protocol!=="foundation_walk_v1"||report.acceptance_scope!=="foundation_walk"||report.mode!=="offline"||report.counts_as_skill_success!==false||report.ablation||!report.condition_recipe||report.execute_steps!==5||report.semantic_refresh_steps!==5||!report.action_padding||!report.inference_precision||!Array.isArray(episodes)||episodes.length!==seeds||score?.episodes!==seeds||score.distinct_conditions!==seeds||!Number.isInteger(score.successes)||score.successes<0||score.successes>seeds||score.strict_rate!==score.successes/seeds)return null;
 if(episodes.some(row=>row.task!=="walk_forward_stop"||!Number.isInteger(row.seed)||!row.initial_condition_sha256)||new Set(episodes.map(row=>row.seed)).size!==seeds)return null;
 return {protocol:report.protocol,condition_recipe:report.condition_recipe,mode:report.mode,execute_steps:report.execute_steps,semantic_refresh_steps:report.semantic_refresh_steps,action_padding:report.action_padding,inference_precision:report.inference_precision,episodes:episodes.map(row=>[row.seed,row.initial_condition_sha256]).sort((a,b)=>a[0]-b[0])};
}
async function loadObjectiveComparison(snapshot,read=readComparisonArtifact){
 const huber=(snapshot.runs||[]).find(run=>run.scope==="walk_flow_objective_v1"&&run.training_recipe?.flow_loss==="huber");
 if(!huber)return null;
 const artifacts=snapshot.artifacts||[],artifact=(run,name)=>artifacts.find(item=>item.run_id===run.id&&item.path===`${run.source}/${name}`);
 const result={huber_id:huber.id,verified:false,error:"配对证据尚未核验"};
 try{
  const configArtifact=artifact(huber,"pilot-config.json"),config=await read(configArtifact);
  const reference=(snapshot.runs||[]).find(run=>run.id===config.reference_pilot?.split("/").filter(Boolean).at(-1));
  if(!reference)throw new Error("登记的 MSE 参考批次尚未同步");result.reference_id=reference.id;
  const recipeArtifact=artifact(reference,"training-recipe.json"),referenceConfigArtifact=artifact(reference,"pilot-config.json"),auditArtifact=artifact(reference,"data-audit.json"),reportArtifact=artifact(reference,"pilot-report.json");
  if(config.protocol!=="walk_flow_objective_v1"||config.validation_loss_objective!=="mse"||config.reference_recipe_sha256!==recipeArtifact?.sha256||config.data_audit_sha256!==auditArtifact?.sha256||config.reference_report_sha256!==reportArtifact?.sha256)throw new Error("登记的参考配方或数据证据不匹配");
  const [recipe,referenceConfig,audit]=await Promise.all([read(recipeArtifact),read(referenceConfigArtifact),read(auditArtifact)]);
  const training=config.training_config,omit=new Set(["output_dir","flow_loss","flow_huber_delta"]),shared=value=>Object.fromEntries(Object.entries(value||{}).filter(([key])=>!omit.has(key)));
  const required=["dataset_index","model_path","normalization_path","seed","closed_loop_seeds","max_val_batches","eval_every","executed_prefix_weight","validation_sampling"];
  if(!training||required.some(key=>training[key]==null)||training.flow_loss!=="huber"||(recipe.flow_loss||"mse")!=="mse"||!sameEvidence(config.training_only_changes?.flow_loss,["mse","huber"])||!sameEvidence(shared(training),shared(recipe)))throw new Error("两组登记的训练条件不一致，不能作为损失函数对照");
  if(!config.initial_checkpoint_sha256||config.initial_checkpoint_sha256!==referenceConfig.checkpoint_sha256||!config.normalization_sha256||config.normalization_sha256!==referenceConfig.normalization_sha256||!config.dataset_index_sha256||config.dataset_index_sha256!==audit.index_sha256)throw new Error("初始化、归一化或数据身份未匹配");
  for(const [run,registered]of [[huber,training],[reference,recipe]]){
   for(const [key,value]of Object.entries(run.training_recipe||{}))if(value!=null&&!sameEvidence(value,registered[key]))throw new Error("当前训练配置与登记配方不一致");
   const sampling=run.validation_sampling,population=sampling?.population;
   if(sampling?.carried_forward_values_removed!==true||sampling.protocol!==registered.validation_sampling||sampling.eval_every!==registered.eval_every||sampling.dagger_sampling_fraction!==registered.dagger_sampling_fraction||population?.dataset_sha256!==config.dataset_index_sha256||!Array.isArray(population.indices)||!population.indices.length)throw new Error("验证样本或测量时序尚未核验");
  }
  if(!sameEvidence(huber.validation_sampling.population,reference.validation_sampling.population))throw new Error("两组验证样本不一致");
  result.verified=true;result.error=null;result.evidence_ids=[configArtifact.id,recipeArtifact.id];result.paired=pairedObjectiveMetrics(huber,reference);result.validation_samples=huber.validation_sampling.population.samples;result.interval=training.closed_loop_every;result.seeds=training.closed_loop_seeds;result.fixed={};
  const reports=await Promise.all([reference,huber].flatMap(run=>(run.checkpoints||[]).map(async projected=>{
   try{const source=artifacts.find(item=>item.id===projected.artifact_id),raw=await read(source);return {run_id:run.id,step:projected.step,artifact_id:projected.artifact_id,report:raw,identity:fixedValidationIdentity(raw,result.seeds)};}
   catch{return {run_id:run.id,step:projected.step,artifact_id:projected.artifact_id,identity:null};}
  })));
  const baseline=reports.find(item=>item.run_id===reference.id&&item.identity)?.identity;
  for(const item of reports){result.fixed[`${item.run_id}:${item.step}`]={artifact_id:item.artifact_id,verified:!!baseline&&!!item.identity&&sameEvidence(baseline,item.identity),score:item.report?.tasks?.walk_forward_stop};}
  return result;
 }catch(error){result.error=error.message;return result;}
}
function renderObjectiveComparison(run){
 const comparison=data?.objective_comparison,visible=comparison&&[comparison.huber_id,comparison.reference_id].includes(run?.id);
 $("objective-comparison-section").hidden=!visible;if(!visible)return;
 $("objective-comparison-content").hidden=!comparison.verified;
 $("objective-comparison-context").textContent=comparison.verified?`Huber：${comparison.huber_id}；MSE 参考：${comparison.reference_id}。已核对登记配方、初始化、数据、归一化和 ${number(comparison.validation_samples,0)} 个验证样本；仅训练损失函数不同。快照采集于 ${date(data.generated_at)}。`:comparison.error;
 if(!comparison.verified)return;
 const huber=data.runs.find(item=>item.id===comparison.huber_id),reference=data.runs.find(item=>item.id===comparison.reference_id),rows=comparison.paired;
 for(const [metric,label]of [["mse","同一步数的验证 MSE"],["prefix","同一步数的前五步动作误差"]])chart(`objective-${metric}-chart`,`objective-${metric}-legend`,rows,[{key:`huber_${metric}`,label:"Huber 训练",color:colors[0]},{key:`reference_${metric}`,label:"MSE 训练",color:colors[1]}],{label});
 const latest=rows.at(-1),summary=$("objective-latest");summary.replaceChildren();
 if(latest){for(const [label,h,m]of [["验证 MSE",latest.huber_mse,latest.reference_mse],["前 5 步 RMSE（rad）",latest.huber_prefix,latest.reference_prefix]]){const tr=element("tr");tr.append(element("td",`step ${number(latest.step,0)} · ${label}`),element("td",number(h,6)),element("td",number(m,6)));summary.append(tr);}}
 else{const tr=element("tr"),td=element("td","等待两组同一步数的实际验证记录；尚无可配对指标。");td.colSpan=3;tr.append(td);summary.append(tr);}
 $("objective-paired-note").textContent=`Huber 最新 step ${number(huber.step,0)}，MSE 最新 step ${number(reference.step,0)}；当前 ${number(rows.length,0)} 个配对测量点。曲线只保留双方同一步数的实际测量，不沿用前次验证值；数值越低表示对教师动作的拟合误差越小。`;
 const fixed=$("objective-fixed-rows");fixed.replaceChildren();
 const steps=[...new Set([comparison.interval,...[huber,reference].flatMap(item=>(item.checkpoints||[]).map(report=>report.step))])].filter(step=>Number.isInteger(step)&&step>0).sort((a,b)=>a-b);
 for(const step of steps){const tr=element("tr");tr.append(element("td",`step ${number(step,0)}`));for(const item of [huber,reference]){const entry=comparison.fixed[`${item.id}:${step}`],td=element("td");if(entry?.verified){td.append(element("strong",`${percent(entry.score.strict_rate)} · ${number(entry.score.successes,0)} / ${number(entry.score.episodes,0)}`),evidenceLink(entry.artifact_id,"固定验证报告"));}else td.append(element("span",entry?"报告条件或证据尚未核验":item.step<step?`待评测 · 尚未到第 ${number(step,0)} 步`:"待评测报告同步","pending"));tr.append(td);}fixed.append(tr);}
 const development=$("objective-development");development.replaceChildren();
 for(const [item,label]of [[huber,"Huber"],[reference,"MSE 参考"]]){const report=item.development_report,score=report?.tasks?.walk_forward_stop;const card=element("article",undefined,"baseline");card.append(element("h3",`${label} · 开发集`));if(score){card.append(element("div",`${percent(score.strict_rate)} · ${number(score.successes,0)} / ${number(score.episodes,0)}`,"rate"),element("small",`所选 ${report.checkpoint?.split("/").at(-1)||"模型"}；${number(score.distinct_conditions,0)} 个不同条件。`),evidenceLink(report.artifact_id,"开发评测报告"));}else card.append(element("p","待开发评测；不以固定验证结果代填。"));development.append(card);}
 $("objective-comparison-evidence").replaceChildren(...comparison.evidence_ids.map((id,index)=>evidenceLink(id,index?"MSE 参考配方":"Huber 登记配方")));
}
function renderMatrix(){
 const candidate=data.runs?.find(run=>run.id===data.selected_run_id)?.best_validation,note=$("matrix-candidate");
 note.replaceChildren();note.hidden=!candidate?.checkpoint||!candidate?.checkpoint_sha256;
 if(!note.hidden){note.append(element("strong",`本表候选：${candidate.checkpoint.split("/").pop()}（按固定验证选择）。`),element("span"," 上方曲线保留所有 checkpoint；最新保存的权重可能比候选模型退步。每行的验证、开发或正式测试范围以对应报告为准。 "),evidenceLink(candidate.artifact_id,"候选验证报告"));}
 const body=$("matrix-rows");body.replaceChildren();for(const row of data.matrix||[]){const tr=element("tr"),name=element("td",row.label);name.append(element("small",row.task));tr.append(name);
  for(const key of ["teacher","student"]){const score=row[key],td=element("td",score?percent(score.strict_rate):"未评测",score?"":"pending");if(score){td.append(element("small",counts(score)));if(key==="teacher"){const c=score.conditions;td.append(element("small",score.teacher_version||c?.teacher_version||c?.version||score.protocol||"条件版本见报告"));}td.append(evidenceLink(score.artifact_id));}tr.append(td);}
  for(const mode of ["offline","realtime"]){const formal=mode==="offline"&&data.registered_acceptance?data.registered_acceptance.matrix.find(r=>r.task===row.task):row;const accepted=formal?.[`${mode}_accepted`],td=element("td",accepted?"已验收":"未验收",accepted?"accepted":"pending");if(accepted)td.append(evidenceLink(formal[`${mode}_evidence`]?.artifact_id,"验收证据"));if(mode==="offline"&&data.registered_acceptance&&formal?.student){td.append(element("small",counts(formal.student)));if(!accepted){td.firstChild.textContent="未通过";td.append(evidenceLink(formal.student.artifact_id,"验收报告"));}}tr.append(td);}body.append(tr);
}const offline=(data.registered_acceptance?.matrix||data.matrix).filter(r=>r.offline_accepted).length,realtime=data.matrix.filter(r=>r.realtime_accepted).length;$("acceptance").textContent=`离线 ${offline} / 13`;$("acceptance-detail").textContent=`实时 ${realtime} / 13 · ${data.registered_acceptance?"离线为已登记模型，非所选训练批次":"所选候选，互不替代"}`;}
function renderRun(run){const metrics=run?.metrics||[],checkpoints=run?.checkpoints||[];renderRefitComparison(run);renderObjectiveComparison(run);renderMixedCollection(run);renderLossTailDiagnostic(run);renderDevelopmentProgress(run);renderCheckpointOutcomes(run);renderPaddingComparison(run);renderReselection(run);renderPrecisionProgress(run);renderPaddingDevelopment(run);
  const cpuCollection=run?.allocation?.hardware==="cpu"&&(run.collection_tasks||[]).length>0;
  const collectionOnly=!metrics.length&&!checkpoints.length&&(cpuCollection||run?.prefix_collection||run?.prefix_preflight||run?.collection_recipe?.teacher_rollin_probability>0||(run?.scope==="walk_dagger_v1"&&run?.collection_recipe?.teacher_rollin_probability===0)||["collect","collection","dagger_collection","data_audit","data_preflight"].includes(run?.stage));
  $("learning-plots").hidden=collectionOnly||Boolean((run?.padding_comparison||run?.padding_reselection||run?.precision_progress||run?.padding_development)&&!metrics.length);$("checkpoint-details").hidden=!checkpoints.length;$("collection-only-note").hidden=!collectionOnly||Boolean(run?.padding_comparison||run?.padding_reselection||run?.precision_progress||run?.padding_development);
  $("collection-only-note").textContent=run?.prefix_preflight?"此 CPU 作业只核查数据与物理重放，没有模型参数更新或学习曲线。":cpuCollection||run?.prefix_collection?"此 CPU 作业只采集和审计示范，不更新模型参数，因此没有训练损失曲线。":"本批次尚无参数更新记录。数据采集与审计见上方；产生训练指标后显示学习曲线。";
  $("run-context").textContent=run?`${run.id} · ${statusNames[run.status]||run.status} · ${run.padding_development?"开发集闭环评测，无参数更新":`${number(metrics.length,0)} 条损失记录`}${run.metrics_truncated?"（仅显示最近 10,000 条）":""}${run.best_validation?.checkpoint?" · 固定验证最佳模型 "+run.best_validation.checkpoint.split("/").pop():""}`:"暂无训练记录";
  $("validation-sampling-note").textContent=run?.validation_sampling?.carried_forward_values_removed?"横轴：优化器步数。验证仅显示实际评估点，日志中重复沿用的值已去除。":"横轴：优化器步数。按日志展示；未取得验证时序配置时，验证值可能沿用上次评估结果。";
  const sampling=run?.validation_sampling; if(sampling?.protocol==="stratified_v1") $("validation-sampling-note").textContent += ` 动作误差按任务与轨迹阶段均匀取样，共 ${number(sampling.population?.samples,0)} 个样本；与旧批次首批样本误差不可直接横比。`; else if(metrics.length) $("validation-sampling-note").textContent += " 此旧批次的动作误差仅测量第一批验证样本。";
  const scope=run?.scope||run?.pilot_report?.protocol||run?.manifests?.[0]?.acceptance_scope;
  const foundation=String(scope||"").includes("foundation_stability")||run?.id?.startsWith("foundation-pilot");
  const walkCurriculum=String(scope||"").includes("foundation_walk")||scope==="walk_dagger_v1"||scope==="walk_refit_pair_v1"||scope==="walk_flow_objective_v1";
  const plan=qualifiedPlan(run),runScope=$("run-scope");runScope.hidden=!plan&&!foundation&&!walkCurriculum;runScope.replaceChildren();
  if(plan){runScope.append(element("strong",`本轮本体：${plan.body==="legs"?"腿式":"滚轮"}（${plan.body}） · ${number(plan.tasks.length,0)} 项任务`),element("p",plan.tasks.map(taskLabel).join("、")),element("span",`开发评测计划：${number(plan.tasks.length,0)} 项任务 × ${number(plan.development_seeds.length,0)} 个条件，共 ${number(plan.total_development_episodes,0)} 条。开发评测不计入 13 项正式行为验收。 `),evidenceLink(plan.plan_artifact_id,"本轮计划"),document.createTextNode(" · "),evidenceLink(plan.request_artifact_id,"作业登记"));}
  else runScope.textContent=walkCurriculum?"本批次是 walk_forward_stop 行走课程。课程开发验证与原 walk_forward 的正式行为验收分开，不能自动提升下方验收状态。":"本批次仅验证基础站立稳定性，不能视为行走、踢球、视觉导航或 13 项行为验收。";
  const recipe=run?.training_recipe;$("training-recipe").hidden=!recipe?.max_steps;const weight=recipe?.executed_prefix_weight??1;const robust=recipe?.flow_loss==="huber";$("training-recipe").textContent=recipe?.max_steps?`本轮目标：${weight===1?"完整动作块等权":`前 5 步权重 ×${number(weight,0)}，其余有效步骤权重 1`}；${robust?`训练使用 2×Huber（δ=${number(recipe.flow_huber_delta??1,2)}），验证保持 MSE；`:""}${recipe.closed_loop_every?`每 ${number(recipe.closed_loop_every,0)} 步闭环检查`:"本轮未配置训练中闭环检查"}，最多 ${number(recipe.max_steps,0)} 步。不同权重或损失函数的 loss 不直接横比，请比较动作误差和闭环结果。`:"";
  chart("loss-chart","loss-legend",metrics,[{key:"train_loss",label:robust?"训练 2×Huber":"训练 loss"},{key:"validation_loss",label:robust?"验证 MSE":"验证 loss"}]);
  chart("rmse-chart","rmse-legend",metrics,[{key:"first_rmse_rad",label:"第一步"},{key:"prefix5_rmse_rad",label:"前 5 步"},{key:"full_rmse_rad",label:"完整动作块"}]);
  const names=[...new Set(checkpoints.flatMap(c=>Object.keys(c.tasks||{})))];
  const rows=checkpoints.map(c=>Object.assign({step:c.step},Object.fromEntries(names.map(name=>[name,c.tasks?.[name]?.strict_rate]))));
  chart("success-chart","success-legend",rows,names.map(name=>({key:name,label:(data.matrix||[]).find(r=>r.task===name)?.label||({stand_still:"基础站立",walk_forward_stop:"行走课程（非正式验收）"}[name])||name})),{rate:true});
  const body=$("checkpoint-rows");body.replaceChildren();for(const report of checkpoints){for(const [task,score]of Object.entries(report.tasks||{})){const row=element("tr"),failure=element("td",JSON.stringify(score.failed_checks||{}));const timing=score.failure_timing;if(timing?.protocol==="walk_failure_timing_v1"){failure.append(element("small",timing.observed_falls?`首次跌倒中位时间 ${number(timing.first_fall_median_s_among_fallen,2)} 秒（仅 ${number(timing.observed_falls,0)} 个已观测跌倒样本）；用于诊断。`:`${number(timing.no_fall_observed,0)} 个样本在观测期间未跌倒。`));if(timing.episodes_without_timing)failure.append(element("small",`${number(timing.episodes_without_timing,0)} 个样本未记录跌倒时间。`));}row.append(element("td",`step ${report.step}`),element("td",task),element("td",percent(score.strict_rate)),element("td",counts(score)),failure);const source=element("td");source.append(evidenceLink(report.artifact_id));row.append(source);body.append(row);}}
  const result=run?.pilot_report,student=result?.development,baseline=result?.constant_mean_baseline||run?.constant_mean_baseline?.summary||run?.constant_mean_baseline;
  const developmentSummary=$("development-summary");developmentSummary.hidden=!walkCurriculum||!student;developmentSummary.replaceChildren();if(walkCurriculum&&student){const falls=student.failure_timing?.observed_falls??student.failed_checks?.no_fall;developmentSummary.append(element("span",`所选模型开发评测：${percent(student.strict_rate)} · ${counts(student)}${Number.isInteger(falls)?` · ${number(falls,0)} 条发生跌倒`:""} · ${result.eligible_for_final_curriculum_test===true?"达到本课程下一阶段门槛":"本课程门槛未通过"}。此结果不计入 13 项正式行为验收。 `),evidenceLink(result.artifact_id,"开发评测报告"));}
  const qualifiedReport=qualifiedDevelopment(run,plan);if(qualifiedReport){developmentSummary.hidden=false;developmentSummary.replaceChildren(element("strong","所选模型开发评测已完成。以下为各任务开发结果，不计入 13 项正式行为验收。 "),evidenceLink(qualifiedReport.artifact_id,"开发评测报告"));for(const task of plan.tasks){const score=qualifiedReport.tasks[task];developmentSummary.append(element("br"),element("span",`${taskLabel(task)}：${percent(score.strict_rate)} · ${counts(score)}`));}}
  $("baseline-section").hidden=!foundation||(!student&&baseline?.strict_rate==null);
  $("baseline-context").textContent=run?`${run.id} · 基础站立开发条件；不属于正式技能测试集。`:"";
  $("student-baseline").textContent=percent(student?.strict_rate);$("student-baseline-count").textContent=counts(student);
  $("constant-baseline").textContent=percent(baseline?.strict_rate);$("constant-baseline-count").textContent=counts(baseline);
  renderInitializationComparison(run);renderCollection(run);if(run?.correction_labels&&run?.data_audit?.eligible_samples){const audit=run.data_audit;$("correction-note").textContent+=` 已审计训练样本：原始示范 ${number(audit.eligible_samples.teacher,0)} 行＋纠偏 ${number(audit.eligible_samples.dagger,0)} 行。${audit.holdouts_unchanged===true?"原验证与测试集保持不变。":""}${audit.normalization_recomputed===false?"沿用原模型归一化参数。":""}`;}renderCriticalFit(run);renderOnsetFits(run);renderPairedReplay(run);renderTrainingCoverage(run);renderReplayDiagnostic(run);renderArtifacts(run);
}
function artifactItem(a){const li=element("li");li.append(a.available?evidenceLink(a.id,({log:"日志",report:"报告",video:"视频"}[a.kind]||"证据")+" · "+a.path.split("/").slice(-2).join("/")):element("span",a.path));li.append(element("small",`${a.available?"已同步":"尚未同步"} · ${date(a.modified_at)}${a.kind==="log"?" · 仅最后 64 KiB":""}`));return li;}
function videoCaption(path){
 const parts=path.split("/"),precision=parts.find(p=>["bf16","fp32"].includes(p));
 const view=parts.at(-1)==="observer.mp4"?"鸭子与环境":parts.at(-1)==="camera.mp4"?"头部相机":"记录画面";
 const episode=parts.at(-2)?.match(/^(.*)-(\d+)$/);
 return [precision?.toUpperCase(),view,episode?.[1],episode?`seed ${episode[2]}`:null].filter(Boolean).join(" · ");
}
function renderArtifacts(run){const items=(data.artifacts||[]).filter(a=>a.run_id===run?.id);$("artifacts").replaceChildren();$("videos").replaceChildren();$("diagnostic-artifacts").replaceChildren();
 const replay=selectedReplayDiagnostic(run),shownVideos=new Set(replay?.verified?replay.videos.map(({artifact})=>artifact?.id):[]);
 for(const r of data?.paired_replays||[]){if(r.verified&&r.owner_id===run?.id)for(const p of r.pairs)if(p.media)shownVideos.add(p.media.id);}
 for(const a of items){if(shownVideos.has(a.id))continue;if(a.kind==="video"&&a.available){const figure=element("figure"),video=element("video"),caption=element("figcaption",videoCaption(a.path));video.controls=true;video.playsInline=true;video.preload="metadata";video.src=a.url;video.setAttribute("aria-label",videoCaption(a.path));caption.append(element("small",a.path));figure.append(video,caption);$("videos").append(figure);}else $("artifacts").append(artifactItem(a));}
 if(!items.length)$("artifacts").append(element("li","该批次暂无已登记证据"));
 for(const a of data.artifacts||[]){if(!a.run_id)$("diagnostic-artifacts").append(artifactItem(a));}
}
function render(){const freshness=data.freshness||{},f=$("freshness");f.className="freshness";if(!data.available){f.textContent=data.error||"暂无快照";f.classList.add("warn");$("run-select").replaceChildren();$("provenance").textContent=JSON.stringify(data,null,2);renderRun(null);renderPrefixDiagnostic();renderMatrix();return;}
 const state={fresh:"快照已更新",stale:"同步可能已中断：当前显示旧快照",disconnected:"读取 H200 失败：当前显示最后一次快照"}[freshness.state]||"同步状态未知";f.textContent=`${state} · 采集于 ${date(freshness.generated_at)} · ${number(Math.floor(freshness.age_seconds||0),0)} 秒前`;if(freshness.state!=="fresh")f.classList.add("warn");
 const active=data.active_job||{},activeRun=(data.runs||[]).find(run=>run.id===active.id)||active;$("active-stage").textContent=jobStage(activeRun);$("active-job").textContent=active.id||"尚无分配记录";$("active-progress").textContent=jobProgress(activeRun);$("active-status").textContent=`${statusNames[active.status]||active.status||"未知"} · ${active.allocation?.hardware==="cpu"?"CPU 作业":"GPU 分配"} ${statusNames[active.allocation?.status]||active.allocation?.status||"未知"}${active.total_tasks?` · 已完成 ${number(active.completed_tasks,0)}/${number(active.total_tasks,0)} 项`:""}`;
 const b=data.budget||{},unlimited=b.unlimited===true||("limit_hours" in b&&b.limit_hours===null),limit=b.limit_hours;
 $("budget").textContent=b.consumed_hours!=null?`${number(b.consumed_hours,2)}${unlimited?"":limit!=null?` / ${number(limit,0)}`:""} h`:"未读取账本";
 $("budget-track").hidden=unlimited||!Number.isFinite(limit)||limit<=0;
 $("budget-policy").textContent=unlimited?"不设累计上限 · 持续记录实际用量":limit!=null?`总上限 ${number(limit,0)} 小时 · 验收保留 ${number(b.reserved_acceptance_hours,0)} 小时`:"预算配置尚未读取";
 $("budget-fill").style.width=`${limit>0?Math.max(0,Math.min(100,(b.consumed_hours||0)/limit*100)):0}%`;
 $("budget-reserve").style.width=`${limit>0?Math.max(0,Math.min(100,(b.reserved_acceptance_hours||0)/limit*100)):0}%`;
 $("budget-reserve").title=`验收保留 ${number(b.reserved_acceptance_hours,0)} 小时`;
 const select=$("run-select");select.replaceChildren();for(const run of data.runs||[]){const option=element("option",`${run.id} · ${statusNames[run.status]||run.status}`);option.value=run.id;select.append(option);}select.value=data.selected_run_id||"";
 renderRunning();renderPrefixDiagnostic();renderRun((data.runs||[]).find(r=>r.id===data.selected_run_id));renderMatrix();$("provenance").textContent=JSON.stringify({schema:data.schema_version,source:data.source,freshness:data.freshness,budget:data.budget,warnings:data.warnings},null,2);
}
async function refresh(){if(loading){queuedRefresh=true;queuedRun=$("run-select").value;return;}loading=true;$("refresh").disabled=true;try{const run=$("run-select").value;const response=await fetch(`/api/training${run?"?run="+encodeURIComponent(run):""}`,{cache:"no-store"});if(!response.ok)throw new Error(`HTTP ${response.status}`);const snapshot=await response.json();[snapshot.objective_comparison,snapshot.replay_diagnostics,snapshot.training_coverage,snapshot.critical_fits,snapshot.initialization_comparison,snapshot.onset_fits,snapshot.paired_replays]=await Promise.all([loadObjectiveComparison(snapshot),loadReplayDiagnostics(snapshot),loadTrainingCoverage(snapshot),loadCriticalFit(snapshot),loadInitializationComparison(snapshot),loadOnsetFits(snapshot),loadPairedReplays(snapshot),loadDevelopmentPlans(snapshot),loadReplayProgress(snapshot)]);if(!queuedRefresh){data=snapshot;render();}}catch(error){$("freshness").textContent=`页面读取失败：${error.message}。已有画面可能过期。`;$("freshness").className="freshness error";}finally{loading=false;$("refresh").disabled=false;if(queuedRefresh){queuedRefresh=false;$("run-select").value=queuedRun;refresh();}}}
$("refresh").addEventListener("click",refresh);$("run-select").addEventListener("change",refresh);setInterval(()=>{if($("auto-refresh").checked&&!document.hidden)refresh();},30000);refresh();
$("replay-diagnostic-metric").addEventListener("change",()=>renderReplayDiagnostic((data?.runs||[]).find(run=>run.id===data.selected_run_id)));
let resizeTimer;window.addEventListener("resize",()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{if(data?.available)renderRun(data.runs.find(r=>r.id===data.selected_run_id));},120);});

$("critical-fit-state").addEventListener("change",()=>renderCriticalFit((data?.runs||[]).find(run=>run.id===data.selected_run_id)));

$("onset-fit-arm").addEventListener("change",()=>renderOnsetFits((data?.runs||[]).find(run=>run.id===data.selected_run_id)));

$("paired-replay-select").addEventListener("change",()=>renderPairedReplay((data?.runs||[]).find(run=>run.id===data.selected_run_id)));
$("outcome-checkpoint").addEventListener("change",()=>{const run=(data?.runs||[]).find(run=>run.id===data.selected_run_id);if(run){outcomeSelections.set(run.id,$("outcome-checkpoint").value);renderCheckpointOutcomes(run);}});
