"use strict";
(() => {
const $=id=>document.getElementById(id), make=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
const num=(v,d=0)=>Number.isFinite(v)?v.toLocaleString('zh-CN',{maximumFractionDigits:d}):'—';
const names={'history-grid-training-preflight-v1':'历史画面 · 训练链路验证','history-grid-resume-preflight-v1':'历史画面 · 断点恢复验证','clear-room-history-grid-bc-v1':'历史画面 · 寻球试训','history-grid-step000250-dev-v1':'历史画面 · 250 步候选','history-grid-step000500-dev-v1':'历史画面 · 500 步候选','clear-room-startup-recovery-bc-v1':'B 组 · 起步纠偏','clear-room-origin-sampling-control-bc-v1':'A 组 · 纠偏采样对照','clear-room-bc-v3':'历史模型 · 初始模仿','origin-sampling-control-dev-v1':'A 组 · 纠偏采样对照','startup-recovery-dev-v1':'B 组 · 起步纠偏','clear-dev-graph-v1':'历史模型 · 初始模仿','settled-recovery-dev-v1':'停止纠偏模型','current-policy-recovery-dev-v1':'上一轮纠偏模型'};
const label=id=>names[id]||id;
names['guarded-rehearsal-pilot-v1']='保留寻球能力 · 单障碍纠偏训练';
names['guarded-rehearsal-pilot-v1-development']='单障碍纠偏候选 · 81 案例开发评测';
names['target-memory-paired-retention-pilot-v1-preflight-v1']='目标记忆 · 两步训练检查';
names['target-memory-paired-retention-pilot-v1-resume-v1']='目标记忆 · 保存恢复检查';
names['target-memory-paired-retention-pilot-v1']='目标记忆 · 250 步候选训练';
names['memory-continuation-pilot-v1']='扩大数据覆盖 · 续训到 1,000 步';
names['memory-continuation-pilot-v1-development']='扩大数据覆盖 · 81 案例开发评测';
names['realtime-current-memory-corrective-pilot-v1']='实时失败纠偏 · 250 步候选训练';
names['realtime-current-memory-corrective-pilot-v1-preflight']='实时失败纠偏 · 两步训练检查';
names['realtime-current-memory-corrective-pilot-v1-restore']='实时失败纠偏 · 保存恢复检查';
names['realtime-current-memory-corrective-pilot-v1-development']='实时失败纠偏 · 81 案例离线评测';
for(const [arm,title] of [['control','原权重对照'],['weighted','加强横向控制']]){
 const id=`memory-axis-${arm}-pilot-v1`;
 names[id]=`${title} · 250 步候选训练`;
 names[id+'-preflight']=`${title} · 两步训练检查`;
 names[id+'-restore']=`${title} · 保存恢复检查`;
 names[id+'-development']=`${title} · 81 案例开发评测`;
}
let data=null,busy=false,selected=null,realtimeSelected=null;
function jobFamily(job){
 if(job?.model_family)return job.model_family;
 const name=job?.docker_name||job?.name||'';
 for(const family of ['vlm','hybrid'])if(name.startsWith(`microduck-fastwam-${family}-`))return family;
 return 'fastwam';
}
function fastwamJob(){return data.current_job&&jobFamily(data.current_job)==='fastwam'?data.current_job:null;}
function renderModelFamilies(){
 const panel=$('other-models'),rows=$('other-models-rows');if(!panel)return;
 const classified=data.gpu_accounting?.schema==='microduck_model_gpu_accounting_v1';
 const legacy=Array.from(new Map([...(data.active||[]),...(data.completed_containers||[]),...(data.current_job?[data.current_job]:[])].filter(j=>jobFamily(j)!=='fastwam').map(j=>[j.id||j.docker_name||j.name,j])).values());
 const groups=classified?data.other_models||{}:Object.fromEntries(['vlm','hybrid','unclassified'].map(f=>[f,{active:legacy.filter(j=>jobFamily(j)===f&&j.running),completed_containers:legacy.filter(j=>jobFamily(j)===f&&!j.running)}]));
 const visible=Object.entries(groups).filter(([,v])=>(v.active||[]).length||(v.completed_containers||[]).length||v.gpu_hours>0);
 panel.hidden=!visible.length;rows.replaceChildren();
 $('other-models-note').textContent=classified?'VLM 与组合实验的用量单独列出，不计入 FastWAM。':'上一份同步记录尚未按模型分类；混合用量暂不显示为 FastWAM 用量，等待下一次小时同步。';
 for(const [family,v] of visible){
  const jobs=v.active||[],live=jobs.filter(j=>j.running&&j.pid_live),ended=v.completed_containers||[];
  const state=live.length?`${live.length} 个作业运行中`:jobs.length?'进程状态待核查':`已结束 ${ended.length} 个作业，结果另验`;
  const row=make('tr');row.append(make('td',({vlm:'VLM',hybrid:'VLM + 世界动作模型',unclassified:'归类待核查'})[family]||'归类待核查'),make('td',state),make('td',classified?num(v.gpu_hours,2):'等待分类同步'));Array.from(row.children).forEach((td,i)=>td.dataset.label=['','作业状态','GPU 小时'][i]);rows.append(row);
 }
 if(!classified&&legacy.length){$('gpu').textContent='等待分类同步';$('gpu-scope').textContent='旧记录包含其他模型，暂不展示混合数值';}
 else{$('gpu').replaceChildren(document.createTextNode(num(data.gpu_hours,2)+' '),make('span','GPU 小时'));$('gpu-scope').textContent=classified?'仅 FastWAM · VLM 另列 · 不含 TinyVLA':'H200 分配时长 · 不含 TinyVLA';}
}
function score(node,e){const text=e?.groups?e.groups.map(g=>`${g.candidate_successes}/${g.total}`).join(' · '):e?`${e.passed} / ${e.total}`:'未评测';node.replaceChildren(make('span',text));node.firstChild.style.cssText='font-size:inherit;font-weight:inherit;color:inherit';}
function chart(run){
 const box=$('loss-chart');box.replaceChildren();const raw=(run?.metrics||[]).filter(p=>Number.isFinite(p.step)&&Number.isFinite(p['train/loss']));
 $('curve-note').textContent=raw.length?`${label(run.id)} · ${num(raw.length)} 次损失记录 · 横轴为优化器步数`:'当前批次尚无可用损失记录。';
 if(raw.length<2){box.append(make('div','等待足够的训练记录','empty'));return;}
 const W=570,H=250,L=48,R=14,T=15,B=34,x0=raw[0].step,x1=Math.max(x0+1,raw.at(-1).step),y1=Math.max(.001,...raw.map(p=>p['train/loss']));
 const X=v=>L+(v-x0)/(x1-x0)*(W-L-R),Y=v=>H-B-v/y1*(H-B-T);
 const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('role','img');svg.setAttribute('aria-label',`${label(run.id)}的训练损失，步数 ${x0} 至 ${x1}`);
 function el(tag,attrs,text){const e=document.createElementNS(svg.namespaceURI,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));if(text!==undefined)e.textContent=text;svg.append(e);return e;}
 for(let i=0;i<=4;i++){let v=y1*i/4;el('line',{x1:L,x2:W-R,y1:Y(v),y2:Y(v),stroke:'#304740'});el('text',{x:L-8,y:Y(v)+4,'text-anchor':'end'},num(v,2));}
 for(let i=0;i<=4;i++){const s=x0+(x1-x0)*i/4;el('text',{x:X(s),y:H-10,'text-anchor':'middle'},num(Math.round(s)));}
 const stride=Math.max(1,Math.ceil(raw.length/350));const draw=raw.filter((_,i)=>i%stride===0||i===raw.length-1);
 const avg=raw.map((p,i)=>({step:p.step,v:raw.slice(Math.max(0,i-19),i+1).reduce((s,q)=>s+q['train/loss'],0)/Math.min(20,i+1)})).filter((_,i)=>i%stride===0||i===raw.length-1);
 for(const [pts,key,color,width] of [[draw,'train/loss','#688477',1],[avg,'v','#a4e5ba',2.3]])el('path',{d:pts.map((p,i)=>`${i?'L':'M'}${X(p.step).toFixed(2)},${Y(p[key]).toFixed(2)}`).join(' '),fill:'none',stroke:color,'stroke-width':width});
 box.append(svg);
}
function comparison(e,name,retained=false){const c=make('div',undefined,`compare ${retained?'retained':''}`);if(e?.groups){c.append(make('div',name));for(const g of e.groups)c.append(comparison({passed:g.candidate_successes,total:g.total},g.scene==='room_clear'?'空房间寻球':'单障碍寻球'));return c;}const row=make('div',undefined,'row');row.append(make('span',name),make('b',e?`${e.passed} / ${e.total}`:'待评测'));const track=make('div',undefined,'track'),fill=make('div',undefined,'fill');fill.style.width=e?`${100*e.passed/e.total}%`:'0%';track.append(fill);c.append(row,track);return c;}
function renderRealtime(){
 const options=data.realtime_comparisons||{legacy:data.realtime_comparison},select=$('realtime-trial');
 const trials=Object.entries(options).filter(([,value])=>value);
 if(!trials.some(([id])=>id===realtimeSelected))realtimeSelected=data.active_realtime_trial||trials[0]?.[0]||null;
 select.replaceChildren();for(const [id,value] of trials){const option=make('option',value.label||'历史：单帧保留模型');option.value=id;select.append(option);}
 if(realtimeSelected)select.value=realtimeSelected;select.disabled=trials.length<2;
 const value=options[realtimeSelected];
 $('realtime-model').textContent=value?`${value.label||'历史：单帧保留模型'} · 权重 ${value.checkpoint_sha256.slice(0,12)} · ${value.input_history?'带历史观测':'单帧观测'}`:'对照批次尚未同步。';
 realtimeComparison(value);
}
function realtimeComparison(value){
 const benchmark=$('realtime-benchmark'),rows=$('realtime-rows');benchmark.replaceChildren();rows.replaceChildren();
 $('realtime-note').textContent='实时对照数据尚未同步。';$('realtime-details').textContent='暂无记录。';if(!value)return;
 const labels=value.variant_labels||{plain:'普通推理',optimized:'上下文缓存 + CUDA Graph'},states={pending:'等待运行',running:'进行中',complete:'已完成',invalid:'记录需核查',failed:'运行失败'};
 if(value.benchmark){
  benchmark.append(make('h3','固定输入 · 推理 p95'));const maximum=Math.max(...Object.values(value.benchmark).map(v=>v.p95_ms),1);
  for(const name of ['plain','optimized']){const v=value.benchmark[name],box=make('div',undefined,'compare'),row=make('div',undefined,'row'),track=make('div',undefined,'track'),fill=make('div',undefined,'fill');row.append(make('span',`${labels[name]} · ${v.calls} 次`),make('b',`${num(v.p95_ms,1)} ms`));fill.style.width=(100*v.p95_ms/maximum)+'%';track.append(fill);box.append(row,track);benchmark.append(box);}
  const a=value.benchmark.plain.p95_ms,b=value.benchmark.optimized.p95_ms;benchmark.append(make('p',a>0?`推理 p95 ${b<=a?'减少':'增加'} ${num(Math.abs(100*(1-b/a)),1)}%；固定输入测试不代表实时任务收益。`:'固定输入耗时不代表实时任务收益。','limits'));
 }
 for(const name of ['plain','optimized']){const v=value.variants[name],s=v.score,tr=make('tr');tr.append(make('td',labels[name]),make('td',`${states[v.state]||'状态待核查'}${Number.isInteger(v.completed)?` · ${v.completed}/${v.total}`:''}`),make('td',s?`${s.passed}/${s.total}`:'待完整评测'),make('td',s?`${s.fallback_episodes}/${s.total} 例`:'—'),make('td',s?num(s.minimum_real_time_factor,3):'—'));Array.from(tr.children).forEach((td,i)=>td.dataset.label=['','评测进度','实时通过','发生接管','最低实时倍率'][i]);rows.append(tr);}
 $('realtime-note').textContent=value.paired?`同案例对照：加速版新增 ${value.paired.improvements} 例成功，${value.paired.regressions} 例退步。这是开发对照，尚非独立能力验收。`:value.complete?'两组记录已完成，对照核查尚未通过，暂不判断任务收益。':'等待两个版本完成全部案例；未完成的结果不显示为 0 分。';
 $('realtime-details').textContent=JSON.stringify({模型摘要:value.checkpoint_sha256,批次:value.trial_id,观测历史:value.input_history,独立核验:value.assessment_source,协议:value.protocol_source,记录:Object.fromEntries(Object.entries(value.variants).map(([k,v])=>[labels[k],{拒绝动作块:v.score?.rejected_plans??null,各集推理p95范围ms:v.score?.episode_p95_range_ms??null,时钟达标案例:v.score?.timing_qualified??null,来源:v.source??null}])),固定输入来源:value.benchmark?{普通版:value.benchmark.plain.source,加速版:value.benchmark.optimized.source}:null},null,2);
}
function renderExpanded(){
 const result=data.expanded_realtime,box=$('expanded-realtime');box.hidden=!result;if(!result)return;
 const bars=$('expanded-bars');bars.replaceChildren();$('expanded-rows').replaceChildren();$('expanded-latency').textContent='';
 $('expanded-details').textContent=JSON.stringify(result.source||{},null,2);
 if(result.state!=='complete'){$('expanded-summary').textContent='新增布局的对照来源缺失或发生变化，成绩暂不展示。';$('expanded-filter').disabled=true;$('expanded-filter-note').textContent='等待完整核验。';return;}
 $('expanded-filter').disabled=false;
 const a=result.variants.plain,b=result.variants.optimized;
 $('expanded-summary').textContent=`普通版 ${a.passed}/${a.total}，加速版 ${b.passed}/${b.total}；新增 ${result.paired_gains} 例成功，${result.paired_regressions} 例退步。完整轨迹与控制时钟已复核。`;
 bars.append(comparison(a,'普通推理'),comparison(b,'语言缓存 + 上下文复用 + CUDA Graph'));
 $('expanded-latency').textContent=`稳态推理 p95：${num(a.steady_inference_p95_ms,1)} → ${num(b.steady_inference_p95_ms,1)} ms。发生控制接管：普通版 ${a.fallback_episodes}/${a.total} 例，加速版 ${b.fallback_episodes}/${b.total} 例。`;
 const selected=result.layout_results.filter(r=>$('expanded-filter').value!=='failures'||r.successes.optimized<r.trials_per_variant);
 $('expanded-filter-note').textContent=`显示 ${selected.length} / ${result.distinct_layouts} 张布局；筛选仅影响明细，上方始终为完整批次。`;
 for(const r of selected){const row=make('tr');row.append(make('td',String(r.seed)),make('td',`${r.successes.plain}/${r.trials_per_variant}`),make('td',`${r.successes.optimized}/${r.trials_per_variant}`));Array.from(row.children).forEach((td,i)=>td.dataset.label=['','普通版通过','加速版通过'][i]);$('expanded-rows').append(row);}
 $('expanded-details').textContent=JSON.stringify({权重:result.checkpoint_sha256,独立复核:result.source,协议:result.protocol_source,运行记录:{普通版:a.source,加速版:b.source},延迟口径:'分别合并各版本已完成的稳态推理请求计算p95；不含相机、网络与冷启动，各集请求数量不同。',统计单位:'20张布局，每张3个颜色任务；不视为60张独立地图。'},null,2);
}
function renderRepeated(){
 const result=data.repeated_realtime,box=$('repeated-realtime');box.hidden=!result;if(!result)return;
 const bars=$('repeated-bars'),rows=$('repeated-rows');bars.replaceChildren();rows.replaceChildren();
 $('repeated-details').textContent=JSON.stringify(result.source||{},null,2);
 if(result.state!=='complete'){$('repeated-summary').textContent='三轮对照记录需核查，暂不显示成绩。';return;}
 $('repeated-summary').textContent=`相同权重、物理持续运行：普通版 ${result.passed.plain}/${result.total_per_variant}，加速版 ${result.passed.optimized}/${result.total_per_variant}。全部六次运行已完成独立复核。`;
 for(const [key,label] of [['plain','普通推理'],['optimized','语言缓存 + 上下文复用 + CUDA Graph']]){const item=make('div',undefined,'compare'),row=make('div',undefined,'row'),track=make('div',undefined,'track'),fill=make('div',undefined,'fill');row.append(make('span',label),make('b',`${result.passed[key]}/${result.total_per_variant}`));fill.style.width=(100*result.passed[key]/result.total_per_variant)+'%';track.append(fill);item.append(row,track);bars.append(item);}
 for(const r of result.repeats){const tr=make('tr'),a=r.variants.plain,b=r.variants.optimized;tr.append(make('td',`第 ${r.repeat+1} 轮`),make('td',`${a.passed}/${a.total}`),make('td',`${b.passed}/${b.total}`),make('td',`${num(a.steady_inference_p95_ms,1)} → ${num(b.steady_inference_p95_ms,1)} ms`));Array.from(tr.children).forEach((td,i)=>td.dataset.label=['','普通版通过','加速版通过','推理 p95'][i]);rows.append(tr);}
 $('realtime-title').textContent='历史单轮实时对照';
 $('repeated-details').textContent=JSON.stringify({权重:result.checkpoint_sha256,独立复核:result.source,每轮记录:result.repeats,延迟口径:'每轮合并已完成稳态推理请求的p95；不含相机、网络与冷启动时间。'},null,2);
}
function renderTeacherDevelopment(){
 const t=data.teacher_development,box=$('teacher-development');box.hidden=!t;if(!t)return;
 const current=$('teacher-current-experiment'),e=t.current_experiment,comparison=t.completed_teacher_comparison,q=t.qualified_teacher,capture=t.data_collection;
 const colorLabels={find_red_ball:'红球',find_blue_ball:'蓝球',find_yellow_ball:'黄球'};
 current.replaceChildren();current.hidden=!e&&!comparison&&!q&&!capture;
 if(q){
  current.append(make('h3','单障碍教师资格'));
  if(q.state!=='verified')current.append(make('p','教师资格证据需核查，暂不作达标判断。'));
  else{
   current.append(make('p',`已通过 · ${q.successes}/${q.total} · ${q.layouts} 个独立布局 × 3 种颜色`));
   current.append(make('p',q.colors.map(r=>`${colorLabels[r.task]} ${r.successes}/${r.total}`).join('　·　')));
   current.append(make('p',`每种颜色均达到 ${q.minimum_per_color}/100 门槛，全部轨迹已独立重放。教师可以开始采集新示范；这些资格验证案例不进入训练。`,'limits'));
  }
 }
 if(capture){
  current.append(make('h3','当前阶段：独立示范采集与数据审计'));
  if(capture.state!=='snapshot')current.append(make('p','示范采集证据需核查，暂不显示数量。'));
  else{
   const phases={running:'核查时正在采集与重放',auditing:'采集结束，正在整组数据审计',complete:'数据审计通过，可进入训练准备',failed:'采集或审计异常',unavailable:'进程状态待核查'};
   current.append(make('p',`${phases[capture.phase]} · ${capture.completed}/${capture.total} 例`));
   const track=make('div',undefined,'track'),fill=make('div',undefined,'fill');fill.style.width=`${100*capture.completed/capture.total}%`;track.setAttribute('role','progressbar');track.setAttribute('aria-label','独立示范采集与回放进度');track.setAttribute('aria-valuemin','0');track.setAttribute('aria-valuemax',String(capture.total));track.setAttribute('aria-valuenow',String(capture.completed));track.append(fill);current.append(track);
   current.append(make('p',`严格成功 ${capture.successes} 例 · 失败 ${capture.failures} 例 · ${capture.training_data_released?'已入集':'待入集'}的成功示范 ${capture.eligible} 例`));
   current.append(make('p',`${capture.layouts} 个新布局 × 3 种颜色，与教师资格案例分开。${capture.training_data_released?'数据已经放行；后续仍需模型训练和能力评测。':'所有案例完成重放与整组审计后才放行训练。'} 本阶段没有更新模型权重。`,'limits'));
   current.append(make('small',`${new Date(capture.observed_unix*1000).toLocaleString('zh-CN',{hour12:false})} 核查 · 每小时同步`));
  }
 }
 if(comparison){
  current.append(make('h3','教师开发对照 · 与模型成绩分开统计'));
  if(comparison.state!=='verified')current.append(make('p','完整教师对照证据需核查，暂不显示成绩。'));
  else{
   current.append(make('p',`原教师 ${comparison.baseline}/${comparison.total} → ${comparison.candidate_label||'遮挡搜索教师'} ${comparison.candidate}/${comparison.total} · 新增 ${comparison.gains} 项成功 · 零退步、碰撞、跌倒或接管`));
   const colors={find_red_ball:'红球',find_blue_ball:'蓝球',find_yellow_ball:'黄球'};
   current.append(make('p',comparison.colors.map(r=>`${colors[r.task]} ${r.baseline}/${r.total} → ${r.candidate}/${r.total}`).join('　·　')));
   current.append(make('p',`20 个开发布局 × 3 种颜色，完整轨迹已重放核对。${q?'这是此前的教师开发对照，最新资格结果见上方。':'这一对照本身不构成新布局教师资格验证。'} 教师成绩不代表 FastWAM 模型已学会避障。`,'limits'));
  }
 }
 if(e){
  const spatial=e.id==='spatial-utility-expanded-v1';
  const guardedQualification=e.id==='guarded-occlusion-qualification-v1';
  const qualification=['spatial-utility-qualification-v1','occlusion-view-qualification-v1','guarded-occlusion-qualification-v1'].includes(e.id);
  const occlusion=e.id==='occlusion-view-expanded-v1',pilot=['occlusion-view-pilot-v1','physics-shortlist-pilot-v1'].includes(e.id);
  current.append(make('h3',qualification?'当前实验：新布局教师资格验证':occlusion?'当前实验：遮挡后重新观察 · 完整布局对照':spatial?'当前实验：补齐搜索盲区':pilot?'当前实验：物理预测辅助搜索 · 小样本对照':'搜索实验：细化搜索网格'));
  if(e.state==='awaiting_snapshot')current.append(make('p','前一阶段已完成，新布局验证已开始；采集数量等待下一次每小时同步。'));
  else if(e.state!=='snapshot')current.append(make('p','当前实验的进度证据需核查。'));
  else{
   current.append(make('p',`${guardedQualification?'已采集并重放核对':'已采集'} ${e.completed} / ${e.total} 集 · ${new Date(e.observed_unix*1000).toLocaleString('zh-CN',{hour12:false})} 核查 · 每小时同步`));
   const labels={room_curriculum_one:'单障碍',room_curriculum_two:'双障碍'},states={running:'核查时运行中',auditing:'采集结束，正在审计',queued:'等待前组完成',complete:'采集结束，需查看审计',failed:'运行异常',unavailable:'进程状态待核查'};
   for(const r of e.rows){const item=make('div',undefined,'compare'),line=make('div',undefined,'row'),track=make('div',undefined,'track'),fill=make('div',undefined,'fill');line.append(make('span',`${labels[r.scene]} · ${states[r.state]}`),make('b',`${r.completed}/${r.total}`));fill.style.width=`${100*r.completed/r.total}%`;track.setAttribute('role','progressbar');track.setAttribute('aria-label',`${labels[r.scene]}搜索实验采集进度`);track.setAttribute('aria-valuemin','0');track.setAttribute('aria-valuemax',String(r.total));track.setAttribute('aria-valuenow',String(r.completed));track.append(fill);item.append(line,track);current.append(item);}
   current.append(make('p',qualification?'100 个新布局 × 3 颜色，逐颜色成功率需达到 95%。完成数包含失败案例，不是通过率；只验证单障碍教师，不代表模型已掌握能力，目前未放行训练。':occlusion?'完整 20 布局 × 3 颜色对照，包含复用的 9 集前置实验。比较换位置重新观察是否改善寻球；必须有提升且不损失原有成功案例，才进入新布局教师验证。完成数包含失败，不是模型通过率。':spatial?'完整 20 布局 × 3 颜色对照，包含复用的 3 集前置实验。完成数包含失败案例，不是通过率；仍需独立教师资格验证，目前未放行训练。':pilot?'3 个开发布局 × 3 颜色，比较物理预测辅助搜索策略。完成数包含失败；小样本对照不等于教师资格或模型能力验收。':'仅比较搜索网格密度；采集完成数包含失败案例，不是通过率。两类场景均改善后仍需独立资格验证，目前未放行训练。','limits'));
  }
 }
 const rows=$('teacher-curricula');rows.replaceChildren();
 $('teacher-time').textContent=t.observed_unix?`状态核查于 ${new Date(t.observed_unix*1000).toLocaleString('zh-CN',{hour12:false})} · 每小时同步`:'状态核查时间尚未取得';
 $('teacher-progress-summary').textContent=t.state==='invalid'?'教师进度证据需核查，暂不显示完成数量。':`历史资格验证已完成 ${t.completed} / ${t.total} 集。完成数量包含失败案例，尚待整组审计。`;
 const gate=t.qualification_upper_bound,warning=$('teacher-qualification-warning');
 warning.hidden=!gate;
 if(gate){
  const scenes={room_curriculum_one:'单障碍',room_curriculum_two:'双障碍'},tasks={find_red_ball:'寻红球',find_blue_ball:'寻蓝球',find_yellow_ball:'寻黄球'};
  warning.textContent=gate.state==='verified'?`历史教师验证 · ${scenes[gate.scene]}${tasks[gate.task]}：已重放核实 ${gate.verified_failures} 次失败，当时版本最高只能达到 ${gate.maximum_successes}/${gate.total}，低于 ${gate.required_successes}/${gate.total} 门槛，因此该组数据未放行。这是历史成功数上限，不是上方新版本的验证结果。`:'历史教师资格上限证据需核查，暂不作达标判断。';
 }
 const states={running:'核查时正在运行',queued:'等待前一课程完成',complete:'运行结束，待查看审计',failed:'运行异常，待检查'};
 for(const r of t.curricula||[]){const item=make('div',undefined,'compare'),line=make('div',undefined,'row'),track=make('div',undefined,'track'),fill=make('div',undefined,'fill');line.append(make('span',r.label),make('b',`${r.completed} / ${r.total}`));fill.style.width=`${100*r.completed/r.total}%`;track.setAttribute('role','progressbar');track.setAttribute('aria-label',`${r.label}验证进度`);track.setAttribute('aria-valuemin','0');track.setAttribute('aria-valuemax',String(r.total));track.setAttribute('aria-valuenow',String(r.completed));track.append(fill);item.append(line,track,make('small',`${states[r.state]||'状态待核查'} · ${r.layouts} 种布局 × ${r.instructions} 种颜色指令`));rows.append(item);}
 $('teacher-pilot').textContent=t.pilot?`前置小样本：单障碍 ${t.pilot.successes.room_curriculum_one}/9，双障碍 ${t.pilot.successes.room_curriculum_two}/9，均已独立复核；每组仅 ${t.pilot.layouts_per_scene} 种布局。`:'前置小样本的独立审计证据尚未取得。';
 const c=t.corrected_case;
 $('teacher-case').hidden=!c;
 $('teacher-case').textContent=c?`一次已复核的修正：四障碍布局 ${c.seed} 寻蓝球，从失败变为通过；${num(c.first_observation_s,1)} 秒发现，${num(c.completion_s,2)} 秒完成。${t.comparison_state==='running'?'九案例完整对照正在进行。':'九案例整体结论须查看独立报告。'}`:'';
 $('teacher-evidence').textContent=JSON.stringify({最新教师资格:q?.source,资格证书:q?.certificate,独立示范采集:capture?.source,采集计划:capture?.plan,已放行数据:capture?.release_source,完整教师开发对照:comparison?.source,当前搜索实验快照:t.current_experiment?.source,历史资格验证协议:t.protocol_source,历史资格上限复核:t.qualification_upper_bound?.source,历史小样本审计:t.pilot?.source,修正案例审计:c?.source,口径:'CPU 教师实验；完成进度不是通过率，教师成绩不计入 FastWAM 模型成绩。状态按核查时间展示，不代表此刻实时进程。'},null,2);
}
function renderCloseout(){
 const c=data.closeout,panel=$('campaign-closeout');panel.hidden=!c;if(!c)return;
 $('closeout-rows').replaceChildren();$('closeout-evidence').textContent='';$('closeout-next').textContent='';
 if(c.state!=='complete'){$('closeout-summary').textContent='收尾报告来源需核查，暂不显示汇总成绩。';return;}
 $('closeout-summary').textContent='暂停前的评测、轨迹审计和数据导出已完成。下方为当时的收尾记录；最新运行状态以页面顶部为准。';
 for(const r of c.summary_rows){const tr=make('tr');for(const [key,label] of [['label','结果范围'],['result','当前结果'],['note','结论']]){const td=make('td',r[key]);td.dataset.label=label;tr.append(td);}$('closeout-rows').append(tr);}
 $('closeout-next').textContent='待恢复后再做：完整模型保存恢复检查 → 障碍任务训练与开发评测 → 独立布局及实时验收。以上结果均不代表完整自主导航或真机验收。';
 $('closeout-evidence').textContent=JSON.stringify({收尾报告:c.source,证据:c.sources,完整离线对照:c.navigation_baselines,未启动:c.not_run,GPU用量:c.gpu_accounting},null,2);
}
function renderGuardedCandidate(){
 const c=data.guarded_candidate_development,panel=$('guarded-candidate');if(!panel)return;panel.hidden=!c;if(!c)return;
 $('guarded-candidate-title').textContent=c.run_id==='target-memory-paired-retention-pilot-v1'?'目标记忆候选 · 与保留模型对照':['action-scope-continuation-pilot-v2','action-scope-epoch-pilot-v2'].includes(c.run_id)?'单障碍续训候选 · 与保留模型对照':c.run_id==='guarded-rehearsal-pilot-v1'?'单障碍纠偏候选 · 训练前后对照':'单障碍候选 · 训练前后对照';
 const status={pending:`训练 ${c.training_step??0}/${c.training_target_step??250} 步；等待固定案例评测`,evaluating:`已评测 ${c.completed}/81 个案例，尚未汇总成绩`,awaiting_replay:'模型评测已结束，等待独立重放核验',replaying:`正在重放核验${Number.isInteger(c.replayed)?` · ${c.replayed}/81`:''}`,failed:'本轮执行失败，结果待核查',invalid:'评测依据不一致，暂不显示成绩',verified:'81 个案例已完成独立重放核验；以下为离线开发结果'};
 $('guarded-candidate-status').textContent=status[c.state]||'等待核验';$('guarded-candidate-rows').replaceChildren();
 if(c.state==='verified')for(const g of c.groups){const tr=make('tr');for(const [title,text] of [['任务组',g.scene==='room_clear'?'空房间寻球':'单障碍寻球'],['训练前',`${g.baseline_successes}/${g.total}`],['训练后',`${g.candidate_successes}/${g.total}`],['新增通过 / 退步',`+${g.gained} / −${g.regressed}`]]){const td=make('td',text);td.dataset.label=title;tr.append(td);}$('guarded-candidate-rows').append(tr);}
 $('guarded-candidate-evidence').textContent=JSON.stringify(c.sources||{状态:c.state,验收:false},null,2);
}
function renderCheckpointReview(){
 const c=data.checkpoint_review,panel=$('checkpoint-review');if(!panel)return;panel.hidden=!c;if(!c)return;
 const step=c.selected_checkpoint_step,known=Number.isInteger(step);
 const stage={bind:'正在加载所选模型',evaluation_preflight:'正在检查评测环境',development_evaluation:'正在运行开发案例',replay_config:'正在准备独立重放',independent_replay:'正在独立重放核验',finish:'正在汇总结果'};
 const status={pending:stage[c.stage]||'等待评测记录',evaluating:`已评测 ${c.completed}/81 个案例`,awaiting_replay:'81 个案例已结束，等待独立重放',replaying:`正在独立重放${Number.isInteger(c.replayed)?` · ${c.replayed}/81`:''}`,verified:'81 个案例已完成独立重放；以下为离线开发结果',failed:'评测执行失败，结果待核查',invalid:'模型或评测依据不一致，暂不显示成绩'};
 $('checkpoint-review-title').textContent=known?`第 ${step} 步版本 · 中间权重复测`:'中间权重复测 · 等待核查';
 $('checkpoint-review-status').textContent=status[c.state]||'等待核验';
 $('checkpoint-review-note').textContent=known?`来自已完成 ${c.training_completed_step} 步的训练批次。本次仅评测第 ${step} 步保存的权重，没有新增训练步数。权重：${c.checkpoint_sha256.slice(0,12)}。`:'所选版本尚未取得完整证据。';
 const rows=$('checkpoint-review-rows');rows.replaceChildren();
 if(c.state==='verified')for(const g of c.groups){const tr=make('tr');for(const [title,text] of [['任务组',g.scene==='room_clear'?'空房间寻球':'单障碍寻球'],['保留模型',`${g.baseline_successes}/${g.total}`],['所选版本',`${g.candidate_successes}/${g.total}`],['新增通过 / 退步',`+${g.gained} / −${g.regressed}`]]){const td=make('td',text);td.dataset.label=title;tr.append(td);}rows.append(tr);}
 $('checkpoint-review-evidence').textContent=JSON.stringify({训练批次:c.training_run_id,所选步数:step,模型摘要:c.checkpoint_sha256,阶段:c.stage,实际进程存活:c.live,选择依据:c.selection_sources,评测依据:c.sources,正式验收:false},null,2);
 if(c.live&&c.state!=='invalid'&&c.state!=='failed'){
  const replaying=c.state==='replaying',completed=replaying?(c.replayed??0):(c.completed??0);
  const stale=data.freshness?.state!=='fresh';
  $('current-title').textContent=`${stale?'最近记录：':''}第 ${step} 步版本 · ${status[c.state]||stage[c.stage]||'复测中'}`;
  $('current-note').textContent='比较同一次训练保存的不同版本，检查增加训练步数是否改善闭环表现。完整重放核验后才显示分数。';
  $('step-label').textContent=`第 ${step} 步版本 · ${replaying?'重放核验':'开发评测'}进度`;
  $('steps').textContent=`${completed} / 81`;$('step-fill').style.width=`${100*completed/81}%`;
  $('step-progress').setAttribute('aria-valuenow',String(completed));$('step-progress').setAttribute('aria-valuemin','0');$('step-progress').setAttribute('aria-valuemax','81');
  $('step-note').textContent=`已结束案例数，不是成功数；训练已完成 ${c.training_completed_step} 步，本次新增 0 步。`;
 }
}
function renderMemoryPreflight(){
 const c=data.memory_preflight,panel=$('memory-preflight');if(!panel)return;panel.hidden=!c;if(!c)return;
 const correction=c.runtime_contract==='current_policy_corrections',title=correction?'纠偏模型':'目标记忆';
 $('memory-preflight-title').textContent=`${title} · 训练链路检查`;
 const states={waiting_for_data:'等待纠偏数据与训练配置核验',waiting_for_review:'等待中间权重复测及独立重放',waiting_for_gpu:'等待空闲 GPU',checking:'正在核对训练检查结果',training:'正在执行两步训练检查',restoring:'正在验证保存恢复',complete:'两步训练与保存恢复检查通过',skipped:'已跳过，等待中间权重采用判断',failed:'训练检查异常，需处理后再推进',invalid:'状态依据不一致，暂不作完成判断',unavailable:'进程或容器状态待核查'};
 const known=!['invalid','unavailable'].includes(c.state),step=known&&Number.isInteger(c.optimizer_steps)?c.optimizer_steps:null;
 const status=states[c.state]||'状态待核查';
 $('memory-preflight-status').textContent=`${status} · 本阶段 GPU：${c.gpu_allocated?'核查时使用中':known?'核查时未占用':'待核查'}`;
 const rows=$('memory-preflight-stages');rows.replaceChildren();
 for(const [name,value] of [[correction?'先完成纠偏数据核验':'先完成版本复测',['waiting_for_review','waiting_for_data'].includes(c.state)?'等待中':correction?'以核验记录为准':'以复测结果为准'],['模型实际更新',step===null?'待核查':`${step} / 2 步`],['保存与恢复',c.state==='complete'?'已通过':c.state==='restoring'?'验证中':known?'待验证':'待核查'],['正式候选训练','本检查不启动']]){
  const row=make('div',undefined,'compare'),line=make('div',undefined,'row');line.append(make('span',name),make('b',value));row.append(line);rows.append(row);
 }
 $('memory-preflight-note').textContent=c.state==='complete'?'完整模型更新两步并完成恢复核验；后续候选仍需正式训练及同一组 81 案例比较。目前没有新增能力成绩。':c.state==='skipped'?'本次检查没有执行训练；中间权重是否采用须依据独立评测。':correction?'用本轮纠偏数据检查训练和恢复是否正常；寻球与避障能力的变化仍需后续闭环评测。':'尝试让模型在球离开当前视野后，仍能参考先前看见球的画面。两步检查只验证训练链路；改善寻球能力需要后续闭环评测。';
 $('memory-preflight-evidence').textContent=JSON.stringify({状态:c.state,实际进程或容器存活:c.live,本阶段新增训练步数:step,最大训练步数:2,本检查负责启动候选:false,模型已替换:false,正式验收:false,依据:c.sources},null,2);
 if(c.live&&known&&!['failed','complete','skipped'].includes(c.state)&&!(correction&&data.correction_data_progress?.live)&&!data.checkpoint_review?.live&&!data.current_evaluation&&(!fastwamJob()||c.gpu_allocated)&&data.execution_control?.status!=='paused_by_user'){
  $('current-title').textContent=`${data.freshness?.state==='fresh'?'核查时：':'最近记录：'}${title} · ${status}`;
  $('current-note').textContent='先检查两步更新、保存和恢复。保留模型与能力分数保持原记录。';
  $('step-label').textContent=`${title} · 两步训练检查`;$('steps').textContent=step===null?'待核查':`${step} / 2`;
  $('step-fill').style.width=step===null?'0%':`${100*step/2}%`;
  $('step-progress').setAttribute('aria-valuemax','2');
  if(step===null)$('step-progress').removeAttribute('aria-valuenow');else $('step-progress').setAttribute('aria-valuenow',String(step));
  $('step-note').textContent=c.state==='restoring'?'恢复时不新增训练步数；正在核对模型和训练状态。':'这是运行检查；两步完成不代表任务能力达标。';
 }
}
function renderMemoryCandidate(){
 const c=data.memory_candidate,panel=$('memory-candidate');if(!panel)return;panel.hidden=!c;if(!c)return;
 const correction=c.runtime_contract==='current_policy_corrections',title=correction?'纠偏模型':'目标记忆';
 panel.querySelector('h2').textContent=`${title} · 候选训练`;
 const labels={training:'候选训练进行中',checking:'正在核查训练结果',complete:'250 步训练完成，能力待评测',waiting_for_preflight:'已登记，等待数据与完整模型预检',waiting_for_gpu:'等待空闲 GPU',failed:'本轮训练异常',invalid:'训练依据需核查',unavailable:'进程或容器状态待核查'};
 const phases={initializing_model:'加载模型',loading_full_model:'加载完整模型',initializing_optimizer:'准备优化器',training:'更新模型',saving_checkpoint:'保存训练状态',offline_tensor_validation:'核对输出'};
 const valid=!['invalid','unavailable'].includes(c.state),step=valid&&Number.isInteger(c.optimizer_steps)?c.optimizer_steps:null;
 const status=labels[c.state]||'状态待核查',detail=c.state==='training'&&phases[c.stage]?` · ${phases[c.stage]}`:'';
 $('memory-candidate-status').textContent=`${status}${detail} · ${step===null?'步数待核查':`${step} / 250 步`}`;
 const rows=$('memory-candidate-stages');rows.replaceChildren();
 const queued=c.continuations||{},stateLabels={not_registered:'尚未登记',waiting:'已登记，等待前置结果',running:'流程运行中',ended:'流程已结束，结果另行核验',failed:'流程异常',invalid:'依据需核查',unavailable:'进程状态待核查'};
 const fit=queued.training_fit,fitPhases={action_fit_retained:'比较保留模型输出',action_fit_candidate:'比较候选模型输出',independent_action_fit_assessment:'核算两套模型的动作误差'};
 const fitStatus=fit?.state==='ended'&&fit.skipped?'诊断已跳过，查看评测结果':fit?.state==='running'&&fitPhases[fit.stage]?fitPhases[fit.stage]:stateLabels[fit?.state]||'尚未取得状态';
 const alignmentStatus=queued.alignment?.state==='ended'&&queued.alignment.skipped?'开发门槛未通过，已跳过':stateLabels[queued.alignment?.state]||'尚未取得状态';
 const realtimeStatus=queued.realtime?.state==='ended'&&queued.realtime.skipped?'前置评测未通过，已跳过':stateLabels[queued.realtime?.state]||'尚未取得状态';
 const stageRows=[[`${title}训练`,step===null?'待核查':`${step} / 250 步`],['21 个空房间 + 60 个障碍案例',stateLabels[queued.development?.state]||'尚未取得状态'],['达到开发门槛后比较三种推理实现',alignmentStatus]];
 if(correction)stageRows.push(['三种实现连续运行，比较成功率与延迟',realtimeStatus]);
 stageRows.push(['未达开发门槛时检查训练样本拟合',fitStatus]);
 for(const [name,value] of stageRows){
  const row=make('div',undefined,'compare'),line=make('div',undefined,'row');line.append(make('span',name),make('b',value));row.append(line);rows.append(row);
 }
 $('memory-candidate-note').textContent=correction?'针对本轮学生运行中的失败状态补充教师示范，在完整保留旧数据的基础上训练 250 步。等待状态表示尚未分配训练 GPU。完成后仍需同一组 81 案例的闭环评测与独立重放；训练完成不会自动替换保留模型。':'本轮检验：球离开当前视野后，参考先前看见球的画面，是否改善绕障且保住寻球能力。相同 81 案例完成独立重放后再判断。若未达门槛，将比较新旧模型在 48 个固定训练样本上的搜索和停止输出，排查是否学会示范；这项诊断不新增训练，也不作为能力验收。';
 $('memory-candidate-evidence').textContent=JSON.stringify({状态:c.state,实际进程或容器存活:c.live,本阶段GPU分配:c.gpu_allocated,训练步数:step,后续流程:queued,模型已替换:false,正式验收:false,依据:c.sources},null,2);
 if(valid&&c.live&&c.gpu_allocated&&data.execution_control?.status!=='paused_by_user'){
  $('current-title').textContent=`${data.freshness?.state==='fresh'?'核查时：':'最近记录：'}${title} · ${status}${detail}`;
  $('current-note').textContent='保留模型继续作为对照；本轮先训练 250 步，再评测相同的 81 个案例。新候选尚未得到能力成绩。';
  $('step-label').textContent=`${title} · 候选训练`;$('steps').textContent=step===null?'待核查':`${step} / 250`;
  $('step-fill').style.width=step===null?'0%':`${100*step/250}%`;
  $('step-progress').setAttribute('aria-valuemax','250');
  if(step===null)$('step-progress').removeAttribute('aria-valuenow');else $('step-progress').setAttribute('aria-valuenow',String(step));
  $('step-note').textContent='每小时同步；训练完成后仍需闭环评测。';
 }
}
function render(){
 const fresh=data.freshness||{},bad=fresh.state!=='fresh';$('freshness').classList.toggle('warn',bad);
 const msg={fresh:'已连接 H200',stale:'同步已过期，显示最后记录',disconnected:'H200 连接中断，显示最后记录',unavailable:'尚未取得 FastWAM 数据'}[fresh.state]||'同步状态未知';
 $('fresh-text').replaceChildren(make('i',undefined,'dot'),document.createTextNode(msg+(fresh.observed_unix?` · ${new Date(fresh.observed_unix*1000).toLocaleString('zh-CN',{hour12:false})} · ${num(fresh.age_seconds)} 秒前`:'')));
 $('content').hidden=!data.available;if(!data.available)return;
 renderRealtime();renderRepeated();renderExpanded();renderTeacherDevelopment();renderCloseout();renderGuardedCandidate();
 const current=data.runs.find(r=>r.id===data.current_run_id),active=fastwamJob(),latest=data.latest_candidate,retained=data.retained;
 const paused=data.execution_control?.status==='paused_by_user'&&data.execution_control.gpu_execution_allowed===false;
 $('gpu-pause-note').hidden=!paused;
 $('gpu-pause-note').textContent=paused?(active||current?.live?'已要求暂停 GPU 作业 · 进程状态待核查':`GPU 训练已暂停${data.closeout?.state==='complete'?' · 阶段收尾已完成':''} · 结果与录屏可继续查看`):'';
 const latestScore=latest?.state==='verified'?latest:null;
 $('step-label').textContent=current?.live?(bad?'最近记录的训练步数':'本轮训练步数'):'最近一轮训练步数';
 $('steps').textContent=current?`${num(current.step)} / ${num(current.max_steps)}`:'—';
 const ev=data.current_evaluation;
 if(ev){$('step-label').textContent=ev.mode==='realtime'?'本轮实时对照进度':'本轮开发评测进度';$('steps').textContent=`${num(ev.completed)} / ${num(ev.total)}`;}
 const pct=ev?.total?Math.min(100,100*ev.completed/ev.total):current?.max_steps?Math.min(100,100*current.step/current.max_steps):0;
 $('step-fill').style.width=pct+'%';$('step-progress').setAttribute('aria-valuenow',String(ev?.completed??current?.step??0));$('step-progress').setAttribute('aria-valuemin','0');$('step-progress').setAttribute('aria-valuemax',String(ev?.total??current?.max_steps??100));
 $('step-note').textContent=ev?`${label(ev.id)} · 未完成前不计入成绩`:current?`${label(current.id)}${current.max_steps?' · '+num(pct,1)+'%':''}`:'等待训练记录';
 const modelName=m=>m?`${m.label||label(m.run_id)} · ${m.checkpoint_sha256?.slice(0,8)||'权重待保存'}`:'版本待核查';
 score($('best-score'),retained);score($('latest-score'),latestScore);
 $('best-name').textContent=modelName(retained);$('latest-name').textContent=modelName(latest);
 $('best-scope').textContent=retained?.scope_label||'缺少当前模型的已核验评测';
 $('latest-scope').textContent=latestScore?.scope_label||'最新训练版本的评测尚未核验';
 $('best-note').textContent=retained?'用于后续开发 · 尚非独立验收':'不会以旧模型分数代替';
 $('latest-note').textContent=latestScore?(latest.decision.status==='not_adopted'?'未采用 · 原因见下方':latest.decision.status==='retained_for_development'?'当前保留版本':'已评测 · 待决定'):latest?.state==='invalid'?'来源不一致，待核查':'待评测 · 不回填旧候选成绩';
 const guarded=data.guarded_candidate_development;
 $('latest-score').classList.remove('mixed-score');
 if(guarded?.state==='verified'&&guarded.run_id===latest?.run_id&&guarded.checkpoint_sha256===latest?.checkpoint_sha256){
  const clear=guarded.groups.find(g=>g.scene==='room_clear'),obstacle=guarded.groups.find(g=>g.scene==='room_curriculum_one');
  if(clear&&obstacle){
   $('latest-score').textContent=`${clear.candidate_successes}/${clear.total} · ${obstacle.candidate_successes}/${obstacle.total}`;
   $('latest-score').classList.add('mixed-score');
   $('latest-scope').textContent='空房间 / 单障碍 · 离线开发';
   $('latest-note').textContent=latestScore?.decision.status==='not_adopted'?'未采用 · 已完成独立重放，原因见下方':'已完成独立重放 · 与保留模型的对照见下方';
  }
 }
 renderModelFamilies();
  const stages={training:'正在训练',initializing_optimizer:'正在准备优化器',initializing_model:'正在加载模型',loading_full_model:'正在加载完整模型',offline_tensor_validation:'正在验证模型输出',saving_checkpoint:'正在保存训练状态',verifying_restored_training_state:'正在核对断点恢复',resume_verified:'断点恢复已验证',complete:'训练已结束'};
 $('current-title').textContent=bad?'以下为最近一次成功同步的状态':current?.live?`${label(current.id)} · ${stages[current.stage]||current.stage}`:active?active.name.includes('align')?'正在核对参考实现与原生实现':active.name.includes('realtime')?'正在运行同权重实时加速对照':active.name.includes('dev')?'正在运行开发集闭环评测':'H200 实验正在运行':current?.execution?.exit_code?'最近一轮异常退出，请查看训练记录':'当前没有运行中的 FastWAM GPU 作业';
 $('current-note').textContent=current?.live?(current.id.includes('preflight')?'正在验证完整模型前后向、保存与断点恢复；这是运行检查，不是候选能力成绩。':'本轮先完成训练，再核对数值并运行同一组 21 个开发案例。训练完成不等于任务通过。'):active?active.name:'完成记录保留在下方；后续新训练和评测会自动同步。';
 if(active?.activity==='preflight'){
  $('current-title').textContent=bad?'最近记录：模型训练与恢复检查':'正在检查模型训练与恢复';
  $('current-note').textContent='验证真实模型更新、保存和准确恢复；这是训练链路检查，不是候选能力成绩。检查通过后才启动候选训练。';
 }
 if(active?.activity==='offline_evaluation'){
  $('current-title').textContent=bad?'最近记录：81 案例离线评测':'正在运行 81 案例离线评测';
  $('current-note').textContent='比较同一组 21 个空房间与 60 个单障碍案例；完成后还需独立重放。这一阶段不更新模型，也不代表实时或正式验收。';
 }
 if(current?.live&&current.id==='realtime-current-memory-corrective-pilot-v1'){
  $('current-title').textContent=`实时失败纠偏 · ${stages[current.stage]||current.stage}`;
  $('current-note').textContent='学习实际连续运行时的失败纠偏示范，同时保留原训练数据。本轮训练 250 步，随后比较相同的 81 个离线开发案例；实时能力另行评测。';
 }
 if(active?.activity==='unverified'){
  $('current-title').textContent='GPU 作业与训练记录尚未匹配';
  $('current-note').textContent='已发现 GPU 进程，尚未核实对应阶段；暂不将它标为训练或实时评测。';
  $('step-label').textContent='当前作业进度';$('steps').textContent='待核查';
  $('step-fill').style.width='0%';$('step-progress').removeAttribute('aria-valuenow');
  $('step-note').textContent='保留历史成绩，不借用上一轮训练步数表示当前进度。';
 }
 if(current?.live&&current.id==='guarded-navigation-pilot-v1')$('current-note').textContent='本轮上限 250 步；随后分别比较 21 个空房间案例和 60 个单障碍案例，并重放核验。训练完成不等于避障通过。';
 if(current?.live&&current.id==='guarded-rehearsal-pilot-v1')$('current-note').textContent='已恢复训练。本轮保留全部 9,932 个原采样项（含纠偏数据），加入 9,100 个障碍采样项，先训练 250 步，再比较 21 个空房间和 60 个单障碍案例并独立重放核验。后续按结果继续改进；累计用量不设 48 小时上限。';
 if(current?.live&&/^memory-axis-(control|weighted)-pilot-v1(?:-preflight|-restore)?$/.test(current.id)){
  const checking=current.id.endsWith('-preflight')||current.id.endsWith('-restore');
  $('current-note').textContent=checking?'检查实际更新、保存和准确恢复；完成检查后才启动对应的 250 步候选训练。能力成绩保持原记录。':'两组使用相同初始模型、数据和采样顺序，只比较横向动作的训练权重。每组训练 250 步后，评测 21 个空房间与 60 个单障碍案例并独立重放核验。';
 }
 if(current?.live&&current.id==='memory-continuation-pilot-v1'){
  $('current-note').textContent='从第 250 步的完整训练状态继续到第 1,000 步，新增 750 步；保留数据采样进度，继续学习此前未采样的部分。模型、数据、损失与学习率不变，之后比较同一组 21 个空房间和 60 个单障碍案例并独立重放。新候选尚未替换保留模型。';
 }
  if(current?.live&&/^expanded-corrective-pilot-v[12]-(250|1000|epoch)$/.test(current.id)){
  $('current-title').textContent=`扩展纠偏数据训练 · ${stages[current.stage]||current.stage}`;
  $('current-note').textContent=`本轮按已核验配置训练 ${num(current.max_steps)} 步；随后比较同一组 21 个空房间和 60 个单障碍案例，并独立重放核验。新候选尚未替换保留模型。`;
 }
 const queue=data.queue||{},queueCount=Number.isInteger(queue.completed_jobs)&&Number.isInteger(queue.total_jobs)&&queue.total_jobs>0;
 if(queueCount&&!ev&&!current?.live&&!active){
  $('step-label').textContent='实时对照队列 · 已结束作业';$('steps').textContent=`${queue.completed_jobs} / ${queue.total_jobs}`;
  $('step-fill').style.width=`${100*queue.completed_jobs/queue.total_jobs}%`;
  $('step-progress').setAttribute('aria-valuenow',String(queue.completed_jobs));$('step-progress').setAttribute('aria-valuemax',String(queue.total_jobs));
  $('step-note').textContent='按作业统计；结束数量不是任务通过数量。';
 }
 if(!bad&&!active&&!current?.live&&queueCount){
  if(queue.status==='complete'){
   $('current-title').textContent='本轮实时对照已结束';$('current-note').textContent='完整轨迹与资源审计另行核对，作业结束不代表能力达标。';
  }else if(queue.process_state==='live'&&queue.stage==='waiting_for_gpu'){
   $('current-title').textContent='核查时正在等待 GPU 空闲';$('current-note').textContent='评测队列仍在运行，资源空闲后按原顺序继续；当前没有新增训练。';
  }else if(queue.process_state!=='live'){
   $('current-title').textContent='评测队列进程状态待核查';$('current-note').textContent='保留最后的完成记录；未将缺失进程或观测错误解释为成功。';
  }
 }
  const correction=data.correction_data_progress;
  const correctionHasPriority=!current?.live&&!ev&&correction&&(!active||active.id===correction.container_id);
  if(correctionHasPriority&&correction.state==='snapshot'&&correction.live){
  const phases={collection:'采集学生纠偏轨迹',replay:'独立重放采集轨迹',anchors:'定位可纠偏状态',teacher:'教师纠偏与示范筛选',audit:'训练数据合并与审计',preparation:'检查训练数据与配置',repair_check:'核验采集流程修复'};
  $('current-title').textContent=`${bad?'最近记录：':correction.waiting?'等待：':'核查时：'}${phases[correction.phase]||'处理纠偏数据'}`;
  $('current-note').textContent='本阶段处理训练数据，没有更新模型权重。完成数包含失败轨迹，不是任务通过率；页面每小时同步。';
  const hasCount=Number.isInteger(correction.completed)&&Number.isInteger(correction.total)&&correction.total>0;
   $('step-label').textContent=correction.phase==='collection'?'本轮纠偏采集进度':correction.phase==='teacher'?'教师已处理的训练案例':'本轮数据处理阶段';
  $('steps').textContent=hasCount?`${correction.completed} / ${correction.total}`:phases[correction.phase]||'进行中';
  $('step-fill').style.width=hasCount?`${100*correction.completed/correction.total}%`:'0%';
  if(hasCount){$('step-progress').setAttribute('aria-valuenow',String(correction.completed));$('step-progress').setAttribute('aria-valuemax',String(correction.total));}
  else{$('step-progress').removeAttribute('aria-valuenow');$('step-progress').removeAttribute('aria-valuemax');}
   $('step-note').textContent=hasCount&&correction.phase==='teacher'
    ?`教师续跑已完成 ${correction.branches_completed} / ${correction.branches_total} 条，其中 ${correction.teacher_successful_branches} 条满足任务成功条件；整批示范还需合并与审计。`
    :hasCount?(Number.isInteger(correction.layout_count)&&Number.isInteger(correction.task_count)
      ?`${correction.layout_count} 个训练布局 × ${correction.task_count} 种任务；采集完成后还需重放、教师纠偏和数据审计。`
      :'采集完成后还需重放、教师纠偏和数据审计。'):'正在准备下一轮训练所需的数据，尚无新增模型成绩。';
 }
 if(!current?.live&&!ev&&!active&&['invalid','unavailable','failed'].includes(correction?.state)){
  $('current-title').textContent=correction.state==='failed'?'纠偏数据作业异常':'纠偏数据进度待核查';
  $('current-note').textContent='保留历史结果；未将进程缺失或证据不一致解释为作业完成。';
  $('step-label').textContent='本轮数据处理阶段';$('steps').textContent='待核查';
  $('step-fill').style.width='0%';$('step-progress').removeAttribute('aria-valuenow');
  $('step-note').textContent='不使用旧实时对照队列的完成数代替当前数据进度。';
 }
 if(paused){
  $('current-title').textContent=active||current?.live?'已要求暂停 · 核查时仍有作业':'GPU 训练已暂停 · 等待你明确恢复';
  $('current-note').textContent=active||current?.live?'暂停请求和进程状态不一致，需核查；不会将暂停请求当作作业已停止。':`${bad?'按最近同步记录，':''}没有运行中的 GPU 作业。训练、GPU 评测与完整模型检查均不自动启动；历史结果与录屏继续可看。`;
 }
 const retest=data.authorized_retest;
 const diagnostic=retest?.diagnostic,diagnosticLive=diagnostic?.live&&active?.id===diagnostic.container_id;
  if(paused&&retest&&!correctionHasPriority&&!current?.live&&((retest.live&&active?.id===retest.container_id)||diagnosticLive||(!retest.live&&!active))){
  $('gpu-pause-note').textContent='GPU 训练已暂停 · 已单独授权 1000ms 未加速复测';
  const retestState=diagnosticLive?'正在排查运行异常':diagnostic?.waiting?'等待空闲 GPU 排查异常':retest.live?(retest.stage==='loading'?'加载模型中':'正在运行'):retest.status==='failed'?'运行异常退出':retest.status==='complete'?'已结束，待结果核查':'进程已结束，待核查';
  $('current-title').textContent=`${bad?'最近记录：':'核查时：'}1000ms 未加速寻球复测 · ${retestState}`;
  $('current-note').textContent=diagnosticLive?'正在运行必要的环境诊断，诊断不计入寻球成绩；训练继续暂停。':diagnostic?.waiting?'等待空闲 GPU 运行必要的环境诊断；训练继续暂停。':retest.status==='failed'?'首轮在完成第一集前退出，尚无新增寻球成绩。诊断复现了相机与 CUDA 交替运行时的卡顿；修复并通过实时检查后才能继续复测。本次 GPU 已释放，训练继续暂停。':'同一保留权重、60 个无障碍开发案例；仅延长动作有效期。结果须在完整复测后核查，不属于训练或正式验收。';
  if(diagnosticLive&&diagnostic.deployment==='local_render_remote_model'){
   $('current-title').textContent=`${bad?'最近记录：':'核查时：'}1000ms 未加速复测 · 跨主机运行检查`;
   $('current-note').textContent='本地 GTX 1660 Ti 渲染相机，H200 推理同一未加速模型；正在验证运行链路。该配置单独记录，下方 0/60 为原单卡复测，不是新配置成功率。训练继续暂停。';
  }
  $('step-label').textContent='1000ms 未加速复测进度';$('steps').textContent=`${num(retest.completed)} / ${num(retest.total)}`;
  $('step-fill').style.width=`${100*retest.completed/retest.total}%`;
  $('step-progress').setAttribute('aria-valuenow',String(retest.completed));$('step-progress').setAttribute('aria-valuemax',String(retest.total));
  $('step-note').textContent='已结束案例数，不是成功数；页面继续每小时同步。';
  if(diagnostic?.deployment==='local_render_remote_model'&&diagnostic.evaluation_progress){
   const progress=diagnostic.evaluation_progress;
   $('current-title').textContent=`${bad?'最近记录：':'核查时：'}1000ms 未加速复测 · 本地渲染＋H200 推理`;
   $('current-note').textContent='复测使用同一保留模型与同一批开发案例，渲染改由本地 GTX 1660 Ti 承担。新配置单独记录；原单卡复测异常退出。训练继续暂停，完整结果仍需审计。';
   $('step-label').textContent='跨主机复测进度';$('steps').textContent=`${num(progress.completed)} / ${num(progress.total)}`;
   $('step-fill').style.width=`${100*progress.completed/progress.total}%`;
   $('step-progress').setAttribute('aria-valuenow',String(progress.completed));$('step-progress').setAttribute('aria-valuemax',String(progress.total));
   $('step-note').textContent=`已结束案例数，不是成功数。记录时间：${new Date(progress.observed_unix*1000).toLocaleString('zh-CN')}；保留每小时同步。`;
   if(!diagnosticLive){
    $('current-title').textContent=`${bad?'最近记录：':'核查时：'}跨主机复测进程已结束 · 等待结果审计`;
    $('current-note').textContent='H200 推理服务已结束，案例数来自最近记录。完整结果尚未通过展示核查；训练保持暂停。';
   }
  }
  if(diagnostic?.evaluation_result){
   const result=diagnostic.evaluation_result;
   $('gpu-pause-note').textContent='GPU 训练已暂停 · 1000ms 复测已结束';
   $('current-title').textContent='1000ms 未加速复测 · 已完成时序审计';
   $('current-note').textContent=`寻球完成 ${num(result.task_successes)}/${num(result.total)}；无回退且满足实时要求 ${num(result.realtime_successes)}/${num(result.total)}。模型动作占控制时间 ${(100*result.model_ticks/result.control_ticks).toFixed(1)}%，其余切回官方站立。本地渲染＋H200 推理，与旧单机结果分开记录；属于开发集复测，训练保持暂停。${diagnosticLive?'核查时推理服务仍在释放过程中。':'核查时推理服务已结束。'}`;
   $('step-label').textContent='跨主机复测已完成';$('steps').textContent=`${num(result.total)} / ${num(result.total)}`;
   $('step-fill').style.width='100%';
   $('step-progress').setAttribute('aria-valuenow',String(result.total));$('step-progress').setAttribute('aria-valuemax',String(result.total));
   $('step-note').textContent='完整开发案例与执行时序已核查；不是独立最终验收，也不构成只改变 TTL 的单变量对比。';
  }
 }
 const noAge=data.no_age_retest;
  if(noAge&&noAge.state!=='invalid'&&!correctionHasPriority&&!current?.live&&(!active||active.name==='microduck-fastwam-no-age-split-service-v1')){
  $('gpu-pause-note').hidden=false;
  $('gpu-pause-note').textContent='单障碍训练已获准恢复 · 当前先完成未加速无时限复测';
  $('current-title').textContent='未加速 · 关闭观测年龄限制 · '+(noAge.live?'复测进行中':'结果待核查');
  $('current-note').textContent='同一模型、60 个空房间开发案例。超过 1000ms 的动作也执行，仍保留有限动作块、取消和会话隔离。物理持续运行；单独比较任务完成与实际动作延迟。';
  $('step-label').textContent='取消超时限制复测';$('steps').textContent=`${num(noAge.completed)} / 60`;
  $('step-fill').style.width=`${100*noAge.completed/60}%`;
  $('step-progress').setAttribute('aria-valuenow',String(noAge.completed));$('step-progress').setAttribute('aria-valuemax','60');
  $('step-note').textContent=`已结束案例数，不是成功数。记录时间：${new Date(noAge.observed_unix*1000).toLocaleString('zh-CN')}；完整结果另行审计。`;
  if(noAge.result){
   const result=noAge.result;
   $('current-title').textContent='未加速 · 取消观测年龄限制 · 完整复测已核查';
   $('current-note').textContent=`寻球完成 ${result.task_successes}/60；无回退且满足实时要求 ${result.realtime_successes}/60。执行了 ${num(result.executed_older_than_1000ms)} 个观测年龄超过 1000ms 的动作。成功和失败录屏见下方“未加速 · 取消时限”。`;
   $('step-note').textContent='同一模型与开发案例，独立记录本次实际延迟；不计为独立最终验收。';
  }

 }
 renderCheckpointReview();renderMemoryPreflight();renderMemoryCandidate();
 const select=$('run-select'),want=selected||data.current_run_id;select.replaceChildren();for(const r of [...data.runs].reverse()){const o=make('option',`${label(r.id)} · ${num(r.step)} 步`);o.value=r.id;select.append(o);}select.value=data.runs.some(r=>r.id===want)?want:data.current_run_id;chart(data.runs.find(r=>r.id===select.value));
 const comparisons=$('comparisons');comparisons.replaceChildren();if(retained)comparisons.append(comparison(retained,'保留模型 · 历史离线开发',true));
 if(latest?.checkpoint_sha256!==retained?.checkpoint_sha256)comparisons.append(comparison(latestScore,'最新候选 · 本轮离线开发'));
 $('comparison-note').textContent=data.model_selection?.comparison_note||'当前模型登记尚未同步，历史评测不能替代当前版本。';
 $('selection-reason').textContent=latestScore?.decision.reason||'等待最新候选完成评测与采用决定。';
 const t=data.teacher,teacherTasks=Object.values(t.tasks||{}),passed=teacherTasks.reduce((s,r)=>s+(r.eligible_successes||0),0),total=teacherTasks.reduce((s,r)=>s+(r.episodes||0),0);
 const milestones=[['空房间教师',t.qualified?'已通过':'待核查',total?`${passed}/${total} 条示范检查通过。教师使用仿真信息。`:'暂无资格报告',t.qualified],['视觉任务学习',current?.live?'训练中':'持续改进','学生仅使用图像、语言和身体状态。',false],['静态障碍任务',t.obstacles_qualified?'教师已通过':'教师未通过','先建立可靠示范，再训练避障。',t.obstacles_qualified],['独立任务验收','尚未验收',`至少 ${data.protocol?.acceptance?.minimum_independent_layouts||200} 个独立布局，目标成功率 ≥80%。`,false],['实时加速收益',data.repeated_realtime?.state==='complete'?'开发对照已验证':'任务收益待验证','同一权重、物理持续运行；独立布局泛化另行验收。',data.repeated_realtime?.state==='complete']];
 if(paused)milestones[1][1]='已暂停';
 if(data.closeout?.state==='complete'&&data.closeout.teacher_qualification?.passed===294){milestones[2][1]='单障碍教师已通过';milestones[2][2]='288 条合格示范已准备；学生的避障能力仍未验收。';milestones[2][3]=true;}
 $('milestones').replaceChildren();milestones.forEach(([name,state,desc,done],i)=>{const c=make('article',undefined,'milestone');c.append(make('div',String(i+1).padStart(2,'0'),'num'),make('h3',name),make('span',state,`badge ${done?'':'pending'}`),make('p',desc));$('milestones').append(c);});
 $('history').replaceChildren();for(const e of [...data.evaluations].reverse()){const tr=make('tr');tr.append(make('td',label(e.id)),make('td',`${e.passed}/${e.total}`));for(const color of ['red','blue','yellow']){const c=e.colors['find_'+color+'_ball'];tr.append(make('td',c?`${c.passed}/${c.total}`:'—'));}$('history').append(tr);}
 $('sources').textContent=JSON.stringify({采集时间:new Date(data.observed_unix*1000).toISOString(),模型版本登记:data.model_selection?.source,开发协议:data.protocol_source,当前训练:current?.source,单独授权复测:data.authorized_retest,最近评测:latestScore,保留模型:retained,指标范围:'FastWAM 独立统计；GPU 时间为容器分配时间'},null,2);
}
async function refresh(){if(busy)return;busy=true;$('refresh').disabled=true;try{const response=await fetch('/api/training/fastwam',{cache:'no-store'});if(!response.ok)throw Error('HTTP '+response.status);data=await response.json();render();}catch(e){$('freshness').classList.add('warn');$('fresh-text').textContent='页面数据读取失败，保留已显示内容；将自动重试。';}finally{busy=false;$('refresh').disabled=false;}}
$('expanded-filter').addEventListener('change',renderExpanded);$('realtime-trial').addEventListener('change',()=>{realtimeSelected=$('realtime-trial').value;renderRealtime();});$('refresh').addEventListener('click',refresh);$('run-select').addEventListener('change',()=>{selected=$('run-select').value;chart(data.runs.find(r=>r.id===selected));});refresh();setInterval(refresh,3600000);
})();
