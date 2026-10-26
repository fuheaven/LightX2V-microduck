'use strict';
const $ = id => document.getElementById(id);
const state = {catalog: null, group: null, clip: null, health: null, tab: 'gallery', color: 'red', job: null, busy: false, sequence: null, epoch: 0};
const colors = {red: '红球', blue: '蓝球', yellow: '黄球'};
const active = new Set(['starting','verifying','loading','warming','running','cancelling','saving']);
const phases = {starting:'准备运行',verifying:'校验模型工件',loading:'加载 FastWAM',warming:'场景与推理预热',running:'实时运行中',cancelling:'正在停止',saving:'保存运行结果',complete:'运行结束',cancelled:'已停止',failed:'运行出错',interrupted:'运行中断',idle:'等待开始'};
const groupCopy = {
  'clear-room': ['无障碍寻球', '开发集表现稳定', '实时 · 20 个布局', 'CLEAR ROOM', '观察颜色、转向寻找目标，接近后保持停止。'],
  'single-obstacle': ['单障碍寻球', '能力仍在实验', '离线 · 20 个布局', 'ONE OBSTACLE', '同一模型在单障碍房间中的迁移测试。离线运行允许等待推理，不代表实时避障已经达标。'],
  'random-obstacles': ['随机障碍', '能力仍在实验', '实时 · 7 个布局', 'RANDOM OBSTACLES', '复杂布局仍是明显短板。保留成功与碰撞案例，展示目前的能力边界。']
};
function text(id, value) { $(id).textContent = value; }
function element(tag, cls, value) { const node = document.createElement(tag); if (cls) node.className=cls; if (value!==undefined) node.textContent=value; return node; }
async function api(path, payload) {
  const response = await fetch(path, payload===undefined ? {cache:'no-store'} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `请求失败 (${response.status})`);
  return result;
}
function error(message) { $('page-error').hidden=!message; text('page-error', message || ''); }
function showTab(tab) {
  state.tab=tab==='live'?'live':'gallery';
  $('gallery-page').hidden=state.tab!=='gallery'; $('live-page').hidden=state.tab!=='live';
  document.querySelectorAll('[data-tab]').forEach(b=>{b.classList.toggle('active',b.dataset.tab===state.tab);b.setAttribute('aria-pressed',String(b.dataset.tab===state.tab));});
  if (state.tab==='live') $('recording').pause();
}
function selectGroup(group) {
  state.group=group;
  document.querySelectorAll('[data-group]').forEach(b=>{b.classList.toggle('active',b.dataset.group===group.id);b.setAttribute('aria-pressed',String(b.dataset.group===group.id));});
  $('clips').replaceChildren();
  group.clips.forEach(clip=>{
    const button=element('button','clip-card'); button.dataset.clip=clip.id; button.setAttribute('aria-label',clip.title);
    const thumbnail=element('div','clip-thumbnail'), image=element('img'); image.src='/media/'+encodeURIComponent(clip.poster); image.alt=''; image.loading='lazy';
    thumbnail.append(image,element('span','clip-play','▶'));
    const label=element('div','clip-label',clip.title), info=element('div','clip-state');
    info.append(element('span',clip.passed?'passed':'failed',clip.passed?'本集完成':'未完成'),document.createTextNode(` · seed ${clip.seed}`));
    button.append(thumbnail,label,info); button.addEventListener('click',()=>selectClip(clip)); $('clips').append(button);
  });
  text('clip-count',`${group.clips.length} 段精选记录 · 完整成绩见报告`);
  $('assessment-link').href='/media/'+encodeURIComponent(group.assessment);
  selectClip(group.clips[0]);
}
function selectClip(clip) {
  state.clip=clip;
  const group=state.group, copy=groupCopy[group.id];
  document.querySelectorAll('[data-clip]').forEach(b=>{b.classList.toggle('active',b.dataset.clip===clip.id);b.setAttribute('aria-pressed',String(b.dataset.clip===clip.id));});
  const video=$('recording'); video.pause(); video.poster='/media/'+encodeURIComponent(clip.poster); video.src='/media/'+encodeURIComponent(clip.video); video.load();
  text('recording-mode',group.mode==='realtime'?'历史实时运行记录':'历史离线运行记录');
  text('clip-category',copy[3]); text('clip-title',clip.title);
  text('clip-result',clip.passed?'✓ 本集完成目标':'本集未完成目标'); $('clip-result').classList.toggle('failed',!clip.passed);
  const explanation=clip.variant==='plain'?'同一模型关闭三项推理优化。在此部署的 500ms 时效下，过期动作会触发站立回退；这不等同于模型没有学到动作。':copy[4];
  text('clip-explanation',explanation); text('clip-target',colors[clip.task.split('_')[1]] || clip.task);
  text('clip-duration',`${clip.duration_s.toFixed(1)} 秒`); text('clip-fallback',`${clip.fallback_ticks} tick`);
  text('clip-evidence',`seed ${clip.seed} · ${clip.variant==='plain'?'未加速':'LightX2V 加速'} · 模型 SHA256 ${state.catalog.checkpoint_sha256}`);
  $('clip-report').href='/media/'+encodeURIComponent(clip.report);
}
function renderCatalog(catalog) {
  state.catalog=catalog; $('capability-tabs').replaceChildren();
  catalog.groups.forEach(group=>{
    const copy=groupCopy[group.id], button=element('button','capability-card'); button.dataset.group=group.id;
    const name=element('div','cap-name',copy[0]); name.append(element('span','',copy[1]));
    const count=element('div','cap-count'); count.append(element('strong','',group.passed),element('span','',`/ ${group.total}`),element('em','','开发案例通过'));
    const track=element('div','progress-track'), fill=element('i'); fill.style.width=`${100*group.passed/group.total}%`; track.append(fill);
    button.append(name,count,track,element('div','cap-sub',copy[2]));
    button.addEventListener('click',()=>selectGroup(group)); $('capability-tabs').append(button);
  });
  selectGroup(catalog.groups[0]);
}
function chooseColor(color) {
  state.color=color;
  document.querySelectorAll('[data-color]').forEach(b=>{b.classList.toggle('selected',b.dataset.color===color);b.setAttribute('aria-pressed',String(b.dataset.color===color));});
}
function refreshControls() {
  const health=state.health || {}, policy=$('live-policy').value, running=state.busy || !!(state.job && active.has(state.job.status));
  $('navigation-options').hidden=policy==='official'; $('variant-field').hidden=policy==='official';
  text('policy-note',policy==='official'?'官方 ONNX 的站立、前进演示，不执行视觉寻球，也不使用 FastWAM。':'FastWAM 观察相机并选择移动命令，官方策略维持行走和平衡。');
  $('start-live').disabled=running || !health[policy+'_available']; $('stop-live').disabled=!state.job || !active.has(state.job.status) || state.job.status==='cancelling';
  document.querySelectorAll('.controls-panel select,.controls-panel input,[data-color]').forEach(e=>e.disabled=running);
  $('runner-notice').hidden=!!health[policy+'_available'];
  text('runner-notice',policy==='fastwam'?(health.fastwam_message || health.message || '正在检查运行环境…'):(health.message || '正在检查运行环境…'));
}
async function health() {
  try {
    state.health=await api('/api/health');
    text('runner-status',state.health.fastwam_available?'FastWAM 运行器就绪':state.health.official_available?'官方仿真就绪 · FastWAM 待配置':'录屏就绪 · 实时运行器待配置');
    refreshControls();
  } catch(e) { text('runner-status','无法连接服务'); state.health={}; refreshControls(); }
}
function imageFrame(id,url,seq) {
  const image=$(id); if (!url) {image.hidden=true;return;}
  image.onload=()=>{image.hidden=false;}; image.onerror=()=>{image.hidden=true;};
  image.src=`${url}?frame=${seq}`;
}
function drawJob(job) {
  if(active.has(job.status) && job.id!==state.job?.id && job.config) {
    $('live-policy').value=job.config.policy; $('live-seed').value=job.config.seed;
    $('live-seconds').value=String(job.config.seconds);
    if(job.config.policy==='fastwam') {
      $('live-scene').value=job.config.scene; $('live-variant').value=job.config.variant; chooseColor(job.config.color);
    }
  }
  state.job=job; const live=job.live || {}, running=active.has(job.status), hasFrame=!!job.frames?.observer;
  const badge=$('live-badge'); badge.replaceChildren(element('i','tiny-dot'),document.createTextNode(phases[job.status] || job.status)); badge.classList.toggle('running',job.status==='running');
  const sim=(running?live.sim_s:(job.report?.sim_s ?? live.sim_s)) || 0;
  text('live-clock',`${String(Math.floor(sim/60)).padStart(2,'0')}:${String(Math.floor(sim%60)).padStart(2,'0')}`);
  $('live-phase').hidden=job.status==='idle' || (job.status==='running'&&hasFrame) || (!running&&hasFrame);
  text('live-phase',job.stage_message && running ? job.stage_message : phases[job.status] || '');
  $('live-empty').hidden=hasFrame;
  const seq=`${job.id}:${live.sequence}`;
  if (seq!==state.sequence) {
    state.sequence=seq; imageFrame('live-observer',job.frames?.observer,live.sequence); imageFrame('live-camera',job.frames?.camera,live.sequence);
  }
  text('camera-label',job.config?.policy==='official'?'此模式不使用视觉输入':live.camera_tick!=null?`相机 · tick ${live.camera_tick}`:'头部相机 · 等待画面');
  const factor=running?live.real_time_factor:(job.report?.real_time_factor ?? live.real_time_factor);
  text('live-rtf',factor!=null?`${factor.toFixed(2)}×`:'—'); text('live-latency',live.latency_ms!=null?`${Math.round(live.latency_ms)} ms`:'—');
  const src=live.execution?.execution_source || live.execution?.source;
  text('live-source',src==='official_onnx'?'官方行走/站立':src==='navigation_model' || src==='navigation_policy' || src==='model' || src==='policy'?'FastWAM':src?.includes('fallback')?'官方站立回退':src || '—');
  const fallbacks=running?live.fallback_ticks:(job.report?.fallback_control_ticks ?? live.fallback_ticks);
  text('live-fallback',job.config?.policy==='official'?'不适用':`${fallbacks || 0} tick`);
  text('frame-freshness',live.published_unix ? (running?`画面更新于 ${Math.max(0,Math.round(Date.now()/1000-live.published_unix))} 秒前`:'本次运行已结束 · 最后一帧'):'');
  $('live-result').hidden=running || job.status==='idle';
  const report=job.report;
  let result=job.status==='cancelled'?'运行已停止。停止的会话不计为任务成功。':job.error || '';
  if (job.status==='complete') {
    result=job.config?.policy==='official'?'官方 ONNX 实时行走演示已结束。这是底层运动演示，不是 FastWAM 寻球结果。':
      report?.realtime_success?'本次实时任务完成：达到目标并停稳，且通过运行时钟检查。单次成功不代表正式验收。':
      report?.strict_model_success?'本集达到任务条件，但运行速度未通过实时门槛。请查看报告中的时钟与延迟数据。':
      '本次任务未通过。请结合碰撞、超时与站立接管记录查看原因。失败记录已保存。';
  }
  text('live-result',result); $('live-result').classList.toggle('error',job.status==='failed' || (job.status==='complete'&&job.config?.policy==='fastwam'&&!report?.realtime_success));
  $('live-report').hidden=!report; if(job.report_url) $('live-report').href=job.report_url;
  $('live-log').hidden=!job.id; if(job.log_url) $('live-log').href=job.log_url;
  refreshControls();
}
async function poll() {
  const epoch=state.epoch;
  try {
    const job=await api(state.job?.id?`/api/live/${state.job.id}`:'/api/live');
    if(epoch===state.epoch) drawJob(job);
  } catch(e) {
    if(state.tab==='live') text('frame-freshness','连接中断，正在重试；浏览器断连不会暂停仿真。');
  } finally { setTimeout(poll,state.tab==='live'?350:2000); }
}
document.querySelectorAll('[data-tab]').forEach(b=>b.addEventListener('click',()=>{location.hash=b.dataset.tab;}));
window.addEventListener('hashchange',()=>showTab(location.hash.slice(1)));
document.querySelectorAll('[data-color]').forEach(b=>b.addEventListener('click',()=>chooseColor(b.dataset.color)));
$('live-policy').addEventListener('change',refreshControls);
$('recording').addEventListener('error',()=>error('录屏加载失败。请确认仓库中的示例视频完整，并检查服务日志。'));
$('try-clip').addEventListener('click',()=>{
  if (!active.has(state.job?.status)) {
    $('live-policy').value='fastwam'; $('live-scene').value=state.group.scene; $('live-seed').value=state.clip.seed;
    $('live-variant').value=state.clip.variant==='plain'?'plain':'optimized'; chooseColor(state.clip.task.split('_')[1]); refreshControls();
  }
  location.hash='live'; window.scrollTo({top:0,behavior:'smooth'});
});
$('start-live').addEventListener('click',async()=>{
  state.busy=true; state.epoch++; refreshControls(); error(null);
  try {
    const result=await api('/api/live',{policy:$('live-policy').value,scene:$('live-scene').value,color:state.color,
      seed:Number($('live-seed').value),seconds:Number($('live-seconds').value),variant:$('live-variant').value});
    state.sequence=null; drawJob(result);
  } catch(e) {error(e.message);} finally{state.busy=false;refreshControls();}
});
$('stop-live').addEventListener('click',async()=>{
  if(!state.job?.id) return; state.epoch++; $('stop-live').disabled=true;
  try{drawJob(await api(`/api/live/${state.job.id}/stop`,{}));}catch(e){error(e.message);refreshControls();}
});
(async()=>{
  showTab(location.hash.slice(1) || (location.pathname==='/live'?'live':'gallery'));
  try {renderCatalog(await api('/api/catalog'));}catch(e){error('能力记录加载失败：'+e.message);}
  await health(); setInterval(health,15000); poll();
})();
