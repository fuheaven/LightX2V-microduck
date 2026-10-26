'use strict';
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let catalog, parsed = null, jobs = [], active = null, page = 0, packet = null, polling = false;
const labels = {queued:'等待启动',running:'正在运行',complete:'运行完成',cancelled:'已停止',failed:'运行出错',interrupted:'进程中断'};
const qualification = {passed_small_test:'通过小规模测试',not_passed:'未通过测试',incomplete_test:'测试未完成',not_evaluated:'尚未测试',no_complete_generation:'尚无完整训练代'};
const modes = {train:'教学',resume:'续训',evaluate:'独立测试',live:'实时运行'};
async function api(path, payload) {
  const response = await fetch('/api/school/' + path, payload === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const value = await response.json(); if (!response.ok) throw Error(value.error || response.status); return value;
}
function error(err) { $('error').textContent = err?.message || ''; }
function settings() { return {seed:Number($('seed').value),budget_s:Number($('budget').value)}; }
function isRunning(job) { return ['queued','running'].includes(job.status); }
async function preview() {
  error(null); parsed = null; $('teach').disabled = true;
  const instruction = $('instruction').value;
  const value = await api('preview', {instruction});
  if ($('instruction').value !== instruction) return;
  parsed = value;
  const c = value.course;
  $('preview').hidden = false;
  $('preview').textContent = `${value.label} · ${c.family === 'rgb' ? '真实 RGB + 本体状态' : '仿真真值课程'}\n${c.family === 'rgb' ? '寻找' + ({red:'红',blue:'蓝',yellow:'黄'}[c.color]) + '球，在距球表面 0.25–0.40 米内停稳 2 秒。平地课程，包含初始视野外目标。' : '底层官方 ONNX 不变；学习速度与转向命令。终止条件包含跌倒、碰撞及超时。'}\n${c.family === 'goto' || c.family === 'explore' || c.family === 'ball' ? '距离参数：' + c.distance_m + ' 米；' : ''}${['circle','spin'].includes(c.family) ? '方向：' + (c.direction === 1 ? '左' : '右') + '；圈数：' + c.turns + '；' : ''}${c.family === 'circle' ? '半径：' + c.radius_m + ' 米；' : ''}每集最多 ${c.duration_s} 秒。`;
  $('teach').disabled = jobs.some(isRunning);
}
async function launch(mode, skill_id) {
  error(null);
  const payload = {mode,...settings(),skill_id,instruction:$('instruction').value};
  const job = await api('run', payload);
  active = job.id; localStorage.setItem('microduck-school-job', active);
  packet = null; $('live-frame').removeAttribute('src'); $('session').hidden = false; $('report').replaceChildren();
  $('session').scrollIntoView({behavior:'smooth',block:'start'});
  await refresh(); await poll();
}
function chart(history) {
  $('chart').hidden = !history?.length;
  $('chart-note').hidden = !history?.length;
  if (!history?.length) { $('chart').replaceChildren(); return; }
  const values = history.flatMap(x => [x.train_best,x.validation_reward]);
  const low = Math.min(...values) - .1, high = Math.max(...values) + .1;
  const point = (x, k, value) => `${45 + k * 725 / Math.max(1,history.length-1)},${150 - (value-low)/(high-low)*125}`;
  $('chart').innerHTML = `<text x="4" y="20" fill="#bdccbf" font-size="12">${high.toFixed(2)}</text><text x="4" y="155" fill="#bdccbf" font-size="12">${low.toFixed(2)}</text><path d="M40 15V155H785" stroke="#688474" fill="none"/>` +
    ['train_best','validation_reward'].map((key,i) => `<polyline points="${history.map((x,k)=>point(x,k,x[key])).join(' ')}" fill="none" stroke="${i?'#7abded':'#9cdda6'}" stroke-width="2"/>` + history.map((x,k)=>`<circle cx="${point(x,k,x[key]).split(',')[0]}" cy="${point(x,k,x[key]).split(',')[1]}" r="3" fill="${i?'#7abded':'#9cdda6'}"/>`).join('')).join('') +
    `<text x="45" y="176" fill="#bdccbf" font-size="12">第 ${history[0].generation} 代</text><text x="700" y="176" fill="#bdccbf" font-size="12">第 ${history.at(-1).generation} 代</text>`;
}
function report(value) {
  if (!value) return $('report').replaceChildren();
  const summaries = value.summaries || {selected:value.summary};
  $('report').innerHTML = `<h3>测试结果${value.complete ? '' : ' · 未完成全部场景'}</h3><div class="table-wrap"><table><thead><tr><th>控制器</th><th>实际集数</th><th>成功率</th><th>平均奖励</th><th>失败原因</th></tr></thead><tbody>${Object.entries(summaries).map(([key,s])=>`<tr><td>${({learned:'训练选中控制器',rule:'固定规则基线',selected:'选定技能'}[key])}</td><td>${s.episodes}</td><td>${s.success_rate === null ? '—' : (s.success_rate*100).toFixed(0)+'%'}</td><td>${s.mean_reward?.toFixed(3) ?? '—'}</td><td>${esc(JSON.stringify(s.failures))}</td></tr>`).join('')}</tbody></table></div><p class="muted">${value.split === 'demo' ? '本次为实时演示，不参与独立测试资格。' : '测试场景与训练场景分离；测试结果不用于 CEM 选优。'}</p>`;
}
function showFrame() {
  const frame = packet?.frames?.[$('view').value];
  if (frame) $('live-frame').src = frame;
}
async function poll() {
  if (!active || polling) return; polling = true;
  const identifier = active;
  try {
    const job = await api('job?id=' + encodeURIComponent(identifier));
    if (identifier !== active) return;
    $('session').hidden = false;
    const p = job.progress || {};
    const stage = p.phase?.includes('validation') ? '验证' : p.phase?.startsWith('generation') ? '候选评估' : p.phase === 'testing' ? '独立测试' : '';
    $('status').textContent = `${modes[job.mode]} · ${job.instruction} · ${labels[job.status]}${p.generation !== undefined ? ' · 已完成 ' + p.generation + ' 代' : ''}${p.elapsed_s !== undefined ? ' · '+p.elapsed_s.toFixed(0)+' / '+p.budget_s+' 秒' : ''}${p.completed !== undefined ? ' · '+stage+' '+p.completed+'/'+p.total : ''}${job.error ? '\n'+job.error : ''}`;
    $('cancel').disabled = !isRunning(job); chart(p.history); report(job.evaluation);
    $('evidence').textContent = JSON.stringify(job,null,2);
    $('live-area').hidden = job.mode !== 'live';
    if (job.mode === 'live') {
      const latest = await api('live?id=' + encodeURIComponent(identifier));
      if (identifier !== active) return;
      packet = latest; showFrame();
      const metrics = packet.result?.metrics;
      const distance = metrics && ['goto','rgb'].includes(job.course.family) ? ' · ' + (job.course.family === 'rgb' ? '距球表面 ' + metrics.surface_distance_m.toFixed(2) : '距标记 ' + metrics.target_distance_m.toFixed(2)) + ' 米' : '';
      $('live-status').textContent = `${isRunning(job) ? '实时画面' : '运行已结束 · 最后一帧'} · 仿真 ${Number(packet.sim_time_s || 0).toFixed(1)} 秒 · 速度 ${Number(packet.real_time_factor || 0).toFixed(2)}×${distance}`;
    }
    const previous = jobs.find(x=>x.id===job.id);
    if (!previous || previous.status !== job.status) await refresh();
  } catch (err) { error(err); }
  finally { polling = false; }
}
function renderJobs() {
  const filtered = jobs.filter(j=>$('filter').value === 'all' || j.course.family === $('filter').value);
  page = Math.min(page,Math.max(0,Math.ceil(filtered.length/6)-1));
  $('skills').innerHTML = filtered.slice(page*6,page*6+6).map(j=>`<article><h3>${esc(j.instruction)} <span class="badge">${j.course.family === 'rgb' ? 'RGB 课程' : '仿真真值'}</span></h3><small>${esc(new Date(j.created_at*1000).toLocaleString())} · ${modes[j.mode]} · ${labels[j.status]} · seed ${j.seed}${j.skill ? ' · '+(qualification[j.skill.qualification] || j.skill.qualification) : ''}${j.skill?.selected_source === 'rule_initialization' ? ' · 当前选中规则初始参数' : ''}</small><div class="row"><button data-id="${j.id}" data-action="detail">查看结果</button>${j.skill && !isRunning(j) ? ['live','evaluate','resume'].map(m=>`<button data-id="${j.id}" data-action="${m}" ${jobs.some(isRunning)?'disabled':''}>${modes[m]}</button>`).join('') : ''}</div></article>`).join('') || '<p>还没有课程记录。从上方选择一个目标开始。</p>';
  $('page').textContent = `${filtered.length} 条记录 · 第 ${page+1} / ${Math.max(1,Math.ceil(filtered.length/6))} 页`;
  $('previous').disabled = page === 0; $('next').disabled = (page+1)*6 >= filtered.length;
  $('teach').disabled = !parsed || jobs.some(isRunning);
}
async function refresh() { jobs = await api('jobs'); renderJobs(); }
$('lesson-form').onsubmit = e=>{e.preventDefault(); preview().catch(error);};
$('instruction').oninput = ()=>{parsed=null;$('teach').disabled=true;$('preview').hidden=true;};
$('teach').onclick = ()=>launch('train').catch(error);
$('cancel').onclick = ()=>api('cancel',{id:active}).then(poll).catch(error);
$('refresh').onclick = ()=>refresh().catch(error);
$('view').onchange = showFrame;
$('filter').onchange = ()=>{page=0;renderJobs();};
$('previous').onclick = ()=>{page--;renderJobs();}; $('next').onclick = ()=>{page++;renderJobs();};
$('skills').onclick = e=>{const b=e.target.closest('button[data-id]');if(!b)return;if(b.dataset.action==='detail'){active=b.dataset.id;packet=null;localStorage.setItem('microduck-school-job',active);poll();$('session').scrollIntoView({behavior:'smooth'});}else launch(b.dataset.action,b.dataset.id).catch(error);};
async function init() {
  catalog = await api('catalog');
  for (const course of catalog.courses) {
    const button=document.createElement('button');button.textContent=course.label;button.onclick=()=>{$('instruction').value=course.example;preview().catch(error);};$('examples').append(button);
    const option=document.createElement('option');option.value=course.id;option.textContent=course.label;$('filter').append(option);
  }
  await refresh(); await preview();
  active = jobs.find(isRunning)?.id || localStorage.getItem('microduck-school-job');
  if (active && jobs.some(j=>j.id===active)) await poll(); else active=null;
  setInterval(poll,700);
}
init().catch(error);
