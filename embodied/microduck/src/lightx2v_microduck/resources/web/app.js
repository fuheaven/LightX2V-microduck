'use strict';
const el = id => document.getElementById(id);
let catalog, category = 'presets', workflow = 'presets', records = [], details = {}, playing = false;
let historyCategory = 'presets', historyPage = 0, activeJob = null, livePacket = null;
const selected = {left: null, right: null}, loadVersions = {left: 0, right: 0};
const PAGE_SIZE = 8;
let refreshVersion = 0, pollingVersion = 0;
let modelChoiceTouched = false;
const terminal = status => ['complete', 'failed', 'cancelled'].includes(status);
const modeLabel = mode => ({live: '实时运行', offline: '离线评测', robotd_bam: 'robotd 协议评测'}[mode] || mode);

async function api(path, options) {
  const response = await fetch(path, options), value = await response.json();
  if (!response.ok) throw new Error(value.error || `HTTP ${response.status}`);
  return value;
}
const post = (path, value) => api(path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(value)});
function options(select, values, label, preserve = false) {
  const previous = select.value;
  select.replaceChildren(...values.map(value => new Option(label(value), value)));
  if (preserve && values.includes(previous)) select.value = previous;
}
function setCategory(next) {
  category = next;
  const flows = Object.keys(catalog.workflows).filter(id => catalog.workflows[id].category === category);
  options(el('project'), flows, id => catalog.workflows[id].label);
  el('project-field').hidden = flows.length < 2;
  setWorkflow(flows[0]);
}
function setWorkflow(next) {
  workflow = next; el('project').value = next;
  const flow = catalog.workflows[workflow];
  el('workflow-description').textContent = flow.description;
  options(el('tasks'), flow.tasks, id => `${catalog.tasks[id].label} · ${id}`);
  options(el('policy'), flow.policies, id => catalog.policies[id].label);
  options(el('scene'), flow.scenes, id => catalog.scenes[id]);
  el('policy-field').hidden = flow.policies.length === 1;
  el('scene-field').hidden = flow.scenes.length === 1;
  updatePolicy();
}
function updatePolicy() {
  const policy = catalog.policies[el('policy').value], model = Boolean(policy.model);
  el('policy-description').textContent = policy.description;
  el('model-fields').hidden = !model; el('model-fields').disabled = !model;
  el('planner-fields').hidden = !policy.planner; el('planner-fields').disabled = !policy.planner;
  const supportsPlannerContext = Boolean(policy.planner && policy.context_modes?.includes('feedback') && policy.context_modes.includes('current_frame'));
  el('planner-context-field').hidden = !supportsPlannerContext;
  el('planner-context').disabled = !supportsPlannerContext;
  const embodiment = catalog.tasks[el('tasks').value].embodiment;
  const projectScope = catalog.workflows[workflow].acceptance_scope === 'project_tasks_v2';
  const models = (catalog.models || []).filter(m => m.embodiment === embodiment && m.tasks.includes(el('tasks').value) && (m.acceptance_scope === 'project_tasks_v2') === projectScope);
  const modelMap = Object.fromEntries(models.map(m => [m.id, m]));
  options(el('model-version'), [...models.map(m => m.id), 'local'], id => id === 'local' ? '高级：本机模型工件' : modelMap[id].label + (modelMap[id].ready ? '' : ' · 尚未部署'), true);
  if (!modelChoiceTouched && models.length) el('model-version').value = models.find(m => m.ready)?.id || models[0].id;
  const modes = ['live', 'offline'];
  if (model && catalog.workflows[workflow].realtime) modes.push('robotd_bam');
  options(el('mode'), modes, id => id === 'live' ? '实时运行 · 边跑边看' : id === 'offline' ? '离线评测 · 结束后回放' : 'robotd 协议评测', true);
  el('mode-hint').textContent = '同一场景、任务和种子可用于策略比较。实时模式按仿真时钟推进，浏览器只接收当前画面。';
  updateLaunch();
}
function updateLaunch() {
  if (!catalog) return;
  const task = catalog.tasks[el('tasks').value], model = Boolean(catalog.policies[el('policy').value].model);
  const planner = Boolean(catalog.policies[el('policy').value].planner);
  const registered = (catalog.models || []).find(m => m.id === el('model-version').value);
  el('local-model-paths').hidden = Boolean(registered);
  el('model-runtime-options').hidden = Boolean(registered);
  for (const select of el('model-runtime-options').querySelectorAll('select')) select.disabled = Boolean(registered);
  for (const id of ['model-path', 'backbone-path']) {el(id).required = model && !registered; el(id).disabled = Boolean(registered);}
  const score = registered?.scores?.[el('tasks').value];
  el('model-score').textContent = !registered ? '模型尚未登记时，可用完整本机工件做实验；不代表行为已验收。' : `${registered.label} · ${registered.ready ? '服务可用' : '当前不可运行'} · ${score?.offline_accepted ? '离线已验收' : '实验技能'} · ${score?.realtime_accepted ? '实时已验收' : '实时未验收'}${score?.episodes ? ` · ${score.episodes} 集，成功率 ${(score.strict_rate * 100).toFixed(1)}%` : ''}`;
  const missing = (model && (registered ? !registered.ready : (!el('model-path').value.trim() || !el('backbone-path').value.trim()))) || (planner && (!el('planner-endpoint').value.trim() || !el('planner-model').value.trim()));
  const seedOK = /^\d+$/.test(el('seed').value), live = el('mode').value === 'live';
  el('task-hint').textContent = `本体：${task.embodiment === 'rollers' ? '滚轮鸭子' : '腿式鸭子'}${model ? '；模型工件必须与本体匹配。' : '。'}`;
  el('launch-summary').textContent = `将运行：${task.label} · ${catalog.scenes[el('scene').value]} · ${catalog.policies[el('policy').value].label} · seed ${el('seed').value} · ${modeLabel(el('mode').value)}`;
  if (planner && !el('planner-context').disabled) el('launch-summary').textContent += el('planner-context').value === 'feedback' ? ' · 动作反馈与记忆' : ' · 当前画面对照';
  el('run-hint').textContent = activeJob ? '当前实验尚未结束。实时运行可用下方按钮停止。' : missing ? (planner ? '请填写已启动的 VLM 服务地址和模型名称。' : '请选择已部署模型、本机完整工件或官方对照。') : seedOK ? '' : '随机种子需要是非负整数。';
  el('run').disabled = Boolean(activeJob) || missing || !seedOK;
  el('run').textContent = live ? '开始实时运行' : '运行离线评测';
  el('launch-note').textContent = live ? '持续显示正在推进的仿真画面，结束或停止后保存记录。' : '尽快完成计算，结束后在历史列表查看回放。';
}
function categoryRecords() {
  return records.filter(r => historyCategory === 'all' || r.category === historyCategory);
}
function historyTabs() {
  const categories = [...catalog.categories, {id: 'other', label: '其他记录'}, {id: 'all', label: '全部'}];
  el('history-tabs').replaceChildren();
  for (const item of categories) {
    const count = records.filter(r => item.id === 'all' || r.category === item.id).length;
    if (!count && item.id === 'other') continue;
    const button = document.createElement('button'); button.type = 'button'; button.dataset.category = item.id;
    button.textContent = `${item.label} (${count})`; button.setAttribute('aria-pressed', String(historyCategory === item.id));
    button.onclick = () => {
      historyCategory = item.id; historyPage = 0;
      for (const id of ['history-task', 'history-scene', 'history-policy', 'history-result']) el(id).value = 'all';
      rebuildHistory();
    };
    el('history-tabs').append(button);
  }
}
function rebuildHistory(preferred) {
  historyTabs();
  let available = categoryRecords();
  const values = key => [...new Set(available.map(r => r[key]))].sort();
  options(el('history-task'), ['all', ...values('task')], id => id === 'all' ? '全部任务' : catalog.tasks[id]?.label || id, true);
  if (el('history-task').value !== 'all') available = available.filter(r => r.task === el('history-task').value);
  options(el('history-scene'), ['all', ...values('scene')], id => id === 'all' ? '全部场景' : catalog.scenes[id] || id, true);
  if (el('history-scene').value !== 'all') available = available.filter(r => r.scene === el('history-scene').value);
  options(el('history-policy'), ['all', ...values('policy')], id => id === 'all' ? '全部策略' : id, true);
  renderRecords(preferred);
}
function matchingRecords() {
  return categoryRecords().filter(r => {
    for (const [id, key] of [['history-task', 'task'], ['history-scene', 'scene'], ['history-policy', 'policy']]) {
      if (el(id).value !== 'all' && el(id).value !== r[key]) return false;
    }
    if (el('history-camera').checked && !r.has_camera && !r.has_observer) return false;
    const result = el('history-result').value, cancelled = r.termination_reason === 'cancelled_by_user';
    return result === 'all' || (result === 'cancelled' ? cancelled : result === 'success' ? r.success === true : r.success === false && !cancelled);
  });
}
function recordLabel(r) {
  return `${catalog.tasks[r.task]?.label || r.task}${r.execution_scope === 'project_tasks_v2' ? ' · 项目动作版本' : ''} · ${catalog.scenes[r.scene] || r.scene} · ${r.policy} · seed ${r.seed} · ${r.path.split('/')[0]}`;
}
function cell(row, text, small) {
  const td = document.createElement('td'); td.textContent = text;
  if (small) {const sub = document.createElement('small'); sub.textContent = small; td.append(sub);}
  row.append(td); return td;
}
function renderRecords(preferred) {
  const shown = matchingRecords(), pages = Math.max(1, Math.ceil(shown.length / PAGE_SIZE));
  if (preferred) {const index = shown.findIndex(r => r.path === preferred); if (index >= 0) historyPage = Math.floor(index / PAGE_SIZE);}
  historyPage = Math.min(historyPage, pages - 1);
  el('replay-count').textContent = shown.length ? `筛选到 ${shown.length} 条记录 / 历史共 ${records.length} 条。每页最多 ${PAGE_SIZE} 条。` : '没有符合筛选条件的记录。可切换任务、场景，或取消“只看有画面”。';
  el('history-rows').replaceChildren();
  const groups = new Map();
  for (const r of shown.slice(historyPage * PAGE_SIZE, (historyPage + 1) * PAGE_SIZE)) {
    if (!groups.has(r.task)) groups.set(r.task, []);
    groups.get(r.task).push(r);
  }
  for (const [task, group] of groups) {
    const heading = document.createElement('tr'); heading.className = 'group';
    const td = cell(heading, catalog.tasks[task]?.label || task); td.colSpan = 5; el('history-rows').append(heading);
    for (const r of group) {
      const row = document.createElement('tr'); row.dataset.record = r.path;
      cell(row, r.recorded_at ? new Date(r.recorded_at).toLocaleString('zh-CN', {hour12: false}) : '未记录时间', r.path.split('/')[0]);
      cell(row, catalog.scenes[r.scene] || r.scene, r.policy);
      cell(row, `seed ${r.seed}`, modeLabel(r.mode));
      const result = r.termination_reason === 'cancelled_by_user' ? '手动停止' : r.success === true ? '成功' : r.success === false ? '未通过' : '未记录';
      cell(row, result, r.has_camera || r.has_observer ? '' : '无画面').className = r.success ? 'success' : 'failure';
      const actions = document.createElement('td'), buttons = document.createElement('div'); buttons.className = 'history-actions';
      for (const side of ['left', 'right']) {
        const button = document.createElement('button'); button.type = 'button'; button.dataset.side = side;
        button.textContent = side === 'left' ? '放到左侧' : '放到右侧';
        button.setAttribute('aria-label', `${button.textContent}：${recordLabel(r)}`);
        button.setAttribute('aria-pressed', String(selected[side] === r.path));
        button.onclick = () => {selectRecord(side, r.path); renderRecords();}; buttons.append(button);
      }
      actions.append(buttons); row.append(actions); el('history-rows').append(row);
    }
  }
  el('history-page').textContent = `${historyPage + 1} / ${pages} 页`;
  el('history-prev').disabled = historyPage === 0; el('history-next').disabled = historyPage >= pages - 1;
  const camera = shown.filter(r => r.has_camera || r.has_observer);
  for (const side of ['left', 'right']) {
    if (!selected[side] && camera.length) selectRecord(side, camera[side === 'right' && camera.length > 1 ? 1 : 0].path);
  }
}
async function refresh(preferredJob) {
  const version = ++refreshVersion, next = await api('/api/episodes');
  if (version !== refreshVersion) return;
  records = next;
  const preferred = preferredJob && records.find(r => r.path.startsWith(preferredJob + '/'));
  if (preferred) {
    historyCategory = preferred.category; el('history-result').value = 'all';
    el('history-camera').checked = Boolean(preferred.has_camera || preferred.has_observer);
    for (const id of ['history-task', 'history-scene', 'history-policy']) el(id).value = 'all';
    rebuildHistory(); el('history-task').value = preferred.task; rebuildHistory(); el('history-scene').value = preferred.scene;
    selectRecord('left', preferred.path);
  }
  rebuildHistory(preferred?.path);
}
function setPlaying(value) {
  playing = value; el('play').textContent = value ? '暂停回放' : '播放回放'; el('play').setAttribute('aria-pressed', String(value));
}
function selectRecord(side, path) {
  if (selected[side] === path) return;
  selected[side] = path; setPlaying(false); el('position').value = 0; load(side);
}
function updateDuration() {
  const duration = Math.max(0, ...Object.values(details).filter(Boolean).map(r => Number(r.final?.tick || r.duration_ticks || 0)));
  el('position').max = duration; el('position').disabled = duration === 0; el('play').disabled = duration === 0;
  if (Number(el('position').value) > duration) el('position').value = duration;
  timeline();
}
async function load(side) {
  const version = ++loadVersions[side], path = selected[side]; details[side] = null;
  el(side + 'Image').hidden = true; el(side + 'Image').removeAttribute('src');
  for (const suffix of ['Context', 'Status', 'View', 'Info']) el(side + suffix).textContent = '';
  const row = records.find(r => r.path === path); el(side + 'Title').textContent = row ? recordLabel(row) : '从上方列表选择记录';
  if (!path) {updateDuration(); return;}
  try {
    const r = await api('/api/episode?path=' + encodeURIComponent(path));
    if (version !== loadVersions[side]) return;
    details[side] = {...r, _path: path, duration_ticks: row?.duration_ticks};
    el(side + 'Context').textContent = `场景：${catalog.scenes[r.scene] || r.scene || '未记录'} · ${modeLabel(r.mode || (r.profile === 'robotd_bam' ? 'robotd_bam' : 'offline'))} · seed ${r.seed}`;
    const teacherLabel = r.policy?.uses_privileged_state ? ' · 真值教师' : (r.teacher_only || r.policy?.metadata?.teacher_only) ? ' · 教师参考' : '';
    el(side + 'Status').textContent = `${r.termination_reason === 'cancelled_by_user' ? '手动停止' : r.success ? '任务成功' : '任务未通过'} · ${r.termination_reason || '未记录终止原因'}${teacherLabel}${r.eligible_for_primary_benchmark === false ? ' · 辅助比较' : ''}`;
    el(side + 'Status').className = `status ${r.success ? 'success' : 'failure'}`;
    el(side + 'Info').textContent = JSON.stringify({record: path, policy: r.policy, profile: r.profile, scene: r.scene, inference_ms: r.inference_ms, live_execution: r.live_execution, strict_result: r.final?.strict_preset_result}, null, 2);
    updateDuration();
  } catch (error) {if (version === loadVersions[side]) el(side + 'Status').textContent = '读取失败：' + error.message;}
}
function frame(side) {
  if (!details[side]) return;
  const record = records.find(r => r.path === selected[side]);
  const view = el('view').value === 'observer' && record?.has_observer ? 'observer' : 'camera';
  const available = view === 'observer' || record?.has_camera, duration = details[side].final?.tick || details[side].duration_ticks || 0;
  el(side + 'Image').hidden = !available;
  el(side + 'View').textContent = available ? (view === 'observer' ? '观察者视角' : '机器人摄像头记录') + (Number(el('position').value) >= duration ? ' · 本条回放已结束' : '') : '这条记录没有相机画面';
  if (available) el(side + 'Image').src = '/api/frame?path=' + encodeURIComponent(selected[side]) + '&view=' + view + '&tick=' + el('position').value;
}
function timeline() {
  el('tick').textContent = `${(Number(el('position').value) / 50).toFixed(1)} / ${(Number(el('position').max) / 50).toFixed(1)} 秒 · tick ${el('position').value}`;
  frame('left'); frame('right');
}
function showLive(packet) {
  livePacket = packet;
  el('live-diagnostics').textContent = JSON.stringify(packet.diagnostics || {}, null, 2);
  const job = packet.job; el('live-panel').hidden = false;
  el('live-context').textContent = `${catalog.tasks[job.task]?.label || job.task} · ${catalog.scenes[job.scene] || job.scene} · ${job.policy} · ${job.id}`;
  const ended = terminal(job.status);
  el('live-heading').textContent = ended ? '本次实时运行已结束' : '正在实时运行';
  el('stop-live').disabled = ended || job.cancel_requested;
  el('stop-live').textContent = job.cancel_requested && !ended ? '正在停止…' : '停止本次运行';
  el('live-status').textContent = job.status === 'cancelled' ? '已停止' : job.status === 'failed' ? '运行报错' : ended ? (packet.success ? '任务成功，记录已保存' : '运行结束，结果见历史记录') : job.cancel_requested ? '正在保存并停止' : packet.status === 'initializing' ? (job.status === 'queued' ? '等待运行' : '正在加载模型与场景') : '实时画面';
  el('live-metrics').textContent = packet.sim_time_s == null ? '' : `仿真 ${packet.sim_time_s.toFixed(2)} 秒 · 实际 ${packet.wall_time_s.toFixed(2)} 秒 · 速度 ${packet.real_time_factor.toFixed(2)}× · tick ${packet.tick}${packet.execution?.fallback_ticks ? ` · 本地接管 ${packet.execution.fallback_ticks} ticks` : ''}${packet.max_lag_s > 0.25 ? ' · 当前计算跟不上实时' : ''}`;
  const source = packet.frames?.[el('live-view').value];
  el('live-image').hidden = !source;
  if (source) el('live-image').src = source;
}
async function followJob(job) {
  activeJob = job; updateLaunch();
  const version = ++pollingVersion;
  localStorage.setItem('microduck-active-job', JSON.stringify({id: job.id, mode: job.mode}));
  const poll = async () => {
    if (version !== pollingVersion) return;
    try {
      let current;
      if (job.mode === 'live') {
        const packet = await api('/api/live?id=' + encodeURIComponent(job.id));
        if (version !== pollingVersion) return;
        current = packet.job; showLive(packet);
      } else current = await api('/api/job?id=' + encodeURIComponent(job.id));
      const status = {queued: '等待运行', running: job.mode === 'live' ? '实时运行中' : '正在计算', complete: '运行结束，记录已保存', failed: '运行报错', cancelled: '已停止'}[current.status] || current.status;
      el('job').textContent = `${status}（${job.id}）${current.error ? '\n' + current.error : ''}`;
      if (terminal(current.status)) {
        activeJob = null; localStorage.removeItem('microduck-active-job'); updateLaunch(); await refresh(job.id);
      } else setTimeout(poll, job.mode === 'live' ? 50 : 1000);
    } catch (error) {
      el('job').textContent = '读取运行状态失败：' + error.message;
      // A transient disconnect does not stop or restart the server simulation.
      if (version === pollingVersion) setTimeout(poll, 2000);
    }
  };
  poll();
}
el('launch').onsubmit = async event => {
  event.preventDefault(); if (el('run').disabled) return;
  const payload = Object.fromEntries(new FormData(event.target)); el('run').disabled = true;
  try {
    const job = await post('/api/run', payload);
    if (job.mode === 'live') {setPlaying(false); livePacket = null; el('live-image').hidden = true; el('live-metrics').textContent = ''; el('live-panel').hidden = false; el('live-status').textContent = '正在启动'; el('live-panel').scrollIntoView({behavior: 'smooth', block: 'start'});}
    followJob(job);
  } catch (error) {el('job').textContent = '无法启动：' + error.message; updateLaunch();}
};
el('stop-live').onclick = async () => {
  if (!activeJob || activeJob.mode !== 'live') return;
  el('stop-live').disabled = true;
  try {await post('/api/cancel', {id: activeJob.id});}
  catch (error) {el('live-status').textContent = '停止失败：' + error.message; el('stop-live').disabled = false;}
};
el('live-view').onchange = () => {if (livePacket) showLive(livePacket);};
el('view').onchange = timeline; el('position').oninput = timeline;
el('play').onclick = () => {if (!playing && Number(el('position').value) >= Number(el('position').max)) el('position').value = 0; setPlaying(!playing);};
setInterval(() => {
  if (playing) {
    const next = Math.min(Number(el('position').max), Number(el('position').value) + 5);
    el('position').value = next; timeline();
    if (next >= Number(el('position').max)) setPlaying(false);
  }
}, 100);
for (const id of ['history-task', 'history-scene', 'history-policy', 'history-result', 'history-camera']) el(id).onchange = () => {historyPage = 0; rebuildHistory();};
el('history-prev').onclick = () => {historyPage--; renderRecords();};
el('history-next').onclick = () => {historyPage++; renderRecords();};
async function init() {
  try {
    catalog = await api('/api/catalog');
    for (const item of catalog.categories) {
      const label = document.createElement('label'); label.className = 'category';
      const input = document.createElement('input'); input.type = 'radio'; input.name = 'category'; input.value = item.id; input.checked = item.id === category;
      input.onchange = () => setCategory(item.id);
      const text = document.createElement('span'), title = document.createElement('strong'), description = document.createElement('small');
      title.textContent = item.label; description.textContent = item.description;
      text.append(title, description); label.append(input, text); el('categories').append(label);
    }
    el('project').onchange = () => setWorkflow(el('project').value); el('policy').onchange = updatePolicy;
    el('model-version').onchange = () => {modelChoiceTouched = true; updateLaunch();};
    el('tasks').onchange = updatePolicy;
    for (const id of ['scene', 'mode', 'planner-context']) el(id).onchange = updateLaunch;
    for (const id of ['seed', 'model-path', 'backbone-path', 'planner-endpoint', 'planner-model']) el(id).oninput = updateLaunch;
    setCategory(category);
    // Gallery links preselect only supported combinations; opening a link never starts a job.
    const preset = new URLSearchParams(location.search), flow = preset.get('workflow');
    if (flow && catalog.workflows[flow]) {
      setCategory(catalog.workflows[flow].category); setWorkflow(flow);
      for (const input of document.querySelectorAll('input[name=category]')) input.checked = input.value === category;
      for (const [key, id, choices] of [['task', 'tasks', 'tasks'], ['policy', 'policy', 'policies'], ['scene', 'scene', 'scenes']]) {
        if (catalog.workflows[flow][choices].includes(preset.get(key))) el(id).value = preset.get(key);
      }
      updatePolicy();
      if ([...el('mode').options].some(o => o.value === preset.get('mode'))) el('mode').value = preset.get('mode');
      updateLaunch();
    }
    await refresh();
    try {
      const saved = JSON.parse(localStorage.getItem('microduck-active-job') || 'null');
      if (saved) {const job = await api('/api/job?id=' + encodeURIComponent(saved.id)); if (!terminal(job.status)) followJob(job); else localStorage.removeItem('microduck-active-job');}
    } catch (_) {localStorage.removeItem('microduck-active-job');}
  } catch (error) {el('job').textContent = '页面加载失败：' + error.message;}
}
init();
