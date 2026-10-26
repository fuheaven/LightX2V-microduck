'use strict';
const $ = id => document.getElementById(id);
let data, catalog, activeJob = null, pollTimer = null, selectedSkill = null;
let livePacket = null, liveViewTouched = false;
const groups = {brain:'自主决策',motion:'运动技能',perception:'视觉感知',world:'仿真世界',training:'训练工具'};
const statusLabels = {native:'本地实验',adapter:'需配置服务',scene:'场景已接入',upstream:'上游成果 · 待适配',tool:'本地课程'};
const endStates = ['complete','cancelled','failed'];
const node = (tag, text, cls) => {const n = document.createElement(tag); if (text != null) n.textContent=text; if (cls) n.className=cls; return n;};
const link = (text, href, cls='text-link') => {const n=node('a',text,cls);n.href=href;if (href.startsWith('https://')) {n.target='_blank';n.rel='noopener noreferrer';}return n;};
async function api(path, body) {
 const r = await fetch(path, body ? {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)} : {});
 const value = await r.json();if(!r.ok){const error=new Error(value.error || `HTTP ${r.status}`);error.status=r.status;throw error;}return value;
}
function route() {
 const name=['skills','official','community'].includes(location.hash.slice(1)) ? location.hash.slice(1) : 'skills';
 for(const n of document.querySelectorAll('[data-panel]'))n.hidden=n.dataset.panel!==name;
 for(const n of document.querySelectorAll('[data-tab]')) {if(n.dataset.tab===name)n.setAttribute('aria-current','page');else n.removeAttribute('aria-current');}
}
function reportDate(value) {return value ? new Date(value*1000).toLocaleDateString('zh-CN') : '暂无记录';}
function displayError(error) {$('page-error').hidden=false;$('page-error').querySelector('span').textContent='暂时无法读取完整页面：'+error.message;}
function markVideoError(video, message) {
 video.addEventListener('error',()=>{message.textContent='视频暂时不可播放，请打开原始报告或稍后重试。';message.hidden=false;});
}
function feature(skill) {
 const clip=skill.videos.find(v=>v.view==='observer'&&v.success);
 if(!clip) return;
 $('featured-video').onloadedmetadata=()=>{$('featured-video').currentTime=Math.min(.2,$('featured-video').duration/2);};
 $('featured-video').src=clip.url;$('featured-video').hidden=false;$('live-image').hidden=true;
 $('preview-title').textContent=skill.label;$('preview-kind').textContent='TinyVLA · 离线回放';
 $('preview-note').textContent=`独立测试 ${skill.score.successes}/${skill.score.episodes} · 当前视频 seed ${clip.seed}`;
 $('preview-empty').hidden=true;
}
function showSkills() {
 $('skill-grid').replaceChildren();$('remaining-grid').replaceChildren();
 const accepted=data.skills.filter(c=>c.status==='accepted');
 for(const skill of accepted) {
  const fragment=$('skill-template').content.cloneNode(true), card=fragment.querySelector('article');
  card.dataset.task=skill.id;
  const button=card.querySelector('button');button.setAttribute('aria-label',`观看 ${skill.label} 的验收回放`);button.onclick=()=>openSkill(skill);
  card.querySelector('h3').textContent=skill.label;card.querySelector('.description').textContent=skill.description;
  card.querySelector('.body-label').textContent=skill.embodiment==='rollers'?'滚轮本体':'腿式本体';
  card.querySelector('.card-score strong').textContent=`${skill.score.successes} / ${skill.score.episodes}`;
  const video=card.querySelector('video'),clip=skill.videos.find(v=>v.view==='observer'&&v.success);
  if(clip){video.src=clip.url;video.addEventListener('loadedmetadata',()=>{video.currentTime=video.duration*.45;},{once:true});}
  else {video.hidden=true;card.querySelector('.media-label').textContent='查看成绩 · 暂无视频';}
  $('skill-grid').append(fragment);
 }
 const others=data.skills.filter(c=>c.status!=='accepted');
 for(const skill of others){
  const row=node('div',null,'remaining-item');row.append(node('strong',skill.label));
  if(skill.status==='failed') {const b=node('button',`${skill.score.successes}/${skill.score.episodes} · 未通过`);b.onclick=()=>openSkill(skill);row.append(b);}
  else row.append(node('span',skill.status==='evidence_unavailable'?'证据缓存待同步':'尚未正式验收'));
  $('remaining-grid').append(row);
 }
 $('remaining-count').textContent=`${others.length} 项`;$('skills-empty').hidden=accepted.length>0;
 $('campaign-note').textContent=`验收批次 ${data.campaign.id || '暂无'} · ${reportDate(data.campaign.completed_at)} · 成绩只属于报告登记的模型。`;
 const chosen=accepted.find(c=>c.id==='walk_forward')||accepted[0];
 if(chosen&&!activeJob)feature(chosen);
 if(!chosen&&!activeJob)$('preview-empty').textContent='暂无可核验的 TinyVLA 视频，可先体验官方运动。';
}
function openSkill(skill) {
 selectedSkill=skill;$('dialog-title').textContent=skill.label;$('dialog-description').textContent=skill.description;
 $('dialog-score').textContent=`${skill.score.successes} / ${skill.score.episodes} · ${skill.status==='accepted'?'离线通过':'未通过'}`;
 $('dialog-model').textContent=skill.model_sha256.slice(0,16);$('dialog-model').title=skill.model_sha256;
 $('dialog-report').href=skill.report_url;
 $('clip-select').replaceChildren(...skill.videos.map((clip,i)=>new Option(`${clip.success?'成功':'失败'}样例 · seed ${clip.seed} · ${clip.view==='observer'?'观察者视角':'鸭子摄像头'}`,String(i))));
 $('clip-select').disabled=!skill.videos.length;setClip();$('skill-dialog').showModal();
}
function setClip(){
 const clip=selectedSkill.videos[Number($('clip-select').value)];
 $('dialog-video').pause();$('dialog-video').hidden=!clip;
 if(!clip){$('dialog-video').removeAttribute('src');$('clip-note').textContent='本地尚无对应视频，成绩可通过原始报告核验。';return;}
 $('dialog-video').src=clip.url;
 $('clip-note').textContent=(clip.success?'本条任务成功。':'本条任务失败。')+(clip.view==='observer'?' 观察者视角不作为模型输入。':' 模型使用的头部摄像头记录。')+' 这是已保存的离线实验。';
}
function renderOfficial(){
 const tasks=['walk_forward','circle_left','figure_eight','sit_walk','kick_left','roller_glide'];
 $('official-grid').replaceChildren();
 for(const task of tasks){
  const card=node('article',null,'community-card');card.append(node('span','官方 ONNX','pill'),node('h3',catalog.tasks[task].label));
  card.append(node('p',data.skills.find(s=>s.id===task)?.description || catalog.tasks[task].instruction));
  const actions=node('div',null,'card-actions'),b=node('button','实时运行','button primary');b.dataset.launch='';b.onclick=()=>startLive('presets',task,'官方 ONNX');
  actions.append(b,link('更多设置',`/experiments?workflow=presets&task=${task}&mode=live`));card.append(actions);$('official-grid').append(card);
 }
}
function renderCommunity(){
 const group=$('community-group').value,status=$('community-status').value,q=$('community-search').value.trim().toLowerCase();
 const cards=data.community.projects.filter(c=>(group==='all'||c.group===group)&&(status==='all'||status==='upstream'&&c.integration==='upstream'||status==='new'&&c.is_new||status==='local'&&(c.launch_ready||c.internal_url))&&[c.title,c.description,c.source_url].join(' ').toLowerCase().includes(q));
 $('community-grid').replaceChildren();
 for(const c of cards){
  const article=node('article',null,'community-card');article.dataset.project=c.id;
  const top=node('div',null,'card-top'),state=c.asset_errors?.length?'资源待准备':statusLabels[c.integration];
  top.append(node('span',state,'pill'+(c.integration==='upstream'?' source':c.requires_service||c.asset_errors?.length?' warn':'')),node('span',groups[c.group],'group-label'));
  article.append(top);const title=node('h3',c.title);if(c.is_new)title.append(node('span','NEW','new-tag'));article.append(title,node('p',c.description));
  const details=node('details'),summary=node('summary','接入说明与版本');details.append(summary,node('p',c.note));
  if(c.local_check)details.append(node('p','本地检查：'+c.local_check));
  if(c.local_revision)details.append(node('p','本地：'+c.local_revision.slice(0,16)));
  details.append(node('p','上游：'+c.latest_revision.slice(0,12)+' · '+c.upstream_updated_at.slice(0,10)));
  if(c.asset_errors?.length)details.append(node('p','资源校验未通过，请在服务器完成固定资产准备。'));
  article.append(details);
  const actions=node('div',null,'card-actions');
  if(c.launch_ready){const b=node('button','实时体验','button primary');b.dataset.launch='';b.onclick=()=>startLive(c.workflow,c.task,c.title);actions.append(b);}
  else if(c.requires_service&&c.launch_url)actions.append(link('配置模型服务',c.launch_url,'button secondary'));
  else if(c.internal_url)actions.append(link('打开课程',c.internal_url,'button secondary'));
  else if(c.demo_url)actions.append(link('查看上游演示',c.demo_url,'button secondary'));
  actions.append(link('项目来源 ↗',c.source_url));article.append(actions);
  if(c.extra_workflow)article.append(link('打开滚轮绕桩 →',`/experiments?workflow=${c.extra_workflow}`));
  $('community-grid').append(article);
 }
 $('community-result').textContent=`显示 ${cards.length} / ${data.community.projects.length} 个项目。上游成绩不计入 TinyVLA 能力。`;
 $('community-empty').hidden=cards.length>0;setLaunchButtons();
}
function setLaunchButtons(){for(const b of document.querySelectorAll('[data-launch],#quick-live'))b.disabled=!catalog||Boolean(activeJob);}
function liveFrame(){
 if(!livePacket)return;
 const view=$('live-view').value,frame=livePacket.frames?.[view];
 if(frame){$('live-image').src=frame;$('live-image').hidden=false;$('preview-empty').hidden=true;
  $('live-image').alt=view==='observer'?'实时观察者视角':view==='perception'?'实时摄像头检测结果':'实时鸭子摄像头';}
}
$('live-view').onchange=()=>{liveViewTouched=true;liveFrame();};
async function startLive(workflow, task, title){
 if(activeJob||!catalog)return;
 const flow=catalog.workflows[workflow];if(!flow)return;
 activeJob={id:null};setLaunchButtons();clearTimeout(pollTimer);
 livePacket=null;liveViewTouched=false;$('live-view').value='observer';$('live-controls').hidden=false;
 $('featured-video').pause();$('featured-video').hidden=true;$('live-image').hidden=true;
 $('preview-empty').textContent='正在启动 MuJoCo 仿真…';$('preview-empty').hidden=false;
 $('preview-kind').textContent='实时仿真';$('preview-title').textContent=title;
 $('preview-note').textContent='等待当前仿真的第一帧…';$('live-status').hidden=false;$('live-status').textContent='正在创建实验';
 document.querySelector('.preview').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth',block:'center'});
 try{
  activeJob=await api('/api/run',{workflow,category:flow.category,task:task||flow.tasks[0],scene:flow.scenes[0],policy:flow.policies[0],mode:'live',seed:2});
  sessionStorage.setItem('microduck-showcase-live',JSON.stringify({id:activeJob.id,title}));
  $('stop-live').hidden=false;$('stop-live').disabled=false;pollLive();
 }catch(e){activeJob=null;setLaunchButtons();$('preview-empty').textContent='启动失败：'+e.message;$('live-status').textContent='没有运行中的实验';}
}
async function pollLive(){
 if(!activeJob?.id)return;
 try{
  const packet=await api('/api/live?id='+encodeURIComponent(activeJob.id));activeJob=packet.job;
  livePacket=packet;$('live-controls').hidden=false;
  $('live-view').querySelector('[value=perception]').disabled=!packet.frames?.perception;
  if(packet.frames?.perception&&!liveViewTouched)$('live-view').value='perception';
  liveFrame();
  if(packet.tick!=null)$('preview-note').textContent=`仿真 ${(packet.sim_time_s||0).toFixed(1)} 秒 · 实际 ${(packet.wall_time_s||0).toFixed(1)} 秒 · tick ${packet.tick} · ${(packet.real_time_factor||0).toFixed(2)}×`;
  $('live-status').textContent=activeJob.status==='queued'?'已排队，等待当前实验结束':activeJob.cancel_requested?'正在停止并保存记录':packet.tick==null?'正在加载模型与场景':'物理持续运行中 · 当前画面';
  if(endStates.includes(activeJob.status)){
   $('live-status').textContent=activeJob.status==='failed'?'实验报错：'+(activeJob.error||'请查看实验室日志'):activeJob.status==='cancelled'?'已停止，记录已保存':'运行结束 · '+(packet.success?'本次任务成功':'本次任务未通过')+' · 单次运行不代表能力验收';
   $('preview-kind').textContent='本次实时实验已结束';$('stop-live').hidden=true;activeJob=null;sessionStorage.removeItem('microduck-showcase-live');setLaunchButtons();return;
  }
 }catch(e){
  if(e.status===400||e.status===404){
   $('live-status').textContent='该实时会话已不可用，请在实验室查看已保存记录或重新运行。';
   $('preview-kind').textContent='实时连接已结束';$('stop-live').hidden=true;
   activeJob=null;sessionStorage.removeItem('microduck-showcase-live');setLaunchButtons();return;
  }
  $('live-status').textContent='连接中断，正在重试：'+e.message;
 }
 pollTimer=setTimeout(pollLive,100);
}
$('stop-live').onclick=async()=>{if(!activeJob?.id)return;$('stop-live').disabled=true;try{await api('/api/cancel',{id:activeJob.id});}catch(e){$('live-status').textContent='停止失败：'+e.message;$('stop-live').disabled=false;}};
$('quick-live').onclick=()=>startLive('presets','walk_forward','官方 ONNX · 向前走并停下');
$('close-dialog').onclick=()=>$('skill-dialog').close();$('skill-dialog').addEventListener('close',()=>$('dialog-video').pause());
$('clip-select').onchange=setClip;markVideoError($('dialog-video'),$('clip-note'));
$('featured-video').addEventListener('error',()=>{if(!activeJob){$('preview-empty').textContent='视频暂时不可播放，请打开能力卡片查看报告。';$('preview-empty').hidden=false;}});
for(const id of ['community-group','community-status'])$(id).onchange=()=>{if(data)renderCommunity();};
$('community-search').oninput=()=>{if(data)renderCommunity();};
window.addEventListener('hashchange',route);route();setLaunchButtons();
async function init(){
 $('page-error').hidden=true;
 try{
  [data,catalog]=await Promise.all([api('/api/showcase'),api('/api/catalog')]);
  $('accepted-count').textContent=`${data.summary.accepted} / ${data.summary.total}`;$('tested-count').textContent=`${data.summary.tested} / ${data.summary.total}`;
  $('community-count').textContent=data.community.projects.filter(c=>c.launch_ready||c.internal_url).length+' 项';
  $('new-count').textContent=data.community.projects.filter(c=>c.is_new).length+' 新';
  const fresh=data.freshness||{};
  $('freshness').textContent=(fresh.state==='fresh'?'验收数据已同步':'使用已保存的验收记录 · 同步待更新')+' · '+(fresh.generated_at?new Date(fresh.generated_at*1000).toLocaleString('zh-CN'):'暂无快照');
  $('community-date').textContent='核查于 '+data.community.checked_at+'。可运行实验、服务接口与上游候选分别展示。';
  const d=data.community.discovery;$('discovery-note').textContent=`本次发现 ${d.hf_models} 个 HF 模型、${d.hf_spaces} 个 Space；GitHub 检索匹配 ${d.github_total_matches} 个仓库，查看最近更新的 ${d.github_inspected_search_results} 条。目录是筛选后的快照。`;
  showSkills();renderOfficial();renderCommunity();
  try{
   const saved=JSON.parse(sessionStorage.getItem('microduck-showcase-live')||'null');
   if(saved){const job=await api('/api/job?id='+encodeURIComponent(saved.id));if(!endStates.includes(job.status)){
    activeJob=job;$('featured-video').pause();$('featured-video').hidden=true;$('stop-live').hidden=false;$('preview-kind').textContent='实时仿真';$('preview-title').textContent=saved.title;$('live-status').hidden=false;setLaunchButtons();pollLive();
   }else sessionStorage.removeItem('microduck-showcase-live');}
  }catch(_){sessionStorage.removeItem('microduck-showcase-live');}
 }catch(e){displayError(e);$('freshness').textContent='页面数据暂不可用';}
}
$('retry').onclick=init;init();
