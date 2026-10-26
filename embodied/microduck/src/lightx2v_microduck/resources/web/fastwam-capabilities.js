(() => {
  const host = document.getElementById('fastwam-capabilities');
  if (!host) return;
  const el = (tag, text, className) => {const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(className)e.className=className;return e;};
  const media = (id,kind) => `/api/fastwam/media?id=${encodeURIComponent(id)}&kind=${kind}`;
  async function load() {
    try {
      const response=await fetch('/api/fastwam/capabilities');if(!response.ok)throw Error('能力证据暂时无法读取');
      const data=await response.json();host.replaceChildren();
      const heading=el('div',undefined,'fw-heading'),intro=el('div'),badge=el('span',undefined,'fw-badge');
      intro.append(el('h2','能力测试与录屏'),el('p','FastWAM 根据图像、颜色指令和自身状态选择动作，官方运动策略负责行走与平衡。'));
      heading.append(intro,badge);host.append(heading);
      const groups=[{...data,id:'clear',label:'无障碍 · 实时',mode:'realtime',scene:'无障碍房间',
        evidence_label:'实时开发集已验证',
        scope_note:'已验证范围：无障碍仿真房间。带障碍能力、200 个独立布局的最终验收与真机能力尚未通过。',
        assessment_id:'assessment'},...(data.obstacle_groups||[]),...(data.retest_groups||[])];
      const groupSelector=el('div',undefined,'fw-groups');groupSelector.setAttribute('aria-label','选择测试场景');
      const panel=el('div',undefined,'fw-panel'),groupButtons=[];host.append(groupSelector,panel);
      function showGroup(group, updateURL=true){
        host.querySelector('video')?.pause();panel.replaceChildren();host.dataset.group=group.id;
        badge.textContent=group.evidence_label;badge.classList.toggle('fw-pending',group.id!=='clear');
        for(const [button,id] of groupButtons)button.setAttribute('aria-pressed',String(id===group.id));
        if(updateURL){const url=new URL(location.href);url.searchParams.set('capability',group.id);history.replaceState(null,'',url);}
        const layout=el('div',undefined,'fw-layout'),viewer=el('div'),results=el('div',undefined,'fw-results');
        const video=el('video');video.controls=true;video.playsInline=true;video.preload='metadata';video.setAttribute('aria-label',`${group.scene}仿真测试录屏`);
        const current=el('div',undefined,'fw-current'),title=el('strong'),link=el('a','下载这段录屏');link.download='';current.append(title,link);
        const meta=el('p',undefined,'fw-meta'),outcome=el('p',undefined,'fw-outcome');outcome.setAttribute('aria-live','polite');
        const selector=el('div',undefined,'fw-selector');selector.setAttribute('aria-label','选择测试录屏');
        const error=el('p','录屏加载失败，请刷新或使用下载链接。','fw-error');error.hidden=true;video.addEventListener('error',()=>{error.hidden=false;});
        const views=el('div',undefined,'fw-views');views.append(el('span','鸭子与环境 · 状态重绘'),el('span','鸭子摄像头 · 原始记录'));
        viewer.append(views,video,current,meta,outcome,error,selector,el('p',group.recording_note||data.recording_note,'fw-caption'));
        const score=el('div',undefined,'fw-score');score.append(document.createTextNode(`${group.passed}`),el('small',` / ${group.total} 案例通过`));
        const topline=el('div');topline.append(el('h3',group.label),score,el('p',`${(100*group.passed/group.total).toFixed(1)}% · ${group.layouts} 个开发布局 × 3 种颜色`));results.append(topline);
        const byColor=el('div');for(const row of group.color_scores){const item=el('div',undefined,'fw-task'),name=el('span'),dot=el('i',undefined,'fw-dot');dot.style.background={red:'#ee7a78',blue:'#85b8f2',yellow:'#e8cc65'}[row.color];name.append(dot,document.createTextNode(`寻找${row.label}球`));item.append(name,el('strong',`${row.passed} / ${row.total}`));byColor.append(item);}results.append(byColor);
        const steps=el('div',undefined,'fw-steps');for(const text of group.id==='clear'?['按颜色指令寻球','接近目标','停稳 ≥ 2 秒']:['检查碰撞','检查接近与停止','成功与失败均保留'])steps.append(el('span',text));results.append(steps);
        results.append(el('p',group.scope_note,'fw-limit'));
        const evidence=el('div',undefined,'fw-evidence'),report=el('a','查看本批完整评测报告');report.href=media(group.assessment_id||group.id,'report');report.target='_blank';report.rel='noopener';
        const details=el('details'),summary=el('summary','模型版本与录屏来源'),source=el('p',`${group.model||data.model} · 权重 ${(group.checkpoint_sha256||data.checkpoint_sha256).slice(0,12)}。${group.selection_rule} 单段演示不替代整体成功率。`,'fw-source');
        const episodeReport=el('a','查看当前录屏的实验记录');episodeReport.target='_blank';episodeReport.rel='noopener';details.append(summary,source,episodeReport);evidence.append(report,details);results.append(evidence);
        const buttons=[];
        function choose(clip){
          video.pause();error.hidden=true;video.poster=media(clip.id,'poster');video.src=media(clip.id,'video');video.load();
          title.textContent=clip.title;link.href=media(clip.id,'video');link.download=clip.id+'.mp4';episodeReport.href=media(clip.id,'report');
          meta.textContent=`${group.mode==='offline'?'离线评测':'实时评测'} · ${group.scene} · seed ${clip.seed} · ${clip.variant==='optimized'?'加速版':'未加速对照'} · ${clip.duration_s.toFixed(1)} 秒`;
          const reasons=clip.failure_reasons||[];
          outcome.textContent=clip.passed?'本集通过任务检查。':`本集未通过${reasons.length?'：'+reasons.join('；'):'。'}${clip.fallback_ticks&&!reasons.some(r=>r.includes('接管'))?' · 发生官方控制接管':''}`;
          outcome.classList.toggle('fw-failed',!clip.passed);
          for(const [button,id] of buttons)button.setAttribute('aria-pressed',String(id===clip.id));
        }
        for(const clip of group.clips){const button=el('button',clip.title);button.type='button';button.addEventListener('click',()=>choose(clip));buttons.push([button,clip.id]);selector.append(button);}
        layout.append(viewer,results);panel.append(layout);
        choose(group.clips.find(c=>c.id===group.default_clip_id)||group.clips[0]);
      }
      for(const group of groups){const button=el('button',group.label);button.type='button';button.dataset.group=group.id;button.addEventListener('click',()=>showGroup(group));groupButtons.push([button,group.id]);groupSelector.append(button);}
      showGroup(groups.find(g=>g.id===new URLSearchParams(location.search).get('capability'))||groups[0],false);
      host.dataset.ready='true';
    } catch(error) {host.replaceChildren(el('h2','能力测试与录屏'),el('p',String(error.message),'fw-error'));const retry=el('button','重新加载');retry.addEventListener('click',load);host.append(retry);}
  }
  load();
})();
