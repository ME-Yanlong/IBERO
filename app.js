/* 本地展示层：不运行仿真、不依赖网络；视频来源由构建器校验。 */
(() => {
  'use strict';
  const data = window.IBERO_SHOWCASE;
  if (!data?.demos?.length) return;
  const publicPreview = data.publicPreview === true;
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const escape = value => String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
  const families = {routing:'柔性布线', release:'卡扣拆卸', machining:'铣削加工', foundation:'基础操作'};
  const formatTime = seconds => `${Math.floor(Math.round(seconds) / 60)}:${String(Math.round(seconds) % 60).padStart(2, '0')}`;
  let filter = 'all';
  let selected = null;
  let returnFocus = null;
  let detail = 'task';
  let toastTimer;
  const dialog = $('#demo-dialog');
  const video = $('#demo-video');
  const hero = $('#hero-video');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

  function toast(message) {
    const notification = $('#toast');
    // 原生 dialog 位于浏览器顶层，提示也放入同一层，复制结果才能看见。
    const host = dialog.open ? dialog : document.body;
    if (notification.parentElement !== host) host.append(notification);
    notification.textContent = message;
    notification.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => $('#toast').classList.remove('show'), 2600);
  }

  async function copy(text) {
    try {
      if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(text);
      else {
        // file:// 下也提供复制；系统禁用剪贴板时保留可选中的原命令。
        const input = document.createElement('textarea');
        input.value = text;
        input.style.cssText = 'position:fixed;left:0;top:0;width:1px;height:1px;opacity:0';
        (dialog.open ? dialog : document.body).append(input);
        input.focus(); input.select();
        const copied = document.execCommand('copy');
        input.remove();
        if (!copied) throw new Error('clipboard blocked');
      }
      toast('命令已复制');
    } catch {
      toast('浏览器未允许自动复制，请选中页面上的命令手动复制');
    }
  }
  document.addEventListener('click', event => {
    const button = event.target.closest('[data-copy]');
    if (button) copy(button.dataset.copy);
  });

  const visibleDemos = () => data.demos.filter(demo => filter === 'all' || demo.family === filter);
  function renderCards() {
    $('#task-grid').innerHTML = visibleDemos().map(demo => `
      <article class="task-card">
        <button class="card-media" type="button" data-demo="${demo.id}" aria-label="播放 ${escape(demo.subtitle)}">
          <img src="${demo.poster}" alt="${escape(demo.subtitle)}的真实仿真广角与工件特写" decoding="async" width="640" height="400">
          <span class="card-index">${demo.code} / ${families[demo.family]}</span><span class="play-circle" aria-hidden="true">▶</span>
          <span class="card-record">RECORDED SIMULATION</span><span class="card-duration">${formatTime(demo.duration)}</span>
        </button>
        <div class="task-meta"><span>${escape(demo.subtitle)}</span><span class="${demo.historical ? '' : 'latest'}">${demo.historical ? '历史版本录制' : '最新完整轨迹'} · seed ${demo.seed}</span></div>
        <h3>${escape(demo.title)}</h3><p>${escape(demo.description)}</p>
      </article>`).join('');
  }
  $$('.filters button').forEach(button => button.addEventListener('click', () => {
    filter = button.dataset.filter;
    $$('.filters button').forEach(item => {
      const active = item === button;
      item.classList.toggle('active', active);
      item.setAttribute('aria-pressed', String(active));
    });
    renderCards();
  }));
  $('#task-grid').addEventListener('click', event => {
    const button = event.target.closest('[data-demo]');
    if (button) openDemo(button.dataset.demo, button);
  });

  // 所有分页式组件遵循相同键盘规则：方向键/Home/End 切换并移动焦点。
  function wireTabs(selector, onSelect) {
    const buttons = $$(selector);
    const select = button => {
      buttons.forEach(item => { item.setAttribute('aria-selected', String(item === button)); item.tabIndex = item === button ? 0 : -1; });
      onSelect(button);
    };
    buttons.forEach((button, index) => {
      button.addEventListener('click', () => select(button));
      button.addEventListener('keydown', event => {
        let next;
        if (event.key === 'ArrowRight' || event.key === 'ArrowDown') next = (index + 1) % buttons.length;
        if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') next = (index + buttons.length - 1) % buttons.length;
        if (event.key === 'Home') next = 0;
        if (event.key === 'End') next = buttons.length - 1;
        if (next !== undefined) { event.preventDefault(); select(buttons[next]); buttons[next].focus(); }
      });
    });
    return select;
  }

  const svgShell = content => `<svg class="mechanism-diagram" viewBox="0 0 600 175" role="img" aria-label="物理机制示意，非实时仿真">${content}</svg>`;
  const cableDiagram = svgShell(`<defs><pattern id="grid-cable" width="25" height="25" patternUnits="userSpaceOnUse"><path d="M25 0H0V25" fill="none" stroke="#33412d" stroke-width=".5"/></pattern></defs><rect width="600" height="175" fill="url(#grid-cable)"/><path d="M30 95Q120 16 210 87T390 88Q480 11 570 68" fill="none" stroke="#bde67d" stroke-width="13" opacity=".15"/><path d="M30 95Q120 16 210 87T390 88Q480 11 570 68" fill="none" stroke="#c5f878" stroke-width="2"/>${[[30,95],[72,68],[117,58],[162,65],[210,87],[255,116],[300,126],[345,115],[390,88],[434,64],[480,48],[526,49],[570,68]].map(([x,y])=>`<circle cx="${x}" cy="${y}" r="4" fill="#111913" stroke="#c5f878"/>`).join('')}<path d="M210 52v20m0-20-4 6m4-6 4 6M390 124v-20m0 20-4-6m4 6 4-6" stroke="#f1a561" stroke-width="1.5"/><text x="218" y="48" fill="#f1a561" font-size="9">EI / BENDING</text><text x="66" y="147" fill="#8eaa80" font-size="9">FREE NODES + ELASTIC EDGES</text><text x="425" y="146" fill="#c5f878" font-size="9">EA / AXIAL RESPONSE</text>`);
  const contactDiagram = svgShell(`<path d="M110 142V100Q110 40 160 45l36 26m-86 71h130V100Q240 40 190 45l-25 18" stroke="#799167" stroke-width="9" fill="none"/><path d="M110 142V100Q110 40 160 45l36 26m44 71V100Q240 40 190 45l-25 18" stroke="#c5f878" stroke-width="2" fill="none"/><circle cx="176" cy="109" r="20" fill="#f1a56155" stroke="#f1a561" stroke-width="3"/><path d="M176 5v24m0 0-5-6m5 6 5-6" stroke="#f1a561" stroke-width="2"/><path d="M285 86h55m0 0-7-4m7 4-7 4" stroke="#c5f878"/><path d="M382 139V99Q382 44 419 67l15 15M500 139V99Q500 44 463 67l-15 15" stroke="#c5f878" fill="none" stroke-width="7"/><circle cx="441" cy="110" r="20" fill="#f1a56155" stroke="#f1a561" stroke-width="3"/><path d="M382 142h118" stroke="#799167" stroke-width="7"/><text x="100" y="169" fill="#9baf91" font-size="9">CONTACT / DEFLECTION</text><text x="382" y="169" fill="#c5f878" font-size="9">RECOVERY / RETENTION</text>`);
  const cuttingDiagram = svgShell(`<g transform="translate(66 43)">${Array.from({length:8},(_,x)=>Array.from({length:4},(_,y)=>`<rect x="${x*25}" y="${y*25}" width="22" height="22" fill="${x>2&&x<6&&y<3?'#c5f87822':'#759264'}" stroke="#b0ca9a" stroke-width=".5"/>`).join('')).join('')}</g><path d="M174 5v66" stroke="#e0e6db" stroke-width="12"/><path d="m165 38 17-9m-17 23 17-9m-17 23 17-9" stroke="#759264" stroke-width="2"/><path d="M295 88h50m0 0-7-4m7 4-7 4" stroke="#c5f878"/><path d="M382 51h145v91H382ZM416 51v57h75V51" fill="#69865b33" stroke="#a9c58e" stroke-width="2"/><path d="M425 56h56v42h-56Z" fill="#0b100e" stroke="#c5f878" stroke-dasharray="4 4"/><text x="65" y="167" fill="#a0b193" font-size="9">TOOL SWEPT VOLUME</text><text x="382" y="167" fill="#c5f878" font-size="9">UPDATED STOCK GEOMETRY</text>`);
  const evalDiagram = svgShell(`<path d="M30 40h135v90H30Z" fill="#1c2b18" stroke="#829c6d"/><text x="47" y="66" fill="#9caf8f" font-size="9">PHYSICAL STATE</text><path d="m48 101 22-18 20 30 29-43 26 13" stroke="#c5f878" fill="none" stroke-width="2"/><path d="M185 86h45m0 0-7-4m7 4-7 4" stroke="#c5f878"/><path d="M250 40h130v90H250Z" fill="#1c2b18" stroke="#829c6d"/><text x="268" y="66" fill="#9caf8f" font-size="9">TASK SPEC</text><text x="268" y="92" fill="#c5f878" font-size="11">path + hold</text><text x="268" y="112" fill="#c5f878" font-size="11">+ clearance</text><path d="M399 86h45m0 0-7-4m7 4-7 4" stroke="#c5f878"/><circle cx="512" cy="85" r="43" fill="#c5f87816" stroke="#c5f878"/><path d="m492 85 14 14 26-29" fill="none" stroke="#c5f878" stroke-width="3"/><text x="35" y="164" fill="#8eaa80" font-size="9">CONTACTS / FORCES / GEOMETRY</text><text x="408" y="164" fill="#c5f878" font-size="9">RESULT + SOURCE RECORD</text>`);
  const coreContent = {
    cable: {name:'FLEXIBLE MATERIAL', number:'01 / 04', diagram:cableDiagram, title:'不是一条动画曲线，是会响应受力的线束。', body:'MuJoCo 一维 Flex 表达线束中心线，自由节点与原生弹性边承担轴向响应，能量一致的中心线弯曲力表达 EI。长度、外径、单位长度质量与材料约束由场景配置；抓取和压入通过真实接触发生。', tags:['1D Flex','EA / axial stiffness','EI / bending stiffness','scene-defined parameters']},
    contact: {name:'CONTACT & COMPLIANCE', number:'02 / 04', diagram:contactDiagram, title:'按压、形变、回弹，决定它能否被释放或固定。', body:'MuJoCo 负责刚体动力学、摩擦与接触；降阶弹性结构补充锁舌和线夹的恢复力。夹口受压张开、回弹后保持线束，连接器则需要先压低锁舌才能拔出。当前使用 G1 固定基座双臂与可选末端工具。', tags:['rigid-body dynamics','passive elasticity','frictional grasp','geometry-based interaction']},
    cutting: {name:'MATERIAL REMOVAL', number:'03 / 04', diagram:cuttingDiagram, title:'刀具走过之后，材料占据状态真的改变。', body:'以体素表示毛坯，用刀具扫掠区域更新剩余材料，形成孔、槽和浅腔。平均工艺力与力矩耦合回执行器，支撑结构继续参与接触。该机制用于工序级交互，飞屑是视觉效果，尚不模拟微观断裂与切屑流动。', tags:['voxel stock','tool swept volume','process force coupling','hole / slot / pocket']},
    eval: {name:'TASK SEMANTICS & AUDIT', number:'04 / 04', diagram:evalDiagram, title:'到达一个位置，不等于完成一道工序。', body:'独立 TaskSpec 从物理状态判断夹点留存、顺序、路径、端区、双手退出与稳定保持；约束与审计记录受力、弯曲和碰撞。视频附带 seed、源码指纹与结果记录，把“看到了”连接到“如何复现”。', tags:['independent TaskSpec','semantic success','force / bend / collision','traceable evidence']}
  };
  function renderCore(key) {
    const entry = coreContent[key];
    $('#core-panel').setAttribute('aria-labelledby', `core-tab-${key}`);
    $('#core-panel').innerHTML = `<div class="core-panel-head"><span>${entry.name}</span><span>${entry.number} · 机制示意</span></div>${entry.diagram}<h3>${entry.title}</h3><p>${entry.body}</p><div class="tech-tags">${entry.tags.map(tag=>`<span>${tag}</span>`).join('')}</div>`;
  }
  wireTabs('.core-tabs button', button => renderCore(button.dataset.core));

  const localRecipes = {
    scene: `<span class="code-comment"># 场景：物料参数 + 工位 + 机器人</span>\n<span class="code-key">机器人</span>: G1 双臂 / 可选末端\n<span class="code-key">线束</span>:\n  长度: 按目标路径与供线余量设置\n  外径: 按物料设置\n  单位长度质量: 按物料设置\n  轴向刚度_EA: 材料参数\n  弯曲刚度_EI: 材料参数\n<span class="code-key">工位</span>: 板 / 弹性线夹 / 供线机构\n<span class="code-comment"># 实际键名与单位以仓库内场景为准</span>`,
    task: `<span class="code-comment"># task_spec.py：读取状态，独立判定</span>\n<span class="code-key">观察</span>: 线束形状 / 夹口历史 / 手部位置\n\n<span class="code-key">成功条件</span> = (\n    所有目标夹点有效保持\n    and 夹点顺序与目标路径正确\n    and 线端进入目标区\n    and 双手已经退出\n    and 持续保持达到设定时间\n)\n<span class="code-comment"># 裁判不负责移动或吸附工件</span>`,
    constraints: `<span class="code-comment"># constraints.yaml：冻结物理门槛</span>\n<span class="code-key">材料边界</span>:\n  最小弯曲半径: 20 mm <span class="code-comment"># B3 / B4</span>\n  允许张力: 按物料与场景设置\n<span class="code-key">接触边界</span>:\n  夹口允许穿透: 按场景设置\n  禁止碰撞对: 机器人 / 工位 / 躯干\n<span class="code-key">审计</span>:\n  记录复位与运行过程中的峰值\n  保存越限事件与来源指纹\n<span class="code-comment"># 材料参数仍需后续实物标定</span>`
  };
  const publicRecipes = {
    scene: `<span class="code-comment">SCENE / 场景描述</span>\n\n机器人与末端工具\n物料种类及物理参数\n工位、夹具与目标布局\n\n<span class="code-comment">公开页仅介绍组件分工。</span>\n<span class="code-comment">具体配置与仿真实现暂未发布。</span>`,
    task: `<span class="code-comment">TASK / 任务目标</span>\n\n定义一道工序需要完成什么\n判断物料的最终状态与路径\n检查双臂退出后是否稳定保持\n\n<span class="code-comment">这里展示设计概念，不包含任务源码。</span>`,
    constraints: `<span class="code-comment">CONSTRAINTS / 工业约束</span>\n\n材料允许受力与形变范围\n机器人和工位的接触边界\n工序过程中的越限记录\n\n<span class="code-comment">具体参数与验证实现暂未发布。</span>`
  };
  const recipes = publicPreview ? publicRecipes : localRecipes;
  function renderRecipe(key) {
    $('#recipe-panel').setAttribute('aria-labelledby', `recipe-tab-${key}`);
    $('#recipe-panel').innerHTML = `<pre><code>${recipes[key]}</code></pre>`;
  }
  wireTabs('.code-tabs button', button => renderRecipe(button.dataset.recipe));

  function renderDetail() {
    if (!selected) return;
    const demo = selected;
    const panel = $('#detail-panel');
    panel.setAttribute('aria-labelledby', `detail-tab-${detail}`);
    if (detail === 'task') {
      panel.innerHTML = `<h3>${escape(demo.subtitle)}</h3><p>${escape(demo.description)}</p><ol class="step-list">${demo.steps.map(step=>`<li>${escape(step)}</li>`).join('')}</ol><span class="detail-label">WHAT COUNTS AS DONE</span><p>${escape(demo.criteria)}</p>`;
    } else if (detail === 'mechanism' && publicPreview) {
      panel.innerHTML = `<h3>这一交互如何发生？</h3><p>${escape(demo.mechanism)}</p><span class="detail-label">RESEARCH PREVIEW</span><p>本站公开任务演示与能力概述。仿真器、任务控制器、场景配置与训练相关实现暂未完整公开，后续将结合论文与项目进展安排发布。</p>`;
    } else if (detail === 'mechanism') {
      panel.innerHTML = `<h3>这一交互如何发生？</h3><p>${escape(demo.mechanism)}</p><span class="detail-label">RUN LOCALLY</span><div class="demo-command">${escape(demo.command)}</div><button class="button secondary" data-copy="${escape(demo.command)}" type="button">复制运行命令 ⧉</button><span class="detail-label">SCOPE</span><p>视频由参考脚本控制器产生。运行仿真需要安装项目环境与资产；历史视频可能与当前代码表现不同。转角任务命令已指定本次完成的局部工位。</p>`;
    } else if (publicPreview) {
      const metrics = demo.metrics;
      const retained = metrics.retained_count !== undefined ? `${metrics.retained_count} / ${metrics.clip_count}` : '—';
      panel.innerHTML = `<h3>关于这段演示</h3><p>${demo.historical ? '历史研发版本录制。' : '当前研发版本的完整演示。'} 单条完成记录不代表多种子成功率。</p><dl class="record-grid"><div><dt>录制结果</dt><dd>完成</dd></div><div><dt>Seed</dt><dd>${demo.seed}</dd></div><div><dt>视频长度</dt><dd>${formatTime(demo.duration)}</dd></div><div><dt>夹点保持</dt><dd>${retained}</dd></div></dl><p>完整视频为 1× 时间尺度（包含首尾停留）。当前由参考脚本控制器执行；材料参数尚未实测标定。公开记录仅含演示信息与视频校验值，不包含内部轨迹或源代码。</p><div class="evidence-links"><a href="${demo.evidence}" target="_blank" rel="noopener">查看演示记录 ↗</a><a href="${demo.video}" target="_blank" rel="noopener">单独打开视频 ↗</a></div>`;
    } else {
      const metrics = demo.metrics;
      const retained = metrics.retained_count !== undefined ? `${metrics.retained_count} / ${metrics.clip_count}` : '—';
      panel.innerHTML = `<h3>这段演示来自哪里？</h3><p>${demo.historical ? '历史版本录制，保留来源指纹，便于独立追溯。' : '最新局部工位的 seed 0 完整轨迹。'} 单条完成记录不代表多种子成功率。</p><dl class="record-grid"><div><dt>录制结果</dt><dd>完成</dd></div><div><dt>Seed</dt><dd>${demo.seed}</dd></div><div><dt>视频长度</dt><dd>${formatTime(demo.duration)}</dd></div><div><dt>夹点保持</dt><dd>${retained}</dd></div></dl><span class="detail-label">SOURCE FINGERPRINT</span><div class="source-hash">commit ${escape(demo.commit)}<br>source ${escape(demo.sourceHash)}</div><p>完整视频为 1× 时间尺度（包含首尾停留）。来源记录含视频 SHA-256、仿真时间与当次任务结果；材料参数尚未实测标定。</p><div class="evidence-links"><a href="${demo.evidence}" target="_blank" rel="noopener">查看来源 JSON ↗</a><a href="${demo.video}" target="_blank" rel="noopener">单独打开视频 ↗</a></div>`;
    }
  }
  const selectDetail = wireTabs('.detail-tabs button', button => { detail = button.dataset.detail; renderDetail(); });

  function setRate(rate) {
    video.playbackRate = rate;
    $$('.rate-controls button').forEach(button => {
      const active = Number(button.dataset.rate) === rate;
      button.classList.toggle('active', active); button.setAttribute('aria-pressed', String(active));
    });
  }
  function openDemo(id, trigger) {
    const demo = data.demos.find(item => item.id === id);
    if (!demo) return;
    if (!dialog.open) {
      returnFocus = trigger || document.activeElement;
      dialog.showModal(); document.body.classList.add('modal-open');
      hero.pause();
    }
    selected = demo;
    $('#demo-code').textContent = `${demo.code} / ${families[demo.family]} · RECORDED SIMULATION`;
    $('#demo-title').textContent = demo.title;
    $('#player-caption').textContent = `原始时间尺度 1× · ${demo.historical ? '历史版本' : '最新完整轨迹'} · seed ${demo.seed} · ${formatTime(demo.duration)}（含首尾停留）`;
    $('#video-error').hidden = true;
    video.pause(); video.src = demo.video; video.poster = demo.poster; video.load(); setRate(1);
    selectDetail($('#detail-tab-task'));
    const list = visibleDemos();
    $('#demo-position').textContent = `${String(list.findIndex(item=>item.id===id)+1).padStart(2,'0')} / ${String(list.length).padStart(2,'0')}`;
    // 打开即播放，浏览器限制时保留原生播放控件。
    video.play().catch(() => {});
    $('#dialog-close').focus({preventScroll:true});
  }
  function moveDemo(direction) {
    const list = visibleDemos();
    const index = list.findIndex(item => item.id === selected?.id);
    openDemo(list[(index + direction + list.length) % list.length].id);
  }
  $('#demo-prev').addEventListener('click', () => moveDemo(-1));
  $('#demo-next').addEventListener('click', () => moveDemo(1));
  $('#presentation-start').addEventListener('click', event => openDemo(visibleDemos()[0].id, event.currentTarget));
  $('#dialog-close').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => { if (event.target === dialog) { const r = dialog.getBoundingClientRect(); if (event.clientX<r.left || event.clientX>r.right || event.clientY<r.top || event.clientY>r.bottom) dialog.close(); } });
  dialog.addEventListener('close', () => {
    video.pause(); video.removeAttribute('src'); video.load();
    document.body.classList.remove('modal-open');
    $('#toast').classList.remove('show');
    document.body.append($('#toast'));
    returnFocus?.focus({preventScroll:true});
    if (!reducedMotion.matches && !heroUserPaused) hero.play().catch(()=>{});
  });
  video.addEventListener('error', () => { if (video.getAttribute('src')) $('#video-error').hidden = false; });
  video.addEventListener('ratechange', () => setRateIfChanged());
  function setRateIfChanged() {
    $$('.rate-controls button').forEach(button => {
      const active = Number(button.dataset.rate) === video.playbackRate;
      button.classList.toggle('active', active); button.setAttribute('aria-pressed', String(active));
    });
  }
  $$('.rate-controls button').forEach(button => button.addEventListener('click', () => setRate(Number(button.dataset.rate))));
  $('#fullscreen-button').addEventListener('click', async () => {
    try { if (document.fullscreenElement) await document.exitFullscreen(); else if (video.requestFullscreen) await video.requestFullscreen(); else if (video.webkitEnterFullscreen) video.webkitEnterFullscreen(); else toast('当前浏览器未提供全屏接口，可使用视频原生控件'); }
    catch { toast('可通过视频原生控件进入全屏'); }
  });
  dialog.addEventListener('keydown', event => {
    const interactive = event.target.closest('button,a,video,input,textarea,[role="tab"]');
    if (!interactive && event.key === 'ArrowRight') { event.preventDefault(); moveDemo(1); }
    if (!interactive && event.key === 'ArrowLeft') { event.preventDefault(); moveDemo(-1); }
    if (!interactive && event.code === 'Space') { event.preventDefault(); if (video.paused) video.play().catch(()=>{}); else video.pause(); }
  });

  let heroUserPaused = reducedMotion.matches;
  function updateHeroButton() { $('#hero-toggle').textContent = hero.paused ? '▷' : 'Ⅱ'; $('#hero-toggle').setAttribute('aria-label', hero.paused ? '播放首页短片' : '暂停首页短片'); }
  $('#hero-toggle').addEventListener('click', () => {
    heroUserPaused = !hero.paused;
    if (hero.paused) hero.play().catch(()=>{}); else hero.pause();
  });
  hero.addEventListener('play', updateHeroButton); hero.addEventListener('pause', updateHeroButton);
  if (reducedMotion.matches) { hero.autoplay = false; hero.pause(); }
  document.addEventListener('visibilitychange', () => { if (document.hidden) { hero.pause(); video.pause(); } else if (!dialog.open && !heroUserPaused) hero.play().catch(()=>{}); });
  renderCards(); renderCore('cable'); renderRecipe('scene'); updateHeroButton();
})();
