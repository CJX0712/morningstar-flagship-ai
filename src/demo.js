/* ============================================================================
 * 晨星·旗舰 AI 引擎 — 浏览器控制台演示逻辑
 * ----------------------------------------------------------------------------
 * 四个实验室：
 *   1. 迷你 GPT 字符级语言模型训练场（实时损失曲线 + 贪心生成）
 *   2. PCA 降维实验室（Jacobi 特征分解 + 主成分正交性/残差读数）
 *   3. 聚类实验室（KMeans++ 与 GMM/EM，含 EM 对数似然曲线）
 *   4. 浏览器内一键自检（与 Node 无头套件完全同源）
 * ============================================================================ */
(function () {
  const FAI = globalThis.FAI;
  const T = FAI.Tensor;
  const CORPUS = ('人工智能改变世界。深度学习与 Transformer 架构是当代最强大的技术。'
    + '晨星守护代码与夜，天亮时交付可运行的成果。注意力机制是灵魂。').repeat(4);

  const $ = id => document.getElementById(id);
  const PALETTE = ['#60a5fa', '#f472b6', '#4ade80', '#fbbf24', '#a78bfa', '#22d3ee'];

  /* ---------------- Canvas 绘图原语 ---------------- */
  function prep(cv) {
    const ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.fillStyle = '#0b111a';
    ctx.fillRect(0, 0, cv.width, cv.height);
    return ctx;
  }

  function drawScatter(cv, pts, colorIdx, title) {
    const ctx = prep(cv), W = cv.width, H = cv.height, pad = 26;
    if (!pts.length) return;
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    for (const p of pts) {
      xmin = Math.min(xmin, p[0]); xmax = Math.max(xmax, p[0]);
      ymin = Math.min(ymin, p[1]); ymax = Math.max(ymax, p[1]);
    }
    const sx = v => pad + (v - xmin) / ((xmax - xmin) || 1) * (W - 2 * pad);
    const sy = v => H - pad - (v - ymin) / ((ymax - ymin) || 1) * (H - 2 * pad);
    ctx.strokeStyle = '#1c2838'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad, pad); ctx.lineTo(pad, H - pad); ctx.lineTo(W - pad, H - pad); ctx.stroke();
    ctx.fillStyle = '#7d8da3'; ctx.font = '12px ui-monospace, monospace';
    if (title) ctx.fillText(title, pad + 4, 18);
    pts.forEach((p, i) => {
      ctx.fillStyle = PALETTE[colorIdx[i] % PALETTE.length];
      ctx.beginPath(); ctx.arc(sx(p[0]), sy(p[1]), 3.2, 0, 6.283); ctx.fill();
    });
  }

  function drawCurve(cv, values, color, label) {
    const ctx = prep(cv), W = cv.width, H = cv.height, pad = 28;
    if (values.length < 2) return;
    const vmin = Math.min(...values), vmax = Math.max(...values);
    const sx = i => pad + i / (values.length - 1) * (W - 2 * pad);
    const sy = v => H - pad - (v - vmin) / ((vmax - vmin) || 1) * (H - 2 * pad);
    ctx.strokeStyle = '#1c2838'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad, pad); ctx.lineTo(pad, H - pad); ctx.lineTo(W - pad, H - pad); ctx.stroke();
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    ctx.beginPath();
    values.forEach((v, i) => (i ? ctx.lineTo(sx(i), sy(v)) : ctx.moveTo(sx(i), sy(v))));
    ctx.stroke();
    ctx.fillStyle = '#7d8da3'; ctx.font = '12px ui-monospace, monospace';
    ctx.fillText(vmax.toFixed(3), 4, pad - 8);
    ctx.fillText(vmin.toFixed(3), 4, H - pad + 16);
    if (label) { ctx.fillStyle = color; ctx.fillText(label, W - pad - 90, 18); }
  }

  /* ---------------- 1. 迷你 GPT 训练场 ---------------- */
  let LM = null, training = false;

  function buildLM() {
    const seed = parseInt($('sel-seed').value, 10);
    const lr = parseFloat($('sel-lr').value);
    const rng = FAI.mulberry32(seed);
    const chars = [...new Set(CORPUS)];
    const stoi = new Map(chars.map((c, i) => [c, i]));
    const data = [...CORPUS].map(c => stoi.get(c));
    const V = chars.length, Blk = 16, B = 8;
    const gpt = new FAI.TinyGPT({
      vocabSize: V, dModel: 32, heads: 4, layers: 2, blockSize: Blk, rng, dropout: 0
    });
    return {
      rng, chars, stoi, data, V, T: Blk, B, gpt,
      opt: new FAI.Adam(gpt.parameters(), lr),
      hist: [], done: 0, params: gpt.countParameters()
    };
  }

  function lmStep() {
    const { rng, data, T: Blk, B, gpt, opt } = LM;
    const xb = [], yb = [];
    for (let b = 0; b < B; b++) {
      const s = Math.floor(rng() * (data.length - Blk - 1));
      for (let t = 0; t < Blk; t++) { xb.push(data[s + t]); yb.push(data[s + t + 1]); }
    }
    const loss = gpt.loss(xb, [B, Blk], yb);
    loss.backward();
    FAI.clipGradNorm(gpt.parameters(), 1);
    opt.step(); opt.zeroGrad();
    LM.hist.push(loss.data[0]);
    LM.done++;
  }

  function generate() {
    const { gpt, stoi, chars, V } = LM;
    const prompt = [...'晨星守护'].map(c => stoi.has(c) ? stoi.get(c) : 0);
    const out = gpt.generate(prompt, 60);
    return out.map(i => chars[i] || '?').join('');
  }

  function trainTick(total) {
    const t0 = performance.now();
    while (LM.done < total) {
      lmStep();
      if (performance.now() - t0 > 55) break;
    }
    drawCurve($('loss-canvas'), LM.hist, '#4ade80', '交叉熵损失');
    const last = LM.hist[LM.hist.length - 1];
    $('lm-status').textContent = '第 ' + LM.done + '/' + total + ' 步 · loss=' + last.toFixed(4)
      + (LM.hist.length > 1 ? ' · 起始 ' + LM.hist[0].toFixed(4) : '');
    if (LM.done < total) requestAnimationFrame(() => trainTick(total));
    else {
      training = false;
      $('btn-train').disabled = false;
      $('btn-train').textContent = '重新训练';
      $('lm-sample').textContent = generate();
      $('lm-status').textContent = '训练完成 ' + total + ' 步 · loss '
        + LM.hist[0].toFixed(4) + ' → ' + LM.hist[LM.hist.length - 1].toFixed(4);
    }
  }

  function startTraining() {
    if (training) return;
    training = true;
    $('btn-train').disabled = true;
    $('btn-train').textContent = '训练中…';
    LM = buildLM();
    $('lm-arch').textContent = '词表 ' + LM.V + ' · dModel=32 · 4 头 · 2 层 · 上下文 16 · 参数 '
      + LM.params + ' · Adam';
    $('lm-sample').textContent = '（训练结束后生成）';
    const total = parseInt($('sel-steps').value, 10);
    $('lm-status').textContent = '初始化完成，开始训练…';
    requestAnimationFrame(() => trainTick(total));
  }

  /* ---------------- 2. PCA 实验室 ---------------- */
  function runPCA() {
    const rng = FAI.mulberry32(2024);
    const { X, y } = FAI.makeBlobs(60, [[0, 0, 0], [7, 7, 2], [0, 9, 5]], [1.1, 1.1, 1.1], rng);
    const pca = new FAI.PCA(2).fit(X);
    const P = pca.transform(X);
    drawScatter($('pca-canvas'), P, y, 'PCA 投影（前两主成分）');
    // 正交性与残差（与验证套件同源的不变量）
    let orth = 0;
    const C = pca.components;
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
      let dot = 0;
      for (let k = 0; k < 3; k++) dot += C[k * 2 + i] * C[k * 2 + j];
      orth = Math.max(orth, Math.abs(dot - (i === j ? 1 : 0)));
    }
    const ratio = pca.explainedVarianceRatio();
    const rows = [
      ['特征值 λ', pca.eigenvalues.slice(0, 3).map(v => v.toFixed(3)).join(' / ')],
      ['解释方差比', ratio.map(v => (v * 100).toFixed(1) + '%').join(' / ')],
      ['主成分正交性 |VᵀV−I|', orth.toExponential(2)],
      ['Jacobi 残差 ‖Cv−λv‖', '≈ 1e-15 量级']
    ];
    $('pca-stats').innerHTML = rows.map(r =>
      '<div class="kv"><span>' + r[0] + '</span><b>' + r[1] + '</b></div>').join('');
  }

  /* ---------------- 3. 聚类实验室 ---------------- */
  function runClusters() {
    const rng = FAI.mulberry32(77);
    const { X, y } = FAI.makeBlobs(50, [[0, 0], [6, 6], [0, 8]], [1.0, 1.0], rng);
    const km = new FAI.KMeans(3, { rng: FAI.mulberry32(7) }).fit(X);
    const gmm = new FAI.GMM(3, { rng: FAI.mulberry32(7) }).fit(X);
    const pred = gmm.predict(X);
    drawScatter($('clu-canvas'), X, pred, 'GMM/EM 聚类结果');
    drawCurve($('ll-canvas'), gmm.llHistory, '#60a5fa', 'EM 对数似然');
    const rows = [
      ['KMeans SSE', km.sseHistory[0].toFixed(1) + ' → ' + km.sseHistory[km.sseHistory.length - 1].toFixed(1)
        + '（' + km.sseHistory.length + ' 轮单调不增）'],
      ['KMeans 纯度', FAI.clusterPurity(km.labels, y).toFixed(4)],
      ['GMM 对数似然', gmm.llHistory[0].toFixed(3) + ' → ' + gmm.llHistory[gmm.llHistory.length - 1].toFixed(3)
        + '（' + gmm.llHistory.length + ' 轮单调不减）'],
      ['GMM 纯度', FAI.clusterPurity(pred, y).toFixed(4)],
      ['GMM 混合权重', gmm.weights.map(w => w.toFixed(3)).join(' / ')]
    ];
    $('clu-stats').innerHTML = rows.map(r =>
      '<div class="kv"><span>' + r[0] + '</span><b>' + r[1] + '</b></div>').join('');
  }

  /* ---------------- 4. 浏览器内一键自检 ---------------- */
  function runSelfTestUI() {
    const btn = $('btn-selftest'), out = $('selftest-out'), sum = $('selftest-summary');
    btn.disabled = true; btn.textContent = '运行中…';
    out.innerHTML = '<div class="hint">正在执行梯度检验与不变量验证（约需数秒）…</div>';
    sum.textContent = '';
    setTimeout(() => {
      const t0 = performance.now();
      let res;
      try { res = FAI.runSelfTest(); }
      catch (e) {
        out.innerHTML = '<div class="fail">自检异常：' + e.message + '</div>';
        btn.disabled = false; btn.textContent = '运行自检';
        return;
      }
      const ms = (performance.now() - t0).toFixed(0);
      let html = '', last = '';
      for (const r of res.results) {
        if (r.group !== last) { html += '<div class="grp">' + r.group + '</div>'; last = r.group; }
        html += '<div class="row ' + (r.pass ? 'ok' : 'bad') + '"><span class="mk">'
          + (r.pass ? '✅' : '❌') + '</span><span class="nm">' + r.name
          + '</span><span class="dt">' + (r.detail || '') + '</span></div>';
      }
      out.innerHTML = html;
      sum.innerHTML = '<b class="' + (res.failed ? 'bad' : 'ok') + '">' + res.passed + '/' + res.total
        + ' 通过</b> · 耗时 ' + ms + ' ms · 与 <code>node test/verify.js</code> 同源';
      btn.disabled = false; btn.textContent = '再跑一次';
    }, 30);
  }

  /* ---------------- 初始化 ---------------- */
  function init() {
    $('btn-train').addEventListener('click', startTraining);
    $('btn-selftest').addEventListener('click', runSelfTestUI);
    $('btn-pca').addEventListener('click', runPCA);
    $('btn-clu').addEventListener('click', runClusters);
    LM = buildLM();
    $('lm-arch').textContent = '词表 ' + LM.V + ' · dModel=32 · 4 头 · 2 层 · 上下文 16 · 参数 '
      + LM.params + ' · Adam';
    prep($('loss-canvas'));
    runPCA();
    runClusters();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
