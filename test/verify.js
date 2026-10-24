/* ============================================================================
 * 晨星·旗舰 AI — 无头验证套件（Headless Verification Suite）
 * ----------------------------------------------------------------------------
 * 验证总纲：每个算法至少有一条可被独立实现交叉验证的「不变量」——
 *   · 解析梯度 vs 中心有限差分（相对误差 ≤ 1e-6 ~ 1e-5）
 *   · Jacobi 残差 ‖Av − λv‖ ≈ 机器精度
 *   · KMeans SSE 逐轮单调不增；GMM 对数似然单调不减（EM 定理）
 *   · 概率行和 = 1；softmax 行和 = 1；logsumexp 大数域不溢出
 *   · 同种子 => 同结果（完全确定性）
 *
 * 运行：node test/verify.js     （退出码 0 = 全绿）
 * ============================================================================ */
(function () {
  const G = typeof globalThis !== 'undefined' ? globalThis : this;
  const FAI = G.FAI || (G.FAI = {});
  // Node 环境：按依赖序加载全部源码（浏览器由 <script> 标签依序加载）
  if (typeof module === 'object' && module.exports && typeof require === 'function') {
    require('../src/engine.js');
    require('../src/utils.js');
    require('../src/losses.js');
    require('../src/optim.js');
    require('../src/modules.js');
    require('../src/classical.js');
  }
  const Tensor = FAI.Tensor;

  /* ---------------- 结果收集 ---------------- */
  const results = [];
  function check(group, name, pass, detail) {
    results.push({ group, name, pass: !!pass, detail: detail || '' });
  }
  /** 任意形状（含 0 秩标量）-> 标量：逐轴归约 */
  const sumAll = t => { let s = t; while (s.rank > 0) s = s.sum(0); return s; };
  const sqSum = t => sumAll(t.mul(t));   // 任意形状 -> 标量（平方和）

  /* ---------------- 语料（字符级迷你语言模型） ---------------- */
  const CORPUS = ('人工智能改变世界。深度学习与 Transformer 架构是当代最强大的技术。'
    + '晨星守护代码与夜，天亮时交付可运行的成果。注意力机制是灵魂。').repeat(3);

  function trainGPT(seed, steps) {
    const rng = FAI.mulberry32(seed);
    const chars = [...new Set(CORPUS)];
    const stoi = new Map(chars.map((c, i) => [c, i]));
    const data = [...CORPUS].map(c => stoi.get(c));
    const V = chars.length, T = 8, B = 8;
    const gpt = new FAI.TinyGPT({ vocabSize: V, dModel: 24, heads: 3, layers: 2, blockSize: T, rng });
    const opt = new FAI.Adam(gpt.parameters(), 5e-3);
    const hist = [];
    for (let s = 0; s < steps; s++) {
      const xb = [], yb = [];
      for (let b = 0; b < B; b++) {
        const start = Math.floor(rng() * (data.length - T - 1));
        for (let t = 0; t < T; t++) { xb.push(data[start + t]); yb.push(data[start + t + 1]); }
      }
      const loss = gpt.loss(xb, [B, T], yb);
      loss.backward();
      FAI.clipGradNorm(gpt.parameters(), 1.0);
      opt.step(); opt.zeroGrad();
      hist.push(loss.data[0]);
    }
    return { hist, gpt, chars, stoi };
  }

  /* ================================================================ *
   * A. 梯度检验：解析梯度 vs 中心有限差分                              *
   * ================================================================ */
  function testGradients() {
    const G1 = '梯度检验';
    let n = 0;
    const T1 = FAI.mulberry32(101);

    // A1 Linear
    {
      const lin = new FAI.Linear(3, 2, { rng: T1 });
      const x = Tensor.randn([4, 3], T1);
      const r = FAI.gradCheck(lin.parameters(), () => sqSum(lin.forward(x)));
      check(G1, 'Linear 层 (3→2)', r.ok, 'max|Δ|=' + r.maxAbs.toExponential(2) + ' rel=' + r.maxRel.toExponential(2));
      n++;
    }
    // A2 广播四则运算
    {
      const a = Tensor.randn([4, 3], T1); a.requiresGrad = true;
      const b = Tensor.randn([1, 3], T1); b.requiresGrad = true;
      const c = Tensor.randn([4, 1], T1); c.requiresGrad = true;
      const makeLoss = () => a.mul(b).add(c).sub(b).div(Tensor.constant(3, [1])).neg();
      const r = FAI.gradCheck([a, b, c], () => sqSum(makeLoss()));
      check(G1, '广播四则运算 (4,3)×(1,3)/(4,1)', r.ok, 'max|Δ|=' + r.maxAbs.toExponential(2) + ' rel=' + r.maxRel.toExponential(2));
      n++;
    }
    // A3 规约 sum/mean
    {
      const a = Tensor.randn([3, 5, 4], T1); a.requiresGrad = true;
      const makeLoss = () => sumAll(a.sum(0).mul(a.mean(0))).add(sumAll(a.mean(2, true)));
      const r = FAI.gradCheck([a], makeLoss);
      check(G1, 'sum/mean 多轴规约', r.ok, 'max|Δ|=' + r.maxAbs.toExponential(2) + ' rel=' + r.maxRel.toExponential(2));
      n++;
    }
    // A4 元素级函数
    {
      const fns = {
        tanh: t => t.tanh(), relu: t => t.relu(), sigmoid: t => t.sigmoid(),
        exp: t => t.exp(), sqrt: t => t.sqrt(), log: t => t.log(),
      };
      for (const name in fns) {
        const x = Tensor.randn([6, 5], T1).add(Tensor.constant(3, [1]));
        x.requiresGrad = true;
        const r = FAI.gradCheck([x], () => sqSum(fns[name](x)));
        check(G1, '元素级 ' + name, r.ok, 'max|Δ|=' + r.maxAbs.toExponential(2) + ' rel=' + r.maxRel.toExponential(2));
        n++;
      }
    }
    // A5 matmul + reshape + permute
    {
      const a = Tensor.randn([4, 3], T1); a.requiresGrad = true;
      const b = Tensor.randn([3, 2], T1); b.requiresGrad = true;
      const e = Tensor.randn([2, 3], T1); e.requiresGrad = true;
      const makeLoss = () => {
        const c = a.matmul(b);                    // (4,2)
        const d = c.reshape([2, 4]).permute([1, 0]); // (4,2)
        return sqSum(d.matmul(e));                // (4,3)
      };
      const r = FAI.gradCheck([a, b, e], makeLoss);
      check(G1, 'matmul+reshape+permute 链', r.ok, 'max|Δ|=' + r.maxAbs.toExponential(2) + ' rel=' + r.maxRel.toExponential(2));
      n++;
    }
    // A6 logsumexp / softmax
    {
      const x = Tensor.randn([5, 7], T1); x.requiresGrad = true;
      const r = FAI.gradCheck([x], () => sqSum(x.logsumexp()));
      check(G1, 'logsumexp', r.ok, 'max|Δ|=' + r.maxAbs.toExponential(2) + ' rel=' + r.maxRel.toExponential(2));
      n++;
      const x2 = Tensor.randn([5, 7], T1); x2.requiresGrad = true;
      const r2 = FAI.gradCheck([x2], () => sqSum(FAI.softmax(x2)));
      check(G1, 'softmax（logsumexp 复合）', r2.ok, 'max|Δ|=' + r2.maxAbs.toExponential(2) + ' rel=' + r2.maxRel.toExponential(2));
      n++;
    }
    // A7 MSE / CE
    {
      const W = Tensor.randn([6, 4], T1); W.requiresGrad = true;
      const x = Tensor.randn([5, 6], T1);
      const tg = [0, 1, 2, 3, 0];
      const makeLoss = () => FAI.crossEntropy(x.matmul(W), tg);
      const r = FAI.gradCheck([W], makeLoss);
      check(G1, '交叉熵损失（含 logsumexp 反传）', r.ok, 'max|Δ|=' + r.maxAbs.toExponential(2) + ' rel=' + r.maxRel.toExponential(2));
      n++;
      const W2 = Tensor.randn([6, 4], T1); W2.requiresGrad = true;
      const tgt = Tensor.randn([5, 4], T1);
      const r2 = FAI.gradCheck([W2], () => FAI.mse(x.matmul(W2), tgt));
      check(G1, 'MSE 损失', r2.ok, 'max|Δ|=' + r2.maxAbs.toExponential(2) + ' rel=' + r2.maxRel.toExponential(2));
      n++;
    }
    // A8 LayerNorm
    {
      const ln = new FAI.LayerNorm(4);
      const x = Tensor.randn([3, 5, 4], T1);
      const r = FAI.gradCheck(ln.parameters(), () => sqSum(ln.forward(x)));
      check(G1, 'LayerNorm', r.ok, 'max|Δ|=' + r.maxAbs.toExponential(2) + ' rel=' + r.maxRel.toExponential(2));
      n++;
    }
    // A9 GELU
    {
      const x = Tensor.randn([5, 6], T1); x.requiresGrad = true;
      const r = FAI.gradCheck([x], () => sqSum(FAI.gelu(x)));
      check(G1, 'GELU（tanh 近似）', r.ok, 'max|Δ|=' + r.maxAbs.toExponential(2) + ' rel=' + r.maxRel.toExponential(2));
      n++;
    }
    // A10 多头注意力（因果掩码）
    {
      const attn = new FAI.MultiHeadAttention(8, 2, { rng: T1, mask: Tensor.causalMask(5) });
      const x = Tensor.randn([2, 5, 8], T1);
      const r = FAI.gradCheck(attn.parameters(), () => sqSum(attn.forward(x)));
      check(G1, 'MultiHeadAttention（因果掩码）', r.ok, 'max|Δ|=' + r.maxAbs.toExponential(2) + ' rel=' + r.maxRel.toExponential(2));
      n++;
    }
    // A11 Transformer 块
    {
      const blk = new FAI.TransformerBlock(8, 2, { rng: T1, mask: Tensor.causalMask(5) });
      const x = Tensor.randn([2, 5, 8], T1);
      const r = FAI.gradCheck(blk.parameters(), () => sqSum(blk.forward(x)));
      check(G1, 'TransformerBlock（Pre-LN 残差）', r.ok, 'max|Δ|=' + r.maxAbs.toExponential(2) + ' rel=' + r.maxRel.toExponential(2));
      n++;
    }
    // A12 TinyGPT 端到端
    {
      const rng = FAI.mulberry32(7);
      const gpt = new FAI.TinyGPT({ vocabSize: 11, dModel: 16, heads: 2, layers: 1, blockSize: 8, rng });
      const idx = Array.from({ length: 16 }, () => Math.floor(rng() * 11));
      const tg = idx.map(i => (i + 1) % 11);
      const makeLoss = () => gpt.loss(idx, [2, 8], tg);
      const r = FAI.gradCheck(gpt.parameters(), makeLoss);
      check(G1, 'TinyGPT 端到端（嵌入→注意力→损失）', r.ok, 'max|Δ|=' + r.maxAbs.toExponential(2) + ' rel=' + r.maxRel.toExponential(2));
      n++;
    }
    // A13 注意力 K 偏置平移不变性（数学不变量）
    // softmax 对整行加常数不变：scores = Q(K + 1δᵀ)ᵀ = QKᵀ + (q·δ)1ᵀ
    // => 注意力权重不变 => ∂L/∂b_K = 0（而 ∂L/∂b_V、∂L/∂b_Q ≠ 0）
    {
      const rng = FAI.mulberry32(19);
      const attn = new FAI.MultiHeadAttention(8, 2, { rng });
      const x = Tensor.randn([2, 5, 8], rng);
      sqSum(attn.forward(x)).backward();
      const absMax = a => (a ? Math.max(...Array.from(a).map(Math.abs)) : NaN);
      const gK = absMax(attn.Wk.b.grad), gV = absMax(attn.Wv.b.grad), gQ = absMax(attn.Wq.b.grad);
      check(G1, '注意力 K 偏置平移不变性 ∂L/∂b_K=0', gK < 1e-12 && gV > 1e-6 && gQ > 1e-6,
        '|∂L/∂b_K|=' + gK.toExponential(2) + ' 对照 |∂L/∂b_V|=' + gV.toExponential(2) + ' |∂L/∂b_Q|=' + gQ.toExponential(2));
      n++;
    }
  }

  /* ================================================================ *
   * B. 数值稳定性                                                     *
   * ================================================================ */
  function testStability() {
    const G1 = '数值稳定性';
    // B1 logsumexp 大数域
    {
      const t = Tensor.of2D([[10000, 10001]]);
      const lse = t.logsumexp().data[0];
      const expect = 10001 + Math.log(1 + Math.exp(-1));
      check(G1, 'logsumexp 大数域不溢出', isFinite(lse) && Math.abs(lse - expect) < 1e-9,
        'lse=' + lse.toFixed(6) + ' 期望=' + expect.toFixed(6));
    }
    // B2 softmax 行和 = 1（大激励）
    {
      const x = Tensor.randn([6, 9], FAI.mulberry32(3)).mul(Tensor.constant(100, [1]));
      const s = FAI.softmax(x);
      let maxDev = 0;
      for (let r = 0; r < 6; r++) {
        let sum = 0;
        for (let c = 0; c < 9; c++) sum += s.data[r * 9 + c];
        maxDev = Math.max(maxDev, Math.abs(sum - 1));
      }
      check(G1, 'softmax 行和 = 1（激励 ×100）', maxDev < 1e-12 && !isNaN(s.data[0]),
        'maxDev=' + maxDev.toExponential(2));
    }
    // B3 大 logits 交叉熵有限
    {
      const logits = Tensor.randn([4, 5], FAI.mulberry32(4)).mul(Tensor.constant(1000, [1]));
      const l = FAI.crossEntropy(logits, [0, 1, 2, 3]);
      check(G1, '交叉熵大 logits 无 NaN', isFinite(l.data[0]), 'loss=' + l.data[0].toFixed(4));
    }
    // B4 因果掩码形状正确
    {
      const m = Tensor.causalMask(4);   // 索引 = i*4 + j
      const ok = m.data[0] === 0 && m.data[3] === -1e9 && m.data[12] === 0 && m.data[15] === 0 && m.data[5] === 0;
      check(G1, '因果掩码上三角为 -1e9', ok, '[0,3]=-1e9 [1,0]=0 [3,3]=0');
    }
  }

  /* ================================================================ *
   * C. 优化器与梯度治理                                               *
   * ================================================================ */
  function testOptimizers() {
    const G1 = '优化器';
    // C1 Adam 收敛
    {
      const p = Tensor.zeros([1]); p.requiresGrad = true;
      const opt = new FAI.Adam([p], 0.1);
      for (let i = 0; i < 500; i++) {
        const d = p.sub(Tensor.constant(3, [1]));
        const l = d.mul(d).sum(0);
        l.backward(); opt.step(); opt.zeroGrad();
      }
      check(G1, 'Adam 收敛 (x−3)²', Math.abs(p.data[0] - 3) < 1e-3, 'x=' + p.data[0].toFixed(8));
    }
    // C2 SGD + 动量收敛
    {
      const p = Tensor.zeros([1]); p.requiresGrad = true;
      const opt = new FAI.SGD([p], 0.05, { momentum: 0.9 });
      for (let i = 0; i < 500; i++) {
        const d = p.sub(Tensor.constant(3, [1]));
        const l = d.mul(d).sum(0);
        l.backward(); opt.step(); opt.zeroGrad();
      }
      check(G1, 'SGD+动量收敛 (x−3)²', Math.abs(p.data[0] - 3) < 1e-2, 'x=' + p.data[0].toFixed(8));
    }
    // C3 AdamW 解耦权重衰减
    {
      const p = Tensor.ones([1]); p.requiresGrad = true;
      const opt = new FAI.AdamW([p], 0.1, { weightDecay: 0.5 });
      for (let i = 0; i < 100; i++) {
        const l = sqSum(p);
        l.backward(); opt.step(); opt.zeroGrad();
      }
      check(G1, 'AdamW 权重衰减趋零', Math.abs(p.data[0]) < 0.2, '|x|=' + Math.abs(p.data[0]).toFixed(6));
    }
    // C4 梯度裁剪
    {
      const a = Tensor.zeros([1]); a.requiresGrad = true; a.grad = Float64Array.from([3]);
      const b = Tensor.zeros([1]); b.requiresGrad = true; b.grad = Float64Array.from([4]);
      const before = FAI.clipGradNorm([a, b], 1);
      const after = Math.hypot(a.grad[0], b.grad[0]);
      check(G1, 'clipGradNorm 范数裁剪', Math.abs(before - 5) < 1e-12 && Math.abs(after - 1) < 1e-12,
        'before=' + before.toFixed(4) + ' after=' + after.toFixed(4));
    }
  }

  /* ================================================================ *
   * D. 经典机器学习不变量                                             *
   * ================================================================ */
  function testClassical() {
    const G1 = '经典算法';
    const rng = FAI.mulberry32(7);
    const { X, y } = FAI.makeBlobs(40, [[0, 0, 0], [5, 5, 1], [0, 6, 3]], [0.8, 0.8, 0.8], rng);

    // D1 PCA：特征向量正交归一
    {
      const pca = new FAI.PCA(2).fit(X);
      const C = pca.components; // 3×2
      let maxDev = 0;
      for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
        let dot = 0;
        for (let k = 0; k < 3; k++) dot += C[k * 2 + i] * C[k * 2 + j];
        maxDev = Math.max(maxDev, Math.abs(dot - (i === j ? 1 : 0)));
      }
      check(G1, 'PCA 主成分正交归一', maxDev < 1e-10, 'maxDev=' + maxDev.toExponential(2));
    }
    // D2 PCA：Jacobi 残差 ‖Cv − λv‖
    {
      const pca = new FAI.PCA(2).fit(X);
      // 重建协方差做残差检验
      const n = X.length, d = 3;
      const mu = pca.mean;
      const cov = new Float64Array(d * d);
      for (const r of X) {
        const rc = r.map((v, j) => v - mu[j]);
        for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) cov[i * d + j] += rc[i] * rc[j] / (n - 1);
      }
      let maxRes = 0;
      for (let c = 0; c < 2; c++) {
        const lam = pca.eigenvalues[c];
        const v = Array.from({ length: d }, (_, k) => pca.components[k * 2 + c]);
        for (let i = 0; i < d; i++) {
          let s = 0;
          for (let j = 0; j < d; j++) s += cov[i * d + j] * v[j];
          maxRes = Math.max(maxRes, Math.abs(s - lam * v[i]));
        }
      }
      check(G1, 'PCA Jacobi 残差 ‖Cv−λv‖≈ε', maxRes < 1e-8, 'maxRes=' + maxRes.toExponential(2));
    }
    // D3 PCA：重建误差随 k 下降
    {
      const p1 = new FAI.PCA(1).fit(X), p2 = new FAI.PCA(2).fit(X);
      const err = p => {
        const R = p.reconstruct(X);
        let s = 0;
        for (let i = 0; i < X.length; i++) for (let j = 0; j < 3; j++) s += (R[i][j] - X[i][j]) ** 2;
        return s / X.length;
      };
      const e1 = err(p1), e2 = err(p2);
      check(G1, 'PCA 重建误差 k=2 < k=1', e2 < e1, 'err(k=1)=' + e1.toFixed(4) + ' err(k=2)=' + e2.toFixed(4));
    }
    // D4 KMeans：SSE 单调不增
    {
      const km = new FAI.KMeans(3, { rng: FAI.mulberry32(11), maxIter: 50 }).fit(X);
      const h = km.sseHistory;
      let mono = true;
      for (let i = 1; i < h.length; i++) if (h[i] > h[i - 1] + 1e-9 * Math.max(1, h[i - 1])) mono = false;
      check(G1, 'KMeans SSE 逐轮单调不增', mono && h.length >= 2,
        '轮数=' + h.length + ' SSE ' + h[0].toFixed(2) + '→' + h[h.length - 1].toFixed(2));
    }
    // D5 KMeans：团簇纯度
    {
      const km = new FAI.KMeans(3, { rng: FAI.mulberry32(11) }).fit(X);
      const purity = FAI.clusterPurity(km.labels, y);
      check(G1, 'KMeans 团簇纯度 ≥ 0.95', purity >= 0.95, 'purity=' + purity.toFixed(4));
    }
    // D6 GMM：对数似然单调不减
    {
      const gmm = new FAI.GMM(3, { rng: FAI.mulberry32(13), maxIter: 60 }).fit(X);
      const h = gmm.llHistory;
      let mono = true;
      for (let i = 1; i < h.length; i++) if (h[i] < h[i - 1] - 1e-8) mono = false;
      check(G1, 'GMM 对数似然单调不减（EM 定理）', mono && h.length >= 2,
        '轮数=' + h.length + ' LL ' + h[0].toFixed(4) + '→' + h[h.length - 1].toFixed(4));
    }
    // D7 GMM：责任度行和 = 1
    {
      const gmm = new FAI.GMM(3, { rng: FAI.mulberry32(13), maxIter: 30 }).fit(X);
      const P = gmm.predictProba(X);
      let maxDev = 0;
      for (const row of P) {
        const s = row.reduce((a, b) => a + b, 0);
        maxDev = Math.max(maxDev, Math.abs(s - 1));
      }
      check(G1, 'GMM 责任度每行和 = 1', maxDev < 1e-12, 'maxDev=' + maxDev.toExponential(2));
    }
    // D8 GMM：团簇纯度
    {
      const gmm = new FAI.GMM(3, { rng: FAI.mulberry32(13), maxIter: 60 }).fit(X);
      const pred = gmm.predict(X);
      const purity = FAI.clusterPurity(pred, y);
      check(G1, 'GMM 团簇纯度 ≥ 0.9', purity >= 0.9, 'purity=' + purity.toFixed(4));
    }
    // D9 高斯朴素贝叶斯
    {
      const nb = new FAI.GaussianNB().fit(X, y);
      const pred = nb.predict(X);
      const acc = FAI.accuracy(pred, y);
      const P = nb.predictProba(X);
      let maxDev = 0;
      for (const row of P) maxDev = Math.max(maxDev, Math.abs(row.reduce((a, b) => a + b, 0) - 1));
      check(G1, '高斯朴素贝叶斯（可分数据）', acc >= 0.95 && maxDev < 1e-12,
        'acc=' + acc.toFixed(4) + ' 后验行和偏差=' + maxDev.toExponential(2));
    }
    // D10 逻辑回归
    {
      const { X: Xb, y: yb } = FAI.makeBlobs(50, [[0, 0], [4, 4]], [1, 1], FAI.mulberry32(17));
      const { out } = FAI.zScore(Xb);
      const lr = new FAI.LogisticRegression({ lr: 0.5, iters: 300, reg: 1e-4 }).fit(out, yb);
      const acc = FAI.accuracy(lr.predict(out), yb);
      const h = lr.lossHistory;
      check(G1, '逻辑回归（线性可分）', acc >= 0.95 && h[h.length - 1] < h[0] * 0.5,
        'acc=' + acc.toFixed(4) + ' loss ' + h[0].toFixed(4) + '→' + h[h.length - 1].toFixed(4) + ' |∇|=' + lr.gradNorm.toExponential(2));
    }
    // D11 CART 决策树（用干净可分的团簇，叶节点纯度应达 1.0）
    {
      const { X: Xc, y: yc } = FAI.makeBlobs(30, [[0, 0, 0], [12, 12, 2], [0, 14, 8]], [0.4, 0.4, 0.4], FAI.mulberry32(29));
      const tree = new FAI.DecisionTree({ maxDepth: 8, minSamplesSplit: 2 }).fit(Xc, yc);
      const acc = FAI.accuracy(tree.predict(Xc), yc);
      check(G1, 'CART 决策树训练精度 = 1.0（叶纯）', acc === 1, 'acc=' + acc.toFixed(4));
    }
  }

  /* ================================================================ *
   * E. 端到端训练与确定性                                             *
   * ================================================================ */
  function testEndToEnd() {
    const G1 = '端到端';
    // E1 TinyGPT 训练收敛
    {
      const { hist } = trainGPT(42, 100);
      const drop = hist[0] / hist[hist.length - 1];
      check(G1, 'TinyGPT 字符级训练收敛', hist[hist.length - 1] < hist[0] * 0.85,
        'loss ' + hist[0].toFixed(4) + '→' + hist[hist.length - 1].toFixed(4) + '（降幅 ' + (100 - 100 / drop).toFixed(1) + '%）');
    }
    // E2 确定性：同种子同结果
    {
      const r1 = trainGPT(42, 6), r2 = trainGPT(42, 6);
      const same = r1.hist.every((v, i) => v === r2.hist[i]);
      check(G1, '同种子训练完全确定性', same,
        'run1[0]=' + r1.hist[0].toFixed(10) + ' run2[0]=' + r2.hist[0].toFixed(10));
    }
    // E3 Trainer + MLP
    {
      const rng = FAI.mulberry32(23);
      const { X, y } = FAI.makeBlobs(40, [[0, 0], [5, 5]], [1.2, 1.2], rng);
      const mlp = new FAI.MLP([2, 8, 2], { rng });
      const opt = new FAI.Adam(mlp.parameters(), 1e-2);
      const trainer = new FAI.Trainer(mlp, FAI.crossEntropy, opt, { batchSize: 16, rng });
      trainer.fit(X, y, 40);
      const h = trainer.history;
      check(G1, 'Trainer+MLP 损失下降', h[h.length - 1] < h[0] * 0.6,
        'loss ' + h[0].toFixed(4) + '→' + h[h.length - 1].toFixed(4));
    }
    // E4 生成能力
    {
      const { gpt, chars, stoi } = trainGPT(42, 100);
      const itos = chars;
      const prompt = [...'晨星守护'].map(c => stoi.get(c));
      const out = gpt.generate(prompt, 20);
      const text = out.map(i => itos[i]).join('');
      check(G1, 'TinyGPT 贪心生成可读序列', out.length === prompt.length + 20 && out.every(i => i >= 0 && i < chars.length),
        '生成="' + text + '"');
    }
  }

  /* ---------------- 执行 ---------------- */
  function runSelfTest() {
    results.length = 0;
    testGradients();
    testStability();
    testOptimizers();
    testClassical();
    testEndToEnd();
    const passed = results.filter(r => r.pass).length;
    return { results, passed, failed: results.length - passed, total: results.length };
  }

  FAI.runSelfTest = runSelfTest;

  if (typeof module === 'object' && module.exports && require.main === module) {
    const { results, passed, failed, total } = runSelfTest();
    const line = '='.repeat(72);
    console.log(line);
    console.log(' 晨星·旗舰 AI — 无头验证套件（梯度检验 + 算法不变量 + 确定性）');
    console.log(line);
    let lastGroup = '';
    for (const r of results) {
      if (r.group !== lastGroup) { console.log('\n【' + r.group + '】'); lastGroup = r.group; }
      const mark = r.pass ? '✅' : '❌';
      console.log('  ' + mark + ' ' + r.name.padEnd(34, '　') + (r.detail ? ' ' + r.detail : ''));
    }
    console.log('\n' + '-'.repeat(72));
    console.log(' 总计: ' + passed + '/' + total + ' 通过' + (failed ? '（' + failed + ' 失败）' : '（全绿）'));
    console.log(line);
    process.exit(failed ? 1 : 0);
  }
})();
