/* ============================================================================
 * 晨星·旗舰 AI 引擎 — 工具链（随机数 / 数据 / 训练循环）
 * ----------------------------------------------------------------------------
 * 血统：
 *   - mulberry32 伪随机数     —— G. Slowsay (2014) 快速可复现 PRNG
 *   - Box–Muller 正态采样     —— Box & Muller (1958)
 *   - k-means++ 初始化        —— Arthur & Vassilvitskii (2007)
 *   - z-score 标准化          —— 经典统计预处理
 * ============================================================================ */
(function () {
  const G = typeof globalThis !== 'undefined' ? globalThis : this;
  const FAI = G.FAI || (G.FAI = {});

  /** mulberry32：32 位种子 PRNG，跨平台确定性一致 */
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /** 标准正态采样（Box–Muller） */
  function randn(rng) {
    const u1 = Math.max(rng(), 1e-300), u2 = rng();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  function randint(rng, n) { return Math.floor(rng() * n) % n; }

  /** Fisher–Yates 洗牌（原地） */
  function shuffle(arr, rng) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const t = arr[i]; arr[i] = arr[j]; arr[j] = t;
    }
    return arr;
  }

  /** z-score 标准化：返回 { out, mu, sigma }，可复用于测试集 */
  function zScore(X) {
    const rows = X.length, cols = X[0].length;
    const mu = new Array(cols).fill(0), sigma = new Array(cols).fill(0);
    for (let j = 0; j < cols; j++) { for (let i = 0; i < rows; i++) mu[j] += X[i][j]; mu[j] /= rows; }
    for (let j = 0; j < cols; j++) { for (let i = 0; i < rows; i++) sigma[j] += (X[i][j] - mu[j]) ** 2; sigma[j] = Math.sqrt(sigma[j] / rows); }
    const out = X.map(r => r.map((v, j) => (v - mu[j]) / (sigma[j] || 1)));
    return { out, mu, sigma };
  }

  function applyZScore(X, mu, sigma) {
    return X.map(r => r.map((v, j) => (v - mu[j]) / (sigma[j] || 1)));
  }

  /** 生成各向同性高斯团簇（用于聚类/降维演示，种子确定性） */
  function makeBlobs(nPerCluster, centers, std, rng) {
    const X = [], y = [];
    for (let c = 0; c < centers.length; c++) {
      for (let i = 0; i < nPerCluster; i++) {
        const p = centers[c].map((mu, j) => mu + randn(rng) * std[j]);
        X.push(p); y.push(c);
      }
    }
    return { X, y };
  }

  /** 双月牙数据集（非线性可分，测试逻辑回归边界） */
  function makeMoons(n, noise, rng) {
    const X = [], y = [];
    for (let i = 0; i < n; i++) {
      const t = Math.PI * i / n;
      const c = i % 2;
      const x = Math.cos(t + c * Math.PI) + randn(rng) * noise;
      const yy = Math.sin(t + c * Math.PI) + randn(rng) * noise;
      X.push([x, yy]); y.push(c);
    }
    return { X, y };
  }

  function argmax(arr) {
    let bi = 0;
    for (let i = 1; i < arr.length; i++) if (arr[i] > arr[bi]) bi = i;
    return bi;
  }

  function accuracy(pred, truth) {
    let ok = 0;
    for (let i = 0; i < pred.length; i++) if (pred[i] === truth[i]) ok++;
    return ok / pred.length;
  }

  /** 聚类纯度：按多数投票把簇映射到真实标签 */
  function clusterPurity(pred, truth) {
    const groups = new Map();
    for (let i = 0; i < pred.length; i++) {
      if (!groups.has(pred[i])) groups.set(pred[i], new Map());
      const m = groups.get(pred[i]);
      m.set(truth[i], (m.get(truth[i]) || 0) + 1);
    }
    let ok = 0;
    for (const m of groups.values()) ok += Math.max(...m.values());
    return ok / pred.length;
  }

  function confusionMatrix(pred, truth, K) {
    const M = Array.from({ length: K }, () => new Array(K).fill(0));
    for (let i = 0; i < pred.length; i++) M[truth[i]][pred[i]]++;
    return M;
  }

  /* ------------------------------------------------------------------ *
   * 训练器：小批次循环 + 损失轨迹                                        *
   * ------------------------------------------------------------------ */
  class Trainer {
    /**
     * @param model    拥有 parameters()/zeroGrad() 的模块
     * @param lossFn   (pred, target) => 标量 Tensor
     * @param optimizer 具备 step()/zeroGrad() 的优化器
     * @param opts     { batchSize, rng, clip }
     */
    constructor(model, lossFn, optimizer, opts = {}) {
      this.model = model;
      this.lossFn = lossFn;
      this.optimizer = optimizer;
      this.batchSize = opts.batchSize || 32;
      this.rng = opts.rng || Math.random;
      this.clip = opts.clip || 0;
      this.history = [];
    }
    /** 跑一个 epoch；X: (N,d) 普通数组或 Tensor，y: 标签数组 */
    epoch(X, y) {
      const N = X.length;
      const idx = shuffle(Array.from({ length: N }, (_, i) => i), this.rng);
      let total = 0, nb = 0;
      for (let s = 0; s < N; s += this.batchSize) {
        const bidx = idx.slice(s, s + this.batchSize);
        const xb = FAI.Tensor.of2D(bidx.map(i => X[i]));
        const yb = bidx.map(i => y[i]);
        const loss = this.lossFn(this.model.forward(xb), yb);
        loss.backward();
        if (this.clip > 0) FAI.clipGradNorm(this.model.parameters(), this.clip);
        this.optimizer.step();
        this.optimizer.zeroGrad();
        total += loss.data[0]; nb++;
      }
      const avg = total / nb;
      this.history.push(avg);
      return avg;
    }
    /** 训练 n 个 epoch，onEpoch 回调（可用于 UI 可视化） */
    fit(X, y, epochs, onEpoch) {
      for (let e = 0; e < epochs; e++) {
        const l = this.epoch(X, y);
        if (onEpoch) onEpoch(e, l);
      }
      return this.history;
    }
  }

  FAI.mulberry32 = mulberry32;
  FAI.randn = randn;
  FAI.randint = randint;
  FAI.shuffle = shuffle;
  FAI.zScore = zScore;
  FAI.applyZScore = applyZScore;
  FAI.makeBlobs = makeBlobs;
  FAI.makeMoons = makeMoons;
  FAI.argmax = argmax;
  FAI.accuracy = accuracy;
  FAI.clusterPurity = clusterPurity;
  FAI.confusionMatrix = confusionMatrix;
  FAI.Trainer = Trainer;

  if (typeof module === 'object' && module.exports) module.exports = FAI;
})();
