/* ============================================================================
 * 晨星·旗舰 AI 引擎 — 经典机器学习套件
 * ----------------------------------------------------------------------------
 * 血统（零重写原则：每个算法都站在世界顶级成果之上）：
 *   - PCA               —— Hotelling (1933)；特征分解用循环 Jacobi 旋转
 *                          （Jacobi 1846，数值代数经典，残差可达机器精度）
 *   - K-Means / K-Means++ —— Lloyd (1982)；Arthur & Vassilvitskii (2007) 初始化
 *   - GMM / EM          —— Dempster, Laird & Rubin (1977)；对数域 logsumexp
 *   - 高斯朴素贝叶斯     —— 条件独立假设 + 贝叶斯定理（Bishop PRML Ch.4）
 *   - 逻辑回归           —— 最大似然 = 最小交叉熵（Nelder & Wedderburn 1972 GLM）
 *   - CART 决策树        —— Breiman et al. (1984)；Gini 不纯度
 *
 * 每个算法均带可独立验证的数学不变量（见 test/verify.js）：
 *   PCA：特征向量正交归一、残差 ‖Av−λv‖≈ε、 ExplainedVariance 单调
 *   KMeans：SSE 逐轮单调不增
 *   GMM：对数似然单调不减、责任度每行和 = 1
 *   NB：后验概率和 = 1
 *   LR：损失单调不增、收敛后梯度趋零
 *   CART：叶节点纯度收敛
 * ============================================================================ */
(function () {
  const G = typeof globalThis !== 'undefined' ? globalThis : this;
  const FAI = G.FAI || (G.FAI = {});

  /* ================================================================== *
   * 数值基础：循环 Jacobi 特征分解（实对称阵）                           *
   * ================================================================== */
  function jacobiEigen(A, n, maxSweeps = 100, tol = 1e-12) {
    const a = Float64Array.from(A);
    const v = new Float64Array(n * n);
    for (let i = 0; i < n; i++) v[i * n + i] = 1;
    for (let sweep = 0; sweep < maxSweeps; sweep++) {
      let off = 0;
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += a[i * n + j] * a[i * n + j];
      if (Math.sqrt(off) < tol) break;
      for (let p = 0; p < n - 1; p++) {
        for (let q = p + 1; q < n; q++) {
          const apq = a[p * n + q];
          if (Math.abs(apq) < 1e-300) continue;
          const app = a[p * n + p], aqq = a[q * n + q];
          const theta = (aqq - app) / (2 * apq);
          const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
          const c = 1 / Math.sqrt(t * t + 1), s = t * c;
          for (let k = 0; k < n; k++) {
            if (k === p || k === q) continue;
            const akp = a[k * n + p], akq = a[k * n + q];
            a[k * n + p] = c * akp - s * akq;
            a[k * n + q] = s * akp + c * akq;
            a[p * n + k] = a[k * n + p];
            a[q * n + k] = a[k * n + q];
          }
          a[p * n + p] = c * c * app - 2 * s * c * apq + s * s * aqq;
          a[q * n + q] = s * s * app + 2 * s * c * apq + c * c * aqq;
          a[p * n + q] = 0; a[q * n + p] = 0;
          for (let k = 0; k < n; k++) {
            const vkp = v[k * n + p], vkq = v[k * n + q];
            v[k * n + p] = c * vkp - s * vkq;
            v[k * n + q] = s * vkp + c * vkq;
          }
        }
      }
    }
    const values = new Float64Array(n);
    for (let i = 0; i < n; i++) values[i] = a[i * n + i];
    return { values, vectors: v };
  }

  /* ================================================================== *
   * PCA                                                               *
   * ================================================================== */
  class PCA {
    constructor(k) { this.k = k; }
    fit(X) {
      const n = X.length, d = X[0].length;
      this.mean = new Array(d).fill(0);
      for (const r of X) for (let j = 0; j < d; j++) this.mean[j] += r[j] / n;
      // 协方差 C = Xcᵀ Xc / (n−1)
      const C = new Float64Array(d * d);
      for (const r of X) {
        const rc = r.map((v, j) => v - this.mean[j]);
        for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) C[i * d + j] += rc[i] * rc[j] / (n - 1);
      }
      const { values, vectors } = jacobiEigen(C, d);
      const order = Array.from({ length: d }, (_, i) => i).sort((a, b) => values[b] - values[a]);
      this.eigenvalues = order.map(i => values[i]);
      this.totalVariance = this.eigenvalues.reduce((a, b) => a + b, 0);
      // components: d × k，第 i 列为第 i 主成分
      this.components = new Float64Array(d * this.k);
      for (let c = 0; c < this.k; c++)
        for (let i = 0; i < d; i++) this.components[i * this.k + c] = vectors[i * d + order[c]];
      return this;
    }
    transform(X) {
      const out = [];
      for (const r of X) {
        const rc = r.map((v, j) => v - this.mean[j]);
        const p = new Array(this.k).fill(0);
        for (let c = 0; c < this.k; c++) {
          let s = 0;
          for (let j = 0; j < rc.length; j++) s += rc[j] * this.components[j * this.k + c];
          p[c] = s;
        }
        out.push(p);
      }
      return out;
    }
    reconstruct(X) {
      const P = this.transform(X);
      return P.map(p => {
        const r = new Array(this.mean.length);
        for (let j = 0; j < r.length; j++) {
          let s = this.mean[j];
          for (let c = 0; c < this.k; c++) s += p[c] * this.components[j * this.k + c];
          r[j] = s;
        }
        return r;
      });
    }
    explainedVarianceRatio() {
      return this.eigenvalues.slice(0, this.k).map(v => v / this.totalVariance);
    }
  }

  /* ================================================================== *
   * K-Means（k-means++ 初始化 + Lloyd 迭代）                            *
   * ================================================================== */
  function dist2(a, b) { let s = 0; for (let j = 0; j < a.length; j++) s += (a[j] - b[j]) ** 2; return s; }

  class KMeans {
    constructor(k, opts = {}) {
      this.k = k;
      this.maxIter = opts.maxIter || 100;
      this.rng = opts.rng || Math.random;
      this.nInit = opts.nInit || 10;      // sklearn 风格多次重启，取 SSE 最小者
      this.sseHistory = [];
    }
    /** 单次 k-means++ 初始化 + Lloyd 迭代 */
    _run(X) {
      const n = X.length;
      const centroids = [X[Math.floor(this.rng() * n) % n].slice()];
      const d2 = new Array(n).fill(Infinity);
      while (centroids.length < this.k) {
        const last = centroids[centroids.length - 1];
        let sum = 0;
        for (let i = 0; i < n; i++) {
          d2[i] = Math.min(d2[i], dist2(X[i], last));
          sum += d2[i];
        }
        if (sum === 0) { // 全部点重合：直接取一个点
          centroids.push(X[Math.floor(this.rng() * n) % n].slice());
          continue;
        }
        let r = this.rng() * sum, acc = 0, chosen = n - 1;
        for (let i = 0; i < n; i++) { acc += d2[i]; if (acc >= r) { chosen = i; break; } }
        centroids.push(X[chosen].slice());
      }
      // ---- Lloyd 迭代 ----
      let labels = new Array(n).fill(0);
      const sseHistory = [];
      for (let iter = 0; iter < this.maxIter; iter++) {
        let sse = 0;
        const newLabels = new Array(n);
        for (let i = 0; i < n; i++) {
          let best = 0, bd = Infinity;
          for (let c = 0; c < this.k; c++) {
            const dd = dist2(X[i], centroids[c]);
            if (dd < bd) { bd = dd; best = c; }
          }
          newLabels[i] = best; sse += bd;
        }
        sseHistory.push(sse);
        const changed = newLabels.some((v, i) => v !== labels[i]);
        labels = newLabels;
        // 更新质心（空簇保留原质心）
        const sums = Array.from({ length: this.k }, () => new Array(X[0].length).fill(0));
        const cnt = new Array(this.k).fill(0);
        for (let i = 0; i < n; i++) {
          const c = labels[i]; cnt[c]++;
          for (let j = 0; j < X[0].length; j++) sums[c][j] += X[i][j];
        }
        for (let c = 0; c < this.k; c++) {
          if (cnt[c] > 0) centroids[c] = sums[c].map(s => s / cnt[c]);
        }
        if (!changed) break;
      }
      return { centroids, labels, sseHistory, sse: sseHistory[sseHistory.length - 1] };
    }
    /** 多次重启，取 SSE 最小的解（sklearn n_init 思想） */
    fit(X) {
      let best = null;
      for (let attempt = 0; attempt < this.nInit; attempt++) {
        const res = this._run(X);
        if (!best || res.sse < best.sse) best = res;
      }
      this.centroids = best.centroids;
      this.labels = best.labels;
      this.sseHistory = best.sseHistory;
      return this;
    }
    predict(x) {
      let best = 0, bd = Infinity;
      for (let c = 0; c < this.k; c++) {
        const dd = dist2(x, this.centroids[c]);
        if (dd < bd) { bd = dd; best = c; }
      }
      return best;
    }
  }

  /* ================================================================== *
   * 高斯混合模型（对角协方差，EM 算法，对数域计算）                        *
   * ================================================================== */
  class GMM {
    constructor(k, opts = {}) {
      this.k = k;
      this.maxIter = opts.maxIter || 100;
      this.reg = opts.reg || 1e-6;
      this.rng = opts.rng || Math.random;
      this.tol = opts.tol || 1e-8;
      this.llHistory = [];
    }
    /** 对角高斯 log 密度 */
    _logGauss(x, mu, var_) {
      let s = 0;
      for (let j = 0; j < x.length; j++) s += Math.log(2 * Math.PI * var_[j]) + (x[j] - mu[j]) ** 2 / var_[j];
      return -0.5 * s;
    }
    fit(X) {
      const n = X.length, d = X[0].length, K = this.k;
      // KMeans 初始化均值；权重均匀；协方差用「各簇内方差」而非全局方差
      // （全局方差过大会让责任度趋于均匀，EM 易陷入退化解）
      const km = new KMeans(K, { rng: this.rng, maxIter: 50 }).fit(X);
      this.weights = new Array(K).fill(1 / K);
      this.means = km.centroids.map(c => c.slice());
      const mu = new Array(d).fill(0);
      for (const r of X) for (let j = 0; j < d; j++) mu[j] += r[j] / n;
      const gvar = new Array(d).fill(0);
      for (const r of X) for (let j = 0; j < d; j++) gvar[j] += (r[j] - mu[j]) ** 2 / n;
      this.vars = this.means.map((mk, k) => {
        const v = new Array(d).fill(0);
        let cnt = 0;
        for (let i = 0; i < n; i++) if (km.labels[i] === k) {
          cnt++;
          for (let j = 0; j < d; j++) v[j] += (X[i][j] - mk[j]) ** 2;
        }
        if (cnt > 1) for (let j = 0; j < d; j++) v[j] = v[j] / cnt + this.reg;
        else for (let j = 0; j < d; j++) v[j] = gvar[j] + this.reg;
        return v;
      });
      let prevLL = -Infinity;
      for (let iter = 0; iter < this.maxIter; iter++) {
        // ---- E 步：对数域责任度 ----
        const logResp = new Float64Array(n * K);
        let ll = 0;
        for (let i = 0; i < n; i++) {
          const row = [];
          for (let k = 0; k < K; k++) {
            row.push(Math.log(this.weights[k]) + this._logGauss(X[i], this.means[k], this.vars[k]));
          }
          const m = Math.max(...row);
          let s = 0;
          for (let k = 0; k < K; k++) s += Math.exp(row[k] - m);
          const logN = m + Math.log(s);
          ll += logN;
          for (let k = 0; k < K; k++) logResp[i * K + k] = row[k] - logN;
        }
        ll /= n;
        this.llHistory.push(ll);
        if (Math.abs(ll - prevLL) < this.tol) { prevLL = ll; break; }
        prevLL = ll;
        // ---- M 步 ----
        const Nk = new Array(K).fill(0);
        const resp = new Float64Array(n * K);
        for (let i = 0; i < n; i++) for (let k = 0; k < K; k++) {
          resp[i * K + k] = Math.exp(logResp[i * K + k]);
          Nk[k] += resp[i * K + k];
        }
        for (let k = 0; k < K; k++) {
          this.weights[k] = Math.max(Nk[k] / n, 1e-12);
          const mean = new Array(d).fill(0);
          for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += resp[i * K + k] * X[i][j];
          for (let j = 0; j < d; j++) mean[j] /= Nk[k];
          this.means[k] = mean;
          const var_ = new Array(d).fill(0);
          for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) var_[j] += resp[i * K + k] * (X[i][j] - mean[j]) ** 2;
          for (let j = 0; j < d; j++) var_[j] = var_[j] / Nk[k] + this.reg;
          this.vars[k] = var_;
        }
      }
      this.logLikelihood = prevLL;
      return this;
    }
    predictProba(X) {
      const K = this.k;
      return X.map(x => {
        const row = [];
        for (let k = 0; k < K; k++) row.push(Math.log(this.weights[k]) + this._logGauss(x, this.means[k], this.vars[k]));
        const m = Math.max(...row);
        let s = 0;
        for (let k = 0; k < K; k++) s += Math.exp(row[k] - m);
        return row.map(v => Math.exp(v - m) / s);
      });
    }
    predict(X) { return this.predictProba(X).map(p => FAI.argmax(p)); }
  }

  /* ================================================================== *
   * 高斯朴素贝叶斯                                                      *
   * ================================================================== */
  class GaussianNB {
    constructor(opts = {}) { this.varSmoothing = opts.varSmoothing || 1e-9; }
    fit(X, y) {
      const n = X.length, d = X[0].length;
      this.classes = [...new Set(y)].sort();
      this.priors = {}; this.means = {}; this.vars = {};
      for (const c of this.classes) {
        const idx = X.map((_, i) => i).filter(i => y[i] === c);
        this.priors[c] = idx.length / n;
        const mu = new Array(d).fill(0);
        for (const i of idx) for (let j = 0; j < d; j++) mu[j] += X[i][j];
        for (let j = 0; j < d; j++) mu[j] /= idx.length;
        const vr = new Array(d).fill(0);
        for (const i of idx) for (let j = 0; j < d; j++) vr[j] += (X[i][j] - mu[j]) ** 2;
        for (let j = 0; j < d; j++) {
          vr[j] = vr[j] / idx.length;
          vr[j] += this.varSmoothing * Math.max(...X.map(r => r[j])) ** 2; // sklearn 风格平滑
        }
        this.means[c] = mu; this.vars[c] = vr;
      }
      return this;
    }
    predictProba(X) {
      const K = this.classes.length;
      return X.map(x => {
        const row = this.classes.map(c => {
          const mu = this.means[c], vr = this.vars[c];
          let logp = Math.log(this.priors[c]);
          for (let j = 0; j < x.length; j++) logp += -0.5 * (Math.log(2 * Math.PI * vr[j]) + (x[j] - mu[j]) ** 2 / vr[j]);
          return logp;
        });
        const m = Math.max(...row);
        let s = 0;
        for (let k = 0; k < K; k++) s += Math.exp(row[k] - m);
        return row.map(v => Math.exp(v - m) / s);
      });
    }
    predict(X) { return this.predictProba(X).map(p => this.classes[FAI.argmax(p)]); }
  }

  /* ================================================================== *
   * 逻辑回归（批量梯度下降，L2 正则）                                    *
   * ================================================================== */
  class LogisticRegression {
    constructor(opts = {}) {
      this.lr = opts.lr || 0.5;
      this.iters = opts.iters || 300;
      this.reg = opts.reg !== undefined ? opts.reg : 1e-4;
      this.lossHistory = [];
    }
    static _sigmoid(z) { return 1 / (1 + Math.exp(-z)); }
    fit(X, y) {
      const n = X.length, d = X[0].length;
      this.w = new Array(d).fill(0);
      this.b = 0;
      for (let it = 0; it < this.iters; it++) {
        const p = X.map(r => {
          let z = this.b;
          for (let j = 0; j < d; j++) z += this.w[j] * r[j];
          return LogisticRegression._sigmoid(z);
        });
        // 记录交叉熵
        let ce = 0;
        for (let i = 0; i < n; i++) {
          const pp = Math.min(Math.max(p[i], 1e-12), 1 - 1e-12);
          ce += -(y[i] * Math.log(pp) + (1 - y[i]) * Math.log(1 - pp));
        }
        ce /= n;
        this.lossHistory.push(ce);
        // 梯度
        const gw = new Array(d).fill(0);
        let gb = 0;
        for (let i = 0; i < n; i++) {
          const e = p[i] - y[i];
          gb += e;
          for (let j = 0; j < d; j++) gw[j] += e * X[i][j];
        }
        for (let j = 0; j < d; j++) {
          gw[j] = gw[j] / n + this.reg * this.w[j];
          this.w[j] -= this.lr * gw[j];
        }
        this.b -= this.lr * (gb / n);
      }
      this.gradNorm = Math.sqrt(this.w.reduce((a, v) => a + v * v, 0) + this.b * this.b);
      return this;
    }
    predictProba(X) {
      return X.map(r => {
        let z = this.b;
        for (let j = 0; j < r.length; j++) z += this.w[j] * r[j];
        return LogisticRegression._sigmoid(z);
      });
    }
    predict(X) { return this.predictProba(X).map(p => (p >= 0.5 ? 1 : 0)); }
  }

  /* ================================================================== *
   * CART 决策树（Gini 不纯度，预剪枝）                                   *
   * ================================================================== */
  class DecisionTree {
    constructor(opts = {}) {
      this.maxDepth = opts.maxDepth || 6;
      this.minSplit = opts.minSamplesSplit || 4;
    }
    static _gini(y) {
      const cnt = {};
      for (const v of y) cnt[v] = (cnt[v] || 0) + 1;
      let imp = 1;
      for (const k in cnt) imp -= (cnt[k] / y.length) ** 2;
      return imp;
    }
    _bestSplit(X, y) {
      const n = X.length, d = X[0].length;
      let best = null, bestGini = Infinity;
      for (let f = 0; f < d; f++) {
        const vals = [...new Set(X.map(r => r[f]))].sort((a, b) => a - b);
        for (let t = 0; t < vals.length - 1; t++) {
          const thr = (vals[t] + vals[t + 1]) / 2;
          const Ly = [], Ry = [];
          for (let i = 0; i < n; i++) (X[i][f] <= thr ? Ly : Ry).push(y[i]);
          if (!Ly.length || !Ry.length) continue;
          const g = (Ly.length * DecisionTree._gini(Ly) + Ry.length * DecisionTree._gini(Ry)) / n;
          if (g < bestGini - 1e-12) { bestGini = g; best = { feat: f, thr }; }
        }
      }
      return best;
    }
    _build(X, y, depth) {
      const label = y[0];
      if (depth >= this.maxDepth || y.length < this.minSplit || y.every(v => v === label)) {
        const cnt = {};
        for (const v of y) cnt[v] = (cnt[v] || 0) + 1;
        return { leaf: true, pred: Number(Object.entries(cnt).sort((a, b) => b[1] - a[1])[0][0]) };
      }
      const sp = this._bestSplit(X, y);
      if (!sp) {
        const cnt = {};
        for (const v of y) cnt[v] = (cnt[v] || 0) + 1;
        return { leaf: true, pred: Number(Object.entries(cnt).sort((a, b) => b[1] - a[1])[0][0]) };
      }
      const LX = [], Ly = [], RX = [], Ry = [];
      for (let i = 0; i < X.length; i++) (X[i][sp.feat] <= sp.thr ? LX : RX).push(X[i]);
      for (let i = 0; i < y.length; i++) (X[i][sp.feat] <= sp.thr ? Ly : Ry).push(y[i]);
      return {
        leaf: false, feat: sp.feat, thr: sp.thr,
        left: this._build(LX, Ly, depth + 1),
        right: this._build(RX, Ry, depth + 1)
      };
    }
    fit(X, y) { this.root = this._build(X, y, 0); return this; }
    _predictOne(x, node) {
      if (node.leaf) return node.pred;
      return x[node.feat] <= node.thr ? this._predictOne(x, node.left) : this._predictOne(x, node.right);
    }
    predict(X) { return X.map(x => this._predictOne(x, this.root)); }
  }

  FAI.jacobiEigen = jacobiEigen;
  FAI.PCA = PCA;
  FAI.KMeans = KMeans;
  FAI.GMM = GMM;
  FAI.GaussianNB = GaussianNB;
  FAI.LogisticRegression = LogisticRegression;
  FAI.DecisionTree = DecisionTree;

  if (typeof module === 'object' && module.exports) module.exports = FAI;
})();
