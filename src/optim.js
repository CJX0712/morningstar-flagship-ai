/* ============================================================================
 * 晨星·旗舰 AI 引擎 — 优化器与梯度治理
 * ----------------------------------------------------------------------------
 * 血统：
 *   - SGD +动量              —— Polyak (1964) 重球法；Nesterov (1983) 加速
 *   - Adam                   —— Kingma & Ba (2015) ICLR
 *   - AdamW（解耦权重衰减）   —— Loshchilov & Hutter (2019) ICLR
 *   - 梯度裁剪               —— Pascanu, Mikolov, Bengio (2013)（防 RNN 爆炸）
 * ============================================================================ */
(function () {
  const G = typeof globalThis !== 'undefined' ? globalThis : this;
  const FAI = G.FAI || (G.FAI = {});

  class SGD {
    constructor(params, lr = 0.1, opts = {}) {
      this.params = params; this.lr = lr;
      this.momentum = opts.momentum || 0;
      this.nesterov = !!opts.nesterov;
      this.weightDecay = opts.weightDecay || 0;
      this.v = params.map(() => null);
    }
    step() {
      for (let i = 0; i < this.params.length; i++) {
        const p = this.params[i];
        if (!p.grad) continue;
        const g = p.grad;
        if (!this.v[i]) this.v[i] = new Float64Array(p.size);
        const v = this.v[i];
        for (let k = 0; k < p.size; k++) {
          let gk = g[k] + this.weightDecay * p.data[k];
          v[k] = this.momentum * v[k] + gk;
          const upd = this.nesterov ? gk + this.momentum * v[k] : v[k];
          p.data[k] -= this.lr * upd;
        }
      }
    }
    zeroGrad() { for (const p of this.params) if (p.grad) p.grad.fill(0); }
  }

  class Adam {
    constructor(params, lr = 1e-3, opts = {}) {
      this.params = params; this.lr = lr;
      this.beta1 = opts.beta1 !== undefined ? opts.beta1 : 0.9;
      this.beta2 = opts.beta2 !== undefined ? opts.beta2 : 0.999;
      this.eps = opts.eps !== undefined ? opts.eps : 1e-8;
      this.weightDecay = opts.weightDecay || 0;
      this.decoupled = !!opts.decoupled; // true => AdamW
      this.m = params.map(() => null);
      this.v = params.map(() => null);
      this.t = 0;
    }
    step() {
      this.t++;
      const b1t = 1 - Math.pow(this.beta1, this.t);
      const b2t = 1 - Math.pow(this.beta2, this.t);
      for (let i = 0; i < this.params.length; i++) {
        const p = this.params[i];
        if (!p.grad) continue;
        const g = p.grad;
        if (!this.m[i]) { this.m[i] = new Float64Array(p.size); this.v[i] = new Float64Array(p.size); }
        const m = this.m[i], v = this.v[i];
        for (let k = 0; k < p.size; k++) {
          const gk = this.decoupled ? g[k] : g[k] + this.weightDecay * p.data[k];
          m[k] = this.beta1 * m[k] + (1 - this.beta1) * gk;
          v[k] = this.beta2 * v[k] + (1 - this.beta2) * gk * gk;
          const mh = m[k] / b1t, vh = v[k] / b2t;
          p.data[k] -= this.lr * mh / (Math.sqrt(vh) + this.eps);
          if (this.decoupled && this.weightDecay > 0) p.data[k] -= this.lr * this.weightDecay * p.data[k];
        }
      }
    }
    zeroGrad() { for (const p of this.params) if (p.grad) p.grad.fill(0); }
  }

  class AdamW extends Adam {
    constructor(params, lr = 1e-3, opts = {}) {
      super(params, lr, Object.assign({}, opts, { decoupled: true }));
    }
  }

  /** 全局梯度范数裁剪：||g|| > maxNorm 时整体缩放 */
  function clipGradNorm(params, maxNorm) {
    let total = 0;
    for (const p of params) if (p.grad) for (let i = 0; i < p.grad.length; i++) total += p.grad[i] * p.grad[i];
    total = Math.sqrt(total);
    if (total > maxNorm && total > 0) {
      const scale = maxNorm / total;
      for (const p of params) if (p.grad) for (let i = 0; i < p.grad.length; i++) p.grad[i] *= scale;
    }
    return total;
  }

  /** 余弦退火学习率（Loshchilov & Hutter 2017, SGDR） */
  function cosineLr(baseLr, epoch, totalEpochs, minRatio = 0.0) {
    const minLr = baseLr * minRatio;
    return minLr + 0.5 * (baseLr - minLr) * (1 + Math.cos(Math.PI * epoch / totalEpochs));
  }

  FAI.SGD = SGD;
  FAI.Adam = Adam;
  FAI.AdamW = AdamW;
  FAI.clipGradNorm = clipGradNorm;
  FAI.cosineLr = cosineLr;

  if (typeof module === 'object' && module.exports) module.exports = FAI;
})();
