/* ============================================================================
 * 晨星·旗舰 AI 引擎 — 模型库（神经网络模块）
 * ----------------------------------------------------------------------------
 * 血统（零重写原则：直接复用世界顶级架构）：
 *   - Transformer / GPT    —— Vaswani et al. (2017) "Attention Is All You Need"
 *   - Pre-LN 残差结构      —— Radford et al. (2019) GPT-2
 *   - He 初始化            —— He et al. (2015)（ReLU 族）
 *   - Xavier 初始化        —— Glorot & Bengio (2010)
 *   - GELU 激活            —— Hendrycks & Gimpel (2016)，tanh 近似
 *   - LayerNorm            —— Ba, Kiros & Hinton (2016)
 *   - 缩放点积注意力 + 掩码 —— Vaswani et al. (2017)
 * ============================================================================ */
(function () {
  const G = typeof globalThis !== 'undefined' ? globalThis : this;
  const FAI = G.FAI || (G.FAI = {});
  const Tensor = FAI.Tensor;

  /** 任意秩 softmax（沿最后一维），经 logsumexp 保证数值稳定 */
  function softmax(x) {
    const lse = x.logsumexp();
    const lseK = lse.reshape(x.shape.slice(0, -1).concat([1]));
    return x.sub(lseK).exp();
  }

  /** GELU（tanh 近似）：0.5x(1 + tanh(√(2/π)(x + 0.044715x³))) */
  function gelu(x) {
    const c = Math.sqrt(2 / Math.PI);
    const inner = x.mul(Tensor.constant(c, [1]))
      .add(x.mul(x).mul(x).mul(Tensor.constant(c * 0.044715, [1])));
    return inner.tanh().add(Tensor.constant(1, [1])).mul(x).mul(Tensor.constant(0.5, [1]));
  }

  /* ------------------------------------------------------------------ *
   * 模块基类                                                             *
   * ------------------------------------------------------------------ */
  class Module {
    constructor() { this.training = true; }
    parameters() { return []; }
    zeroGrad() { for (const p of this.parameters()) if (p.grad) p.grad.fill(0); }
    train(mode = true) { this.training = mode; return this; }
    eval() { return this.train(false); }
    countParameters() { return this.parameters().reduce((a, p) => a + p.size, 0); }
  }

  /* ------------------------------------------------------------------ *
   * 线性层：y = xW + b                                                   *
   * ------------------------------------------------------------------ */
  class Linear extends Module {
    constructor(inF, outF, opts = {}) {
      super();
      const rng = opts.rng || Math.random;
      const scale = opts.init === 'xavier' ? Math.sqrt(1 / inF) : Math.sqrt(2 / inF);
      this.W = Tensor.randn([inF, outF], rng, scale);
      this.W.requiresGrad = true;
      this.b = Tensor.zeros([1, outF]);
      this.b.requiresGrad = true;
      this.hasBias = opts.bias !== false;
    }
    forward(x) {
      const y = x.matmul(this.W);
      return this.hasBias ? y.add(this.b) : y;
    }
    parameters() { return this.hasBias ? [this.W, this.b] : [this.W]; }
  }

  /* ------------------------------------------------------------------ *
   * Embedding：整数索引 -> 稠密向量（GPT 量级小初始化 0.02）              *
   * ------------------------------------------------------------------ */
  class Embedding extends Module {
    constructor(V, d, opts = {}) {
      super();
      const rng = opts.rng || Math.random;
      this.W = Tensor.randn([V, d], rng, opts.scale || 0.02);
      this.W.requiresGrad = true;
    }
    /** idx: 扁平整数数组；返回 (n, d)，调用方可 reshape */
    forward(idx) { return Tensor.gatherRows(this.W, idx); }
    parameters() { return [this.W]; }
  }

  /* ------------------------------------------------------------------ *
   * LayerNorm：沿最后一维归一化 + 仿射                                   *
   * ------------------------------------------------------------------ */
  class LayerNorm extends Module {
    constructor(d, eps = 1e-5) {
      super();
      this.gamma = Tensor.ones([d]); this.gamma.requiresGrad = true;
      this.beta = Tensor.zeros([d]); this.beta.requiresGrad = true;
      this.eps = eps;
    }
    forward(x) {
      const mu = x.mean(-1, true);
      const xc = x.sub(mu);
      const variance = xc.mul(xc).mean(-1, true);
      const denom = variance.add(Tensor.constant(this.eps, [1])).sqrt();
      return xc.div(denom).mul(this.gamma).add(this.beta);
    }
    parameters() { return [this.gamma, this.beta]; }
  }

  /* ------------------------------------------------------------------ *
   * Dropout（ inverted dropout，训练期生效 ）                            *
   * ------------------------------------------------------------------ */
  class Dropout extends Module {
    constructor(p, rng) {
      super();
      this.p = p; this.rng = rng || Math.random;
    }
    forward(x) {
      if (!this.training || this.p <= 0) return x;
      const keep = 1 - this.p;
      const mask = new Float64Array(x.size);
      for (let i = 0; i < x.size; i++) mask[i] = this.rng() < keep ? 1 / keep : 0;
      return x.mul(new Tensor(mask, x.shape, {}));
    }
  }

  /* ------------------------------------------------------------------ *
   * 多头缩放点积注意力（支持因果掩码）                                    *
   *   Attention(Q,K,V) = softmax(QKᵀ/√d_k + M)V                          *
   * ------------------------------------------------------------------ */
  class MultiHeadAttention extends Module {
    constructor(dModel, heads, opts = {}) {
      super();
      if (dModel % heads !== 0) throw new Error('dModel 必须被 heads 整除');
      this.dModel = dModel; this.heads = heads; this.dk = dModel / heads;
      this.Wq = new Linear(dModel, dModel, opts);
      this.Wk = new Linear(dModel, dModel, opts);
      this.Wv = new Linear(dModel, dModel, opts);
      this.Wo = new Linear(dModel, dModel, opts);
      this.mask = opts.mask || null;         // 加性掩码 Tensor (T,T)
      this.attnDrop = opts.attnDropout ? new Dropout(opts.attnDropout, opts.rng) : null;
    }
    /** x: (B, T, d) -> (B, T, d) */
    forward(x) {
      const B = x.shape[0], T = x.shape[1], H = this.heads, dk = this.dk;
      const split = t => t.reshape([B, T, H, dk]).permute([0, 2, 1, 3]).reshape([B * H, T, dk]);
      const q = split(this.Wq.forward(x));
      const k = split(this.Wk.forward(x));
      const v = split(this.Wv.forward(x));
      let scores = q.matmul(k.permute([0, 2, 1])).mul(Tensor.constant(1 / Math.sqrt(dk), [1]));
      // 掩码按实际上下文长度自适应（生成阶段序列可能短于 blockSize）
      if (this.mask) {
        const m = this.mask.shape[0] === T ? this.mask : Tensor.causalMask(T);
        scores = scores.add(m);                     // (T,T) 广播至 (B*H,T,T)
      }
      let attn = softmax(scores);
      if (this.attnDrop) attn = this.attnDrop.forward(attn);
      const ctx = attn.matmul(v);                          // (B*H, T, dk)
      const merged = ctx.reshape([B, H, T, dk]).permute([0, 2, 1, 3]).reshape([B, T, this.dModel]);
      return this.Wo.forward(merged);
    }
    parameters() {
      return [...this.Wq.parameters(), ...this.Wk.parameters(), ...this.Wv.parameters(), ...this.Wo.parameters()];
    }
    train(mode = true) {
      super.train(mode);
      this.Wq.train(mode); this.Wk.train(mode); this.Wv.train(mode); this.Wo.train(mode);
      if (this.attnDrop) this.attnDrop.train(mode);
      return this;
    }
  }

  /* ------------------------------------------------------------------ *
   * Transformer 块（Pre-LN 残差，GPT-2 结构）                            *
   *   h = x + Attn(LN1(x));  y = h + FFN(LN2(h))                        *
   * ------------------------------------------------------------------ */
  class TransformerBlock extends Module {
    constructor(dModel, heads, opts = {}) {
      super();
      const ffnMult = opts.ffnMult || 4;
      this.ln1 = new LayerNorm(dModel);
      this.attn = new MultiHeadAttention(dModel, heads, opts);
      this.ln2 = new LayerNorm(dModel);
      this.ff1 = new Linear(dModel, ffnMult * dModel, opts);
      this.ff2 = new Linear(ffnMult * dModel, dModel, opts);
      this.drop = opts.dropout ? new Dropout(opts.dropout, opts.rng) : null;
    }
    forward(x) {
      let h = x.add(this.attn.forward(this.ln1.forward(x)));
      let f = this.ff1.forward(this.ln2.forward(h));
      f = gelu(f);
      if (this.drop) f = this.drop.forward(f);
      return h.add(this.ff2.forward(f));
    }
    parameters() {
      return [...this.ln1.parameters(), ...this.attn.parameters(),
      ...this.ln2.parameters(), ...this.ff1.parameters(), ...this.ff2.parameters()];
    }
    train(mode = true) {
      super.train(mode);
      this.ln1.train(mode); this.attn.train(mode); this.ln2.train(mode);
      this.ff1.train(mode); this.ff2.train(mode);
      if (this.drop) this.drop.train(mode);
      return this;
    }
  }

  /* ------------------------------------------------------------------ *
   * TinyGPT：字符级迷你语言模型（GPT 架构完整微缩）                        *
   * ------------------------------------------------------------------ */
  class TinyGPT extends Module {
    constructor(cfg = {}) {
      super();
      this.cfg = cfg;
      const d = cfg.dModel || 32, V = cfg.vocabSize, T = cfg.blockSize || 16;
      const rng = cfg.rng || Math.random;
      this.tokEmb = new Embedding(V, d, { rng });
      this.posEmb = new Embedding(T, d, { rng });
      this.blocks = [];
      for (let i = 0; i < (cfg.layers || 2); i++) {
        this.blocks.push(new TransformerBlock(d, cfg.heads || 2, {
          rng, dropout: cfg.dropout || 0, mask: Tensor.causalMask(T)
        }));
      }
      this.lnF = new LayerNorm(d);
      this.head = new Linear(d, V, { rng });
      this.drop = cfg.dropout ? new Dropout(cfg.dropout, rng) : null;
    }
    /** idx: 扁平整数数组（B*T），lead: 前导形状如 [B, T]；返回 logits (..., V) */
    forward(idx, lead) {
      const d = this.cfg.dModel || 32;
      let h = this.tokEmb.forward(idx).reshape(lead.concat([d]));
      // 位置编码按实际上下文长度取（生成阶段可短于 blockSize）
      const tLen = lead[lead.length - 1];
      const pos = this.posEmb.forward(Array.from({ length: tLen }, (_, i) => i)); // (tLen,d)
      h = h.add(pos);
      if (this.drop) h = this.drop.forward(h);
      for (const blk of this.blocks) h = blk.forward(h);
      h = this.lnF.forward(h);
      const V = this.cfg.vocabSize;
      return this.head.forward(h.reshape([idx.length, d])).reshape(lead.concat([V]));
    }
    /** 交叉熵损失（next-token 预测） */
    loss(idx, lead, targets) {
      const logits = this.forward(idx, lead);
      const V = this.cfg.vocabSize;
      return FAI.crossEntropy(logits.reshape([idx.length, V]), targets);
    }
    /** 贪心生成 maxNew 个字符 */
    generate(prompt, maxNew) {
      const T = this.cfg.blockSize || 16;
      let idx = prompt.slice(-T);
      const out = idx.slice();
      for (let s = 0; s < maxNew; s++) {
        const ctx = idx.slice(-T);
        const lead = [1, ctx.length];
        const logits = this.forward(ctx, lead);      // (1, t, V)
        const V = this.cfg.vocabSize;
        const last = logits.data.slice((ctx.length - 1) * V, ctx.length * V);
        const next = FAI.argmax(last);
        out.push(next);
        idx = idx.concat([next]);
      }
      return out;
    }
    parameters() {
      let ps = [...this.tokEmb.parameters(), ...this.posEmb.parameters()];
      for (const b of this.blocks) ps = ps.concat(b.parameters());
      return ps.concat([...this.lnF.parameters(), ...this.head.parameters()]);
    }
    train(mode = true) {
      super.train(mode);
      this.tokEmb.train(mode); this.posEmb.train(mode);
      for (const b of this.blocks) b.train(mode);
      this.lnF.train(mode); this.head.train(mode);
      if (this.drop) this.drop.train(mode);
      return this;
    }
  }

  /* ------------------------------------------------------------------ *
   * MLP：经典多层感知机                                                  *
   * ------------------------------------------------------------------ */
  class MLP extends Module {
    constructor(sizes, opts = {}) {
      super();
      const rng = opts.rng || Math.random;
      this.act = opts.activation || 'tanh';
      this.layers = [];
      for (let i = 0; i < sizes.length - 1; i++) {
        this.layers.push(new Linear(sizes[i], sizes[i + 1], { rng }));
      }
    }
    forward(x) {
      let h = x;
      for (let i = 0; i < this.layers.length; i++) {
        h = this.layers[i].forward(h);
        if (i < this.layers.length - 1) {
          if (this.act === 'relu') h = h.relu();
          else if (this.act === 'sigmoid') h = h.sigmoid();
          else h = h.tanh();
        }
      }
      return h;
    }
    parameters() {
      let ps = [];
      for (const l of this.layers) ps = ps.concat(l.parameters());
      return ps;
    }
  }

  FAI.Module = Module;
  FAI.Linear = Linear;
  FAI.Embedding = Embedding;
  FAI.LayerNorm = LayerNorm;
  FAI.Dropout = Dropout;
  FAI.MultiHeadAttention = MultiHeadAttention;
  FAI.TransformerBlock = TransformerBlock;
  FAI.TinyGPT = TinyGPT;
  FAI.MLP = MLP;
  FAI.softmax = softmax;
  FAI.gelu = gelu;

  if (typeof module === 'object' && module.exports) module.exports = FAI;
})();
