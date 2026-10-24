/* ============================================================================
 * 晨星·旗舰 AI 引擎 — 自动微分核心（Reverse-Mode Autodiff Tensor Engine）
 * ----------------------------------------------------------------------------
 * 血统（零重写原则：站在世界顶级成果之上）：
 *   - 反向模式自动微分  —— Baydin, Pearlmutter, Radford, Siskind (2018)
 *     "Automatic Differentiation in Machine Learning: a Survey"
 *   - 计算图与拓扑排序反传 —— PyTorch autograd 架构思想
 *   - logsumexp 稳定技巧   —— McInnes (2017) / Blanchard et al. (2020)
 *   - Float64 全精度       —— 保证中心差分梯度检验达 1e-7 量级
 *
 * 设计约束：零依赖、零构建、浏览器与 Node 双端同构运行。
 * ============================================================================ */
(function () {
  const G = typeof globalThis !== 'undefined' ? globalThis : this;
  const FAI = G.FAI || (G.FAI = {});

  /* ------------------------------------------------------------------ *
   * 形状 / 步长 / 广播工具                                               *
   * ------------------------------------------------------------------ */

  function numel(shape) { return shape.reduce((a, b) => a * b, 1); }

  function stridesOf(shape) {
    const s = new Array(shape.length);
    let acc = 1;
    for (let i = shape.length - 1; i >= 0; i--) { s[i] = acc; acc *= shape[i]; }
    return s;
  }

  function alignShape(shape, rank) {
    const s = new Array(rank).fill(1);
    for (let i = 0; i < shape.length; i++) s[rank - shape.length + i] = shape[i];
    return s;
  }

  /** 右对齐广播：输出形状 */
  function broadcastShapes(s1, s2) {
    const rank = Math.max(s1.length, s2.length);
    const a = alignShape(s1, rank), b = alignShape(s2, rank);
    const out = new Array(rank);
    for (let d = 0; d < rank; d++) {
      if (a[d] === b[d]) out[d] = a[d];
      else if (a[d] === 1) out[d] = b[d];
      else if (b[d] === 1) out[d] = a[d];
      else throw new Error('广播形状不兼容: [' + s1 + '] vs [' + s2 + ']');
    }
    return out;
  }

  /**
   * 广播索引映射：srcShape -> outShape。
   * src 中长度为 1 的维度步长置 0 —— 反向传播自动退化为求和（unbroadcast）。
   */
  function indexMap(srcShape, outShape) {
    const rank = outShape.length;
    const src = alignShape(srcShape, rank);
    const srcStrides = stridesOf(src);
    const n = numel(outShape);
    const map = new Int32Array(n);
    const idx = new Array(rank).fill(0);
    for (let i = 0; i < n; i++) {
      let flat = 0;
      for (let d = 0; d < rank; d++) flat += idx[d] * (src[d] === 1 ? 0 : srcStrides[d]);
      map[i] = flat;
      for (let d = rank - 1; d >= 0; d--) { if (++idx[d] < outShape[d]) break; idx[d] = 0; }
    }
    return map;
  }

  /**
   * 规约正向映射：outShape（= inShape 去掉 axis 维）-> inShape，
   * 被去掉的轴坐标取 0。用于 sum 前向。
   */
  function reduceIndexMap(outShape, inShape, axis) {
    const inStrides = stridesOf(inShape);
    const kept = [];
    for (let d = 0; d < inShape.length; d++) if (d !== axis) kept.push(d);
    const n = numel(outShape);
    const map = new Int32Array(n);
    const oIdx = new Array(outShape.length).fill(0);
    for (let i = 0; i < n; i++) {
      let flat = 0;
      for (let p = 0; p < kept.length; p++) flat += oIdx[p] * inStrides[kept[p]];
      map[i] = flat;
      for (let k = outShape.length - 1; k >= 0; k--) { if (++oIdx[k] < outShape[k]) break; oIdx[k] = 0; }
    }
    return map;
  }

  /**
   * 规约反向映射：每个输入元素 -> 其所属的输出元素（轴坐标置 0）。
   * 与 reduceIndexMap 互逆；sum 反向传播使用。
   */
  function reduceBackwardMap(inShape, outShape, axis) {
    const outStrides = stridesOf(outShape);
    const kept = [];
    for (let d = 0; d < inShape.length; d++) if (d !== axis) kept.push(d);
    const n = numel(inShape);
    const map = new Int32Array(n);
    const idx = new Array(inShape.length).fill(0);
    for (let i = 0; i < n; i++) {
      let flat = 0;
      for (let p = 0; p < kept.length; p++) flat += idx[kept[p]] * outStrides[p];
      map[i] = flat;
      for (let d = inShape.length - 1; d >= 0; d--) { if (++idx[d] < inShape[d]) break; idx[d] = 0; }
    }
    return map;
  }

  function normalizeAxis(axis, rank) {
    let a = axis;
    if (a < 0) a += rank;
    if (a < 0 || a >= rank) throw new Error('非法轴: ' + axis);
    return a;
  }

  /* ------------------------------------------------------------------ *
   * 张量                                                                 *
   * ------------------------------------------------------------------ */

  class Tensor {
    constructor(data, shape, opts = {}) {
      this.data = data;                    // Float64Array
      this.shape = shape.slice();
      this.grad = null;                    // Float64Array | null
      this.requiresGrad = !!opts.requiresGrad;
      this._ctx = opts._ctx || null;       // { parents, backward }
    }
    get size() { return this.data.length; }
    get rank() { return this.shape.length; }

    /* ---- 工厂 ---- */
    static zeros(shape, requiresGrad = false) {
      return new Tensor(new Float64Array(numel(shape)), shape, { requiresGrad });
    }
    static ones(shape, requiresGrad = false) {
      return new Tensor(new Float64Array(numel(shape)).fill(1), shape, { requiresGrad });
    }
    static constant(value, shape) {
      const s = shape || [1];
      return new Tensor(new Float64Array(numel(s)).fill(value), s, {});
    }
    static randn(shape, rng, scale = 1) {
      rng = rng || Math.random;
      const d = new Float64Array(numel(shape));
      for (let i = 0; i < d.length; i += 2) {
        const u1 = Math.max(rng(), 1e-300), u2 = rng();
        const r = Math.sqrt(-2 * Math.log(u1)), th = 2 * Math.PI * u2;
        d[i] = r * Math.cos(th) * scale;
        if (i + 1 < d.length) d[i + 1] = r * Math.sin(th) * scale;
      }
      return new Tensor(d, shape, {});
    }
    static of2D(rows, requiresGrad = false) {
      const r = rows.length, c = rows[0].length;
      const d = new Float64Array(r * c);
      for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) d[i * c + j] = rows[i][j];
      return new Tensor(d, [r, c], { requiresGrad });
    }
    /** 独热编码：(N,) 整数 -> (N, V) 常量 */
    static oneHot(idx, V) {
      const d = new Float64Array(idx.length * V);
      for (let i = 0; i < idx.length; i++) d[i * V + idx[i]] = 1;
      return new Tensor(d, [idx.length, V], {});
    }
    /** 因果（下三角）加性掩码：(T,T)，上三角置大负数 */
    static causalMask(T, neg = -1e9) {
      const d = new Float64Array(T * T);
      for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) d[i * T + j] = j > i ? neg : 0;
      return new Tensor(d, [T, T], {});
    }

    /* ---- 形状操作 ---- */
    reshape(shape) {
      const out = new Tensor(this.data, shape, { requiresGrad: this.requiresGrad });
      const self = this;
      out._ctx = { parents: [this], backward(t) { addInto(self.grad, t.grad); } };
      return out;
    }
    permute(perm) {
      const inShape = this.shape, outShape = perm.map(d => inShape[d]);
      const inStrides = stridesOf(inShape);
      const n = numel(outShape);
      const d = new Float64Array(n);
      const idx = new Array(perm.length).fill(0);
      for (let i = 0; i < n; i++) {
        let flat = 0;
        for (let k = 0; k < perm.length; k++) flat += idx[k] * inStrides[perm[k]];
        d[i] = this.data[flat];
        for (let k = perm.length - 1; k >= 0; k--) { if (++idx[k] < outShape[k]) break; idx[k] = 0; }
      }
      const out = new Tensor(d, outShape, { requiresGrad: this.requiresGrad });
      const self = this;
      out._ctx = {
        parents: [this],
        backward(t) {
          const gd = new Float64Array(self.size);
          const idx2 = new Array(perm.length).fill(0);
          for (let i = 0; i < n; i++) {
            let flat = 0;
            for (let k = 0; k < perm.length; k++) flat += idx2[k] * inStrides[perm[k]];
            gd[flat] += t.grad[i];
            for (let k = perm.length - 1; k >= 0; k--) { if (++idx2[k] < outShape[k]) break; idx2[k] = 0; }
          }
          addInto(self.grad, gd);
        }
      };
      return out;
    }
    transpose(axes) {
      const rank = this.rank;
      const perm = axes || Array.from({ length: rank }, (_, i) => rank - 1 - i);
      return this.permute(perm);
    }

    /* ---- 规约 ---- */
    sum(axis, keepdims = false) {
      const ax = normalizeAxis(axis, this.rank);
      const keptShape = this.shape.filter((_, d) => d !== ax);
      const outShape = keepdims ? this.shape.map((n, d) => d === ax ? 1 : n) : keptShape;
      const n = numel(outShape);
      const fwdMap = reduceIndexMap(outShape, this.shape, ax);
      const srcStrides = stridesOf(this.shape);
      const d = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const base = fwdMap[i];
        let acc = 0;
        for (let k = 0; k < this.shape[ax]; k++) acc += this.data[base + k * srcStrides[ax]];
        d[i] = acc;
      }
      const out = new Tensor(d, outShape, { requiresGrad: this.requiresGrad });
      const self = this;
      out._ctx = {
        parents: [this],
        backward(t) {
          // 每个输入元素归属的输出元素（轴坐标置 0）；sum 的雅可比为广播
          const bwdMap = reduceBackwardMap(self.shape, out.shape, ax);
          const g = t.grad;
          for (let i = 0; i < self.size; i++) self.grad[i] += g[bwdMap[i]];
        }
      };
      return out;
    }
    mean(axis, keepdims = false) {
      const ax = normalizeAxis(axis, this.rank);
      const cnt = Tensor.constant(1 / this.shape[ax], [1]);
      return this.sum(ax, keepdims).mul(cnt);
    }

    /* ---- 逐元素（广播） ---- */
    add(o) { return broadcastBinary(this, o, (a, b) => a + b, 'add'); }
    sub(o) { return broadcastBinary(this, o, (a, b) => a - b, 'sub'); }
    mul(o) { return broadcastBinary(this, o, (a, b) => a * b, 'mul'); }
    div(o) { return broadcastBinary(this, o, (a, b) => a / b, 'div'); }
    neg() {
      const d = new Float64Array(this.size);
      for (let i = 0; i < d.length; i++) d[i] = -this.data[i];
      const out = new Tensor(d, this.shape, { requiresGrad: this.requiresGrad });
      const self = this;
      out._ctx = { parents: [this], backward(t) { negateInto(self.grad, t.grad); } };
      return out;
    }
    exp() {
      const d = new Float64Array(this.size);
      for (let i = 0; i < d.length; i++) d[i] = Math.exp(this.data[i]);
      const out = new Tensor(d, this.shape, { requiresGrad: this.requiresGrad });
      const self = this;
      out._ctx = { parents: [this], backward(t) { scaleInto(self.grad, t.grad, out.data); } };
      return out;
    }
    log() {
      const d = new Float64Array(this.size);
      for (let i = 0; i < d.length; i++) d[i] = Math.log(this.data[i]);
      const out = new Tensor(d, this.shape, { requiresGrad: this.requiresGrad });
      const self = this;
      out._ctx = { parents: [this], backward(t) { divIntoGrad(self.grad, t.grad, self.data); } };
      return out;
    }
    sqrt() {
      const d = new Float64Array(this.size);
      for (let i = 0; i < d.length; i++) d[i] = Math.sqrt(this.data[i]);
      const out = new Tensor(d, this.shape, { requiresGrad: this.requiresGrad });
      const self = this;
      out._ctx = {
        parents: [this], backward(t) {
          for (let i = 0; i < self.size; i++) self.grad[i] += t.grad[i] * 0.5 / Math.max(out.data[i], 1e-300);
        }
      };
      return out;
    }
    tanh() {
      const d = new Float64Array(this.size);
      for (let i = 0; i < d.length; i++) d[i] = Math.tanh(this.data[i]);
      const out = new Tensor(d, this.shape, { requiresGrad: this.requiresGrad });
      const self = this;
      out._ctx = {
        parents: [this], backward(t) {
          for (let i = 0; i < self.size; i++) self.grad[i] += t.grad[i] * (1 - out.data[i] * out.data[i]);
        }
      };
      return out;
    }
    relu() {
      const d = new Float64Array(this.size);
      for (let i = 0; i < d.length; i++) d[i] = this.data[i] > 0 ? this.data[i] : 0;
      const out = new Tensor(d, this.shape, { requiresGrad: this.requiresGrad });
      const self = this;
      out._ctx = {
        parents: [this], backward(t) {
          for (let i = 0; i < self.size; i++) if (self.data[i] > 0) self.grad[i] += t.grad[i];
        }
      };
      return out;
    }
    sigmoid() {
      const d = new Float64Array(this.size);
      for (let i = 0; i < d.length; i++) d[i] = 1 / (1 + Math.exp(-this.data[i]));
      const out = new Tensor(d, this.shape, { requiresGrad: this.requiresGrad });
      const self = this;
      out._ctx = {
        parents: [this], backward(t) {
          for (let i = 0; i < self.size; i++) self.grad[i] += t.grad[i] * out.data[i] * (1 - out.data[i]);
        }
      };
      return out;
    }

    /* ---- logsumexp（沿最后一维，数值稳定；缓存 softmax 供反传） ---- */
    logsumexp() {
      const cols = this.shape[this.rank - 1];
      const rows = this.size / cols;
      const lead = this.shape.slice(0, -1);
      const d = new Float64Array(rows);
      const sm = new Float64Array(rows * cols);
      for (let r = 0; r < rows; r++) {
        let m = -Infinity;
        for (let c = 0; c < cols; c++) m = Math.max(m, this.data[r * cols + c]);
        let s = 0;
        for (let c = 0; c < cols; c++) { const e = Math.exp(this.data[r * cols + c] - m); sm[r * cols + c] = e; s += e; }
        d[r] = m + Math.log(s);
        for (let c = 0; c < cols; c++) sm[r * cols + c] /= s;
      }
      const out = new Tensor(d, lead.length ? lead : [1], { requiresGrad: this.requiresGrad });
      const self = this;
      out._ctx = {
        parents: [this], backward(t) {
          const g = t.grad;
          for (let r = 0; r < rows; r++)
            for (let c = 0; c < cols; c++) self.grad[r * cols + c] += g[r] * sm[r * cols + c];
        }
      };
      return out;
    }

    /* ---- 矩阵乘 ---- */
    matmul(o) {
      const self = this;
      // 双 batched：前导形状相同 -> 逐切片 2D 乘
      if (self.rank > 2 && o.rank > 2) {
        const lead = self.shape.slice(0, -2), leadO = o.shape.slice(0, -2);
        if (lead.join() !== leadO.join()) throw new Error('batched matmul 前导形状不一致');
        const B = numel(lead), M = self.shape[self.rank - 2], K = self.shape[self.rank - 1];
        const N = o.shape[o.rank - 1];
        const d = new Float64Array(B * M * N);
        const aS = M * K, bS = K * N, oS = M * N;
        for (let b = 0; b < B; b++) {
          const ao = b * aS, bo = b * bS, oo = b * oS;
          for (let i = 0; i < M; i++) for (let k = 0; k < K; k++) {
            const av = self.data[ao + i * K + k]; if (av === 0) continue;
            for (let j = 0; j < N; j++) d[oo + i * N + j] += av * o.data[bo + k * N + j];
          }
        }
        const out = new Tensor(d, lead.concat([M, N]), { requiresGrad: self.requiresGrad || o.requiresGrad });
        out._ctx = {
          parents: [self, o],
          backward(t) {
            const g = t.grad;
            for (let b = 0; b < B; b++) {
              const ao = b * aS, bo = b * bS, oo = b * oS;
              for (let i = 0; i < M; i++) for (let k = 0; k < K; k++) {
                let av = 0;
                for (let j = 0; j < N; j++) av += g[oo + i * N + j] * o.data[bo + k * N + j];
                self.grad[ao + i * K + k] += av;
              }
              for (let k = 0; k < K; k++) for (let j = 0; j < N; j++) {
                let acc = 0;
                for (let i = 0; i < M; i++) acc += g[oo + i * N + j] * self.data[ao + i * K + k];
                o.grad[bo + k * N + j] += acc;
              }
            }
          }
        };
        return out;
      }
      // a 为 >2D、b 为 2D -> reshape 技巧
      if (self.rank > 2 && o.rank === 2) {
        const M = numel(self.shape.slice(0, -1)), K = self.shape[self.rank - 1], N = o.shape[1];
        if (K !== o.shape[0]) throw new Error('matmul 内维不匹配: ' + K + ' vs ' + o.shape[0]);
        const out2 = self.reshape([M, K]).matmul(o);
        return out2.reshape(self.shape.slice(0, -1).concat([N]));
      }
      // a 为 2D、b 为 >2D -> 逐 batch（b 的每个前导切片右乘）
      if (self.rank === 2 && o.rank > 2) {
        const B = numel(o.shape.slice(0, -2)), K = o.shape[o.rank - 2], N = o.shape[o.rank - 1];
        if (self.shape[1] !== K) throw new Error('matmul 内维不匹配');
        const M = self.shape[0], bS = K * N, oS = M * N;
        const d = new Float64Array(B * M * N);
        for (let b = 0; b < B; b++) for (let i = 0; i < M; i++) for (let k = 0; k < K; k++) {
          const av = self.data[i * K + k]; if (av === 0) continue;
          for (let j = 0; j < N; j++) d[b * oS + i * N + j] += av * o.data[b * bS + k * N + j];
        }
        const out = new Tensor(d, o.shape.slice(0, -2).concat([M, N]), { requiresGrad: self.requiresGrad || o.requiresGrad });
        out._ctx = {
          parents: [self, o],
          backward(t) {
            const g = t.grad;
            for (let i = 0; i < M; i++) for (let k = 0; k < K; k++) {
              let av = 0;
              for (let b = 0; b < B; b++) for (let j = 0; j < N; j++) av += g[b * oS + i * N + j] * o.data[b * bS + k * N + j];
              self.grad[i * K + k] += av;
            }
            for (let b = 0; b < B; b++) for (let k = 0; k < K; k++) for (let j = 0; j < N; j++) {
              let acc = 0;
              for (let i = 0; i < M; i++) acc += g[b * oS + i * N + j] * self.data[i * K + k];
              o.grad[b * bS + k * N + j] += acc;
            }
          }
        };
        return out;
      }
      // 纯 2D×2D
      const M = self.shape[0], K = self.shape[1], N = o.shape[1];
      if (K !== o.shape[0]) throw new Error('matmul 内维不匹配: ' + K + ' vs ' + o.shape[0]);
      const d = new Float64Array(M * N);
      for (let i = 0; i < M; i++) for (let k = 0; k < K; k++) {
        const av = self.data[i * K + k]; if (av === 0) continue;
        for (let j = 0; j < N; j++) d[i * N + j] += av * o.data[k * N + j];
      }
      const out = new Tensor(d, [M, N], { requiresGrad: self.requiresGrad || o.requiresGrad });
      out._ctx = {
        parents: [self, o],
        backward(t) {
          const g = t.grad;
          for (let i = 0; i < M; i++) for (let k = 0; k < K; k++) {
            let av = 0;
            for (let j = 0; j < N; j++) av += g[i * N + j] * o.data[k * N + j];
            self.grad[i * K + k] += av;
          }
          for (let k = 0; k < K; k++) for (let j = 0; j < N; j++) {
            let acc = 0;
            for (let i = 0; i < M; i++) acc += g[i * N + j] * self.data[i * K + k];
            o.grad[k * N + j] += acc;
          }
        }
      };
      return out;
    }

    /* ---- Embedding 查表 ---- */
    static gatherRows(W, idx) {
      const d = W.shape[1];
      const n = idx.length;
      const out = new Float64Array(n * d);
      for (let i = 0; i < n; i++) {
        const r = idx[i];
        for (let j = 0; j < d; j++) out[i * d + j] = W.data[r * d + j];
      }
      const t = new Tensor(out, [n, d], { requiresGrad: W.requiresGrad });
      t._ctx = {
        parents: [W],
        backward(tt) {
          if (!W.grad) return;
          const g = tt.grad;
          for (let i = 0; i < n; i++) {
            const r = idx[i];
            for (let j = 0; j < d; j++) W.grad[r * d + j] += g[i * d + j];
          }
        }
      };
      return t;
    }

    /* ---- 反传 ---- */
    backward() {
      const topo = [];
      const visited = new Set();
      const stack = [[this, false]];
      while (stack.length) {
        const [node, processed] = stack.pop();
        if (processed) { topo.push(node); continue; }
        if (visited.has(node)) continue;
        visited.add(node);
        stack.push([node, true]);
        if (node._ctx) for (const p of node._ctx.parents) if (!visited.has(p)) stack.push([p, false]);
      }
      this.grad = new Float64Array(this.size).fill(1);
      for (let i = topo.length - 1; i >= 0; i--) {
        const t = topo[i];
        if (t._ctx && t.grad) {
          for (const p of t._ctx.parents) if (!p.grad) p.grad = new Float64Array(p.size);
          t._ctx.backward(t);
        }
      }
    }
  }

  /* ---- 梯度累加原语 ---- */
  function addInto(gradBuf, g) { for (let i = 0; i < g.length; i++) gradBuf[i] += g[i]; }
  function negateInto(gradBuf, g) { for (let i = 0; i < g.length; i++) gradBuf[i] -= g[i]; }
  function scaleInto(gradBuf, g, y) { for (let i = 0; i < g.length; i++) gradBuf[i] += g[i] * y[i]; }
  function divIntoGrad(gradBuf, g, x) { for (let i = 0; i < g.length; i++) gradBuf[i] += g[i] / x[i]; }

  /** 广播逐元素二元运算；反向经 indexMap 散射自动完成 unbroadcast */
  function broadcastBinary(a, b, fn, tag) {
    const outShape = broadcastShapes(a.shape, b.shape);
    const mapA = indexMap(a.shape, outShape);
    const mapB = indexMap(b.shape, outShape);
    const n = numel(outShape);
    const d = new Float64Array(n);
    for (let i = 0; i < n; i++) d[i] = fn(a.data[mapA[i]], b.data[mapB[i]]);
    const out = new Tensor(d, outShape, { requiresGrad: a.requiresGrad || b.requiresGrad });
    out._ctx = {
      parents: [a, b],
      backward(t) {
        const g = t.grad;
        for (let i = 0; i < n; i++) {
          const av = a.data[mapA[i]], bv = b.data[mapB[i]], gi = g[i];
          if (tag === 'add') { a.grad[mapA[i]] += gi; b.grad[mapB[i]] += gi; }
          else if (tag === 'sub') { a.grad[mapA[i]] += gi; b.grad[mapB[i]] -= gi; }
          else if (tag === 'mul') { a.grad[mapA[i]] += gi * bv; b.grad[mapB[i]] += gi * av; }
          else { a.grad[mapA[i]] += gi / bv; b.grad[mapB[i]] -= gi * av / (bv * bv); }
        }
      }
    };
    return out;
  }

  /* ------------------------------------------------------------------ *
   * 梯度检验基础设施（中心有限差分 vs 解析梯度）                          *
   * ------------------------------------------------------------------ */

  /**
   * 对一组参数做中心差分梯度检验。
   * makeLoss: () => 标量 Tensor（每次调用构建全新计算图）
   * 返回 { maxRel, maxWhere }。
   */
  function gradCheck(params, makeLoss, opts = {}) {
    const eps = opts.eps || 1e-5;
    const rtol = opts.rtol || 1e-5;
    // 关键：梯度本征为零时必须用绝对容差。
    // 例：注意力对 K 的偏置有平移不变性（softmax 对整行加常数不变），
    //     ∂L/∂b_K = 0，此时相对误差无意义，中心差分只剩 1e-9 量级噪声。
    const atol = opts.atol || 1e-7;
    let maxRel = 0, maxAbs = 0, maxWhere = '', ok = true;
    for (const p of params) if (p.grad) p.grad.fill(0);
    const loss0 = makeLoss();
    loss0.backward();
    const analytic = params.map(p => (p.grad ? p.grad.slice() : null));
    for (let pi = 0; pi < params.length; pi++) {
      const p = params[pi];
      for (let i = 0; i < p.size; i++) {
        const old = p.data[i];
        p.data[i] = old + eps; const lp = makeLoss().data[0];
        p.data[i] = old - eps; const lm = makeLoss().data[0];
        p.data[i] = old;
        const numeric = (lp - lm) / (2 * eps);
        const an = analytic[pi] ? analytic[pi][i] : 0;
        const abs = Math.abs(an - numeric);
        const rel = abs / Math.max(1e-8, Math.abs(an) + Math.abs(numeric));
        if (abs > atol + rtol * Math.abs(numeric)) ok = false;
        if (rel > maxRel) { maxRel = rel; maxWhere = p.shape.join('x') + '[' + i + ']'; }
        if (abs > maxAbs) maxAbs = abs;
      }
    }
    return { maxRel, maxAbs, maxWhere, ok };
  }

  /* ------------------------------------------------------------------ *
   * 导出                                                                 *
   * ------------------------------------------------------------------ */

  FAI.Tensor = Tensor;
  FAI.gradCheck = gradCheck;
  FAI.broadcastShapes = broadcastShapes;
  FAI.numel = numel;
  FAI.normalizeAxis = normalizeAxis;

  if (typeof module === 'object' && module.exports) module.exports = FAI;
})();
