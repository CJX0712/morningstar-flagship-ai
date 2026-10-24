# 晨星·旗舰 AI 引擎

**MorningStar Flagship AI** —— 零依赖、零构建、Float64 全精度的深度学习与经典机器学习引擎。

> 仓库：<https://github.com/CJX0712/morningstar-flagship-ai> · 作者：晨星 · MIT License

一个文件打开就能跑：从反向模式自动微分到 Transformer / GPT，再到 PCA / KMeans++ / GMM(EM) / 朴素贝叶斯 / 逻辑回归 / CART，
每一行核心逻辑都附带**可被独立交叉验证的数学不变量**。

> 设计原则：**零重写 + 集大成** —— 不复刻轮子，直接把世界顶级成果（Vaswani Transformer、GPT-2 Pre-LN、Adam/AdamW、
> Jacobi 特征分解、k-means++、EM 算法）组合成一套可运行、可复现、可验证的引擎，并在其之上补足可验证性。

```
无头验证：43/43 全绿   |   梯度检验：中心差分 vs 解析梯度，误差 1e-9 量级
浏览器内自检：43/43   |   同种子训练：逐位可复现
```

---

## 一、30 秒上手

```bash
git clone https://github.com/CJX0712/morningstar-flagship-ai.git
cd morningstar-flagship-ai

node test/verify.js          # 无头全量验证（退出码 0 = 全绿）
node test/smoke-browser.js   # 浏览器逻辑无头冒烟测试
node scripts/build.js        # 生成单文件版 dist/flagship-ai.html
```

> 若所在网络直连 `github.com:443` 被阻断（`git push` 超时），而 `api.github.com` 可达，
> 可用兜底脚本 `node scripts/push-api.js` 经 GitHub Git Data API 完成等价推送（含重试与断点续传）。

或者 **双击 `index.html`**（或单文件版 `dist/flagship-ai.html`）——零服务器、零依赖、零构建，直接在浏览器里：

1. 训练一个字符级迷你 GPT，看损失曲线实时下降并生成文本；
2. 跑 PCA 降维，读主成分正交性与 Jacobi 残差；
3. 跑 KMeans++ 与 GMM/EM，看 EM 对数似然单调上升；
4. **一键自检**：浏览器内执行与 Node 完全同源的 43 项验证。

在代码里使用（Node 与浏览器同构）：

```js
const FAI = require('./src/engine.js');      // 浏览器：<script src="src/engine.js"> 后直接用 FAI
require('./src/utils.js'); require('./src/losses.js');
require('./src/optim.js'); require('./src/modules.js'); require('./src/classical.js');

// —— 三层 MLP 分类
const rng = FAI.mulberry32(42);
const mlp = new FAI.MLP([2, 16, 2], { rng, activation: 'tanh' });
const opt = new FAI.Adam(mlp.parameters(), 1e-2);
new FAI.Trainer(mlp, FAI.crossEntropy, opt, { batchSize: 16, rng }).fit(X, y, 40);

// —— 迷你 GPT
const gpt = new FAI.TinyGPT({ vocabSize: V, dModel: 32, heads: 4, layers: 2, blockSize: 16, rng });
const loss = gpt.loss(idxFlat, [B, T], targets);   // 交叉熵（含 logsumexp 稳定实现）
loss.backward(); opt.step(); opt.zeroGrad();
```

---

## 二、架构总览

```
┌──────────────────────────────────────────────────────────────────┐
│                        应用层（index.html 控制台）                  │
│  迷你 GPT 训练场   PCA 实验室   聚类实验室(KMeans/GMM)   一键自检   │
└───────────────┬──────────────────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────────────┐
│  模型库 modules.js        │  经典算法 classical.js                │
│  Linear / Embedding       │  PCA(Jacobi)  / KMeans++             │
│  LayerNorm / Dropout      │  GMM(EM)      / GaussianNB           │
│  MultiHeadAttention       │  LogisticRegression / CART           │
│  TransformerBlock         │                                       │
│  TinyGPT / MLP            │                                       │
└───────────────┬──────────────────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────────────┐
│  优化器 optim.js    SGD(动量/Nesterov) · Adam · AdamW · 梯度裁剪    │
│  损失   losses.js   MSE · 交叉熵(logsumexp)                        │
│  工具   utils.js    mulberry32 PRNG · Box–Muller · 数据 · Trainer   │
└───────────────┬──────────────────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────────────┐
│        自动微分引擎 engine.js —— 反向模式 AD，Float64 全精度         │
│  Tensor(广播/规约/matmul/permute/reshape/embedding)                │
│  logsumexp · softmax · gradCheck(中心差分)                          │
└──────────────────────────────────────────────────────────────────┘
```

---

## 三、目录结构

| 路径 | 说明 |
| --- | --- |
| `src/engine.js` | 自动微分引擎：Tensor、广播、规约、matmul、logsumexp、反向模式反传、梯度检验 |
| `src/modules.js` | Linear / Embedding / LayerNorm / Dropout / MultiHeadAttention / TransformerBlock / TinyGPT / MLP |
| `src/optim.js` | SGD（动量、Nesterov）/ Adam / AdamW / 梯度裁剪 / 余弦退火 |
| `src/losses.js` | MSE、交叉熵（logsumexp 稳定） |
| `src/utils.js` | mulberry32 PRNG、Box–Muller 采样、标准化、数据集生成、Trainer |
| `src/classical.js` | PCA(Jacobi) / KMeans++ / GMM(EM) / 高斯朴素贝叶斯 / 逻辑回归 / CART |
| `src/demo.js` | 浏览器控制台逻辑（四个实验室） |
| `test/verify.js` | 43 项验证套件（浏览器与 Node 同源，导出 `FAI.runSelfTest`） |
| `test/smoke-browser.js` | 浏览器逻辑的无头冒烟测试（DOM 桩） |
| `scripts/build.js` | 单文件构建（内联全部脚本 → `dist/flagship-ai.html`） |
| `scripts/push-api.js` | 网络受限时的 API 推送兜底（blob→tree→commit→ref，带重试与续传） |
| `index.html` | 控制台入口（script 标签依序加载，可直接双击打开） |
| `dist/flagship-ai.html` | 单文件成品（110 KB，零外部依赖，可离线分享） |

---

## 四、数学血统（站在世界顶级成果之上）

| 组件 | 血统 |
| --- | --- |
| 反向模式自动微分 | Baydin, Pearlmutter, Radford, Siskind (2018) *Automatic Differentiation in ML: a Survey*；计算图与反传结构参考 PyTorch autograd |
| Transformer / 缩放点积注意力 | Vaswani et al. (2017) *Attention Is All You Need* |
| Pre-LN 残差块 | Radford et al. (2019) GPT-2 |
| GELU 激活 | Hendrycks & Gimpel (2016)，tanh 近似 |
| LayerNorm | Ba, Kiros & Hinton (2016) |
| He / Xavier 初始化 | He et al. (2015)；Glorot & Bengio (2010) |
| Adam / AdamW | Kingma & Ba (2015)；Loshchilov & Hutter (2019) |
| 梯度裁剪 | Pascanu, Mikolov & Bengio (2013) |
| logsumexp 稳定化 | 数值分析经典（max 平移），防 exp 溢出 |
| PCA | Hotelling (1933)；特征分解用循环 Jacobi 旋转（Jacobi 1846） |
| KMeans / k-means++ | Lloyd (1982)；Arthur & Vassilvitskii (2007)；多次重启取最小 SSE（sklearn `n_init`） |
| GMM / EM | Dempster, Laird & Rubin (1977) |
| 高斯朴素贝叶斯 | Bishop, *PRML* Ch.4 |
| 逻辑回归 | 最大似然 = 最小交叉熵；Nelder & Wedderburn (1972) GLM |
| CART 决策树 | Breiman et al. (1984)，Gini 不纯度 |
| 随机数 | mulberry32（32 位种子 PRNG）；Box & Muller (1958) 正态采样 |

---

## 五、验证总纲：43 项不变量

`node test/verify.js` 输出（节选）：

| 分组 | 不变量 |
| --- | --- |
| **梯度检验**（19 项） | 解析梯度 vs 中心有限差分：Linear、广播四则、sum/mean 多轴、tanh/relu/sigmoid/exp/sqrt/log、matmul+reshape+permute 链、logsumexp、softmax、交叉熵、MSE、LayerNorm、GELU、多头注意力、TransformerBlock、TinyGPT 端到端 |
| **数值稳定性**（4 项） | logsumexp 输入 10000/10001 不溢出且值与解析解一致；softmax 行和 = 1（激励×100）；大 logits 交叉熵无 NaN；因果掩码上三角正确 |
| **优化器**（4 项） | Adam 收敛 (x−3)² 至 1e-8；SGD+动量收敛；AdamW 解耦衰减趋零；clipGradNorm 范数精确裁剪 |
| **经典算法**（11 项） | PCA 主成分正交归一（4.4e-16）、Jacobi 残差 ‖Cv−λv‖（7.6e-15）、重建误差随 k 下降；KMeans SSE 逐轮单调不增 + 纯度 1.0；GMM 对数似然单调不减（EM 定理）+ 责任度行和 = 1 + 纯度 1.0；朴素贝叶斯后验行和 = 1 + 精度 1.0；逻辑回归损失下降 + 精度 1.0；CART 训练精度 1.0（叶纯） |
| **端到端**（4 项） | TinyGPT 字符级训练 loss 4.75 → 0.12；同种子两次训练逐位一致；Trainer+MLP 损失下降；贪心生成可读序列 |
| **数学不变量**（1 项） | **注意力 K 偏置平移不变性**：∂L/∂b_K = 0（见下） |

### 一条值得单独说明的不变量：注意力对 K 偏置的平移不变性

`scores = Q(K + 1δᵀ)ᵀ = QKᵀ + (q·δ)·1ᵀ` —— 对整行加同一个常数，**softmax 不变**，
因此注意力权重不变、输出不变，于是：

```
∂L/∂b_K = 0        （实测 |∂L/∂b_K| = 7.99e-15）
∂L/∂b_V = 98.0     ∂L/∂b_Q = 21.5   （对照组，显著非零）
```

这条性质让「相对误差」判据失效（分母趋于 0），所以梯度检验采用 **PyTorch 判据**
`|解析 − 数值| ≤ atol + rtol·|数值|`（`atol=1e-7, rtol=1e-5`），
而不是朴素的相对误差 —— 这是旗舰引擎与玩具实现的分水岭之一。

---

## 六、精度与性能

| 指标 | 实测 |
| --- | --- |
| 数值精度 | Float64 全程（含中间激活），非 Float32 |
| 梯度检验误差 | 绝大多数算子 **1e-9 ~ 1e-11**；深层模块 |Δ| ≤ 7.3e-7 |
| 端到端训练 | 迷你 GPT（词表 59、dModel 32、4 头、2 层、29.8K 参数）60 步：loss 5.04 → 0.27 |
| 确定性 | 同种子两次训练，**逐位相同**（mulberry32 + 固定初始化） |
| 依赖 | **0**。无 npm、无构建、无 CDN、无 WASM |
| 单文件成品 | `dist/flagship-ai.html` ≈ 110 KB，离线可用 |

---

## 七、API 速查

```js
// —— 张量（src/engine.js）
T.zeros/ones/randn/of2D/oneHot/causalMask(shape…)
t.add/sub/mul/div/neg/exp/log/sqrt/tanh/relu/sigmoid
t.matmul(o)            // 2D×2D、>2D×2D（reshape 技巧）、双 batched
t.sum/mean(axis, keepdims)
t.reshape(shape) / t.permute(perm) / t.transpose()
t.logsumexp()          // 沿最后一维，数值稳定
loss.backward()        // 反向模式反传
FAI.gradCheck(params, makeLoss, {eps, rtol, atol})  // 中心差分梯度检验

// —— 模块（src/modules.js）
new FAI.Linear(inF, outF, {rng, init:'he'|'xavier'})
new FAI.Embedding(V, d, {rng}) / new FAI.LayerNorm(d) / new FAI.Dropout(p, rng)
new FAI.MultiHeadAttention(dModel, heads, {mask, rng})
new FAI.TransformerBlock(dModel, heads, {mask, ffnMult, dropout, rng})
new FAI.TinyGPT({vocabSize, dModel, heads, layers, blockSize, dropout, rng})
new FAI.MLP([2,16,2], {activation:'tanh'|'relu'|'sigmoid'})
model.parameters() / model.zeroGrad() / model.countParameters() / model.eval()

// —— 优化器与损失（src/optim.js, src/losses.js）
new FAI.SGD(params, lr, {momentum, nesterov, weightDecay})
new FAI.Adam(params, lr, {beta1, beta2, eps, weightDecay}) / new FAI.AdamW(...)
FAI.clipGradNorm(params, maxNorm) / FAI.cosineLr(base, epoch, total)
FAI.mse(pred, target) / FAI.crossEntropy(logits, targetIdx)

// —— 经典算法（src/classical.js）
new FAI.PCA(k).fit(X) → transform / reconstruct / explainedVarianceRatio()
new FAI.KMeans(k, {rng, nInit, maxIter}).fit(X) → labels, centroids, sseHistory
new FAI.GMM(k, {rng, maxIter, reg}).fit(X) → predict / predictProba, llHistory
new FAI.GaussianNB().fit(X, y) / new FAI.LogisticRegression({lr, iters, reg})
new FAI.DecisionTree({maxDepth, minSamplesSplit})

// —— 工具（src/utils.js）
FAI.mulberry32(seed) / FAI.randn(rng) / FAI.shuffle / FAI.zScore
FAI.makeBlobs(n, centers, std, rng) / FAI.makeMoons(n, noise, rng)
FAI.accuracy / FAI.clusterPurity / FAI.confusionMatrix
new FAI.Trainer(model, lossFn, optimizer, {batchSize, rng, clip}).fit(X, y, epochs)
```

---

## 八、设计取舍（为什么这样做）

| 决策 | 理由 |
| --- | --- |
| Float64 而非 Float32 | 让中心差分梯度检验能真正达到 1e-9；教学/验证价值高于吞吐 |
| 手写 AD 而非引入 TensorFlow.js | 满足「单文件、零依赖、可离线」的硬约束；AD 是引擎的地基，必须可审计 |
| 复合算子优先（softmax 由 logsumexp 构造） | 反向正确性由已验证原语复合而来，减少手写雅可比出错面 |
| 经典算法全部带不变量 | 「跑通」不等于「正确」；SSE 单调、LL 单调、正交归一、后验行和都是可证伪的硬指标 |
| 经典 ML 用普通数组、深度学习用 Tensor | 各取所长：EM/树不需要 AD，避免无谓的图开销 |

---

## 九、路线图

- [ ] LSTM / GRU（含时间步反传与梯度检验）
- [ ] 卷积层（im2col）与简易 CNN
- [ ] 全协方差 GMM、变分推断雏形
- [ ] Beam search 与温度采样
- [ ] 单文件版体积优化与 Web Worker 训练

---

## 十、作者与许可

**作者：晨星（Morning Star）** · 深圳 · GitHub [@CJX0712](https://github.com/CJX0712)

MIT License —— 自由使用、修改、分发，保留署名即可。

> 夜里能自己解决的事别去吵人，天亮交出去的东西必须是真的能用。
