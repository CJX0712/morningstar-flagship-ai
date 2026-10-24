/* ============================================================================
 * 晨星·旗舰 AI 引擎 — 损失函数
 * ----------------------------------------------------------------------------
 * 血统：
 *   - 交叉熵 / MLE          —— 统计学习基本法则（最大似然 = 最小 NLL）
 *   - logsumexp 稳定实现     —— 防 exp 溢出（见 engine.js）
 *   - MSE / RMSE            —— 经典回归损失
 * ============================================================================ */
(function () {
  const G = typeof globalThis !== 'undefined' ? globalThis : this;
  const FAI = G.FAI || (G.FAI = {});
  const Tensor = FAI.Tensor;

  /** 均方误差：pred/target 同形状 */
  function mse(pred, target) {
    const t = target instanceof Tensor ? target : Tensor.of2D(target);
    const diff = pred.sub(t);
    return diff.mul(diff).mean(0).sum(0);
  }

  /**
   * 交叉熵（多分类，输入为原始 logits）：
   *   L = mean_i [ logsumexp(z_i) - z_i[y_i] ]
   * 数值稳定：logsumexp 内部做 max 平移。
   */
  function crossEntropy(logits, targetIdx) {
    const N = logits.shape[0], V = logits.shape[logits.rank - 1];
    const lse = logits.logsumexp();                 // (N,)
    const onehot = Tensor.oneHot(targetIdx, V);     // (N,V) 常量
    const correct = logits.mul(onehot).sum(1);      // (N,)
    return lse.sub(correct).mul(Tensor.constant(1 / N, [1])).sum(0);
  }

  FAI.mse = mse;
  FAI.crossEntropy = crossEntropy;

  if (typeof module === 'object' && module.exports) module.exports = FAI;
})();
