/* ============================================================================
 * 晨星·旗舰 AI — 浏览器逻辑无头冒烟测试（DOM 桩）
 * ----------------------------------------------------------------------------
 * 目的：在没有浏览器的环境下验证 src/demo.js 的初始化、训练驱动、
 *       PCA/GMM 实验室与自检渲染逻辑不抛异常，且产出真实数据。
 * 运行：node test/smoke-browser.js
 * ============================================================================ */
'use strict';

require('../src/engine.js');
require('../src/utils.js');
require('../src/losses.js');
require('../src/optim.js');
require('../src/modules.js');
require('../src/classical.js');
require('../test/verify.js');

/* ---- 极简 DOM 桩 ---- */
const ctxStub = new Proxy({}, { get: () => () => { } });
const els = {};
const DEFAULT_VALUES = { 'sel-steps': '60', 'sel-lr': '0.003', 'sel-seed': '42' };

function makeEl(id) {
  if (els[id]) return els[id];
  const el = {
    id, value: DEFAULT_VALUES[id] || '', textContent: '', innerHTML: '',
    disabled: false, width: 640, height: 400, _h: {},
    getContext: () => ctxStub,
    addEventListener: (t, f) => { el._h[t] = f; }
  };
  els[id] = el;
  return el;
}

globalThis.document = { readyState: 'complete', getElementById: makeEl, addEventListener: () => { } };
globalThis.requestAnimationFrame = fn => setImmediate(fn);

const sleep = ms => new Promise(r => setTimeout(r, ms));

(async function main() {
  let failed = 0;
  const check = (name, ok, detail) => {
    if (!ok) failed++;
    console.log('  ' + (ok ? '✅' : '❌') + ' ' + name + (detail ? '  ' + detail : ''));
  };

  console.log('='.repeat(68));
  console.log(' 晨星·旗舰 AI — 浏览器逻辑无头冒烟测试');
  console.log('='.repeat(68));

  // 1. 初始化（加载 demo.js 即执行 init）
  require('../src/demo.js');
  check('init 无异常', true, '');
  check('PCA 实验室产出统计', els['pca-stats'].innerHTML.indexOf('正交性') !== -1, '');
  check('聚类实验室产出统计', els['clu-stats'].innerHTML.indexOf('GMM 纯度') !== -1, '');
  check('模型架构信息已渲染', els['lm-arch'].textContent.indexOf('参数') !== -1,
    els['lm-arch'].textContent);

  // 2. 训练驱动
  els['btn-train']._h.click();
  for (let i = 0; i < 200 && els['btn-train'].disabled; i++) await sleep(50);
  const st = els['lm-status'].textContent;
  check('训练完成并收尾', st.indexOf('训练完成') !== -1, st);
  const sample = els['lm-sample'].textContent;
  check('贪心生成文本非空', sample && sample.length > 10, '“' + sample.slice(0, 28) + '…”');

  // 3. 自检渲染
  els['btn-selftest']._h.click();
  await sleep(120);
  for (let i = 0; i < 200 && els['btn-selftest'].textContent === '运行中…'; i++) await sleep(50);
  const sum = els['selftest-summary'].innerHTML;
  check('自检结果已渲染', sum.indexOf('通过') !== -1, sum.replace(/<[^>]+>/g, '').trim());

  console.log('-'.repeat(68));
  console.log(failed ? ' ❌ ' + failed + ' 项失败' : ' ✅ 全部通过');
  console.log('='.repeat(68));
  process.exit(failed ? 1 : 0);
})();
