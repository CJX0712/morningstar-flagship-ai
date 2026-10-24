/* ============================================================================
 * 晨星·旗舰 AI — 单文件构建脚本（零依赖）
 * 把 index.html 中引用的 src/*.js 与 test/verify.js 内联，产出：
 *   dist/flagship-ai.html   —— 可直接双击打开、可离线、可分享的单文件版本
 * 运行：node scripts/build.js
 * ============================================================================ */
'use strict';
const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const entry = path.join(root, 'index.html');
const outFile = path.join(root, 'dist', 'flagship-ai.html');

let html = fs.readFileSync(entry, 'utf8');
let count = 0;
html = html.replace(/<script src="((?:src|test)\/[^"]+)"><\/script>/g, (m, rel) => {
  const code = fs.readFileSync(path.join(root, rel), 'utf8');
  if (code.indexOf('</script>') !== -1) throw new Error('源码含 </script>，需转义: ' + rel);
  count++;
  return '<script>\n/* ===== ' + rel + ' ===== */\n' + code + '\n</script>';
});

if (!count) throw new Error('未内联任何脚本，检查 index.html 的 script 标签');

const stamp = new Date().toISOString().slice(0, 10);
html = html.replace('<title>', '<!-- 单文件构建版 · ' + stamp + ' · 作者：晨星 -->\n<title>');

fs.mkdirSync(path.dirname(outFile), { recursive: true });
fs.writeFileSync(outFile, html);

const kb = (Buffer.byteLength(html, 'utf8') / 1024).toFixed(1);
console.log('已内联 ' + count + ' 个脚本 -> dist/flagship-ai.html（' + kb + ' KB，零外部依赖）');
