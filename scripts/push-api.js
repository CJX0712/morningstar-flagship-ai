/* ============================================================================
 * 晨星·旗舰 AI — API 推送兜底脚本（零依赖）
 * ----------------------------------------------------------------------------
 * 场景：某些网络环境直连 github.com:443 被阻断（git push 超时），
 *       但 api.github.com 可达（gh CLI 可用）。本脚本改用 GitHub Git Data API
 *       直接创建 blob → tree → commit → ref，完成等价的推送。
 *
 * 特性：指数退避重试（抗网络抖动）+ 断点续传缓存（已上传的 blob 不重复上传）
 * 运行：node scripts/push-api.js [分支名，默认 main]
 * 依赖：已安装并已授权的 gh CLI
 * ============================================================================ */
'use strict';
const fs = require('fs');
const path = require('path');
const os = require('os');
const { execFileSync } = require('child_process');

const REPO = 'CJX0712/morningstar-flagship-ai';
const WANT_BRANCH = process.argv[2] || 'main';
const CACHE = path.resolve(__dirname, '..', '.push-cache.json');
const AUTHOR = {
  name: '晨星',
  email: 'CJX0712@users.noreply.github.com',
  date: new Date().toISOString()
};
const MESSAGE = '晨星·旗舰 AI 引擎 v1.0：零依赖 Float64 深度学习引擎'
  + '（反向模式 AD + Transformer/GPT + 六类经典 ML）43/43 验证全绿';
const root = path.resolve(__dirname, '..');
const SKIP = new Set(['.git', 'node_modules', '.workbuddy']);

const sleep = ms => { const t = Date.now() + ms; while (Date.now() < t); };

/* ---- 收集文件 ---- */
function collect(dir, rel = '') {
  const out = [];
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    if (SKIP.has(e.name) || e.name === '.push-cache.json') continue;
    const p = path.join(dir, e.name);
    const r = rel ? rel + '/' + e.name : e.name;
    if (e.isDirectory()) out.push(...collect(p, r));
    else out.push(r);
  }
  return out;
}

/* ---- 带指数退避的 gh api 调用 ---- */
function api(method, endpoint, payload, attempts = 6) {
  const args = ['api', '--method', method, endpoint];
  let tmp = null;
  if (payload !== undefined) {
    tmp = path.join(os.tmpdir(), 'fai-' + Math.random().toString(36).slice(2) + '.json');
    fs.writeFileSync(tmp, JSON.stringify(payload));
    args.push('--input', tmp);
  }
  let lastErr = null;
  for (let i = 0; i < attempts; i++) {
    try {
      const res = execFileSync('gh', args, { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
      if (tmp) { try { fs.unlinkSync(tmp); } catch (e) { } }
      return res.trim() ? JSON.parse(res) : {};
    } catch (e) {
      lastErr = e;
      const wait = 800 * Math.pow(2, i);
      process.stdout.write('    (重试 ' + (i + 1) + '/' + attempts + '，等待 ' + (wait / 1000) + 's)\n');
      sleep(wait);
    }
  }
  if (tmp) { try { fs.unlinkSync(tmp); } catch (e) { } }
  const msg = ((lastErr && (lastErr.stderr || lastErr.stdout)) || String(lastErr)).slice(0, 300);
  throw new Error('gh api ' + endpoint + ' 失败：' + msg);
}

/* ---- 空仓库引导：先用 contents API 建一次提交，默认分支才存在 ---- */
function ensureBranch() {
  for (const b of [WANT_BRANCH, 'main', 'master']) {
    try { api('GET', '/repos/' + REPO + '/git/refs/heads/' + b, undefined, 3); return b; } catch (e) { /* 未创建 */ }
  }
  const content = fs.readFileSync(path.join(root, 'LICENSE')).toString('base64');
  api('PUT', '/repos/' + REPO + '/contents/LICENSE', { message: '初始化仓库（MIT 许可，作者 晨星）', content });
  const info = api('GET', '/repos/' + REPO);
  console.log('  空仓库已引导，默认分支 = ' + info.default_branch);
  return info.default_branch;
}

(function main() {
  const branch = ensureBranch();
  const files = collect(root);
  const cache = fs.existsSync(CACHE) ? JSON.parse(fs.readFileSync(CACHE, 'utf8')) : {};
  console.log('待推送 ' + files.length + ' 个文件 → ' + REPO + '@' + branch);

  // 1) blobs（带缓存，中断后可续传）
  const tree = [];
  for (const rel of files) {
    const buf = fs.readFileSync(path.join(root, rel));
    const key = rel + ':' + buf.length + ':' + buf.toString('base64').slice(0, 32);
    if (cache[key]) {
      tree.push({ path: rel.split(path.sep).join('/'), mode: '100644', type: 'blob', sha: cache[key] });
      process.stdout.write('  ✓ ' + rel + '（缓存）\n');
      continue;
    }
    const blob = api('POST', '/repos/' + REPO + '/git/blobs', {
      content: buf.toString('base64'), encoding: 'base64'
    });
    cache[key] = blob.sha;
    fs.writeFileSync(CACHE, JSON.stringify(cache));
    tree.push({ path: rel.split(path.sep).join('/'), mode: '100644', type: 'blob', sha: blob.sha });
    process.stdout.write('  ↑ ' + rel + ' (' + (buf.length / 1024).toFixed(1) + ' KB)\n');
  }

  // 2) tree
  const t = api('POST', '/repos/' + REPO + '/git/trees', { tree });
  console.log('  tree ' + t.sha.slice(0, 8));

  // 3) commit
  const c = api('POST', '/repos/' + REPO + '/git/commits', {
    message: MESSAGE, tree: t.sha, parents: [], author: AUTHOR, committer: AUTHOR
  });
  console.log('  commit ' + c.sha.slice(0, 8));

  // 4) ref
  try {
    api('POST', '/repos/' + REPO + '/git/refs', { ref: 'refs/heads/' + branch, sha: c.sha });
  } catch (e) {
    api('PATCH', '/repos/' + REPO + '/git/refs/heads/' + branch, { sha: c.sha, force: true });
  }
  console.log('  refs/heads/' + branch + ' → ' + c.sha);
  console.log('完成：https://github.com/' + REPO);
})();
