/* app.js — UI for AIoT Regression Lab */
(function () {
  'use strict';
  const ML = window.ML;
  const $ = id => document.getElementById(id);
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  /* ---------- 格式化 ---------- */
  function fmt(v, d) {
    if (v === null || v === undefined || Number.isNaN(v)) return '—';
    if (!Number.isFinite(v)) return '∞';
    d = d === undefined ? 2 : d;
    return v.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
  }
  function fmtAuto(v) {
    if (!Number.isFinite(v)) return fmt(v);
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e9)) return v.toExponential(3);
    return fmt(v, a >= 1000 ? 2 : a >= 1 ? 4 : 5);
  }
  function fmtP(p) {
    if (!Number.isFinite(p)) return '—';
    if (p < 1e-4) return p.toExponential(2);
    return p.toFixed(4);
  }
  const pct = v => fmt(v * 100, 0) + '%';
  function sigPill(p) {
    if (!Number.isFinite(p)) return '';
    if (p < 0.01) return '<span class="pill good">***</span>';
    if (p < 0.05) return '<span class="pill good">**</span>';
    if (p < 0.1) return '<span class="pill warn">*</span>';
    return '<span class="pill bad">n.s.</span>';
  }
  function kpi(label, value, sub, cls) {
    return `<div class="kpi ${cls || ''}"><span class="eyebrow">${label}</span><span class="v">${value}</span>${sub ? `<span class="s">${sub}</span>` : ''}</div>`;
  }
  function setStatus(msg, cls) { const el = $('ds-status'); el.className = 'status ' + (cls || ''); el.innerHTML = msg; }

  /* ---------- 主題色（給 Chart.js） ---------- */
  function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  const charts = {};
  function chart(id, cfg) {
    if (!window.Chart) return;
    if (charts[id]) charts[id].destroy();
    const ink = css('--muted'), grid = css('--grid');
    Chart.defaults.color = ink;
    Chart.defaults.font.family = css('--font-body') || 'sans-serif';
    Chart.defaults.font.size = 11;
    cfg.options = cfg.options || {};
    cfg.options.responsive = true; cfg.options.maintainAspectRatio = false;
    cfg.options.animation = matchMedia('(prefers-reduced-motion: reduce)').matches ? false : { duration: 250 };
    const sc = cfg.options.scales || {};
    Object.values(sc).forEach(s => { s.grid = Object.assign({ color: grid }, s.grid || {}); s.border = { color: grid }; });
    cfg.options.plugins = Object.assign({ legend: { labels: { boxWidth: 12, boxHeight: 2 } } }, cfg.options.plugins || {});
    charts[id] = new Chart($(id), cfg);
  }
  const line = (label, data, color, extra) => Object.assign({ label, data, borderColor: color, backgroundColor: color, pointRadius: 3, pointHoverRadius: 5, borderWidth: 2, tension: 0 }, extra || {});

  /* ---------- 資料來源 ---------- */
  const SOURCES = {
    mlr: [
      { label: '50_Startups（Avik-Jain）', url: 'https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv' },
      { label: '50_Startups（harimittapalli）', url: 'https://raw.githubusercontent.com/harimittapalli/Mulitple-Linear-Reggression/master/50_Startups.csv' },
      { label: 'Boston Housing', url: 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv' },
      { label: 'Advertising', url: 'https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/master/data/Advertising.csv' },
      { label: '還原內建 50_Startups', builtin: 'startups' }
    ],
    ar: [
      { label: '還原內建 2330.TW 快照', builtin: 'tsmc' },
      { label: 'AirPassengers（月資料）', url: 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv' },
      { label: 'Daily Min Temperatures', url: 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv' }
    ]
  };
  function normalizeUrl(u) {
    u = u.trim();
    let m = u.match(/^https?:\/\/github\.com\/([^/]+)\/([^/]+)\/blob\/(.+)$/);
    if (m) return `https://raw.githubusercontent.com/${m[1]}/${m[2]}/${m[3]}`;
    m = u.match(/^https?:\/\/gist\.github\.com\/([^/]+)\/([0-9a-f]+)\/?$/i);
    if (m) return `https://gist.githubusercontent.com/${m[1]}/${m[2]}/raw`;
    return u;
  }
  async function fetchText(url) {
    let res;
    try { res = await fetch(url, { cache: 'no-store' }); }
    catch (e) { throw new Error('瀏覽器無法讀取這個網址（可能是跨網域 CORS 限制或網路被擋）。請改用 GitHub raw / gist raw 連結，或下載後用「上傳 CSV」。'); }
    if (!res.ok) throw new Error(`伺服器回應 HTTP ${res.status}，請確認網址是否正確、檔案是否公開。`);
    return res.text();
  }

  const state = {
    problem: 'mlr',
    mlr: { raw: null, label: '', url: '', target: null, testSize: 0.2, seed: 0, algo: 'lasso', criterion: 'cv', k: null, cache: {} },
    ar: { raw: null, label: '', url: '', col: null, testFrac: 0.2, p: null, h: 10 }
  };

  /* ====================================================================
   * Problem 2：多元線性回歸
   * ==================================================================== */
  function prepareMLR(raw, target) {
    const n0 = raw.n, notes = [];
    const yRaw = raw.cols[target].map(ML.toNumber);
    const keep = []; for (let i = 0; i < n0; i++) if (Number.isFinite(yRaw[i])) keep.push(i);
    if (keep.length < n0) notes.push(`移除目標值缺漏的 ${n0 - keep.length} 列。`);
    const n = keep.length;
    if (n < 10) throw new Error('有效資料少於 10 列，無法建立回歸模型。');
    const y = keep.map(i => yRaw[i]);
    const numeric = [], categorical = [], ignored = [];
    const colInfo = [];
    raw.header.forEach(h => {
      if (h === target) return;
      const vals = keep.map(i => raw.cols[h][i]);
      const nonMiss = vals.filter(v => !ML.isMissing(v));
      const nums = nonMiss.map(ML.toNumber).filter(Number.isFinite);
      const miss = vals.length - nonMiss.length;
      const uniq = new Set(nonMiss).size;
      const looksId = /^(id|index|no\.?|編號|序號|unnamed:?\s*\d*)$/i.test(h.trim()) ||
        (nums.length === vals.length && nums.every((v, i) => v === nums[0] + i));
      if (looksId && nums.length === nonMiss.length) {
        ignored.push({ h, why: 'ID / 流水號欄' });
        colInfo.push({ h, type: 'ignored', miss, uniq });
      } else if (nonMiss.length && nums.length / nonMiss.length >= 0.95) {
        const parsed = vals.map(ML.toNumber);
        const fin = parsed.filter(Number.isFinite), mu = fin.length ? ML.mean(fin) : 0;
        const missN = parsed.filter(v => !Number.isFinite(v)).length;
        if (missN) notes.push(`<code>${esc(h)}</code> 有 ${missN} 個缺值，以平均數 ${fmtAuto(mu)} 填補。`);
        const filled = parsed.map(v => (Number.isFinite(v) ? v : mu));
        if (ML.std(filled) === 0) { ignored.push({ h, why: '常數欄' }); return; }
        numeric.push({ h, values: filled, mean: ML.mean(filled) });
        colInfo.push({ h, type: 'numeric', miss, uniq });
      } else if (uniq >= 2 && uniq <= Math.min(30, Math.max(2, Math.floor(0.5 * n)))) {
        const cats = Array.from(new Set(vals.map(v => (ML.isMissing(v) ? 'Missing' : v)))).sort();
        categorical.push({ h, values: vals.map(v => (ML.isMissing(v) ? 'Missing' : v)), cats });
        colInfo.push({ h, type: 'categorical', miss, uniq });
      } else {
        ignored.push({ h, why: uniq < 2 ? '只有一種值' : '文字/ID 欄（類別太多）' });
        colInfo.push({ h, type: 'ignored', miss, uniq });
      }
    });
    // 特徵矩陣：數值欄 → one-hot（drop_first）
    const feats = [];
    numeric.forEach(c => feats.push({ name: c.h, src: c.h, kind: 'num' }));
    categorical.forEach(c => c.cats.slice(1).forEach(cat => feats.push({ name: `${c.h}_${cat}`, src: c.h, kind: 'dummy', cat })));
    if (!feats.length) throw new Error('找不到可用的特徵欄位（需要數值或少量類別的欄）。');
    const X = [];
    for (let r = 0; r < n; r++) {
      X.push(feats.map(f => {
        if (f.kind === 'num') return numeric.find(c => c.h === f.src).values[r];
        return categorical.find(c => c.h === f.src).values[r] === f.cat ? 1 : 0;
      }));
    }
    return { n, y, X, feats, numeric, categorical, ignored, notes, keep, colInfo, target };
  }

  function pickDefaultTarget(raw) {
    const numCols = raw.header.filter(h => { const v = raw.cols[h].filter(s => !ML.isMissing(s)); return v.length && v.filter(s => Number.isFinite(ML.toNumber(s))).length / v.length >= 0.95; });
    const pref = numCols.find(h => /^(profit|target|y|price|medv|sales|label)$/i.test(h.trim()));
    return pref || numCols[numCols.length - 1] || raw.header[raw.header.length - 1];
  }

  function mlrCtx() {
    const s = state.mlr, D = s.data;
    const sp = ML.trainTestSplit(D.n, s.testSize, s.seed);
    return {
      D, sp,
      Xtr: sp.train.map(i => D.X[i]), ytr: sp.train.map(i => D.y[i]),
      Xte: sp.test.map(i => D.X[i]), yte: sp.test.map(i => D.y[i])
    };
  }

  function algoAllowed(a, p) { return a !== 'exhaustive' || p <= 12; }

  function getEval(algo) {
    const s = state.mlr, key = `${algo}|${s.criterion}`;
    if (!s.cache[key]) {
      const sets = ML.selectSets(algo, s.ctx, s.criterion);
      s.cache[key] = ML.evaluateSets(sets, s.ctx);
    }
    return s.cache[key];
  }
  const critOf = (e, crit) => (crit === 'test' ? e.test.rmse : e.cv);
  function bestK(ev, crit) { let b = ev[0]; ev.forEach(e => { if (critOf(e, crit) < critOf(b, crit)) b = e; }); return b.k; }

  function loadMLR(raw, label, url) {
    const s = state.mlr;
    s.raw = raw; s.label = label; s.url = url || '';
    const sel = $('m-target');
    sel.innerHTML = raw.header.map(h => `<option>${esc(h)}</option>`).join('');
    s.target = pickDefaultTarget(raw);
    sel.value = s.target;
    s.k = null;
    rebuildMLR();
  }

  function rebuildMLR() {
    const s = state.mlr;
    try {
      s.data = prepareMLR(s.raw, s.target);
      s.ctx = mlrCtx();
      if (s.ctx.ytr.length <= s.data.feats.length + 2) throw new Error(`訓練集只有 ${s.ctx.ytr.length} 筆，但有 ${s.data.feats.length} 個特徵，請降低測試集比例或換資料集。`);
      s.cache = {};
      if (!algoAllowed(s.algo, s.data.feats.length)) s.algo = 'lasso';
      s.k = null;
      renderMLRUnderstanding();
      renderMLRPrep();
      renderMLRModel();
      setStatus(`已載入 <b>${esc(s.label)}</b>：${s.data.n} 列 × ${s.raw.header.length} 欄，目標 <code>${esc(s.target)}</code>，${s.data.feats.length} 個特徵（含 one-hot）。`, 'ok');
    } catch (e) {
      console.error(e);
      setStatus('無法建立模型：' + esc(e.message), 'err');
    }
  }

  function renderMLRUnderstanding() {
    const s = state.mlr, D = s.data, raw = s.raw;
    document.querySelectorAll('.tname').forEach(el => { el.textContent = s.target; });
    const featTxt = D.feats.length <= 8 ? D.numeric.map(c => c.h).concat(D.categorical.map(c => c.h)).map(h => `<code>${esc(h)}</code>`).join('、') : `${D.numeric.length + D.categorical.length} 個欄位`;
    $('m-biz').innerHTML = `目標：用 ${featTxt} 預測 <b>${esc(s.target)}</b>，並找出哪些因素真正影響 ${esc(s.target)}，以最少的特徵達到最好的預測效果（Occam's razor）。`;
    const missing = raw.header.reduce((a, h) => a + raw.cols[h].filter(ML.isMissing).length, 0);
    $('m-kpis').innerHTML = kpi('樣本數', D.n) + kpi('欄位數', raw.header.length) + kpi('數值特徵', D.numeric.length) + kpi('類別特徵', D.categorical.length, D.categorical.map(c => `${esc(c.h)}: ${c.cats.length} 類`).join('、')) + kpi('缺值格數', missing, missing ? '已處理' : '無缺值');
    // head
    const hdr = raw.header;
    $('m-head').innerHTML = `<thead><tr><th>#</th>${hdr.map(h => `<th class="${h === s.target ? '' : ''}">${esc(h)}</th>`).join('')}</tr></thead><tbody>` +
      Array.from({ length: Math.min(8, raw.n) }, (_, i) => `<tr><td>${i}</td>${hdr.map(h => { const v = raw.cols[h][i]; const num = ML.toNumber(v); return Number.isFinite(num) && String(v).trim() !== '' ? `<td>${esc(v)}</td>` : `<td class="txt">${esc(v)}</td>`; }).join('')}</tr>`).join('') + '</tbody>';
    // describe
    const numCols = D.numeric.map(c => ({ h: c.h, v: c.values })).concat([{ h: s.target, v: D.y, t: true }]);
    const q = (arr, p) => { const a = arr.slice().sort((x, y) => x - y); const idx = (a.length - 1) * p, lo = Math.floor(idx); return a[lo] + (a[Math.min(lo + 1, a.length - 1)] - a[lo]) * (idx - lo); };
    $('m-describe').innerHTML = '<thead><tr><th>欄位</th><th>型態</th><th>mean</th><th>std</th><th>min</th><th>25%</th><th>50%</th><th>75%</th><th>max</th></tr></thead><tbody>' +
      numCols.map(c => `<tr><td class="txt">${esc(c.h)}${c.t ? ' <span class="pill acc">y</span>' : ''}</td><td class="txt">數值</td><td>${fmt(ML.mean(c.v))}</td><td>${fmt(ML.std(c.v, 1))}</td><td>${fmt(Math.min(...c.v))}</td><td>${fmt(q(c.v, .25))}</td><td>${fmt(q(c.v, .5))}</td><td>${fmt(q(c.v, .75))}</td><td>${fmt(Math.max(...c.v))}</td></tr>`).join('') +
      D.categorical.map(c => `<tr><td class="txt">${esc(c.h)}</td><td class="txt">類別</td><td colspan="7" class="txt">${c.cats.map(k => `${esc(k)} (${c.values.filter(v => v === k).length})`).join('、')}</td></tr>`).join('') +
      D.ignored.map(c => `<tr><td class="txt">${esc(c.h)}</td><td class="txt">忽略</td><td colspan="7" class="txt">${esc(c.why)}</td></tr>`).join('') + '</tbody>';
    // correlation
    const corr = D.feats.map((f, j) => { const x = D.X.map(r => r[j]); const mx = ML.mean(x), my = ML.mean(D.y); let sxy = 0, sxx = 0, syy = 0; for (let i = 0; i < D.n; i++) { sxy += (x[i] - mx) * (D.y[i] - my); sxx += (x[i] - mx) ** 2; syy += (D.y[i] - my) ** 2; } return { name: f.name, r: sxy / Math.sqrt(sxx * syy), j }; }).sort((a, b) => Math.abs(b.r) - Math.abs(a.r));
    const top = corr.slice(0, 15);
    chart('c-corr', {
      type: 'bar',
      data: { labels: top.map(c => c.name), datasets: [{ label: 'r', data: top.map(c => c.r), backgroundColor: top.map(c => (c.r >= 0 ? css('--train') : css('--test'))), borderRadius: 3 }] },
      options: { indexAxis: 'y', plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => 'r = ' + c.raw.toFixed(3) } } }, scales: { x: { min: -1, max: 1 }, y: { ticks: { autoSkip: false } } } }
    });
    const best = corr[0];
    $('m-scatter-cap').textContent = `${best.name} vs ${s.target}（r = ${best.r.toFixed(3)}）`;
    chart('c-scatter', {
      type: 'scatter',
      data: { datasets: [{ label: s.target, data: D.X.map((r, i) => ({ x: r[best.j], y: D.y[i] })), backgroundColor: css('--train'), pointRadius: 3.5 }] },
      options: { plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: best.name } }, y: { title: { display: true, text: s.target } } } }
    });
  }

  function renderMLRPrep() {
    const s = state.mlr, D = s.data, ctx = s.ctx;
    $('m-test-out').textContent = pct(s.testSize);
    const steps = [];
    steps.push(`分離特徵 X 與目標 y＝<code>${esc(s.target)}</code>。`);
    D.notes.forEach(n => steps.push(n));
    if (D.categorical.length) steps.push(`類別欄 ${D.categorical.map(c => `<code>${esc(c.h)}</code>`).join('、')} 做 One-Hot Encoding，並丟掉第一類（${D.categorical.map(c => esc(c.cats[0])).join('、')}）避免 dummy variable trap：<code>pd.get_dummies(drop_first=True)</code>。`);
    if (D.ignored.length) steps.push(`忽略欄位：${D.ignored.map(c => `<code>${esc(c.h)}</code>（${esc(c.why)}）`).join('、')}。`);
    steps.push(`最終特徵 ${D.feats.length} 個：${D.feats.map(f => `<code>${esc(f.name)}</code>`).join(' ')}`);
    steps.push(`<code>train_test_split(test_size=${s.testSize}, random_state=${s.seed})</code> → 訓練 ${ctx.ytr.length} 筆、測試 ${ctx.yte.length} 筆（與 sklearn 相同的亂數切分；測試集 index：${ctx.sp.test.slice(0, 12).join(', ')}${ctx.sp.test.length > 12 ? '…' : ''}）。`);
    $('m-prep').innerHTML = steps.map(t => `<li>${t}</li>`).join('');
    const url = s.url || 'https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv';
    const cats = D.categorical.map(c => `'${c.h}'`).join(', ');
    $('m-code').textContent =
`import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. Data Understanding
df = pd.read_csv("${url}")
print(df.head()); print(df.describe())

# 2. Data Preparation
y = df["${s.target}"]
X = df.drop(columns=["${s.target}"${D.ignored.length ? ', ' + D.ignored.map(c => `"${c.h}"`).join(', ') : ''}])
${D.categorical.length ? `X = pd.get_dummies(X, columns=[${cats}], drop_first=True).astype(float)\n` : ''}X = X.fillna(X.mean(numeric_only=True))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=${s.testSize}, random_state=${s.seed})

# 3. Modeling：依特徵數 k = 1..${D.feats.length} 擬合 OLS（特徵順序由選擇演算法決定）
def evaluate(cols):
    m = LinearRegression().fit(X_train[cols], y_train)
    p = m.predict(X_test[cols])
    return np.sqrt(mean_squared_error(y_test, p)), r2_score(y_test, p), mean_absolute_error(y_test, p)

# 4. Evaluation
for k in range(1, X.shape[1] + 1):
    cols = ranking[:k]          # ranking 來自 Lasso / Forward / RFE…
    print(k, cols, evaluate(cols))`;
  }

  function renderAlgoButtons() {
    const s = state.mlr, p = s.data.feats.length;
    const keys = Object.keys(ML.ALGOS);
    $('m-algos').innerHTML = keys.map(k => `<button type="button" role="tab" data-algo="${k}" aria-selected="${s.algo === k}" ${algoAllowed(k, p) ? '' : 'disabled title="特徵超過 12 個時窮舉太慢"'}>${ML.ALGOS[k].short}</button>`).join('') +
      `<button type="button" role="tab" class="all" data-algo="all" aria-selected="${s.algo === 'all'}">全部比較</button>`;
  }

  const ALGO_NOTES = {
    lasso: '在標準化後的訓練資料上跑 Lasso 正則化路徑（α 由大到小），特徵越早出現非零係數排名越前；取前 k 名再用 OLS 擬合。',
    enet: 'ElasticNet 同時用 L1 + L2 懲罰（l1_ratio = 0.5），對高度相關的特徵較穩定；依路徑進入順序排名。',
    forward: '從空集合開始，每次加入讓「挑選準則」最小的特徵（greedy forward selection，課堂 Part 2 的做法）。',
    backward: '從全部特徵開始，每次剔除 p-value 最大的特徵（課堂筆記：Backward selection 最常用，p < 0.05 顯著）。',
    rfe: '用標準化係數的絕對值當重要度，每輪剔除最不重要的特徵（Recursive Feature Elimination）。',
    exhaustive: '對每個 k 試遍所有組合（2^p − 1 種），挑準則最小的子集；保證找到最佳但最耗時。',
    all: '同時比較所有特徵選擇演算法在每個 k 的測試集 RMSE。'
  };

  function renderMLRModel() {
    const s = state.mlr, D = s.data, p = D.feats.length;
    renderAlgoButtons();
    const kIn = $('m-k'); kIn.max = p;
    $('m-algo-note').textContent = ALGO_NOTES[s.algo] + (s.criterion === 'test' ? '　⚠ 以測試集挑選特徵會讓測試分數偏樂觀（資料洩漏），這裡保留是為了對照課堂結果。' : '');
    if (s.algo === 'all') return renderCompare();
    const ev = getEval(s.algo);
    const bk = bestK(ev, s.criterion);
    if (s.k === null || s.k > p) s.k = bk;
    kIn.value = s.k; $('m-k-out').textContent = s.k;
    $('m-k-wrap').hidden = false;
    $('m-results').innerHTML = '<thead><tr><th>k</th><th class="txt">Selected Features</th><th>CV RMSE</th><th>Train RMSE</th><th>Test RMSE</th><th>Test R²</th><th>Train Adj R²</th><th>Test MAE</th></tr></thead><tbody>' +
      ev.map(e => `<tr data-k="${e.k}" class="${e.k === s.k ? 'sel' : ''}"><td>${e.k}${e.k === bk ? ' <span class="pill good">best</span>' : ''}</td><td class="txt">${e.idx.map(j => esc(D.feats[j].name)).join(', ')}</td><td>${fmt(e.cv)}</td><td>${fmt(e.train.rmse)}</td><td>${fmt(e.test.rmse)}</td><td>${e.test.r2.toFixed(4)}</td><td>${e.model.adjR2.toFixed(4)}</td><td>${fmt(e.test.mae)}</td></tr>`).join('') + '</tbody>';
    const ks = ev.map(e => e.k);
    const ptR = ks.map(k => (k === s.k ? 6 : 3));
    chart('c-rmse', {
      type: 'line',
      data: { labels: ks, datasets: [line('Test RMSE', ev.map(e => e.test.rmse), css('--test'), { pointRadius: ptR }), line('CV RMSE (train 5-fold)', ev.map(e => e.cv), css('--cv'), { borderDash: [5, 4] }), line('Train RMSE', ev.map(e => e.train.rmse), css('--train'))] },
      options: { interaction: { mode: 'index', intersect: false }, scales: { x: { title: { display: true, text: 'Number of Features' } }, y: { title: { display: true, text: 'RMSE' } } } }
    });
    chart('c-r2', {
      type: 'line',
      data: { labels: ks, datasets: [line('Test R²', ev.map(e => e.test.r2), css('--test'), { pointRadius: ptR }), line('Train R²', ev.map(e => e.train.r2), css('--train')), line('Train Adj R²', ev.map(e => e.model.adjR2), css('--naive'), { borderDash: [5, 4] })] },
      options: { interaction: { mode: 'index', intersect: false }, scales: { x: { title: { display: true, text: 'Number of Features' } }, y: { title: { display: true, text: 'R-squared' } } } }
    });
    const full = ev[ev.length - 1], b = ev[bk - 1];
    const v = $('m-fit-verdict');
    if (bk < p && full.test.rmse > b.test.rmse) {
      v.className = 'callout warn';
      v.innerHTML = `<b>甜蜜點 k = ${bk}</b>（${b.idx.map(j => esc(D.feats[j].name)).join(', ')}）：Test RMSE ${fmt(b.test.rmse)}。全部 ${p} 個特徵時 Test RMSE 升到 ${fmt(full.test.rmse)}（+${fmt(full.test.rmse - b.test.rmse)}），但 Train RMSE 反而降到 ${fmt(full.train.rmse)}：訓練誤差下降、測試誤差上升，是加入雜訊特徵造成的<b>過度擬合</b>。`;
    } else {
      v.className = 'callout good';
      v.innerHTML = `依「${s.criterion === 'cv' ? 'CV RMSE' : 'Test RMSE'}」最佳的是 <b>k = ${bk}</b>，Test RMSE ${fmt(b.test.rmse)}、Test R² ${b.test.r2.toFixed(4)}。訓練與測試誤差差距 ${fmt(Math.abs(b.test.rmse - b.train.rmse))}，可用來判斷過擬合 / 欠擬合。`;
    }
    renderMLREval(ev[s.k - 1]);
  }

  function renderCompare() {
    const s = state.mlr, D = s.data, p = D.feats.length;
    $('m-k-wrap').hidden = true;
    const algos = Object.keys(ML.ALGOS).filter(a => algoAllowed(a, p));
    const evs = algos.map(a => ({ a, ev: getEval(a) }));
    const ks = Array.from({ length: p }, (_, i) => i + 1);
    const palette = [css('--train'), css('--test'), css('--cv'), css('--warn'), css('--bad'), css('--naive')];
    $('m-results').innerHTML = `<thead><tr><th>演算法</th>${ks.map(k => `<th>k=${k}</th>`).join('')}<th class="txt">最佳子集（依準則）</th></tr></thead><tbody>` +
      evs.map(({ a, ev }) => { const bk = bestK(ev, s.criterion); const minT = Math.min(...ev.map(e => e.test.rmse)); return `<tr data-algo="${a}"><td>${ML.ALGOS[a].short}</td>${ev.map(e => `<td>${e.test.rmse === minT ? '<b>' + fmt(e.test.rmse) + '</b>' : fmt(e.test.rmse)}</td>`).join('')}<td class="txt">k=${bk}：${ev[bk - 1].idx.map(j => esc(D.feats[j].name)).join(', ')}</td></tr>`; }).join('') + '</tbody>';
    chart('c-rmse', {
      type: 'line',
      data: { labels: ks, datasets: evs.map(({ a, ev }, i) => line(ML.ALGOS[a].short, ev.map(e => e.test.rmse), palette[i % palette.length], { borderDash: i % 2 ? [5, 4] : [] })) },
      options: { interaction: { mode: 'index', intersect: false }, scales: { x: { title: { display: true, text: 'Number of Features' } }, y: { title: { display: true, text: 'Test RMSE' } } } }
    });
    chart('c-r2', {
      type: 'line',
      data: { labels: ks, datasets: evs.map(({ a, ev }, i) => line(ML.ALGOS[a].short, ev.map(e => e.test.r2), palette[i % palette.length], { borderDash: i % 2 ? [5, 4] : [] })) },
      options: { interaction: { mode: 'index', intersect: false }, scales: { x: { title: { display: true, text: 'Number of Features' } }, y: { title: { display: true, text: 'Test R²' } } } }
    });
    let gb = null; evs.forEach(({ a, ev }) => ev.forEach(e => { if (!gb || e.test.rmse < gb.e.test.rmse) gb = { a, e }; }));
    $('m-fit-verdict').className = 'callout';
    $('m-fit-verdict').innerHTML = `${evs.length} 種演算法中測試集 RMSE 最低：<b>${ML.ALGOS[gb.a].short}, k = ${gb.e.k}</b>（${gb.e.idx.map(j => esc(D.feats[j].name)).join(', ')}），RMSE ${fmt(gb.e.test.rmse)}。點表格任一列可切換到該演算法。`;
    const ev0 = getEval(algos[0]);
    renderMLREval(ev0[bestK(ev0, s.criterion) - 1]);
  }

  function renderMLREval(e) {
    const s = state.mlr, D = s.data, ctx = s.ctx, m = e.model;
    const names = e.idx.map(j => D.feats[j].name);
    $('m-eval-title').textContent = `${s.algo === 'all' ? ML.ALGOS[Object.keys(ML.ALGOS)[0]].short : ML.ALGOS[s.algo].short} · k = ${e.k}`;
    $('m-metrics').innerHTML = kpi('Test R²', e.test.r2.toFixed(4), `${pct(Math.max(0, e.test.r2))} 變異被解釋`, e.test.r2 > 0.9 ? 'good' : e.test.r2 < 0.5 ? 'bad' : '') +
      kpi('Train Adj R²', m.adjR2.toFixed(4), `Train R² ${m.r2.toFixed(4)}`) +
      kpi('Test RMSE', fmt(e.test.rmse), `Train ${fmt(e.train.rmse)}`) +
      kpi('Test MAE', fmt(e.test.mae)) +
      kpi('Test MSE', fmt(e.test.mse, 0)) +
      kpi('F-statistic', fmt(m.F), 'p = ' + fmtP(m.Fp));
    $('m-eq').innerHTML = `<span class="y">ŷ</span>（${esc(s.target)}）= ${fmtAuto(m.beta[0])}` + names.map((nm, i) => { const b = m.beta[i + 1]; return ` ${b < 0 ? '−' : '+'} ${fmtAuto(Math.abs(b))} × ${esc(nm)}`; }).join('');
    const vifs = e.idx.length > 1 ? ML.vif(ML.pick(ctx.Xtr, e.idx)) : [1];
    $('m-coef').innerHTML = '<thead><tr><th>變數</th><th>β 係數</th><th>Std. Error</th><th>t</th><th>p-value</th><th>顯著</th><th>95% CI 下界</th><th>95% CI 上界</th><th>VIF</th></tr></thead><tbody>' +
      m.rows.map((r, i) => { const vf = i === 0 ? null : vifs[i - 1]; return `<tr><td class="txt">${i === 0 ? 'Intercept (β₀)' : esc(names[i - 1])}</td><td>${fmtAuto(r.coef)}</td><td>${fmtAuto(r.se)}</td><td>${fmt(r.t, 3)}</td><td>${fmtP(r.p)}</td><td>${sigPill(r.p)}</td><td>${fmtAuto(r.lo)}</td><td>${fmtAuto(r.hi)}</td><td>${vf === null ? '—' : fmt(vf, 2) + (vf > 10 ? ' <span class="pill bad">高共線</span>' : vf > 5 ? ' <span class="pill warn">注意</span>' : '')}</td></tr>`; }).join('') + '</tbody>';
    const all = ctx.yte.concat(e.pte), lo = Math.min(...all), hi = Math.max(...all);
    chart('c-avp', {
      type: 'scatter',
      data: { datasets: [
        { label: '測試樣本', data: ctx.yte.map((y, i) => ({ x: y, y: e.pte[i] })), backgroundColor: css('--test'), pointRadius: 5 },
        { type: 'line', label: 'y = x（完美預測）', data: [{ x: lo, y: lo }, { x: hi, y: hi }], borderColor: css('--naive'), borderDash: [5, 4], pointRadius: 0, borderWidth: 1.5 }
      ] },
      options: { scales: { x: { title: { display: true, text: 'Actual ' + s.target } }, y: { title: { display: true, text: 'Predicted ' + s.target } } } }
    });
    const res = ctx.yte.map((y, i) => ({ row: s.data.keep[ctx.sp.test[i]], y, p: e.pte[i], r: y - e.pte[i] })).sort((a, b) => Math.abs(b.r) - Math.abs(a.r)).slice(0, 10);
    $('m-resid').innerHTML = '<thead><tr><th>#</th><th>列</th><th>Actual</th><th>Predicted</th><th>Residual</th><th>誤差 %</th><th>狀態</th></tr></thead><tbody>' +
      res.map((r, i) => { const z = Math.abs(r.r) / m.sigma; return `<tr><td>${i + 1}</td><td>${r.row}</td><td>${fmt(r.y)}</td><td>${fmt(r.p)}</td><td>${fmt(r.r)}</td><td>${r.y ? fmt(100 * Math.abs(r.r / r.y), 1) + '%' : '—'}</td><td>${z > 2 ? '<span class="pill bad">離群 >2σ</span>' : z > 1 ? '<span class="pill warn">偏大</span>' : '<span class="pill good">正常</span>'}</td></tr>`; }).join('') + '</tbody>';
    renderPredictor(e);
  }

  function renderPredictor(e) {
    const s = state.mlr, D = s.data;
    s.current = e;
    const srcs = []; e.idx.forEach(j => { const f = D.feats[j]; if (!srcs.includes(f.src)) srcs.push(f.src); });
    const form = $('m-pred-form');
    const prev = {}; form.querySelectorAll('[data-src]').forEach(el => { prev[el.dataset.src] = el.value; });
    form.innerHTML = srcs.map((src, i) => {
      const cat = D.categorical.find(c => c.h === src);
      const id = 'pf-' + i;
      if (cat) return `<label for="${id}">${esc(src)}<select id="${id}" data-src="${esc(src)}">${cat.cats.map(c => `<option${prev[src] === c ? ' selected' : ''}>${esc(c)}</option>`).join('')}</select></label>`;
      const num = D.numeric.find(c => c.h === src);
      const val = prev[src] !== undefined ? prev[src] : Math.round(num.mean * 100) / 100;
      return `<label for="${id}">${esc(src)}<input id="${id}" type="number" step="any" data-src="${esc(src)}" value="${val}"></label>`;
    }).join('');
    updatePrediction();
    const model = {
      problem: 'multiple_linear_regression', target: s.target, algorithm: s.algo === 'all' ? 'lasso' : s.algo, k: e.k,
      features: e.idx.map(j => D.feats[j].name), intercept: e.model.beta[0],
      coefficients: Object.fromEntries(e.idx.map((j, i) => [D.feats[j].name, e.model.beta[i + 1]])),
      encoding: Object.fromEntries(D.categorical.map(c => [c.h, { dropped_first: c.cats[0], categories: c.cats }])),
      split: { test_size: s.testSize, random_state: s.seed }, metrics: { test_rmse: e.test.rmse, test_r2: e.test.r2, test_mae: e.test.mae }
    };
    $('m-json').textContent = JSON.stringify(model, null, 2);
  }

  function updatePrediction() {
    const s = state.mlr, D = s.data, e = s.current; if (!e) return;
    const vals = {}; $('m-pred-form').querySelectorAll('[data-src]').forEach(el => { vals[el.dataset.src] = el.value; });
    let yhat = e.model.beta[0]; const parts = [`β₀ = ${fmtAuto(e.model.beta[0])}`]; let bad = false;
    e.idx.forEach((j, i) => {
      const f = D.feats[j]; let x;
      if (f.kind === 'num') { x = ML.toNumber(vals[f.src]); if (!Number.isFinite(x)) bad = true; }
      else x = vals[f.src] === f.cat ? 1 : 0;
      const c = e.model.beta[i + 1] * x; yhat += c;
      parts.push(`${f.name}: ${fmtAuto(e.model.beta[i + 1])} × ${fmtAuto(x)} = ${fmt(c)}`);
    });
    $('m-pred').textContent = bad ? '請輸入數字' : fmt(yhat);
    $('m-pred-break').innerHTML = parts.map(esc).join('<br>');
  }

  /* ====================================================================
   * Problem 1：AR 股價
   * ==================================================================== */
  function parseDate(s) {
    s = String(s).trim();
    let m = s.match(/^(\d{2,3})\/(\d{1,2})\/(\d{1,2})$/); // 民國
    if (m) return new Date(Date.UTC(+m[1] + 1911, +m[2] - 1, +m[3]));
    m = s.match(/^(\d{4})[-/.](\d{1,2})(?:[-/.](\d{1,2}))?/);
    if (m) return new Date(Date.UTC(+m[1], +m[2] - 1, m[3] ? +m[3] : 1));
    if (/^\d{8}$/.test(s)) return new Date(Date.UTC(+s.slice(0, 4), +s.slice(4, 6) - 1, +s.slice(6, 8)));
    const t = Date.parse(s); return Number.isFinite(t) && /[a-z\-/]/i.test(s) ? new Date(t) : null;
  }
  const ymd = d => d.toISOString().slice(0, 10);

  function prepareAR(raw, col) {
    const dateCol = raw.header.find(h => /date|time|日期|時間|month|day/i.test(h)) || raw.header.find(h => raw.cols[h].slice(0, 20).filter(v => parseDate(v)).length >= 15);
    if (!dateCol) throw new Error('找不到日期欄（欄名需含 Date / 日期，或內容可解析為日期）。');
    const numCols = raw.header.filter(h => h !== dateCol && raw.cols[h].filter(v => Number.isFinite(ML.toNumber(v))).length >= 0.9 * raw.n);
    if (!numCols.length) throw new Error('找不到數值欄。');
    if (!col || !numCols.includes(col)) col = numCols.find(h => /^(adj\s*close|close|收盤價?|price|value)$/i.test(h.trim())) || numCols[numCols.length - 1];
    const pts = [];
    for (let i = 0; i < raw.n; i++) { const d = parseDate(raw.cols[dateCol][i]), v = ML.toNumber(raw.cols[col][i]); if (d && Number.isFinite(v)) pts.push({ d, v }); }
    pts.sort((a, b) => a.d - b.d);
    const uniq = pts.filter((p, i) => i === 0 || +p.d !== +pts[i - 1].d);
    if (uniq.length < 30) throw new Error('有效時間點少於 30 筆，不足以建立 AR 模型。');
    return { dateCol, col, numCols, dates: uniq.map(p => p.d), y: uniq.map(p => p.v), dropped: raw.n - uniq.length };
  }

  function loadAR(raw, label, url) {
    const s = state.ar;
    s.raw = raw; s.label = label; s.url = url || ''; s.col = null; s.p = null;
    rebuildAR();
  }

  function rebuildAR() {
    const s = state.ar;
    try {
      s.data = prepareAR(s.raw, s.col);
      s.col = s.data.col;
      const D = s.data;
      $('a-col').innerHTML = D.numCols.map(h => `<option${h === s.col ? ' selected' : ''}>${esc(h)}</option>`).join('');
      $('a-col-wrap').hidden = D.numCols.length < 2;
      s.maxP = Math.max(1, Math.min(20, Math.floor(D.y.length / 8)));
      s.sweep = [];
      for (let p = 1; p <= s.maxP; p++) {
        try { s.sweep.push(ML.fitAR(D.y, p, s.testFrac, s.maxP)); }
        catch (err) { if (p === 1) throw err; break; }   // 更高階數共線 → 停在可估計的最大 p
      }
      s.maxP = s.sweep.length;
      let best = s.sweep[0]; s.sweep.forEach(f => { if (f.model.aic < best.model.aic) best = f; });
      s.bestP = best.p;
      if (s.p === null || s.p > s.maxP) s.p = s.bestP;
      renderARUnderstanding(); renderARPrep(); renderARModel();
      setStatus(`已載入 <b>${esc(s.label)}</b>：${D.y.length} 個時間點（${ymd(D.dates[0])} → ${ymd(D.dates[D.dates.length - 1])}），欄位 <code>${esc(s.col)}</code>。`, 'ok');
    } catch (e) {
      console.error(e);
      setStatus('無法建立模型：' + esc(e.message), 'err');
    }
  }

  function renderARUnderstanding() {
    const s = state.ar, D = s.data, y = D.y, n = y.length;
    const isTsmc = /2330/.test(s.label);
    $('a-biz').innerHTML = `目標：以 <b>${esc(s.label)}</b> 的歷史 <code>${esc(s.col)}</code> 序列，預測下一期的值，並評估單純用「過去價格」能否打敗最簡單的 naive 預測。${isTsmc ? '台積電是台股權值最大的個股，股價走勢常被當作台股與半導體景氣的觀察指標。' : ''}`;
    const ret = []; for (let i = 1; i < n; i++) ret.push(Math.log(y[i] / y[i - 1]));
    const vol = ret.length && y.every(v => v > 0) ? ML.std(ret, 1) * Math.sqrt(252) : NaN;
    $('a-kpis').innerHTML = kpi('時間點', n) + kpi('期間', `${ymd(D.dates[0]).slice(2)} → ${ymd(D.dates[n - 1]).slice(2)}`) + kpi('最新值', fmt(y[n - 1]), ymd(D.dates[n - 1])) +
      kpi('期間漲跌', (y[n - 1] >= y[0] ? '+' : '') + fmt(100 * (y[n - 1] / y[0] - 1), 1) + '%', `${fmt(y[0])} → ${fmt(y[n - 1])}`) + kpi('最高 / 最低', `${fmt(Math.max(...y), 0)} / ${fmt(Math.min(...y), 0)}`) + kpi('年化波動', Number.isFinite(vol) ? fmt(100 * vol, 1) + '%' : '—', 'log 報酬 × √252');
    $('a-line-cap').textContent = `${s.label} · ${s.col}`;
    chart('c-price', {
      type: 'line',
      data: { labels: D.dates.map(ymd), datasets: [line(s.col, y, css('--train'), { pointRadius: 0, borderWidth: 1.8, fill: false })] },
      options: { interaction: { mode: 'index', intersect: false }, plugins: { legend: { display: false } }, scales: { x: { ticks: { maxTicksLimit: 8, maxRotation: 0 } }, y: { title: { display: true, text: s.col } } } }
    });
    const L = Math.min(20, Math.floor(n / 4)), pa = ML.pacf(y, L), band = 1.96 / Math.sqrt(n);
    chart('c-pacf', {
      type: 'bar',
      data: { labels: pa.map((_, i) => i + 1), datasets: [
        { label: 'PACF', data: pa, backgroundColor: pa.map(v => (Math.abs(v) > band ? css('--train') : css('--naive'))), borderRadius: 2 },
        { type: 'line', label: '+1.96/√n', data: pa.map(() => band), borderColor: css('--test'), borderDash: [4, 4], pointRadius: 0, borderWidth: 1 },
        { type: 'line', label: '−1.96/√n', data: pa.map(() => -band), borderColor: css('--test'), borderDash: [4, 4], pointRadius: 0, borderWidth: 1 }
      ] },
      options: { plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: 'Lag' } }, y: { min: -1, max: 1 } } }
    });
  }

  function renderARPrep() {
    const s = state.ar, D = s.data, f = s.sweep[s.p - 1];
    $('a-test-out').textContent = pct(s.testFrac);
    const steps = [
      `解析日期欄 <code>${esc(D.dateCol)}</code>、依時間排序並去除重複${D.dropped ? `（略過 ${D.dropped} 筆無法解析的列）` : ''}。`,
      `建立落後特徵 <code>lag_1 … lag_p</code>：第 t 天的特徵是前 p 天的 ${esc(D.col)}（<code>df.shift(1..p)</code>）。`,
      `為了公平比較不同 p，所有模型都從第 ${s.maxP + 1} 筆開始建樣本（前 ${s.maxP} 筆只當落後值），共 ${f.nTr + f.nTest} 筆。`,
      `<b>依時間順序切分，不打亂</b>：前 ${f.nTr} 筆訓練（到 ${ymd(D.dates[f.tTrain[f.tTrain.length - 1]])}），最後 ${f.nTest} 筆測試（${ymd(D.dates[f.tTest[0]])} 起）。股價有時間相依性，隨機切分會讓模型偷看到未來。`
    ];
    $('a-prep').innerHTML = steps.map(t => `<li>${t}</li>`).join('');
    const src = s.url || 'data/2330_TW.csv';
    $('a-code').textContent =
`import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("${src}", parse_dates=["${D.dateCol}"]).sort_values("${D.dateCol}")
s = df["${D.col}"].astype(float).reset_index(drop=True)

P_MAX, p = ${s.maxP}, ${s.p}
X = pd.concat({f"lag_{l}": s.shift(l) for l in range(1, p + 1)}, axis=1)
data = pd.concat([X, s.rename("y")], axis=1).iloc[P_MAX:]

n_test = round(len(data) * ${s.testFrac})          # 時間序列：不打亂
train, test = data.iloc[:-n_test], data.iloc[-n_test:]
model = LinearRegression().fit(train.drop(columns="y"), train["y"])
pred = model.predict(test.drop(columns="y"))

print("RMSE", np.sqrt(mean_squared_error(test["y"], pred)))
print("naive RMSE", np.sqrt(mean_squared_error(test["y"], test["lag_1"])))`;
  }

  function renderARModel() {
    const s = state.ar, D = s.data;
    const pIn = $('a-p'); pIn.max = s.maxP; pIn.value = s.p; $('a-p-out').textContent = s.p;
    const ps = s.sweep.map(f => f.p);
    let bestTest = s.sweep[0]; s.sweep.forEach(f => { if (f.test.rmse < bestTest.test.rmse) bestTest = f; });
    $('a-p-note').innerHTML = `用訓練集 AIC 自動選出 <b>p = ${s.bestP}</b>（不看測試集，避免資料洩漏）。測試 RMSE 最低出現在 p = ${bestTest.p}（${fmt(bestTest.test.rmse)}）。拖曳滑桿可比較不同 p。`;
    const ptR = ps.map(p => (p === s.p ? 6 : 3));
    chart('c-a-rmse', {
      type: 'line',
      data: { labels: ps, datasets: [
        line('Test RMSE', s.sweep.map(f => f.test.rmse), css('--test'), { pointRadius: ptR }),
        line('Train RMSE', s.sweep.map(f => f.train.rmse), css('--train')),
        line('Naive（明天=今天）', s.sweep.map(f => f.naive.rmse), css('--naive'), { borderDash: [5, 4], pointRadius: 0 })
      ] },
      options: { interaction: { mode: 'index', intersect: false }, scales: { x: { title: { display: true, text: 'Lag order p' } }, y: { title: { display: true, text: 'RMSE' } } } }
    });
    chart('c-a-aic', {
      type: 'line',
      data: { labels: ps, datasets: [line('AIC', s.sweep.map(f => f.model.aic), css('--cv'), { pointRadius: ps.map(p => (p === s.bestP ? 6 : 3)) })] },
      options: { plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: 'Lag order p' } }, y: { title: { display: true, text: 'AIC' } } } }
    });
    renderAREval();
  }

  function renderAREval() {
    const s = state.ar, D = s.data, f = s.sweep[s.p - 1], m = f.model;
    $('a-eval-title').textContent = `AR(${s.p}) · 測試 ${f.nTest} 期`;
    const beat = f.test.rmse < f.naive.rmse;
    $('a-metrics').innerHTML = kpi('Test RMSE', fmt(f.test.rmse), `Train ${fmt(f.train.rmse)}`, beat ? 'good' : '') +
      kpi('Naive RMSE', fmt(f.naive.rmse), '明天 = 今天') + kpi('Test MAE', fmt(f.test.mae)) + kpi('Test MAPE', fmt(f.test.mape, 2) + '%') +
      kpi('Test R²', f.test.r2.toFixed(4), '水準值 R² 通常偏高') + kpi('方向準確率', fmt(f.dirAcc, 1) + '%', '預測漲跌方向正確比例');
    const v = $('a-verdict');
    if (beat) { v.className = 'callout good'; v.innerHTML = `AR(${s.p}) 的測試 RMSE 比 naive 低 ${fmt(100 * (1 - f.test.rmse / f.naive.rmse), 1)}%，過去價格提供了些微額外資訊。`; }
    else { v.className = 'callout warn'; v.innerHTML = `AR(${s.p}) 的測試 RMSE（${fmt(f.test.rmse)}）<b>沒有打敗</b> naive 基準（${fmt(f.naive.rmse)}）。φ₁ ≈ ${m.beta[1].toFixed(3)} 接近 1，股價接近隨機漫步（random walk）：最好的預測幾乎就是今天的價格。這也是「效率市場」的典型結果。R² 很高只是因為價格水準本身高度自相關，不代表模型有預測力。`; }
    $('a-eq').innerHTML = `<span class="y">ŷ<sub>t</sub></span> = ${fmtAuto(m.beta[0])}` + m.beta.slice(1).map((b, i) => ` ${b < 0 ? '−' : '+'} ${Math.abs(b).toFixed(4)} × y<sub>t−${i + 1}</sub>`).join('');
    $('a-coef').innerHTML = '<thead><tr><th>變數</th><th>係數</th><th>Std. Error</th><th>t</th><th>p-value</th><th>顯著</th><th>95% CI</th></tr></thead><tbody>' +
      m.rows.map((r, i) => `<tr><td class="txt">${i === 0 ? 'Intercept (c)' : 'lag_' + i + '（y<sub>t−' + i + '</sub>）'}</td><td>${fmtAuto(r.coef)}</td><td>${fmtAuto(r.se)}</td><td>${fmt(r.t, 3)}</td><td>${fmtP(r.p)}</td><td>${sigPill(r.p)}</td><td>[${fmtAuto(r.lo)}, ${fmtAuto(r.hi)}]</td></tr>`).join('') + '</tbody>';
    const labels = f.tTest.map(t => ymd(D.dates[t]));
    chart('c-a-test', {
      type: 'line',
      data: { labels, datasets: [
        line('實際', f.yte, css('--ink'), { pointRadius: 2, borderWidth: 2 }),
        line(`AR(${s.p}) 預測`, f.pte, css('--test'), { pointRadius: 2, borderWidth: 2 }),
        line('Naive', f.tTest.map(t => D.y[t - 1]), css('--naive'), { pointRadius: 0, borderDash: [5, 4], borderWidth: 1.5 })
      ] },
      options: { interaction: { mode: 'index', intersect: false }, scales: { x: { ticks: { maxTicksLimit: 8, maxRotation: 0 } }, y: { title: { display: true, text: D.col } } } }
    });
    renderARForecast();
  }

  function nextBizDays(last, h, daily) {
    const out = []; let d = new Date(last);
    while (out.length < h) {
      if (daily) { d = new Date(d.getTime() + 864e5); if (d.getUTCDay() === 0 || d.getUTCDay() === 6) continue; }
      else d = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + 1, d.getUTCDate()));
      out.push(new Date(d));
    }
    return out;
  }

  function renderARForecast() {
    const s = state.ar, D = s.data, f = s.sweep[s.p - 1];
    $('a-h-out').textContent = s.h;
    // 部署：用全部資料重新擬合
    const L = ML.lagMatrix(D.y, s.p, s.maxP);
    const fullModel = ML.ols(L.X, L.y);
    const fc = ML.forecastAR(D.y, fullModel, s.p, s.h);
    const n = D.y.length, gaps = []; for (let i = Math.max(1, n - 30); i < n; i++) gaps.push((D.dates[i] - D.dates[i - 1]) / 864e5);
    const daily = ML.mean(gaps) < 10;
    const fdates = nextBizDays(D.dates[n - 1], s.h, daily);
    const hist = Math.min(60, n);
    const labels = D.dates.slice(n - hist).map(ymd).concat(fdates.map(ymd));
    const pad = Array(hist - 1).fill(null);
    const lastV = D.y[n - 1];
    chart('c-a-fc', {
      type: 'line',
      data: { labels, datasets: [
        line('歷史', D.y.slice(n - hist).concat(Array(s.h).fill(null)), css('--ink'), { pointRadius: 0, borderWidth: 2 }),
        line('95% 上界', pad.concat([lastV], fc.map(x => x.hi)), css('--test'), { pointRadius: 0, borderWidth: 0, fill: '+1', backgroundColor: hexA(css('--test'), 0.15) }),
        line('95% 下界', pad.concat([lastV], fc.map(x => x.lo)), css('--test'), { pointRadius: 0, borderWidth: 0 }),
        line(`AR(${s.p}) 預測`, pad.concat([lastV], fc.map(x => x.mean)), css('--test'), { pointRadius: 2, borderWidth: 2, borderDash: [6, 3] })
      ] },
      options: { interaction: { mode: 'index', intersect: false }, plugins: { legend: { labels: { filter: it => !/界/.test(it.text) } } }, scales: { x: { ticks: { maxTicksLimit: 8, maxRotation: 0 } }, y: { title: { display: true, text: D.col } } } }
    });
    $('a-fc-table').innerHTML = '<thead><tr><th>步數</th><th>日期</th><th>預測值</th><th>95% 下界</th><th>95% 上界</th><th>相對最新值</th></tr></thead><tbody>' +
      fc.map((x, i) => `<tr><td>t+${i + 1}</td><td>${ymd(fdates[i])}</td><td>${fmt(x.mean)}</td><td>${fmt(x.lo)}</td><td>${fmt(x.hi)}</td><td>${x.mean >= lastV ? '+' : ''}${fmt(100 * (x.mean / lastV - 1), 2)}%</td></tr>`).join('') + '</tbody>';
  }
  function hexA(c, a) {
    const m = c.match(/^#([0-9a-f]{6})$/i); if (!m) return c;
    const v = parseInt(m[1], 16); return `rgba(${v >> 16},${(v >> 8) & 255},${v & 255},${a})`;
  }

  /* ====================================================================
   * 資料載入 / 事件
   * ==================================================================== */
  function renderChips() {
    $('ds-chips').innerHTML = '<span class="eyebrow" style="align-self:center">範例：</span>' + SOURCES[state.problem].map((c, i) => `<button type="button" class="chip" data-chip="${i}">${esc(c.label)}</button>`).join('');
  }
  function useText(text, label, url) {
    const raw = ML.parseCSV(text);
    if (state.problem === 'mlr') loadMLR(raw, label, url); else loadAR(raw, label, url);
  }
  function loadBuiltin(which) {
    const D = window.DEFAULT_DATA;
    if (which === 'startups') { $('ds-url').value = SOURCES.mlr[0].url; state.problem === 'mlr' && useText(D.startups, '50_Startups.csv（內建，課程資料夾版本）', SOURCES.mlr[0].url); }
    else { $('ds-url').value = ''; useText(D.tsmc, '2330.TW 台積電（內建 TWSE 快照）', ''); }
  }
  async function loadUrl(url) {
    url = normalizeUrl(url);
    if (!/^https?:\/\//i.test(url)) { setStatus('請輸入以 http(s):// 開頭的 CSV 網址。', 'err'); return; }
    $('ds-url').value = url;
    setStatus('下載中… ' + esc(url), 'busy');
    const btn = $('ds-load'); btn.disabled = true;
    try {
      const text = await fetchText(url);
      if (/^\s*</.test(text)) throw new Error('這個網址回傳的是網頁（HTML），不是 CSV。GitHub 請用 Raw 連結。');
      useText(text, decodeURIComponent(url.split('/').pop().split('?')[0]) || url, url);
    } catch (e) { setStatus(esc(e.message), 'err'); }
    finally { btn.disabled = false; }
  }
  async function loadFinMind() {
    const id = $('stock-id').value.trim();
    if (!/^\d{4,6}[A-Z]?$/i.test(id)) { setStatus('請輸入台股代號，例如 2330、2317、0050。', 'err'); return; }
    const days = +$('stock-days').value;
    const start = new Date(Date.now() - days * 864e5).toISOString().slice(0, 10);
    const url = `https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=${id}&start_date=${start}`;
    setStatus(`向 FinMind 抓取 ${esc(id)} 自 ${start} 起的日收盤價…`, 'busy');
    const btn = $('stock-live'); btn.disabled = true;
    try {
      const js = JSON.parse(await fetchText(url));
      if (!js.data || !js.data.length) throw new Error('FinMind 沒有回傳資料（' + (js.msg || '代號錯誤或超過免費額度') + '）。');
      const csv = 'Date,Close,Open,High,Low,Volume\n' + js.data.map(r => [r.date, r.close, r.open, r.max, r.min, r.Trading_Volume].join(',')).join('\n');
      useText(csv, `${id}.TW（FinMind 即時，${js.data.length} 日）`, url);
    } catch (e) { setStatus(esc(e.message) + ' 已保留目前的資料。', 'err'); }
    finally { btn.disabled = false; }
  }

  function switchProblem(p) {
    state.problem = p;
    $('problem').value = p;
    $('panel-mlr').hidden = p !== 'mlr';
    $('panel-ar').hidden = p !== 'ar';
    $('stock-row').hidden = p !== 'ar';
    $('ds-hint').innerHTML = p === 'mlr'
      ? '貼上任意 CSV 網址（GitHub raw、gist…）或上傳檔案。會自動偵測數值 / 類別欄並做 one-hot，目標變數可在步驟 1 改選。'
      : '貼上含「日期欄 + 數值欄」的 CSV 網址或上傳檔案（Yahoo Finance 匯出的 CSV 可直接用），或輸入台股代號即時抓取。';
    $('ds-url').value = p === 'mlr' ? (state.mlr.url || '') : (state.ar.url || '');
    renderChips();
    const s = state[p];
    if (!s.raw) loadBuiltin(p === 'mlr' ? 'startups' : 'tsmc');
    else { setStatus(`目前資料：<b>${esc(s.label)}</b>`, 'ok'); rerenderCharts(); }
    try { localStorage.setItem('dic2-problem', p); } catch (e) { /* ignore */ }
    try { history.replaceState(null, '', '#' + p); } catch (e) { /* ignore */ }
  }
  function rerenderCharts() {
    if (state.problem === 'mlr' && state.mlr.data) { renderMLRUnderstanding(); renderMLRModel(); }
    if (state.problem === 'ar' && state.ar.data) { renderARUnderstanding(); renderARModel(); }
  }

  function bind() {
    $('problem').addEventListener('change', e => switchProblem(e.target.value));
    $('ds-form').addEventListener('submit', e => { e.preventDefault(); loadUrl($('ds-url').value); });
    $('ds-file').addEventListener('change', e => {
      const f = e.target.files[0]; if (!f) return;
      const rd = new FileReader();
      rd.onload = () => { try { state[state.problem].url = ''; useText(rd.result, f.name, ''); } catch (err) { setStatus(esc(err.message), 'err'); } };
      rd.readAsText(f); e.target.value = '';
    });
    $('ds-chips').addEventListener('click', e => {
      const b = e.target.closest('[data-chip]'); if (!b) return;
      const c = SOURCES[state.problem][+b.dataset.chip];
      if (c.builtin) loadBuiltin(c.builtin); else loadUrl(c.url);
    });
    $('stock-live').addEventListener('click', loadFinMind);
    // MLR controls
    $('m-target').addEventListener('change', e => { state.mlr.target = e.target.value; rebuildMLR(); });
    $('m-test').addEventListener('input', e => { state.mlr.testSize = +e.target.value; rebuildMLR(); });
    $('m-seed').addEventListener('change', e => { const v = Math.max(0, Math.min(4294967295, Math.floor(+e.target.value || 0))); e.target.value = v; state.mlr.seed = v; rebuildMLR(); });
    $('m-crit').addEventListener('change', e => { state.mlr.criterion = e.target.value; state.mlr.k = null; renderMLRModel(); });
    $('m-algos').addEventListener('click', e => { const b = e.target.closest('[data-algo]'); if (!b || b.disabled) return; state.mlr.algo = b.dataset.algo; state.mlr.k = null; setBusy(() => renderMLRModel()); });
    $('m-k').addEventListener('input', e => { state.mlr.k = +e.target.value; renderMLRModel(); });
    $('m-results').addEventListener('click', e => {
      const tr = e.target.closest('tr'); if (!tr) return;
      if (tr.dataset.k) { state.mlr.k = +tr.dataset.k; renderMLRModel(); }
      else if (tr.dataset.algo) { state.mlr.algo = tr.dataset.algo; state.mlr.k = null; renderMLRModel(); }
    });
    $('m-pred-form').addEventListener('input', updatePrediction);
    $('m-dl').addEventListener('click', () => {
      try {
        const blob = new Blob([$('m-json').textContent], { type: 'application/json' });
        const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'mlr_model.json'; document.body.appendChild(a); a.click(); a.remove();
      } catch (err) { /* sandboxed */ }
    });
    // AR controls
    $('a-col').addEventListener('change', e => { state.ar.col = e.target.value; state.ar.p = null; rebuildAR(); });
    $('a-test').addEventListener('input', e => { state.ar.testFrac = +e.target.value; rebuildAR(); });
    $('a-p').addEventListener('input', e => { state.ar.p = +e.target.value; renderARModel(); renderARPrep(); });
    $('a-h').addEventListener('input', e => { state.ar.h = +e.target.value; renderARForecast(); });
    // copy
    document.addEventListener('click', e => {
      const b = e.target.closest('[data-copy]'); if (!b) return;
      const el = $(b.dataset.copy), txt = el.textContent;
      const done = () => { const o = b.textContent; b.textContent = '已複製'; setTimeout(() => { b.textContent = o; }, 1200); };
      const sel = () => { const r = document.createRange(); r.selectNodeContents(el); const s = getSelection(); s.removeAllRanges(); s.addRange(r); };
      try { navigator.clipboard.writeText(txt).then(done, sel); } catch (err) { sel(); }
    });
    // theme change → redraw charts with new token colors
    try { matchMedia('(prefers-color-scheme: dark)').addEventListener('change', rerenderCharts); } catch (e) { /* old browsers */ }
    new MutationObserver(rerenderCharts).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  }
  function setBusy(fn) { document.body.style.cursor = 'progress'; setTimeout(() => { try { fn(); } finally { document.body.style.cursor = ''; } }, 10); }

  function start() {
    bind();
    let p = 'mlr';
    const h = (location.hash || '').replace('#', '');
    if (h === 'ar' || h === 'mlr') p = h;
    else { try { const v = localStorage.getItem('dic2-problem'); if (v === 'ar' || v === 'mlr') p = v; } catch (e) { /* ignore */ } }
    switchProblem(p);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
})();
