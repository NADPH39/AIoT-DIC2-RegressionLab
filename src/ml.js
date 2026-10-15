/* ml.js — 純前端回歸工具庫（無相依套件）
 * - MT19937 + numpy legacy shuffle → 與 sklearn train_test_split(random_state=…) 相同切分
 * - OLS（Householder QR）+ statsmodels 風格推論統計：SE、t、p、95% CI、F、VIF
 * - Lasso / ElasticNet 座標下降路徑、Forward SFS、Backward（p-value）、RFE、Exhaustive
 * - AR(p) 自我迴歸、PACF、遞迴預測與預測區間
 */
(function (root) {
  'use strict';

  /* ---------------- MT19937 (numpy RandomState legacy) ---------------- */
  function MT19937(seed) {
    this.mt = new Uint32Array(624);
    this.idx = 624;
    this.mt[0] = seed >>> 0;
    for (let i = 1; i < 624; i++) {
      const prev = this.mt[i - 1] ^ (this.mt[i - 1] >>> 30);
      // 1812433253 * prev + i  (mod 2^32)，拆成高低 16 位避免精度流失
      this.mt[i] = (((((prev & 0xffff0000) >>> 16) * 1812433253) << 16) + (prev & 0x0000ffff) * 1812433253 + i) >>> 0;
    }
  }
  MT19937.prototype.next32 = function () {
    const mt = this.mt;
    if (this.idx >= 624) {
      for (let k = 0; k < 624; k++) {
        const y = (mt[k] & 0x80000000) | (mt[(k + 1) % 624] & 0x7fffffff);
        mt[k] = (mt[(k + 397) % 624] ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0)) >>> 0;
      }
      this.idx = 0;
    }
    let y = mt[this.idx++];
    y ^= y >>> 11;
    y = (y ^ ((y << 7) & 0x9d2c5680)) >>> 0;
    y = (y ^ ((y << 15) & 0xefc60000)) >>> 0;
    y ^= y >>> 18;
    return y >>> 0;
  };
  MT19937.prototype.interval = function (max) {
    if (max === 0) return 0;
    let mask = max;
    mask |= mask >>> 1; mask |= mask >>> 2; mask |= mask >>> 4; mask |= mask >>> 8; mask |= mask >>> 16;
    mask >>>= 0;
    let v;
    while ((v = (this.next32() & mask) >>> 0) > max) { /* reject */ }
    return v;
  };
  function permutation(n, seed) {
    const rng = new MT19937(seed);
    const a = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i >= 1; i--) {
      const j = rng.interval(i);
      const t = a[i]; a[i] = a[j]; a[j] = t;
    }
    return a;
  }
  /** 與 sklearn.model_selection.train_test_split(test_size, random_state) 相同 */
  function trainTestSplit(n, testSize, seed) {
    const nTest = Math.ceil(testSize * n);
    const perm = permutation(n, seed);
    return { test: perm.slice(0, nTest), train: perm.slice(nTest) };
  }

  /* ---------------- CSV ---------------- */
  function parseCSV(text) {
    text = text.replace(/^﻿/, '');
    const firstLine = text.split(/\r?\n/, 1)[0] || '';
    const cand = [',', ';', '\t'];
    let delim = ',', best = -1;
    for (const c of cand) { const k = firstLine.split(c).length; if (k > best) { best = k; delim = c; } }
    const rows = [];
    let row = [], field = '', q = false;
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (q) {
        if (ch === '"') { if (text[i + 1] === '"') { field += '"'; i++; } else q = false; }
        else field += ch;
      } else if (ch === '"') q = true;
      else if (ch === delim) { row.push(field); field = ''; }
      else if (ch === '\n' || ch === '\r') {
        if (ch === '\r' && text[i + 1] === '\n') i++;
        row.push(field); field = '';
        if (row.length > 1 || row[0] !== '') rows.push(row);
        row = [];
      } else field += ch;
    }
    if (field !== '' || row.length) { row.push(field); if (row.length > 1 || row[0] !== '') rows.push(row); }
    if (!rows.length) throw new Error('檔案是空的');
    const header = rows[0].map((h, i) => (h.trim() || ('col_' + (i + 1))));
    const data = rows.slice(1).filter(r => r.some(v => v.trim() !== ''));
    const cols = {};
    header.forEach((h, j) => { cols[h] = data.map(r => (r[j] === undefined ? '' : r[j].trim())); });
    return { header, cols, n: data.length };
  }
  function toNumber(s) {
    if (s === null || s === undefined) return NaN;
    if (typeof s === 'number') return s;
    const t = String(s).replace(/[,$\s%]/g, '');
    if (t === '' || /^(na|nan|null|none|-|--)$/i.test(t)) return NaN;
    const v = Number(t);
    return Number.isFinite(v) ? v : NaN;
  }
  function isMissing(s) { return s === '' || /^(na|nan|null|none)$/i.test(String(s).trim()); }

  /* ---------------- 統計分配 ---------------- */
  function lgamma(x) {
    const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    let y = x, tmp = x + 5.5;
    tmp -= (x + 0.5) * Math.log(tmp);
    let ser = 1.000000000190015;
    for (let j = 0; j < 6; j++) ser += c[j] / ++y;
    return -tmp + Math.log(2.5066282746310005 * ser / x);
  }
  function betacf(a, b, x) {
    const MAXIT = 300, EPS = 3e-16, FPMIN = 1e-300;
    let qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - qab * x / qap;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    d = 1 / d; let h = d;
    for (let m = 1; m <= MAXIT; m++) {
      const m2 = 2 * m;
      let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d; h *= d * c;
      aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d;
      const del = d * c; h *= del;
      if (Math.abs(del - 1) < EPS) break;
    }
    return h;
  }
  function betai(a, b, x) {
    if (x <= 0) return 0; if (x >= 1) return 1;
    const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
    return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
  }
  /** 雙尾 t 檢定 p 值 */
  function tPValue(t, df) {
    if (!Number.isFinite(t)) return t === t ? 0 : NaN;
    return betai(df / 2, 0.5, df / (df + t * t));
  }
  function tCDF(t, df) { const p = tPValue(t, df) / 2; return t >= 0 ? 1 - p : p; }
  function tInv(p, df) {
    let lo = -1e3, hi = 1e3;
    for (let i = 0; i < 200; i++) { const mid = (lo + hi) / 2; if (tCDF(mid, df) < p) lo = mid; else hi = mid; }
    return (lo + hi) / 2;
  }
  /** F 分配右尾機率 */
  function fPValue(F, d1, d2) {
    if (!(F > 0)) return 1;
    return betai(d2 / 2, d1 / 2, d2 / (d2 + d1 * F));
  }

  /* ---------------- 基本工具 ---------------- */
  const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
  function std(a, ddof = 0) { const m = mean(a); return Math.sqrt(a.reduce((s, v) => s + (v - m) * (v - m), 0) / (a.length - ddof)); }
  function metrics(y, p) {
    const n = y.length; let sse = 0, sae = 0, sape = 0, nape = 0;
    const m = mean(y); let sst = 0;
    for (let i = 0; i < n; i++) {
      const e = y[i] - p[i]; sse += e * e; sae += Math.abs(e); sst += (y[i] - m) * (y[i] - m);
      if (y[i] !== 0) { sape += Math.abs(e / y[i]); nape++; }
    }
    return { mse: sse / n, rmse: Math.sqrt(sse / n), mae: sae / n, r2: sst > 0 ? 1 - sse / sst : NaN, mape: nape ? 100 * sape / nape : NaN, sse, n };
  }
  const adjR2 = (r2, n, k) => (n - k - 1 > 0 ? 1 - (1 - r2) * (n - 1) / (n - k - 1) : NaN);

  /* ---------------- OLS（Householder QR） ---------------- */
  /** X: rows (n×k，不含截距)；回傳含截距的係數與推論統計 */
  function ols(X, y, opts) {
    opts = opts || {};
    const n = y.length, k = X.length ? X[0].length : 0, p = k + 1;
    if (n <= p) throw new Error('樣本數不足以估計 ' + p + ' 個參數');
    // A: column-major copy with intercept
    const A = [];
    A.push(new Float64Array(n).fill(1));
    for (let j = 0; j < k; j++) { const c = new Float64Array(n); for (let i = 0; i < n; i++) c[i] = X[i][j]; A.push(c); }
    const b = Float64Array.from(y);
    const R = Array.from({ length: p }, () => new Float64Array(p));
    const scale = A.map(c => Math.sqrt(c.reduce((s, v) => s + v * v, 0)) || 1);
    for (let j = 0; j < p; j++) {
      const col = A[j];
      let norm = 0; for (let i = j; i < n; i++) norm += col[i] * col[i];
      norm = Math.sqrt(norm);
      if (norm < 1e-10 * scale[j]) { const e = new Error('共線性：無法估計（設計矩陣奇異）'); e.singular = j; throw e; }
      const alpha = col[j] > 0 ? -norm : norm;
      const v = new Float64Array(n);
      v[j] = col[j] - alpha; for (let i = j + 1; i < n; i++) v[i] = col[i];
      let vv = 0; for (let i = j; i < n; i++) vv += v[i] * v[i];
      const apply = c => { let d = 0; for (let i = j; i < n; i++) d += v[i] * c[i]; d = 2 * d / vv; for (let i = j; i < n; i++) c[i] -= d * v[i]; };
      for (let jj = j; jj < p; jj++) apply(A[jj]);
      apply(b);
      for (let jj = j; jj < p; jj++) R[j][jj] = A[jj][j];
    }
    // back-substitution
    const beta = new Float64Array(p);
    for (let i = p - 1; i >= 0; i--) { let s = b[i]; for (let jj = i + 1; jj < p; jj++) s -= R[i][jj] * beta[jj]; beta[i] = s / R[i][i]; }
    // Rinv
    const Ri = Array.from({ length: p }, () => new Float64Array(p));
    for (let i = p - 1; i >= 0; i--) {
      Ri[i][i] = 1 / R[i][i];
      for (let jj = i + 1; jj < p; jj++) { let s = 0; for (let m = i + 1; m <= jj; m++) s += R[i][m] * Ri[m][jj]; Ri[i][jj] = -s / R[i][i]; }
    }
    const fitted = new Float64Array(n);
    for (let i = 0; i < n; i++) { let s = beta[0]; for (let j = 0; j < k; j++) s += beta[j + 1] * X[i][j]; fitted[i] = s; }
    let sse = 0; const ym = mean(y); let sst = 0;
    for (let i = 0; i < n; i++) { sse += (y[i] - fitted[i]) ** 2; sst += (y[i] - ym) ** 2; }
    const df = n - p, s2 = sse / df;
    const tcrit = tInv(0.975, df);
    const rows = [];
    for (let j = 0; j < p; j++) {
      let d = 0; for (let m = j; m < p; m++) d += Ri[j][m] * Ri[j][m];
      const se = Math.sqrt(d * s2), t = beta[j] / se;
      rows.push({ coef: beta[j], se, t, p: tPValue(t, df), lo: beta[j] - tcrit * se, hi: beta[j] + tcrit * se });
    }
    const r2 = 1 - sse / sst;
    const F = k > 0 ? ((sst - sse) / k) / s2 : NaN;
    return {
      beta: Array.from(beta), rows, sse, sst, r2, adjR2: adjR2(r2, n, k), df, s2, sigma: Math.sqrt(s2),
      F, Fp: k > 0 ? fPValue(F, k, df) : NaN, n, k,
      aic: n * Math.log(sse / n) + 2 * p,
      predict: Xn => Xn.map(r => { let s = beta[0]; for (let j = 0; j < k; j++) s += beta[j + 1] * r[j]; return s; })
    };
  }
  function vif(X) {
    const k = X[0].length; if (k < 2) return Array(k).fill(1);
    const out = [];
    for (let j = 0; j < k; j++) {
      const Xo = X.map(r => r.filter((_, m) => m !== j)), yj = X.map(r => r[j]);
      try { const f = ols(Xo, yj); out.push(1 / Math.max(1e-12, 1 - f.r2)); } catch (e) { out.push(Infinity); }
    }
    return out;
  }
  const pick = (X, idx) => X.map(r => idx.map(j => r[j]));

  /* ---------------- 標準化 + 座標下降（Lasso / ElasticNet） ---------------- */
  function standardize(X) {
    const n = X.length, k = X[0].length, mu = [], sd = [];
    for (let j = 0; j < k; j++) { const c = X.map(r => r[j]); mu.push(mean(c)); sd.push(std(c) || 1); }
    return { Z: X.map(r => r.map((v, j) => (v - mu[j]) / sd[j])), mu, sd };
  }
  /** sklearn 目標函數：1/(2n)||y-Zw||² + α·l1·||w||₁ + ½α(1-l1)||w||² */
  function enetPath(X, y, l1Ratio, nAlphas) {
    nAlphas = nAlphas || 100;
    const { Z } = standardize(X);
    const n = Z.length, k = Z[0].length, ym = mean(y), yc = y.map(v => v - ym);
    const cols = Array.from({ length: k }, (_, j) => Z.map(r => r[j]));
    const ss = cols.map(c => c.reduce((s, v) => s + v * v, 0));
    let amax = 0; for (let j = 0; j < k; j++) amax = Math.max(amax, Math.abs(cols[j].reduce((s, v, i) => s + v * yc[i], 0)) / (n * l1Ratio));
    const alphas = []; for (let i = 0; i < nAlphas; i++) alphas.push(amax * Math.pow(1e-4, i / (nAlphas - 1)));
    const w = new Float64Array(k), r = Float64Array.from(yc), path = [];
    for (const a of alphas) {
      for (let it = 0; it < 2000; it++) {
        let maxd = 0, maxw = 0;
        for (let j = 0; j < k; j++) {
          const c = cols[j]; let rho = 0;
          for (let i = 0; i < n; i++) rho += c[i] * r[i];
          rho += ss[j] * w[j];
          const thr = n * a * l1Ratio;
          const nw = (rho > thr ? rho - thr : rho < -thr ? rho + thr : 0) / (ss[j] + n * a * (1 - l1Ratio));
          const d = nw - w[j];
          if (d !== 0) { for (let i = 0; i < n; i++) r[i] -= d * c[i]; w[j] = nw; }
          maxd = Math.max(maxd, Math.abs(d)); maxw = Math.max(maxw, Math.abs(nw));
        }
        if (maxd <= 1e-9 * Math.max(1, maxw)) break;
      }
      path.push(Array.from(w));
    }
    return { alphas, path };
  }
  function rankByPath(X, y, l1Ratio) {
    const { alphas, path } = enetPath(X, y, l1Ratio);
    const k = X[0].length, entry = Array(k).fill(Infinity);
    path.forEach((w, i) => w.forEach((v, j) => { if (v !== 0 && entry[j] === Infinity) entry[j] = i; }));
    const last = path[path.length - 1];
    const order = Array.from({ length: k }, (_, j) => j).sort((a, b) => entry[a] - entry[b] || Math.abs(last[b]) - Math.abs(last[a]));
    return { order, alphas, path };
  }

  /* ---------------- 特徵選擇 ---------------- */
  function kfoldRMSE(X, y, idx, folds) {
    folds = folds || 5;
    const n = y.length; let sse = 0;
    // sklearn KFold(shuffle=False)：前 n % k 折多 1 筆
    let start = 0;
    for (let f = 0; f < folds; f++) {
      const size = Math.floor(n / folds) + (f < n % folds ? 1 : 0);
      const te = new Set(); for (let i = start; i < start + size; i++) te.add(i);
      start += size;
      const Xtr = [], ytr = [], Xte = [], yte = [];
      for (let i = 0; i < n; i++) { const r = idx.map(j => X[i][j]); if (te.has(i)) { Xte.push(r); yte.push(y[i]); } else { Xtr.push(r); ytr.push(y[i]); } }
      try { const m = ols(Xtr, ytr); const p = m.predict(Xte); for (let i = 0; i < p.length; i++) sse += (yte[i] - p[i]) ** 2; }
      catch (e) { return Infinity; }
    }
    return Math.sqrt(sse / n);
  }
  function makeScorer(ctx, criterion) {
    const { Xtr, ytr, Xte, yte } = ctx;
    return idx => {
      if (criterion === 'test') {
        try { const m = ols(pick(Xtr, idx), ytr); return metrics(yte, m.predict(pick(Xte, idx))).rmse; } catch (e) { return Infinity; }
      }
      return kfoldRMSE(Xtr, ytr, idx, 5);
    };
  }
  function forward(ctx, score) {
    const k = ctx.Xtr[0].length, sel = [], sets = [];
    while (sel.length < k) {
      let best = null, bs = Infinity;
      for (let j = 0; j < k; j++) if (!sel.includes(j)) { const s = score(sel.concat(j)); if (s < bs) { bs = s; best = j; } }
      if (best === null) break;
      sel.push(best); sets.push(sel.slice());
    }
    return sets;
  }
  function backwardP(ctx) {
    const k = ctx.Xtr[0].length; let cur = Array.from({ length: k }, (_, j) => j);
    const sets = [cur.slice()], pvals = [];
    while (cur.length > 1) {
      let worst = -1, wp = -1, m;
      try { m = ols(pick(ctx.Xtr, cur), ctx.ytr); } catch (e) { m = null; }
      cur.forEach((j, pos) => { const p = m ? m.rows[pos + 1].p : 1; if (p > wp) { wp = p; worst = j; } });
      pvals.push({ removed: worst, p: wp });
      cur = cur.filter(j => j !== worst); sets.unshift(cur.slice());
    }
    return { sets, steps: pvals };
  }
  function rfe(ctx) {
    const { Z } = standardize(ctx.Xtr);
    const k = Z[0].length; let cur = Array.from({ length: k }, (_, j) => j); const sets = [cur.slice()];
    while (cur.length > 1) {
      let m; try { m = ols(pick(Z, cur), ctx.ytr); } catch (e) { m = null; }
      let worst = cur[0], wv = Infinity;
      cur.forEach((j, pos) => { const v = m ? Math.abs(m.beta[pos + 1]) : 0; if (v < wv) { wv = v; worst = j; } });
      cur = cur.filter(j => j !== worst); sets.unshift(cur.slice());
    }
    return sets;
  }
  function exhaustive(ctx, score) {
    const k = ctx.Xtr[0].length, best = Array(k).fill(null).map(() => ({ s: Infinity, set: null }));
    for (let mask = 1; mask < (1 << k); mask++) {
      const idx = []; for (let j = 0; j < k; j++) if (mask & (1 << j)) idx.push(j);
      const s = score(idx); const b = best[idx.length - 1];
      if (s < b.s) { b.s = s; b.set = idx; }
    }
    return best.map(b => b.set);
  }
  const ALGOS = {
    lasso: { name: 'Lasso (L1 路徑排序)', short: 'Lasso' },
    enet: { name: 'ElasticNet (L1+L2, l1_ratio=0.5)', short: 'ElasticNet' },
    forward: { name: 'Forward 逐步選擇 (greedy)', short: 'Forward' },
    backward: { name: 'Backward 逐步剔除 (p-value)', short: 'Backward' },
    rfe: { name: 'RFE 遞迴特徵消去', short: 'RFE' },
    exhaustive: { name: 'Exhaustive 窮舉最佳子集', short: 'Exhaustive' }
  };
  function nestedFromOrder(order) { return order.map((_, i) => order.slice(0, i + 1)); }
  function selectSets(algo, ctx, criterion) {
    const score = makeScorer(ctx, criterion);
    switch (algo) {
      case 'lasso': return nestedFromOrder(rankByPath(ctx.Xtr, ctx.ytr, 1).order);
      case 'enet': return nestedFromOrder(rankByPath(ctx.Xtr, ctx.ytr, 0.5).order);
      case 'forward': return forward(ctx, score);
      case 'backward': return backwardP(ctx).sets;
      case 'rfe': return rfe(ctx);
      case 'exhaustive': return exhaustive(ctx, score);
    }
    throw new Error('unknown algo ' + algo);
  }
  /** 對每個 k 擬合 OLS，回傳訓練 / CV / 測試指標 */
  function evaluateSets(sets, ctx) {
    return sets.map((idx, i) => {
      const m = ols(pick(ctx.Xtr, idx), ctx.ytr);
      const ptr = m.predict(pick(ctx.Xtr, idx)), pte = m.predict(pick(ctx.Xte, idx));
      const tr = metrics(ctx.ytr, ptr), te = metrics(ctx.yte, pte);
      return { k: i + 1, idx, model: m, train: tr, test: te, cv: kfoldRMSE(ctx.Xtr, ctx.ytr, idx, 5), testAdjR2: adjR2(te.r2, te.n, idx.length), pte };
    });
  }

  /* ---------------- 時間序列：AR(p) ---------------- */
  function lagMatrix(series, p, start) {
    const X = [], y = [], t = [];
    for (let i = Math.max(p, start || 0); i < series.length; i++) { const r = []; for (let l = 1; l <= p; l++) r.push(series[i - l]); X.push(r); y.push(series[i]); t.push(i); }
    return { X, y, t };
  }
  function acf(series, maxLag) {
    const n = series.length, m = mean(series); let c0 = 0;
    for (const v of series) c0 += (v - m) ** 2;
    const out = [];
    for (let l = 1; l <= maxLag; l++) { let c = 0; for (let i = l; i < n; i++) c += (series[i] - m) * (series[i - l] - m); out.push(c / c0); }
    return out;
  }
  /** Durbin–Levinson */
  function pacf(series, maxLag) {
    const r = [1].concat(acf(series, maxLag)), out = [];
    let phiPrev = [];
    for (let k = 1; k <= maxLag; k++) {
      let num = r[k], den = 1;
      for (let j = 1; j < k; j++) { num -= phiPrev[j - 1] * r[k - j]; den -= phiPrev[j - 1] * r[j]; }
      const pkk = num / den, phi = [];
      for (let j = 1; j < k; j++) phi.push(phiPrev[j - 1] - pkk * phiPrev[k - j - 1]);
      phi.push(pkk); phiPrev = phi; out.push(pkk);
    }
    return out;
  }
  /** 依時間順序切分：最後 testFrac 為測試集（不打亂） */
  function fitAR(series, p, testFrac, start) {
    const { X, y, t } = lagMatrix(series, p, start);
    const nTest = Math.max(1, Math.round(y.length * testFrac)), nTr = y.length - nTest;
    const Xtr = X.slice(0, nTr), ytr = y.slice(0, nTr), Xte = X.slice(nTr), yte = y.slice(nTr);
    const m = ols(Xtr, ytr);
    const ptr = m.predict(Xtr), pte = m.predict(Xte);
    const naive = Xte.map(r => r[0]);
    const te = metrics(yte, pte);
    let hit = 0; for (let i = 0; i < yte.length; i++) { const prev = Xte[i][0]; if (Math.sign(pte[i] - prev) === Math.sign(yte[i] - prev)) hit++; }
    return { p, model: m, train: metrics(ytr, ptr), test: te, naive: metrics(yte, naive), dirAcc: 100 * hit / yte.length, tTest: t.slice(nTr), tTrain: t.slice(0, nTr), pte, ptr, yte, ytr, nTr, nTest };
  }
  /** 遞迴預測 h 步，附 95% 預測區間（ψ 權重） */
  function forecastAR(series, model, p, h) {
    const hist = series.slice(), out = [];
    const phi = model.beta.slice(1);
    const psi = [1];
    for (let j = 1; j < h; j++) { let s = 0; for (let i = 1; i <= Math.min(j, p); i++) s += phi[i - 1] * psi[j - i]; psi.push(s); }
    let cum = 0; const z = 1.959963984540054;
    for (let step = 0; step < h; step++) {
      let v = model.beta[0]; for (let l = 1; l <= p; l++) v += phi[l - 1] * hist[hist.length - l];
      hist.push(v); cum += psi[step] * psi[step];
      const se = model.sigma * Math.sqrt(cum);
      out.push({ mean: v, lo: v - z * se, hi: v + z * se });
    }
    return out;
  }

  root.ML = {
    MT19937, permutation, trainTestSplit, parseCSV, toNumber, isMissing,
    tPValue, tInv, fPValue, mean, std, metrics, adjR2, ols, vif, pick, standardize, enetPath, rankByPath,
    kfoldRMSE, selectSets, evaluateSets, backwardP, ALGOS,
    lagMatrix, acf, pacf, fitAR, forecastAR
  };
})(typeof window !== 'undefined' ? window : globalThis);
