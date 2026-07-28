// animate3.js — MODEL 3: "delay-coded tables + a listening aggregator population".
// Each table stays a self-contained model-1 black box: detector -> complement ->
// row-select -> a DELAY-coded output stage that emits Dout latency spikes
// (t = GZ_t + ALPHA + BETA*O_t[r_t][j], bigger value = LATER, like single_table).
// A SEPARATE population of Dout aggregator neurons LISTENS to every table's output
// spikes: each incoming spike switches on a SUSTAINED constant current I from its
// arrival until a fixed readout clock T_agg, so
//     V_j(T_agg) = I * sum_t (T_agg - t_{t,j}) = const - I*BETA*S_j
// is AFFINE in the true sum S_j = sum_t O_t[r_t][j]. The aggregator reads V_j at
// T_agg and re-emits a latency spike at t_out_j = T_agg + ALPHA_A + BETA_A*S_j.
// Contrast with model 2 (charge summation on SHARED outputs): here the tables do
// not share outputs at all — they emit their own spikes and a downstream
// population sums them. Single rAF / clear-each-frame / size-once + resize/DPR fix.
(function () {
  'use strict';

  function showErr(msg) {
    let bar = document.getElementById('errbar');
    if (!bar) { bar = document.createElement('div'); bar.id = 'errbar'; document.body.prepend(bar); }
    bar.style.display = 'block'; bar.textContent = '⚠ Aggregator page error — ' + msg;
    if (window.console) console.error(msg);
  }
  window.addEventListener('error', (e) => showErr((e.error && e.error.stack) || e.message));
  window.addEventListener('unhandledrejection', (e) => showErr('promise: ' + ((e.reason && e.reason.message) || e.reason)));

  const LS = window.LUTSpiking;
  if (!LS) { showErr('lut_spiking.js did not load'); return; }

  const m1 = LS.buildModel({ SEED: 0 });
  const m2 = LS.buildModel({ SEED: 1 });
  const { D, K, Dout } = m1.cfg;
  const NT = 2;
  // per-table output latency code (value in conduction delay, bigger value = later)
  const ALPHA = 5.0, BETA = 1.0, SETTLE = 0.6;
  // aggregator: sustained current I from each arrival until T_agg; then re-emit
  // t_out_j = T_agg + ALPHA_A + BETA_A*S_j (same bigger=later convention, composable).
  const I_CUR = 1.0, SETTLE_A = 0.6, ALPHA_A = 5.0, BETA_A = 1.0;
  const MIN_TRAVEL = 0.6, FLASH = 0.35;
  let x = new Array(D).fill(0);

  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const fmt = (v) => (v >= 0 ? '+' : '') + v.toFixed(3);
  const bitBox = (b) => `<span class="bit ${b ? 'on' : 'off'}">${b}</span>`;
  function fillVec(id, vec) {
    const e = document.getElementById(id); if (!e) return;
    const mx = Math.max(1e-9, ...vec.map(Math.abs));
    e.innerHTML = vec.map((v) => {
      const wp = (Math.abs(v) / mx) * 50;
      return `<div class="small mono">${fmt(v)}</div><div class="vecbar"><i style="left:${v < 0 ? 50 - wp : 50}%;width:${wp}%;background:${v < 0 ? css('--bad') : css('--spike')}"></i></div>`;
    }).join('');
  }
  const visualArrival = (e) => e.tDepart + Math.max(e.tArrive - e.tDepart, MIN_TRAVEL);
  function nodeV(INC, id, t) {
    const inc = INC[id]; if (!inc) return 0;
    let V = 0; for (const e of inc) { if (t < e.tv) continue; V += e.kind === 'ramp' ? e.weight * (t - e.tv) : e.weight; }
    return V;
  }
  function weightColor(MAXW, wt) {
    const mag = Math.min(1, Math.abs(wt) / MAXW);
    return { color: Math.abs(wt) < 1e-9 ? '#6b7280' : (wt < 0 ? '#5b9dff' : '#ff6b6b'), alpha: 0.12 + 0.55 * mag };
  }
  function nodeColor(id) {
    if (id === 'START') return css('--muted'); if (id[0] === 'x') return css('--spike');
    if (id === 'CLK') return css('--ok'); if (id[0] === 'H') return css('--addr');
    if (id[0] === 'C' && id[1] !== 'L') return css('--warn'); return css('--row');
  }
  const AGGCOL = '#e6b3ff';   // aggregator population — distinct light purple

  let maxv = 1e-6;
  for (const O of [m1.V, m2.V]) for (const row of O) for (const v of row) maxv = Math.max(maxv, Math.abs(v));

  // ---- canvases sized ONCE (dpr-capped, width-guarded — mobile-safe) ---------
  const canv = {};
  ['graph', 'atimeline'].forEach((id) => { const cv = document.getElementById(id); if (cv) canv[id] = { cv, ctx: cv.getContext('2d'), w: 0, h: 0, hCSS: +cv.getAttribute('height') || 300 }; });
  let needResize = true;
  function resizeAll() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    for (const id in canv) {
      const c = canv[id];
      const w = Math.round(c.cv.clientWidth || (c.cv.parentElement && c.cv.parentElement.clientWidth) || 820);
      const h = c.hCSS;
      const bw = Math.max(1, Math.round(w * dpr));
      if (w === c.w && c.cv.width === bw) continue;
      c.cv.style.height = h + 'px';
      c.cv.width = bw; c.cv.height = Math.round(h * dpr);
      c.ctx.setTransform(dpr, 0, 0, dpr, 0, 0); c.w = w; c.h = h;
    }
    needResize = false;
  }

  // ---- per-table state: reuse the model-1 detector/decoder machinery ---------
  function tableState(m) {
    const S = LS.simulateCircuit(m, x);
    const TOPO = LS.circuitTopology(m).filter((e) => e.dst[0] !== 'o');   // skeleton, minus row->output (drawn separately)
    const MAXW = Math.max(1e-6, ...TOPO.map((e) => Math.abs(e.weight)));
    const INCOMING = {}, FIREVISUAL = {};
    for (const e of S.events) {
      if (e.dst[0] === 'o') continue;
      const tv = visualArrival(e);
      (INCOMING[e.dst] || (INCOMING[e.dst] = [])).push({ kind: e.kind, weight: e.weight, tv });
    }
    for (const id in S.thr) {
      if (id[0] === 'o') continue;
      const thr = S.thr[id], evs = (INCOMING[id] || []).slice().sort((a, b) => a.tv - b.tv);
      let V = 0, slope = 0, last = 0, fv = Infinity;
      for (const e of evs) { V += slope * (e.tv - last); last = e.tv; if (e.kind === 'ramp') slope += e.weight; else V += e.weight; if (V > thr) { fv = e.tv; break; } }
      FIREVISUAL[id] = (S.st[id] && S.st[id].fired) ? fv : Infinity;
    }
    FIREVISUAL['START'] = 0; for (let i = 0; i < D; i++) FIREVISUAL['x' + i] = S.tin[i];
    const bits = LS.lutBits(m, x), r = LS.bitsToRow(bits);
    return { m, S, TOPO, MAXW, INCOMING, FIREVISUAL, r, out: m.V[r] };
  }

  let T1, T2, Tmax = 1, agg;
  let simT = 0, playing = false, speed = 1, lastTs = null, dirty = true;
  const RATE = 2;

  // aggregator membrane at time t: sustained current from each arrival until T_agg
  function aggV(j, t) {
    const tt = Math.min(t, agg.T_agg); let V = 0;
    for (const T of [T1, T2]) { const ts = T.tout[j]; if (tt > ts) V += I_CUR * (tt - ts); }
    return V;
  }

  function recompute() {
    T1 = tableState(m1); T2 = tableState(m2);
    const tabs = [T1, T2];
    for (const T of tabs) {
      T.GZt = T.FIREVISUAL['r' + T.r] + SETTLE;               // per-table output ground zero
      T.tout = T.out.map((v) => T.GZt + ALPHA + BETA * v);    // Dout output-spike times (delay code)
    }
    const allt = []; for (const T of tabs) for (const t of T.tout) allt.push(t);
    const T_agg = Math.max(...allt) + SETTLE_A;               // fixed aggregator readout clock
    const C0 = tabs.reduce((a, T) => a + (T.GZt + ALPHA), 0);  // j-independent constant
    const Strue = [], Shat = [], Vend = [], outT = [];
    for (let j = 0; j < Dout; j++) {
      let s = 0, V = 0;
      for (const T of tabs) { s += T.out[j]; V += I_CUR * (T_agg - T.tout[j]); }
      Strue.push(s); Vend.push(V);
      const sh = (I_CUR * (NT * T_agg - C0) - V) / (I_CUR * BETA);   // recover the sum from V_j
      Shat.push(sh);
      outT.push(T_agg + ALPHA_A + BETA_A * sh);                     // re-emit as a latency spike
    }
    agg = { T_agg, C0, Strue, Shat, Vend, outT };
    Tmax = Math.max(...outT) + 0.8;
    agg.NTICKS = Math.ceil(Tmax - 0.8 - 1e-9);
    const sc = document.getElementById('scrub'); sc.max = Tmax.toFixed(2); sc.step = (Tmax / 700).toFixed(4);
    fillLUT('1', m1); fillLUT('2', m2); fillGtTable(); updateAggInfo();
    dirty = true;
  }

  // ---- a single band node (base disc + membrane fill + flash + border) -------
  function drawNode(ctx, p, id, col, fv, thr, INC, R) {
    const DRAIN = 0.5;
    const fired = isFinite(fv) && simT >= fv;
    const flashing = isFinite(fv) && Math.abs(simT - fv) < FLASH;
    let frac = 0;
    if (thr) frac = fired ? Math.max(0, 1 - (simT - fv) / DRAIN) : Math.max(0, Math.min(1, nodeV(INC, id, simT) / thr));
    if (flashing) { const a = 1 - Math.abs(simT - fv) / FLASH; ctx.globalAlpha = 0.5 * a; ctx.fillStyle = col; ctx.beginPath(); ctx.arc(p.x, p.y, R + 6 * a, 0, 7); ctx.fill(); ctx.globalAlpha = 1; }
    ctx.beginPath(); ctx.arc(p.x, p.y, R, 0, 7); ctx.fillStyle = '#20262f'; ctx.fill();
    if (frac > 0.002) { ctx.save(); ctx.beginPath(); ctx.arc(p.x, p.y, R, 0, 7); ctx.clip(); ctx.globalAlpha = 0.85; ctx.fillStyle = col; const fh = 2 * R * frac; ctx.fillRect(p.x - R, p.y + R - fh, 2 * R, fh); ctx.globalAlpha = 1; ctx.restore(); }
    ctx.beginPath(); ctx.arc(p.x, p.y, R, 0, 7); ctx.strokeStyle = fired ? col : css('--edge'); ctx.lineWidth = fired ? 2 : 1; ctx.stroke(); ctx.lineWidth = 1;
    ctx.fillStyle = frac > 0.55 ? '#05080d' : css('--muted'); ctx.font = 'bold 8px ui-monospace'; ctx.textAlign = 'center'; ctx.fillText(id, p.x, p.y + 3); ctx.textAlign = 'left';
  }

  // ---- ONE unified graph: inputs -> two delay-output tables -> aggregator -----
  function drawUnifiedGraph() {
    const c = canv.graph; if (!c || !T1 || !T2 || !agg) return;
    const { ctx, w, h } = c; ctx.clearRect(0, 0, w, h);
    const NC = 7, padL = 34, padR = 44, padT = 26, padB = 14, R = 8, CLOCKCOL = css('--ok');
    const colX = (ci) => padL + (ci / NC) * (w - padL - padR);
    const fullTop = padT + 16, fullBot = h - padB, midY = (fullTop + fullBot) / 2;
    const shp = {};
    shp['START'] = { x: colX(0), y: midY };
    for (let i = 0; i < D; i++) shp['x' + i] = { x: colX(1), y: fullTop + (i + 0.5) * (fullBot - fullTop) / D };
    shp['CLK'] = { x: colX(2), y: midY };
    for (let j = 0; j < Dout; j++) shp['a' + j] = { x: colX(7), y: midY - (Dout - 1) * 15 + j * 30 };   // aggregator column
    const band = (bt, bb) => {
      const p = {};
      for (let k = 0; k < K; k++) p['H' + k] = { x: colX(3), y: bt + (k + 0.5) * (bb - bt) / K };
      for (let k = 0; k < K; k++) p['C' + k] = { x: colX(4), y: bt + (k + 0.5) * (bb - bt) / K };
      for (let a = 0; a < (1 << K); a++) p['r' + a] = { x: colX(5), y: bt + (a + 0.5) * (bb - bt) / (1 << K) };
      for (let j = 0; j < Dout; j++) p['oT' + j] = { x: colX(6), y: bt + (j + 0.5) * (bb - bt) / Dout };   // per-table output
      return p;
    };
    const pA = band(fullTop, midY - 10), pB = band(midY + 10, fullBot);
    const bandCol = [css('--spike'), css('--out')];
    const sharedId = (id) => id[0] === 'x' || id === 'START' || id === 'CLK' || id[0] === 'a';
    const posOf = (tbl, id) => sharedId(id) ? shp[id] : (tbl === 1 ? pA : pB)[id];
    const tables = [[1, T1, pA], [2, T2, pB]];

    tables.forEach(([tbl, T]) => {
      ctx.lineWidth = 1;
      for (const e of T.TOPO) {
        const a = posOf(tbl, e.src), b = posOf(tbl, e.dst); if (!a || !b) continue;
        const inhib = e.src[0] === 'H' && e.dst[0] === 'C';
        if (inhib) { ctx.strokeStyle = css('--lut'); ctx.globalAlpha = 0.7; ctx.setLineDash([3, 3]); ctx.lineWidth = 1.4; }
        else { const wc = weightColor(T.MAXW, e.weight); ctx.strokeStyle = wc.color; ctx.globalAlpha = wc.alpha; }
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
        if (inhib) { ctx.setLineDash([]); ctx.lineWidth = 1; }
      }
      ctx.globalAlpha = 1;
      // winning row -> this table's own output neurons (delay-coded value)
      const rp = (tbl === 1 ? pA : pB)['r' + T.r];
      for (let j = 0; j < Dout; j++) {
        const op = (tbl === 1 ? pA : pB)['oT' + j];
        ctx.strokeStyle = bandCol[tbl - 1]; ctx.globalAlpha = simT >= T.GZt ? 0.5 : 0.12; ctx.lineWidth = 1.3;
        ctx.beginPath(); ctx.moveTo(rp.x, rp.y); ctx.lineTo(op.x, op.y); ctx.stroke();
      }
      ctx.globalAlpha = 1; ctx.lineWidth = 1;
      // aggregator listening edges: this table's output j -> aggregator j
      for (let j = 0; j < Dout; j++) {
        const op = (tbl === 1 ? pA : pB)['oT' + j], ap = shp['a' + j];
        ctx.strokeStyle = AGGCOL; ctx.globalAlpha = simT >= T.tout[j] ? 0.55 : 0.1; ctx.lineWidth = 1.3;
        ctx.beginPath(); ctx.moveTo(op.x, op.y); ctx.lineTo(ap.x, ap.y); ctx.stroke();
      }
      ctx.globalAlpha = 1; ctx.lineWidth = 1;
      // traveling dots inside the table (detector/decoder events)
      for (const e of T.S.events) {
        if (e.dst[0] === 'o') continue;
        const a = posOf(tbl, e.src), b = posOf(tbl, e.dst); if (!a || !b) continue;
        const vspan = Math.max(e.tArrive - e.tDepart, MIN_TRAVEL);
        if (simT < e.tDepart || simT > e.tDepart + vspan) continue;
        const fr = (simT - e.tDepart) / vspan;
        ctx.fillStyle = e.src === 'CLK' ? CLOCKCOL : (e.weight < 0 ? '#5b9dff' : '#ff6b6b');
        ctx.beginPath(); ctx.arc(a.x + (b.x - a.x) * fr, a.y + (b.y - a.y) * fr, e.src === 'CLK' ? 4 : 3.2, 0, 7); ctx.fill();
      }
      // traveling dot: winning row -> oT_j, arriving exactly at the emit time t_{t,j}
      for (let j = 0; j < Dout; j++) {
        const op = (tbl === 1 ? pA : pB)['oT' + j];
        const dep = T.GZt, arr = T.tout[j];
        if (simT < dep || simT > arr) continue;
        const fr = (simT - dep) / Math.max(arr - dep, 1e-6);
        ctx.fillStyle = bandCol[tbl - 1];
        ctx.beginPath(); ctx.arc(rp.x + (op.x - rp.x) * fr, rp.y + (op.y - rp.y) * fr, 3.2, 0, 7); ctx.fill();
      }
      // traveling dot: oT_j -> aggregator j (the aggregator "listening")
      for (let j = 0; j < Dout; j++) {
        const op = (tbl === 1 ? pA : pB)['oT' + j], ap = shp['a' + j];
        const dep = T.tout[j], arr = T.tout[j] + MIN_TRAVEL;
        if (simT < dep || simT > arr) continue;
        const fr = (simT - dep) / MIN_TRAVEL;
        ctx.fillStyle = AGGCOL;
        ctx.beginPath(); ctx.arc(op.x + (ap.x - op.x) * fr, op.y + (ap.y - op.y) * fr, 3.4, 0, 7); ctx.fill();
      }
      // band nodes (detectors, complements, rows)
      const pb = (tbl === 1 ? pA : pB);
      for (const id in pb) { if (id[0] === 'o') continue; drawNode(ctx, pb[id], id, nodeColor(id), T.FIREVISUAL[id], T.S.thr[id], T.INCOMING, R); }
      // this table's own output neurons: fire (flash) at t_{t,j}
      for (let j = 0; j < Dout; j++) {
        const p = pb['oT' + j], fv = T.tout[j], fired = simT >= fv, flashing = Math.abs(simT - fv) < FLASH;
        if (flashing) { const a = 1 - Math.abs(simT - fv) / FLASH; ctx.globalAlpha = 0.5 * a; ctx.fillStyle = bandCol[tbl - 1]; ctx.beginPath(); ctx.arc(p.x, p.y, R + 6 * a, 0, 7); ctx.fill(); ctx.globalAlpha = 1; }
        ctx.beginPath(); ctx.arc(p.x, p.y, R, 0, 7); ctx.fillStyle = fired ? bandCol[tbl - 1] : '#20262f'; ctx.fill();
        ctx.beginPath(); ctx.arc(p.x, p.y, R, 0, 7); ctx.strokeStyle = fired ? bandCol[tbl - 1] : css('--edge'); ctx.lineWidth = fired ? 2 : 1; ctx.stroke(); ctx.lineWidth = 1;
        ctx.fillStyle = fired ? '#05080d' : css('--muted'); ctx.font = 'bold 7px ui-monospace'; ctx.textAlign = 'center'; ctx.fillText('o' + j, p.x, p.y + 2.5); ctx.textAlign = 'left';
      }
    });

    // aggregator neurons: accumulate sustained charge, fire the summed spike
    const Ra = 12, Vmax = Math.max(1e-6, ...agg.Vend);
    for (let j = 0; j < Dout; j++) {
      const p = shp['a' + j];
      const frac = Math.max(0, Math.min(1, aggV(j, simT) / Math.max(agg.Vend[j], 1e-6)));
      const fired = simT >= agg.outT[j], flashing = Math.abs(simT - agg.outT[j]) < FLASH;
      if (flashing) { const a = 1 - Math.abs(simT - agg.outT[j]) / FLASH; ctx.globalAlpha = 0.6 * a; ctx.fillStyle = AGGCOL; ctx.beginPath(); ctx.arc(p.x, p.y, Ra + 9 * a, 0, 7); ctx.fill(); ctx.globalAlpha = 1; }
      ctx.beginPath(); ctx.arc(p.x, p.y, Ra, 0, 7); ctx.fillStyle = '#20262f'; ctx.fill();
      if (frac > 0.002) { ctx.save(); ctx.beginPath(); ctx.arc(p.x, p.y, Ra, 0, 7); ctx.clip(); ctx.globalAlpha = 0.85; ctx.fillStyle = AGGCOL; const fh = 2 * Ra * frac; ctx.fillRect(p.x - Ra, p.y + Ra - fh, 2 * Ra, fh); ctx.globalAlpha = 1; ctx.restore(); }
      ctx.beginPath(); ctx.arc(p.x, p.y, Ra, 0, 7); ctx.strokeStyle = fired ? AGGCOL : css('--edge'); ctx.lineWidth = fired ? 2.5 : 1; ctx.stroke(); ctx.lineWidth = 1;
      ctx.fillStyle = AGGCOL; ctx.font = 'bold 9px ui-monospace'; ctx.textAlign = 'center'; ctx.fillText('Σ' + j, p.x, p.y + 3);
      if (fired) { ctx.fillStyle = AGGCOL; ctx.beginPath(); ctx.arc(p.x + Ra + 7, p.y, 3.5, 0, 7); ctx.fill(); }
      ctx.textAlign = 'left';
    }

    // shared input column
    ['START'].concat(Array.from({ length: D }, (_, i) => 'x' + i)).concat(['CLK'])
      .forEach((id) => drawNode(ctx, shp[id], id, nodeColor(id), T1.FIREVISUAL[id], T1.S.thr[id], T1.INCOMING, R));

    ctx.fillStyle = css('--accent'); ctx.font = 'bold 9px ui-monospace'; ctx.textAlign = 'center';
    ['START', 'INPUT', 'CLOCK', 'DETECT', 'COMPL', 'ROWS', 'TAB-OUT', 'AGGREGATOR'].forEach((t, i) => ctx.fillText(t, colX(i), 12));
    ctx.fillStyle = css('--spike'); ctx.font = 'bold 10px ui-monospace'; ctx.textAlign = 'left';
    ctx.fillText('Table 1', colX(3) - 16, fullTop - 3);
    ctx.fillStyle = css('--out'); ctx.fillText('Table 2', colX(3) - 16, midY + 13);
  }

  // ---- TIMELINE: per-table output spikes -> T_agg -> summed output spikes -----
  function drawAggTimeline() {
    const c = canv.atimeline; if (!c || !agg) return;
    const { ctx, w, h } = c; ctx.clearRect(0, 0, w, h);
    const N = Math.max(1, agg.NTICKS), t0 = 0, t1 = N;
    const padL = 118, padR = 24, padT = 30, padB = 24;
    const X = (t) => padL + ((t - t0) / (t1 - t0)) * (w - padL - padR);
    const laneTop = padT, laneBot = h - padB, nLanes = NT + Dout, laneH = (laneBot - laneTop) / nLanes;
    const laneY = (i) => laneTop + laneH * (i + 0.5);
    ctx.font = '11px ui-monospace';
    for (let t = 0; t <= N; t++) {
      ctx.strokeStyle = css('--edge'); ctx.globalAlpha = (t % 5 === 0) ? .5 : .16;
      ctx.beginPath(); ctx.moveTo(X(t), laneTop - 6); ctx.lineTo(X(t), laneBot); ctx.stroke(); ctx.globalAlpha = 1;
      ctx.fillStyle = css('--muted'); ctx.textAlign = 'center'; ctx.fillText(t, X(t), laneBot + 15); ctx.textAlign = 'left';
    }
    ctx.fillStyle = css('--accent'); ctx.font = 'bold 13px ui-monospace';
    ctx.fillText('⏱ ' + N + ' ticks: START → tables emit → T_agg → summed output', padL, 16);
    ctx.font = '11px ui-monospace';
    const bandCol = [css('--spike'), css('--out')];
    // per-table output-spike lanes
    for (let t = 0; t < NT; t++) {
      const y = laneY(t), T = t === 0 ? T1 : T2;
      ctx.fillStyle = bandCol[t]; ctx.fillText('table ' + (t + 1) + ' out', 6, y + 4);
      ctx.strokeStyle = css('--edge'); ctx.globalAlpha = .3; ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke(); ctx.globalAlpha = 1;
      for (let j = 0; j < Dout; j++) {
        const tt = T.tout[j]; if (simT < tt) continue;
        ctx.strokeStyle = bandCol[t]; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(X(tt), y - laneH * .3); ctx.lineTo(X(tt), y + laneH * .3); ctx.stroke(); ctx.lineWidth = 1;
      }
    }
    // T_agg readout clock
    ctx.setLineDash([4, 3]); ctx.strokeStyle = css('--warn'); ctx.globalAlpha = .95;
    ctx.beginPath(); ctx.moveTo(X(agg.T_agg), laneTop - 6); ctx.lineTo(X(agg.T_agg), laneBot); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
    ctx.fillStyle = css('--warn'); ctx.textAlign = 'center'; ctx.fillText('T_agg ' + agg.T_agg.toFixed(1), X(agg.T_agg), laneTop - 8); ctx.textAlign = 'left';
    // aggregated (summed) output-spike lanes
    for (let j = 0; j < Dout; j++) {
      const y = laneY(NT + j);
      ctx.fillStyle = AGGCOL; ctx.fillText('Σ o' + j, 6, y + 4);
      ctx.strokeStyle = css('--edge'); ctx.globalAlpha = .3; ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke(); ctx.globalAlpha = 1;
      if (simT >= agg.outT[j]) {
        ctx.strokeStyle = AGGCOL; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(X(agg.outT[j]), y - laneH * .32); ctx.lineTo(X(agg.outT[j]), y + laneH * .32); ctx.stroke(); ctx.lineWidth = 1;
        ctx.fillStyle = AGGCOL; ctx.fillText('t=' + agg.outT[j].toFixed(2), X(agg.outT[j]) + 5, y - 3);
      }
    }
    if (simT >= t0) { ctx.strokeStyle = AGGCOL; ctx.lineWidth = 1.5; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(X(Math.min(simT, t1)), laneTop - 6); ctx.lineTo(X(Math.min(simT, t1)), laneBot); ctx.stroke(); ctx.setLineDash([]); ctx.lineWidth = 1; }
  }

  // ---- LUT panels + aggregator info + ground-truth table (DOM) ---------------
  function fillLUT(sfx, m) {
    const pre = LS.lutPreacts(m, x), bits = LS.lutBits(m, x), r = LS.bitsToRow(bits);
    const tb = document.querySelector('#lut' + sfx + '_tests tbody'); tb.innerHTML = '';
    for (let k = 0; k < K; k++) {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>H${k}</td><td class="mono" style="color:${pre[k] > 0 ? '#7ee787' : '#ffa198'}">${fmt(pre[k])}</td><td>${bitBox(bits[k])}</td>`;
      tb.appendChild(tr);
    }
    document.getElementById('lut' + sfx + '_addr').textContent = bits.join('');
    document.getElementById('lut' + sfx + '_row').textContent = r;
    fillVec('lut' + sfx + '_out', m.V[r]);
  }
  function updateAggInfo() {
    const el = document.getElementById('agginfo'); if (!el) return;
    el.innerHTML = '⏱ This resolves in <b>' + agg.NTICKS + ' discrete ticks</b> — each table emits its 4 output spikes '
      + '(t = GZ<sub>t</sub> + α + β·O), the aggregator integrates a sustained current I until the readout clock '
      + '<b>T_agg = ' + agg.T_agg.toFixed(1) + '</b>, then re-emits the summed spikes at t = T_agg + A + B·S. '
      + 'One tick = one latency unit (β=1). Deeper than model 2 because the aggregator waits for the readout clock.';
  }
  function fillGtTable() {
    const tb = document.querySelector('#gtmatch tbody'); if (!tb) return; tb.innerHTML = '';
    const b1 = LS.lutBits(m1, x), r1 = LS.bitsToRow(b1), O1 = m1.V[r1];
    const b2 = LS.lutBits(m2, x), r2 = LS.bitsToRow(b2), O2 = m2.V[r2];
    let maxres = 0;
    for (let j = 0; j < Dout; j++) {
      const strue = O1[j] + O2[j];
      const smem = agg.Shat[j];                                    // read from the aggregator membrane V_j
      const semit = (agg.outT[j] - agg.T_agg - ALPHA_A) / BETA_A;  // decode from the emitted spike time
      const res = Math.max(Math.abs(strue - smem), Math.abs(strue - semit)); maxres = Math.max(maxres, res);
      const ok = res < 1e-9;
      const tr = document.createElement('tr');
      tr.innerHTML = '<td>o' + j + '</td><td class="mono">' + fmt(O1[j]) + '</td><td class="mono">' + fmt(O2[j])
        + '</td><td class="mono"><b>' + fmt(strue) + '</b></td><td class="mono">' + fmt(smem)
        + '</td><td class="mono">' + fmt(semit) + '</td><td class="' + (ok ? 'yes' : 'no') + '">' + (ok ? '✓' : fmt(res)) + '</td>';
      tb.appendChild(tr);
    }
    const note = document.getElementById('gtnote');
    if (note) note.innerHTML = 'Both the aggregator membrane readout and the emitted-spike decode equal the direct LUT sum to <b>'
      + maxres.toExponential(1) + '</b> for this input — the tables never shared an output; a downstream population summed their spikes.';
  }

  // ---- clock + single rAF loop ----------------------------------------------
  function draw() {
    if (needResize) resizeAll();
    drawUnifiedGraph(); drawAggTimeline();
    document.getElementById('simtime').textContent = 't = ' + simT.toFixed(2) + ' / ' + Tmax.toFixed(2);
  }
  function loop(ts) {
    try {
      if (document.hidden) { lastTs = null; requestAnimationFrame(loop); return; }
      if (lastTs == null) lastTs = ts;
      let dt = (ts - lastTs) / 1000; lastTs = ts; if (dt > 0.1) dt = 0.1;
      if (playing) { simT += dt * speed * RATE; if (simT >= Tmax) { simT = Tmax; playing = false; setPlay(false); } document.getElementById('scrub').value = simT; dirty = true; }
      if (dirty || needResize) { draw(); dirty = false; }
    } catch (err) { showErr((err && err.stack) || String(err)); }
    requestAnimationFrame(loop);
  }
  const setPlay = (p) => { document.getElementById('play').textContent = p ? '❚❚ Pause' : '▶ Play'; };

  const slidersEl = document.getElementById('sliders');
  const valEls = [];
  for (let i = 0; i < D; i++) {
    const row = document.createElement('div'); row.className = 'slider-row';
    row.innerHTML = `<label>x${i}</label><input type="range" min="-1" max="1" step="0.01" value="0" data-i="${i}"><span class="val">0.00</span>`;
    slidersEl.appendChild(row);
    const inp = row.querySelector('input'), val = row.querySelector('.val'); valEls.push(val);
    inp.addEventListener('input', () => { x[i] = parseFloat(inp.value); val.textContent = x[i].toFixed(2); recompute(); simT = 0; playing = false; setPlay(false); dirty = true; });
  }
  function syncSliders() { slidersEl.querySelectorAll('input').forEach((inp) => { const i = +inp.dataset.i; inp.value = x[i]; valEls[i].textContent = x[i].toFixed(2); }); }
  document.getElementById('rand').onclick = () => { x = x.map(() => +(2 * Math.random() - 1).toFixed(2)); syncSliders(); recompute(); simT = 0; playing = true; setPlay(true); dirty = true; };
  document.getElementById('reset').onclick = () => { x = new Array(D).fill(0); syncSliders(); recompute(); simT = 0; playing = false; setPlay(false); dirty = true; };
  document.getElementById('play').onclick = () => { if (simT >= Tmax) simT = 0; playing = !playing; setPlay(playing); dirty = true; };
  document.getElementById('restart').onclick = () => { simT = 0; playing = true; setPlay(true); dirty = true; };
  document.getElementById('speed').onchange = (e) => { speed = parseFloat(e.target.value); };
  document.getElementById('scrub').addEventListener('input', (e) => { playing = false; setPlay(false); simT = parseFloat(e.target.value); dirty = true; });
  window.addEventListener('resize', () => { needResize = true; });
  document.addEventListener('visibilitychange', () => { if (!document.hidden) { needResize = true; dirty = true; } });

  recompute();
  requestAnimationFrame(loop);
})();
