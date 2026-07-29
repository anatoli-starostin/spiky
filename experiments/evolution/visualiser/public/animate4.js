// animate4.js — MODEL 4: HIERARCHICAL / TREE aggregation of delay-coded tables.
// Model 3 dumps every table's output spikes onto ONE aggregator (fan-in grows
// with N). Model 4 builds a binary aggregation TREE: each aggregator has fan-in
// EXACTLY 2, so depth is O(log2 N) while every neuron's fan-in stays bounded.
// 4 tables (seeds 0..3) -> L1a=agg(T0,T1), L1b=agg(T2,T3) -> L2=agg(L1a,L1b).
// Every emitter (table OR aggregator) emits Dout latency spikes at
//     t_{c,j} = GZ_c + ALPHA + BETA*value_c[j]      (bigger value = later)
// and every aggregator uses the model-3 sustained-current-until-readout-clock
// rule, so an aggregator's output is itself an emitter the next level consumes.
// Reuses lut_spiking.js exactly like animate3.js. Single rAF / clear-each-frame /
// size-once + DPR-cap + pause-on-hidden + responsive.
(function () {
  'use strict';

  function showErr(msg) {
    let bar = document.getElementById('errbar');
    if (!bar) { bar = document.createElement('div'); bar.id = 'errbar'; document.body.prepend(bar); }
    bar.style.display = 'block'; bar.textContent = '⚠ Tree-aggregator page error — ' + msg;
    if (window.console) console.error(msg);
  }
  window.addEventListener('error', (e) => showErr((e.error && e.error.stack) || e.message));
  window.addEventListener('unhandledrejection', (e) => showErr('promise: ' + ((e.reason && e.reason.message) || e.reason)));

  const LS = window.LUTSpiking;
  if (!LS) { showErr('lut_spiking.js did not load'); return; }

  const models = [0, 1, 2, 3].map((s) => LS.buildModel({ SEED: s }));
  const { D, K, Dout } = models[0].cfg;
  const ALPHA = 5.0, BETA = 1.0, SETTLE = 0.6, I_CUR = 1.0;
  const MIN_TRAVEL = 0.6, FLASH = 0.35;
  const AGGCOL = '#e6b3ff';
  const bandCol = ['#00b4d8', '#ffb84d', '#7ee787', '#f778ba'];
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

  let maxv = 1e-6;
  for (const m of models) for (const row of m.V) for (const v of row) maxv = Math.max(maxv, Math.abs(v));

  // ---- canvases sized ONCE (dpr-capped, width-guarded) -----------------------
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

  function tableState(m) {
    const S = LS.simulateCircuit(m, x);
    const TOPO = LS.circuitTopology(m).filter((e) => e.dst[0] !== 'o');
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

  let tabs, L1a, L1b, L2, Tmax = 1;
  let simT = 0, playing = false, speed = 1, lastTs = null, dirty = true;
  const RATE = 2;

  // one aggregator over a list of emitters (each {GZ, tout[]}); fan-in = children.length
  function aggregate(children) {
    const n = children.length;
    let T = -Infinity; for (const c of children) for (let j = 0; j < Dout; j++) T = Math.max(T, c.tout[j]);
    T += SETTLE;
    const C0 = children.reduce((a, c) => a + (c.GZ + ALPHA), 0);
    const val = [], V = [], tout = [];
    for (let j = 0; j < Dout; j++) {
      let Vj = 0; for (const c of children) Vj += I_CUR * (T - c.tout[j]);
      V.push(Vj);
      const Pj = (I_CUR * (n * T - C0) - Vj) / (I_CUR * BETA);   // recover partial sum
      val.push(Pj); tout.push(T + ALPHA + BETA * Pj);
    }
    return { GZ: T, T, val, V, tout, childTout: children.map((c) => c.tout) };
  }
  // aggregator membrane at time t for dim j (sustained current until its clock T)
  function aggFill(a, j, t) {
    const tt = Math.min(t, a.T); let V = 0;
    for (const ct of a.childTout) if (tt > ct[j]) V += I_CUR * (tt - ct[j]);
    return V;
  }

  function recompute() {
    tabs = models.map((m) => {
      const T = tableState(m);
      T.GZ = T.FIREVISUAL['r' + T.r] + SETTLE;
      T.tout = T.out.map((v) => T.GZ + ALPHA + BETA * v);
      return T;
    });
    L1a = aggregate([tabs[0], tabs[1]]);
    L1b = aggregate([tabs[2], tabs[3]]);
    L2 = aggregate([L1a, L1b]);
    Tmax = Math.max(...L2.tout) + 0.8;
    L2.NTICKS = Math.ceil(Tmax - 0.8 - 1e-9);
    const sc = document.getElementById('scrub'); sc.max = Tmax.toFixed(2); sc.step = (Tmax / 800).toFixed(4);
    for (let t = 0; t < 4; t++) fillLUT('' + t, models[t]);
    fillGtTable(); updateTreeInfo();
    dirty = true;
  }

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
    ctx.fillStyle = frac > 0.55 ? '#05080d' : css('--muted'); ctx.font = 'bold 7px ui-monospace'; ctx.textAlign = 'center'; ctx.fillText(id, p.x, p.y + 2.5); ctx.textAlign = 'left';
  }

  // an aggregator neuron: sustained-current fill toward its emit, flash on fire
  function drawAgg(ctx, p, a, j, label, Ra) {
    const frac = Math.max(0, Math.min(1, aggFill(a, j, simT) / Math.max(a.V[j], 1e-6)));
    const fired = simT >= a.tout[j], flashing = Math.abs(simT - a.tout[j]) < FLASH;
    if (flashing) { const al = 1 - Math.abs(simT - a.tout[j]) / FLASH; ctx.globalAlpha = 0.6 * al; ctx.fillStyle = AGGCOL; ctx.beginPath(); ctx.arc(p.x, p.y, Ra + 9 * al, 0, 7); ctx.fill(); ctx.globalAlpha = 1; }
    ctx.beginPath(); ctx.arc(p.x, p.y, Ra, 0, 7); ctx.fillStyle = '#20262f'; ctx.fill();
    if (frac > 0.002) { ctx.save(); ctx.beginPath(); ctx.arc(p.x, p.y, Ra, 0, 7); ctx.clip(); ctx.globalAlpha = 0.85; ctx.fillStyle = AGGCOL; const fh = 2 * Ra * frac; ctx.fillRect(p.x - Ra, p.y + Ra - fh, 2 * Ra, fh); ctx.globalAlpha = 1; ctx.restore(); }
    ctx.beginPath(); ctx.arc(p.x, p.y, Ra, 0, 7); ctx.strokeStyle = fired ? AGGCOL : css('--edge'); ctx.lineWidth = fired ? 2.4 : 1; ctx.stroke(); ctx.lineWidth = 1;
    ctx.fillStyle = AGGCOL; ctx.font = 'bold 8px ui-monospace'; ctx.textAlign = 'center'; ctx.fillText(label, p.x, p.y + 3);
    if (fired) { ctx.fillStyle = AGGCOL; ctx.beginPath(); ctx.arc(p.x + Ra + 6, p.y, 3.2, 0, 7); ctx.fill(); }
    ctx.textAlign = 'left';
  }

  function listenEdge(ctx, a, b, active) { ctx.strokeStyle = AGGCOL; ctx.globalAlpha = active ? 0.55 : 0.1; ctx.lineWidth = 1.3; ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); ctx.globalAlpha = 1; ctx.lineWidth = 1; }
  function travelDot(ctx, a, b, dep, arr, col, r) { if (simT < dep || simT > arr) return; const fr = (simT - dep) / Math.max(arr - dep, 1e-6); ctx.fillStyle = col; ctx.beginPath(); ctx.arc(a.x + (b.x - a.x) * fr, a.y + (b.y - a.y) * fr, r, 0, 7); ctx.fill(); }

  function drawUnifiedGraph() {
    const c = canv.graph; if (!c || !tabs || !L2) return;
    const { ctx, w, h } = c; ctx.clearRect(0, 0, w, h);
    const NC = 8, padL = 30, padR = 46, padT = 26, padB = 14, R = 7, CLOCKCOL = css('--ok');
    const colX = (ci) => padL + (ci / NC) * (w - padL - padR);
    const fullTop = padT + 16, fullBot = h - padB, midY = (fullTop + fullBot) / 2;
    const shp = {};
    shp['START'] = { x: colX(0), y: midY };
    for (let i = 0; i < D; i++) shp['x' + i] = { x: colX(1), y: fullTop + (i + 0.5) * (fullBot - fullTop) / D };
    shp['CLK'] = { x: colX(2), y: midY };
    const slotH = (fullBot - fullTop) / 4;
    const band = (bt, bb) => {
      const p = {};
      for (let k = 0; k < K; k++) p['H' + k] = { x: colX(3), y: bt + (k + 0.5) * (bb - bt) / K };
      for (let k = 0; k < K; k++) p['C' + k] = { x: colX(4), y: bt + (k + 0.5) * (bb - bt) / K };
      for (let a = 0; a < (1 << K); a++) p['r' + a] = { x: colX(5), y: bt + (a + 0.5) * (bb - bt) / (1 << K) };
      for (let j = 0; j < Dout; j++) p['oT' + j] = { x: colX(6), y: bt + (j + 0.5) * (bb - bt) / Dout };
      return p;
    };
    const bands = [];
    for (let s = 0; s < 4; s++) bands.push(band(fullTop + s * slotH + 4, fullTop + (s + 1) * slotH - 4));
    const cy = (s) => fullTop + (s + 0.5) * slotH;
    const laPos = [], lbPos = [], l2Pos = [];
    const laY = (cy(0) + cy(1)) / 2, lbY = (cy(2) + cy(3)) / 2;
    for (let j = 0; j < Dout; j++) {
      laPos.push({ x: colX(7), y: laY - (Dout - 1) * 11 + j * 22 });
      lbPos.push({ x: colX(7), y: lbY - (Dout - 1) * 11 + j * 22 });
      l2Pos.push({ x: colX(8), y: midY - (Dout - 1) * 13 + j * 26 });
    }

    // --- each table band ---
    for (let t = 0; t < 4; t++) {
      const T = tabs[t], pb = bands[t], col = bandCol[t];
      const posOf = (id) => (id[0] === 'x' || id === 'START' || id === 'CLK') ? shp[id] : pb[id];
      ctx.lineWidth = 1;
      for (const e of T.TOPO) {
        const a = posOf(e.src), b = posOf(e.dst); if (!a || !b) continue;
        const inhib = e.src[0] === 'H' && e.dst[0] === 'C';
        if (inhib) { ctx.strokeStyle = css('--lut'); ctx.globalAlpha = 0.7; ctx.setLineDash([3, 3]); ctx.lineWidth = 1.3; }
        else { const wc = weightColor(T.MAXW, e.weight); ctx.strokeStyle = wc.color; ctx.globalAlpha = wc.alpha; }
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
        if (inhib) { ctx.setLineDash([]); ctx.lineWidth = 1; }
      }
      ctx.globalAlpha = 1;
      const rp = pb['r' + T.r];
      for (let j = 0; j < Dout; j++) { const op = pb['oT' + j]; ctx.strokeStyle = col; ctx.globalAlpha = simT >= T.GZ ? 0.5 : 0.1; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(rp.x, rp.y); ctx.lineTo(op.x, op.y); ctx.stroke(); }
      ctx.globalAlpha = 1; ctx.lineWidth = 1;
      // listening edges: table t out j -> its L1 aggregator
      const agg = (t < 2) ? laPos : lbPos;
      for (let j = 0; j < Dout; j++) listenEdge(ctx, pb['oT' + j], agg[j], simT >= T.tout[j]);
      // within-table travelling dots
      for (const e of T.S.events) {
        if (e.dst[0] === 'o') continue;
        const a = posOf(e.src), b = posOf(e.dst); if (!a || !b) continue;
        const vspan = Math.max(e.tArrive - e.tDepart, MIN_TRAVEL);
        if (simT < e.tDepart || simT > e.tDepart + vspan) continue;
        const fr = (simT - e.tDepart) / vspan;
        ctx.fillStyle = e.src === 'CLK' ? CLOCKCOL : (e.weight < 0 ? '#5b9dff' : '#ff6b6b');
        ctx.beginPath(); ctx.arc(a.x + (b.x - a.x) * fr, a.y + (b.y - a.y) * fr, e.src === 'CLK' ? 3.4 : 3, 0, 7); ctx.fill();
      }
      // row -> oT (arriving at emit) and oT -> aggregator (listening)
      for (let j = 0; j < Dout; j++) {
        const op = pb['oT' + j];
        travelDot(ctx, rp, op, T.GZ, T.tout[j], col, 3);
        travelDot(ctx, op, agg[j], T.tout[j], T.tout[j] + MIN_TRAVEL, AGGCOL, 3.2);
      }
      // nodes
      for (const id in pb) { if (id[0] === 'o') continue; drawNode(ctx, pb[id], id, nodeColor(id), T.FIREVISUAL[id], T.S.thr[id], T.INCOMING, R); }
      for (let j = 0; j < Dout; j++) {
        const p = pb['oT' + j], fv = T.tout[j], fired = simT >= fv, flashing = Math.abs(simT - fv) < FLASH;
        if (flashing) { const a = 1 - Math.abs(simT - fv) / FLASH; ctx.globalAlpha = 0.5 * a; ctx.fillStyle = col; ctx.beginPath(); ctx.arc(p.x, p.y, R + 6 * a, 0, 7); ctx.fill(); ctx.globalAlpha = 1; }
        ctx.beginPath(); ctx.arc(p.x, p.y, R, 0, 7); ctx.fillStyle = fired ? col : '#20262f'; ctx.fill();
        ctx.beginPath(); ctx.arc(p.x, p.y, R, 0, 7); ctx.strokeStyle = fired ? col : css('--edge'); ctx.lineWidth = fired ? 2 : 1; ctx.stroke(); ctx.lineWidth = 1;
        ctx.fillStyle = fired ? '#05080d' : css('--muted'); ctx.font = 'bold 6px ui-monospace'; ctx.textAlign = 'center'; ctx.fillText('o' + j, p.x, p.y + 2); ctx.textAlign = 'left';
      }
      ctx.fillStyle = col; ctx.font = 'bold 9px ui-monospace'; ctx.textAlign = 'left';
      ctx.fillText('Table ' + t, colX(3) - 18, fullTop + t * slotH + 8);
    }

    // --- L1 -> L2 listening edges + dots ---
    for (let j = 0; j < Dout; j++) {
      listenEdge(ctx, laPos[j], l2Pos[j], simT >= L1a.tout[j]);
      listenEdge(ctx, lbPos[j], l2Pos[j], simT >= L1b.tout[j]);
      travelDot(ctx, laPos[j], l2Pos[j], L1a.tout[j], L1a.tout[j] + MIN_TRAVEL, AGGCOL, 3.2);
      travelDot(ctx, lbPos[j], l2Pos[j], L1b.tout[j], L1b.tout[j] + MIN_TRAVEL, AGGCOL, 3.2);
    }
    // --- aggregator nodes ---
    for (let j = 0; j < Dout; j++) {
      drawAgg(ctx, laPos[j], L1a, j, 'a' + j, 10);
      drawAgg(ctx, lbPos[j], L1b, j, 'b' + j, 10);
      drawAgg(ctx, l2Pos[j], L2, j, 'Σ' + j, 12);
    }

    // shared input column
    ['START'].concat(Array.from({ length: D }, (_, i) => 'x' + i)).concat(['CLK'])
      .forEach((id) => drawNode(ctx, shp[id], id, nodeColor(id), tabs[0].FIREVISUAL[id], tabs[0].S.thr[id], tabs[0].INCOMING, R));

    ctx.fillStyle = css('--accent'); ctx.font = 'bold 8px ui-monospace'; ctx.textAlign = 'center';
    ['START', 'INPUT', 'CLOCK', 'DETECT', 'COMPL', 'ROWS', 'TAB-OUT', 'L1-AGG', 'L2-AGG'].forEach((t, i) => ctx.fillText(t, colX(i), 12));
    ctx.textAlign = 'left';
  }

  // ---- TIMELINE: 4 tables -> L1 clocks/partials -> L2 clock -> final sums ------
  function drawAggTimeline() {
    const c = canv.atimeline; if (!c || !L2) return;
    const { ctx, w, h } = c; ctx.clearRect(0, 0, w, h);
    const N = Math.max(1, L2.NTICKS), t0 = 0, t1 = N;
    const padL = 108, padR = 26, padT = 28, padB = 22;
    const X = (t) => padL + ((t - t0) / (t1 - t0)) * (w - padL - padR);
    const rows = [];
    for (let t = 0; t < 4; t++) rows.push({ lab: 'table ' + t, col: bandCol[t], tout: tabs[t].tout });
    rows.push({ lab: 'L1a (0+1)', col: AGGCOL, tout: L1a.tout });
    rows.push({ lab: 'L1b (2+3)', col: AGGCOL, tout: L1b.tout });
    for (let j = 0; j < Dout; j++) rows.push({ lab: 'Σ o' + j, col: AGGCOL, single: L2.tout[j] });
    const laneTop = padT, laneBot = h - padB, laneH = (laneBot - laneTop) / rows.length;
    const laneY = (i) => laneTop + laneH * (i + 0.5);
    ctx.font = '10px ui-monospace';
    for (let t = 0; t <= N; t++) {
      ctx.strokeStyle = css('--edge'); ctx.globalAlpha = (t % 5 === 0) ? .45 : .13;
      ctx.beginPath(); ctx.moveTo(X(t), laneTop - 6); ctx.lineTo(X(t), laneBot); ctx.stroke(); ctx.globalAlpha = 1;
      if (t % 2 === 0 || N <= 24) { ctx.fillStyle = css('--muted'); ctx.textAlign = 'center'; ctx.fillText(t, X(t), laneBot + 14); ctx.textAlign = 'left'; }
    }
    ctx.fillStyle = css('--accent'); ctx.font = 'bold 12px ui-monospace';
    ctx.fillText('⏱ ' + N + ' ticks · depth 2 tree · every aggregator fan-in = 2', padL, 15);
    ctx.font = '10px ui-monospace';
    // clocks
    const clk = (T, lab) => { ctx.setLineDash([4, 3]); ctx.strokeStyle = css('--warn'); ctx.globalAlpha = .9; ctx.beginPath(); ctx.moveTo(X(T), laneTop - 6); ctx.lineTo(X(T), laneBot); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1; ctx.fillStyle = css('--warn'); ctx.textAlign = 'center'; ctx.fillText(lab, X(T), laneTop - 8); ctx.textAlign = 'left'; };
    clk(L1a.T, 'T1a'); clk(L1b.T, 'T1b'); clk(L2.T, 'T2');
    for (let i = 0; i < rows.length; i++) {
      const r = rows[i], y = laneY(i);
      ctx.fillStyle = r.col; ctx.fillText(r.lab, 6, y + 3);
      ctx.strokeStyle = css('--edge'); ctx.globalAlpha = .3; ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke(); ctx.globalAlpha = 1;
      const marks = r.single != null ? [r.single] : r.tout;
      for (const tt of marks) { if (simT < tt) continue; ctx.strokeStyle = r.col; ctx.lineWidth = r.single != null ? 3 : 2.4; ctx.beginPath(); ctx.moveTo(X(tt), y - laneH * .3); ctx.lineTo(X(tt), y + laneH * .3); ctx.stroke(); ctx.lineWidth = 1; }
      if (r.single != null && simT >= r.single) { ctx.fillStyle = r.col; ctx.fillText('t=' + r.single.toFixed(2), X(r.single) + 5, y - 2); }
    }
    if (simT >= t0) { ctx.strokeStyle = AGGCOL; ctx.lineWidth = 1.5; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(X(Math.min(simT, t1)), laneTop - 6); ctx.lineTo(X(Math.min(simT, t1)), laneBot); ctx.stroke(); ctx.setLineDash([]); ctx.lineWidth = 1; }
  }

  // ---- DOM panels ------------------------------------------------------------
  function fillLUT(sfx, m) {
    const pre = LS.lutPreacts(m, x), bits = LS.lutBits(m, x), r = LS.bitsToRow(bits);
    const tb = document.querySelector('#lut' + sfx + '_tests tbody'); if (!tb) return; tb.innerHTML = '';
    for (let k = 0; k < K; k++) {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>H${k}</td><td class="mono" style="color:${pre[k] > 0 ? '#7ee787' : '#ffa198'}">${fmt(pre[k])}</td><td>${bitBox(bits[k])}</td>`;
      tb.appendChild(tr);
    }
    document.getElementById('lut' + sfx + '_addr').textContent = bits.join('');
    document.getElementById('lut' + sfx + '_row').textContent = r;
    fillVec('lut' + sfx + '_out', m.V[r]);
  }
  function updateTreeInfo() {
    const el = document.getElementById('treeinfo'); if (!el) return;
    el.innerHTML = '⏱ This resolves in <b>' + L2.NTICKS + ' discrete ticks</b> over a <b>depth-2</b> tree — 4 tables emit, '
      + 'two L1 aggregators sum pairs (clocks T1a=' + L1a.T.toFixed(1) + ', T1b=' + L1b.T.toFixed(1) + '), then L2 sums the partials '
      + '(clock T2=' + L2.T.toFixed(1) + ') and emits the total. Every aggregator has <b>fan-in exactly 2</b> — independent of table count; '
      + 'depth grows like log₂(N), fan-in stays O(1).';
  }
  function fillGtTable() {
    const tb = document.querySelector('#gtmatch tbody'); if (!tb) return; tb.innerHTML = '';
    const O = models.map((m) => m.V[LS.bitsToRow(LS.lutBits(m, x))]);
    let maxres = 0;
    for (let j = 0; j < Dout; j++) {
      const strue = O[0][j] + O[1][j] + O[2][j] + O[3][j];
      const smem = L2.val[j];                                       // L2 membrane readout
      const semit = (L2.tout[j] - L2.GZ - ALPHA) / BETA;            // decode from final emitted spike
      const res = Math.max(Math.abs(strue - smem), Math.abs(strue - semit)); maxres = Math.max(maxres, res);
      const ok = res < 1e-9;
      const tr = document.createElement('tr');
      tr.innerHTML = '<td>o' + j + '</td>'
        + O.map((Ov) => '<td class="mono">' + fmt(Ov[j]) + '</td>').join('')
        + '<td class="mono"><b>' + fmt(strue) + '</b></td><td class="mono">' + fmt(smem)
        + '</td><td class="mono">' + fmt(semit) + '</td><td class="' + (ok ? 'yes' : 'no') + '">' + (ok ? '✓' : fmt(res)) + '</td>';
      tb.appendChild(tr);
    }
    const note = document.getElementById('gtnote');
    if (note) note.innerHTML = 'Four independent tables, summed through a two-level tree; the L2 membrane readout and the final emitted-spike decode both equal the direct 4-way LUT sum to <b>'
      + maxres.toExponential(1) + '</b> for this input.';
  }
  (function fillScaleTable() {
    const tb = document.querySelector('#scaletable tbody'); if (!tb) return;
    const gi = (n) => n.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    for (const N of [4, 16, 64, 256, 1024]) {
      const depth = Math.ceil(Math.log2(N)), aggs = N - 1;
      const tr = document.createElement('tr'); if (N === 4) tr.className = 'winrow';
      tr.innerHTML = '<td>' + N + (N === 4 ? ' (this demo)' : '') + '</td><td class="mono">' + depth + '</td><td class="mono">2</td><td class="mono">' + gi(aggs) + '</td><td class="mono">' + gi(aggs * Dout) + '</td>';
      tb.appendChild(tr);
    }
    const note = document.getElementById('scalenote');
    if (note) note.innerHTML = 'Depth grows like <b>log₂(N)</b> while every aggregator keeps <b>fan-in 2</b> (bounded). Flat model 3 is depth 1 but one neuron with fan-in N — unbounded membrane load. The tree is the scale-to-hundreds answer.';
  })();

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
