// animate2.js — composition of TWO LUT tables. Each table gets a real animated
// node-graph (detectors -> complements -> 1-of-8 rows) like the single-table page;
// both tables' selected rows deposit their stored output vectors as CHARGE onto a
// SHARED row of output neurons, whose membranes accumulate O1[r1] + O2[r2].
// Mirrors animate.js's drawGraph machinery, generalised per table. Single rAF /
// clear-each-frame / size-once. Reuses lut_spiking.js (buildModel/simulateCircuit/
// circuitTopology). Additive page — nothing else touched.
(function () {
  'use strict';

  function showErr(msg) {
    let bar = document.getElementById('errbar');
    if (!bar) { bar = document.createElement('div'); bar.id = 'errbar'; document.body.prepend(bar); }
    bar.style.display = 'block'; bar.textContent = '⚠ Composition page error — ' + msg;
    if (window.console) console.error(msg);
  }
  window.addEventListener('error', (e) => showErr((e.error && e.error.stack) || e.message));
  window.addEventListener('unhandledrejection', (e) => showErr('promise: ' + ((e.reason && e.reason.message) || e.reason)));

  const LS = window.LUTSpiking;
  if (!LS) { showErr('lut_spiking.js did not load'); return; }

  const m1 = LS.buildModel({ SEED: 0 });
  const m2 = LS.buildModel({ SEED: 1 });
  const { D, K, Dout } = m1.cfg;
  const BASE = 4.0, KAPPA = 0.35;
  // OUTPUT spike readout: after the summed charge settles, each shared output neuron
  // FIRES ONCE at t_out = GZ_out + ALPHA_O + BETA_O*S (same form as the input encoding
  // t_i = alpha + beta*x, bigger value = LATER). ALPHA_O(=5) > max|S|(~4.5) so every
  // output spike lands after the readout ground-zero GZ_out. Decode: S = (ℓ - ALPHA_O)/BETA_O.
  const ALPHA_O = 5.0, BETA_O = 1.0, SETTLE = 0.6;
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
  const MIN_TRAVEL = 0.6, FLASH = 0.35;
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
    if (id[0] === 'C') return css('--warn'); return css('--row');
  }
  // per-table graph columns (no OUTPUT column — outputs are shared)
  const COLS = [['START'], Array.from({ length: D }, (_, i) => 'x' + i), ['CLK'],
    Array.from({ length: K }, (_, k) => 'H' + k), Array.from({ length: K }, (_, k) => 'C' + k),
    Array.from({ length: 1 << K }, (_, a) => 'r' + a)];
  const CAPS = ['START', 'INPUT', 'CLOCK', 'H', 'C', 'ROWS'];

  let maxv = 1e-6;
  for (const O of [m1.V, m2.V]) for (const row of O) for (const v of row) maxv = Math.max(maxv, Math.abs(v));

  // ---- canvases sized ONCE --------------------------------------------------
  const canv = {};
  ['graph', 'oraster'].forEach((id) => { const cv = document.getElementById(id); if (cv) canv[id] = { cv, ctx: cv.getContext('2d'), w: 0, h: 0 }; });
  let needResize = true;
  function resizeAll() {
    const dpr = window.devicePixelRatio || 1;
    for (const id in canv) {
      const c = canv[id];
      const w = c.cv.clientWidth || (c.cv.parentElement && c.cv.parentElement.clientWidth) || 820;
      const h = +c.cv.getAttribute('height');
      c.cv.width = Math.max(1, Math.round(w * dpr)); c.cv.height = Math.round(h * dpr);
      c.ctx.setTransform(dpr, 0, 0, dpr, 0, 0); c.w = w; c.h = h;
    }
    needResize = false;
  }

  // ---- per-table state (recomputed per input) -------------------------------
  function tableState(m) {
    const S = LS.simulateCircuit(m, x);
    const TOPO = LS.circuitTopology(m).filter((e) => e.dst[0] !== 'o');   // drop rows->output (shared)
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

  let T1, T2, Tmax = 1, shared;
  let simT = 0, playing = false, speed = 1, lastTs = null, dirty = true;
  const RATE = 2;

  function recompute() {
    T1 = tableState(m1); T2 = tableState(m2);
    shared = { o1: T1.out, o2: T2.out, fv1: T1.FIREVISUAL['r' + T1.r], fv2: T2.FIREVISUAL['r' + T2.r] };
    // output ground-zero = after both tables' rows have deposited their charge, then each
    // shared output neuron fires once at GZ + ALPHA_O + BETA_O*S.
    shared.GZ = Math.max(shared.fv1 || 0, shared.fv2 || 0) + SETTLE;
    shared.S = shared.o1.map((v, j) => v + shared.o2[j]);
    shared.outT = shared.S.map((s) => shared.GZ + ALPHA_O + BETA_O * s);
    Tmax = 0;
    for (const T of [T1, T2]) for (const n in T.S.st) if (T.S.st[n].fired) Tmax = Math.max(Tmax, T.S.st[n].tf);
    Tmax = Math.max(Tmax, ...shared.outT) + 0.6;
    const sc = document.getElementById('scrub'); sc.max = Tmax.toFixed(2); sc.step = (Tmax / 600).toFixed(4);
    fillLUT('1', m1); fillLUT('2', m2); fillSumRows();
    dirty = true;
  }

  // ---- a single node: base disc + membrane fill + flash + border + label ----
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

  // ---- ONE unified graph: shared inputs -> two bands -> shared summing outputs
  function drawUnifiedGraph() {
    const c = canv.graph; if (!c || !T1 || !T2 || !shared) return;
    const { ctx, w, h } = c; ctx.clearRect(0, 0, w, h);
    const padL = 34, padR = 40, padT = 26, padB = 14, R = 8, CLOCKCOL = css('--ok');
    const colX = (ci) => padL + (ci / 6) * (w - padL - padR);
    const fullTop = padT + 16, fullBot = h - padB, midY = (fullTop + fullBot) / 2;
    const shp = {};
    shp['START'] = { x: colX(0), y: midY };
    for (let i = 0; i < D; i++) shp['x' + i] = { x: colX(1), y: fullTop + (i + 0.5) * (fullBot - fullTop) / D };
    shp['CLK'] = { x: colX(2), y: midY };
    for (let j = 0; j < Dout; j++) shp['o' + j] = { x: colX(6), y: midY - (Dout - 1) * 12 + j * 24 };
    const band = (bt, bb) => {
      const p = {};
      for (let k = 0; k < K; k++) p['H' + k] = { x: colX(3), y: bt + (k + 0.5) * (bb - bt) / K };
      for (let k = 0; k < K; k++) p['C' + k] = { x: colX(4), y: bt + (k + 0.5) * (bb - bt) / K };
      for (let a = 0; a < (1 << K); a++) p['r' + a] = { x: colX(5), y: bt + (a + 0.5) * (bb - bt) / (1 << K) };
      return p;
    };
    const pA = band(fullTop, midY - 10), pB = band(midY + 10, fullBot);
    const sharedId = (id) => id[0] === 'x' || id === 'START' || id === 'CLK' || id[0] === 'o';
    const posOf = (tbl, id) => sharedId(id) ? shp[id] : (tbl === 1 ? pA : pB)[id];
    const tables = [[1, T1, pA], [2, T2, pB]];

    tables.forEach(([tbl, T]) => {
      ctx.lineWidth = 1;
      for (const e of T.TOPO) {
        const a = posOf(tbl, e.src), b = posOf(tbl, e.dst); if (!a || !b) continue;
        const wc = weightColor(T.MAXW, e.weight); ctx.strokeStyle = wc.color; ctx.globalAlpha = wc.alpha;
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }
      ctx.globalAlpha = 1;
      for (const e of T.S.events) {
        if (e.dst[0] === 'o') continue;
        const a = posOf(tbl, e.src), b = posOf(tbl, e.dst); if (!a || !b) continue;
        const vspan = Math.max(e.tArrive - e.tDepart, MIN_TRAVEL);
        if (simT < e.tDepart || simT > e.tDepart + vspan) continue;
        const fr = (simT - e.tDepart) / vspan, wc = weightColor(T.MAXW, e.weight);
        ctx.strokeStyle = wc.color; ctx.globalAlpha = Math.min(0.9, wc.alpha + 0.4); ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); ctx.lineWidth = 1; ctx.globalAlpha = 1;
        const dx = a.x + (b.x - a.x) * fr, dy = a.y + (b.y - a.y) * fr;
        ctx.fillStyle = e.src === 'CLK' ? CLOCKCOL : (e.weight < 0 ? '#5b9dff' : '#ff6b6b');
        ctx.beginPath(); ctx.arc(dx, dy, e.src === 'CLK' ? 4 : 3.2, 0, 7); ctx.fill();
      }
      const KICK = 0.32;
      for (const e of T.S.events) {
        if (sharedId(e.dst)) continue;
        const tv = visualArrival(e), p = posOf(tbl, e.dst); if (!p) continue;
        if (simT < tv || simT > tv + KICK) continue;
        const a = 1 - (simT - tv) / KICK, isClk = e.src === 'CLK';
        ctx.strokeStyle = isClk ? CLOCKCOL : (e.weight < 0 ? '#5b9dff' : css('--spike'));
        ctx.globalAlpha = 0.7 * a; ctx.lineWidth = isClk ? 2.2 : 1.4;
        ctx.beginPath(); ctx.arc(p.x, p.y, R + (isClk ? 10 : 6) * (1 - a), 0, 7); ctx.stroke(); ctx.lineWidth = 1; ctx.globalAlpha = 1;
        if (isClk && a > 0.4) { ctx.fillStyle = CLOCKCOL; ctx.font = 'bold 7px ui-monospace'; ctx.textAlign = 'center'; ctx.fillText('CLK', p.x, p.y - R - 5); ctx.textAlign = 'left'; }
      }
      const pb = (tbl === 1 ? pA : pB);
      for (const id in pb) drawNode(ctx, pb[id], id, nodeColor(id), T.FIREVISUAL[id], T.S.thr[id], T.INCOMING, R);
    });

    // winning rows -> shared outputs (charge-deposit edges)
    tables.forEach(([tbl, T]) => {
      const rp = (tbl === 1 ? pA : pB)['r' + T.r]; if (!rp) return;
      const deposited = simT >= (tbl === 1 ? shared.fv1 : shared.fv2);
      for (let j = 0; j < Dout; j++) {
        const op = shp['o' + j], val = T.out[j];
        ctx.strokeStyle = val < 0 ? '#5b9dff' : '#ff6b6b';
        ctx.globalAlpha = deposited ? Math.min(0.85, 0.25 + Math.abs(val) / (2 * maxv)) : 0.07;
        ctx.lineWidth = deposited ? 1.6 : 1;
        ctx.beginPath(); ctx.moveTo(rp.x, rp.y); ctx.lineTo(op.x, op.y); ctx.stroke();
      }
    });
    ctx.globalAlpha = 1; ctx.lineWidth = 1;

    // shared inputs (fill from T1) then the summing output neurons
    ['START'].concat(Array.from({ length: D }, (_, i) => 'x' + i)).concat(['CLK'])
      .forEach((id) => drawNode(ctx, shp[id], id, nodeColor(id), T1.FIREVISUAL[id], T1.S.thr[id], T1.INCOMING, R));
    const scale = 2 * maxv, Ro = 12;
    for (let j = 0; j < Dout; j++) {
      const p = shp['o' + j];
      const c1 = simT >= (shared.fv1 || Infinity) ? shared.o1[j] : 0;
      const c2 = simT >= (shared.fv2 || Infinity) ? shared.o2[j] : 0;
      const mem = c1 + c2, fired = simT >= shared.outT[j], flashing = Math.abs(simT - shared.outT[j]) < FLASH;
      const frac = Math.max(0, Math.min(1, Math.abs(mem) / scale)), col = mem >= 0 ? '#ff6b6b' : '#5b9dff';
      if (flashing) { const a = 1 - Math.abs(simT - shared.outT[j]) / FLASH; ctx.globalAlpha = 0.6 * a; ctx.fillStyle = css('--out'); ctx.beginPath(); ctx.arc(p.x, p.y, Ro + 9 * a, 0, 7); ctx.fill(); ctx.globalAlpha = 1; }
      ctx.beginPath(); ctx.arc(p.x, p.y, Ro, 0, 7); ctx.fillStyle = '#20262f'; ctx.fill();
      if (frac > 0.002) { ctx.save(); ctx.beginPath(); ctx.arc(p.x, p.y, Ro, 0, 7); ctx.clip(); ctx.globalAlpha = 0.85; ctx.fillStyle = col; const fh = 2 * Ro * frac; ctx.fillRect(p.x - Ro, p.y + Ro - fh, 2 * Ro, fh); ctx.globalAlpha = 1; ctx.restore(); }
      ctx.beginPath(); ctx.arc(p.x, p.y, Ro, 0, 7); ctx.strokeStyle = fired ? css('--out') : css('--edge'); ctx.lineWidth = fired ? 2.5 : 1; ctx.stroke(); ctx.lineWidth = 1;
      ctx.fillStyle = css('--out'); ctx.font = 'bold 9px ui-monospace'; ctx.textAlign = 'center'; ctx.fillText('o' + j, p.x, p.y + 3);
      if (fired) { ctx.fillStyle = css('--out'); ctx.beginPath(); ctx.arc(p.x + Ro + 7, p.y, 3.5, 0, 7); ctx.fill(); }
      ctx.textAlign = 'left';
    }

    ctx.fillStyle = css('--accent'); ctx.font = 'bold 9px ui-monospace'; ctx.textAlign = 'center';
    ['START', 'INPUT', 'CLOCK', 'DETECT', 'COMPL', 'ROWS', 'OUT'].forEach((t, i) => ctx.fillText(t, colX(i), 12));
    ctx.fillStyle = css('--spike'); ctx.font = 'bold 10px ui-monospace'; ctx.textAlign = 'left';
    ctx.fillText('Table 1', colX(3) - 16, fullTop - 3);
    ctx.fillStyle = css('--out'); ctx.fillText('Table 2', colX(3) - 16, midY + 13);
  }

  // ---- OUTPUT RASTER: the 4 emitted output spikes on a time axis --------------
  function drawOutputRaster() {
    const c = canv.oraster; if (!c || !shared) return;
    const { ctx, w, h } = c; ctx.clearRect(0, 0, w, h);
    const t0 = Math.max(0, shared.GZ - 1.0), t1 = Tmax;
    const padL = 62, padR = 18, padT = 14, padB = 20;
    const X = (t) => padL + ((t - t0) / (t1 - t0)) * (w - padL - padR);
    const top = padT, rowH = (h - padT - padB) / Dout;
    ctx.font = '11px ui-monospace';
    for (let t = Math.ceil(t0); t <= t1; t++) { ctx.strokeStyle = css('--edge'); ctx.globalAlpha = .3; ctx.beginPath(); ctx.moveTo(X(t), top); ctx.lineTo(X(t), h - padB); ctx.stroke(); ctx.globalAlpha = 1; ctx.fillStyle = css('--muted'); ctx.fillText('t=' + t, X(t) - 8, h - padB + 14); }
    ctx.setLineDash([4, 3]); ctx.strokeStyle = css('--ok'); ctx.beginPath(); ctx.moveTo(X(shared.GZ), top); ctx.lineTo(X(shared.GZ), h - padB); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = css('--ok'); ctx.fillText('GZ (readout clock)', X(shared.GZ) - 4, top - 1);
    for (let j = 0; j < Dout; j++) {
      const y = top + rowH * (j + 0.5);
      ctx.fillStyle = css('--muted'); ctx.fillText('out o' + j, 6, y + 4);
      ctx.strokeStyle = css('--edge'); ctx.globalAlpha = .3; ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke(); ctx.globalAlpha = 1;
      if (simT >= shared.outT[j]) {
        ctx.strokeStyle = css('--out'); ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(X(shared.outT[j]), y - rowH * .34); ctx.lineTo(X(shared.outT[j]), y + rowH * .34); ctx.stroke(); ctx.lineWidth = 1;
        ctx.fillStyle = css('--out'); ctx.fillText(shared.outT[j].toFixed(2), X(shared.outT[j]) + 5, y - 3);
      }
    }
    if (simT >= t0) { ctx.strokeStyle = css('--out'); ctx.lineWidth = 1.5; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(X(Math.min(simT, t1)), top); ctx.lineTo(X(Math.min(simT, t1)), h - padB); ctx.stroke(); ctx.setLineDash([]); ctx.lineWidth = 1; }
  }

  // ---- LUT panel + sum rows (DOM) -------------------------------------------
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
  const SEG1 = [], SEG2 = [], ENDP = [], TXT = [];
  (function buildRows() {
    const sr = document.getElementById('sumrows');
    for (let j = 0; j < Dout; j++) {
      const row = document.createElement('div'); row.className = 'sumrow';
      row.innerHTML = `<span class="olab mono">o${j}</span><span class="compbar"><i class="cline"></i><i class="cseg s1"></i><i class="cseg s2"></i><i class="cend"></i></span><span class="mono small comptxt"></span>`;
      sr.appendChild(row);
      SEG1[j] = row.querySelector('.s1'); SEG2[j] = row.querySelector('.s2'); ENDP[j] = row.querySelector('.cend'); TXT[j] = row.querySelector('.comptxt');
    }
  })();
  const SCALE = 40 / (2 * maxv);
  function fillSumRows() {
    const o1 = T1.out, o2 = T2.out;
    for (let j = 0; j < Dout; j++) {
      const v1 = o1[j], v2 = o2[j], S = v1 + v2, p1 = 50 + v1 * SCALE, p2 = 50 + S * SCALE;
      SEG1[j].style.left = Math.min(50, p1) + '%'; SEG1[j].style.width = Math.abs(p1 - 50) + '%';
      SEG2[j].style.left = Math.min(p1, p2) + '%'; SEG2[j].style.width = Math.abs(p2 - p1) + '%';
      ENDP[j].style.left = p2 + '%';
      TXT[j].innerHTML = `<span style="color:${css('--spike')}">${fmt(v1)}</span> + <span style="color:${css('--out')}">${fmt(v2)}</span> = <b>${fmt(S)}</b> → spike @ ${shared.outT[j].toFixed(2)}`;
    }
    document.getElementById('cnote').textContent = 'Summed membrane S = [' + o1.map((v, j) => fmt(v + o2[j])).join(', ') + '] = O₁[r₁] + O₂[r₂] (charge adds; latencies do not).';
  }

  // ---- clock + controls -----------------------------------------------------
  function draw() {
    if (needResize) resizeAll();
    drawUnifiedGraph(); drawOutputRaster();
    document.getElementById('simtime').textContent = 't = ' + simT.toFixed(2) + ' / ' + Tmax.toFixed(2);
  }
  function loop(ts) {
    try {
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

  recompute();
  requestAnimationFrame(loop);
})();
