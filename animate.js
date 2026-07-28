// animate.js — real-time replay of the pure spiking circuit. ONE rAF loop,
// single source of truth for sim-time, trace computed ONCE per input, canvases
// sized once (not per frame), clear-and-draw each frame. Zero-dependency.
(function () {
  'use strict';

  function showErr(msg) {
    let bar = document.getElementById('errbar');
    if (!bar) { bar = document.createElement('div'); bar.id = 'errbar'; document.body.prepend(bar); }
    bar.style.display = 'block'; bar.textContent = '⚠ Animation error — ' + msg;
    if (window.console) console.error(msg);
  }
  window.addEventListener('error', (e) => showErr((e.error && e.error.stack) || e.message));
  window.addEventListener('unhandledrejection', (e) => showErr('promise: ' + ((e.reason && e.reason.message) || e.reason)));

  const LS = window.LUTSpiking;
  if (!LS) { showErr('lut_spiking.js did not load'); return; }

  const m = LS.buildModel();
  const { D, K, Dout } = m.cfg;
  const TOPO = LS.circuitTopology(m);   // full fixed wiring — stable skeleton, computed once
  let x = new Array(D).fill(0);

  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const fmt = (v) => (v >= 0 ? ' ' : '') + v.toFixed(3);
  const bitBox = (b) => `<span class="bit ${b ? 'on' : 'off'}">${b}</span>`;
  const fillBits = (id, bits) => { const e = document.getElementById(id); if (e) e.innerHTML = bits.map(bitBox).join(''); };
  function fillVec(id, vec) {
    const e = document.getElementById(id); if (!e) return;
    const mx = Math.max(1e-9, ...vec.map(Math.abs));
    e.innerHTML = vec.map((v) => {
      const wp = (Math.abs(v) / mx) * 50;
      return `<div class="small mono">${fmt(v)}</div><div class="vecbar"><i style="left:${v < 0 ? 50 - wp : 50}%;width:${wp}%;background:${v < 0 ? css('--bad') : css('--spike')}"></i></div>`;
    }).join('');
  }

  // layer columns + raster row groups (input rows stay x0..x5 in order)
  const COLS = [['START'], Array.from({ length: D }, (_, i) => 'x' + i), ['CLK'],
    Array.from({ length: K }, (_, k) => 'H' + k), Array.from({ length: K }, (_, k) => 'C' + k),
    Array.from({ length: 1 << K }, (_, a) => 'r' + a), Array.from({ length: Dout }, (_, j) => 'o' + j)];
  const RGROUPS = [['START', ['START']], ['INPUT', COLS[1]], ['CLOCK', ['CLK']],
    ['DETECTORS H', COLS[3]], ['COMPLEMENTS C', COLS[4]], ['ROW-SELECT r', COLS[5]], ['OUTPUT o', COLS[6]]];
  // interpolate a piecewise-linear membrane trace [{t,V}] at time t
  function Vat(tr, t) {
    if (!tr || !tr.length) return 0;
    if (t <= tr[0].t) return tr[0].V;
    for (let i = 1; i < tr.length; i++) {
      if (t <= tr[i].t) {
        const a = tr[i - 1], b = tr[i];
        return b.t === a.t ? b.V : a.V + (b.V - a.V) * ((t - a.t) / (b.t - a.t));
      }
    }
    return tr[tr.length - 1].V;   // hold last value (neuron decided/frozen)
  }
  function nodeColor(id) {
    if (id === 'START') return css('--muted'); if (id[0] === 'x') return css('--spike');
    if (id === 'CLK') return css('--ok'); if (id[0] === 'H') return css('--addr');
    if (id[0] === 'C') return css('--warn'); if (id[0] === 'r') return css('--row'); return css('--out');
  }

  // ---- canvases sized ONCE (and only on real resize) ------------------------
  const canv = {};
  ['animgraph', 'animraster'].forEach((id) => { const cv = document.getElementById(id); if (cv) canv[id] = { cv, ctx: cv.getContext('2d'), w: 0, h: 0 }; });
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

  // ---- trace state (computed ONCE per input) --------------------------------
  let S = null, Tmax = 1;
  function recompute() {
    S = LS.simulateCircuit(m, x);
    // time scale computed EXACTLY as circuit.js's raster maxT (max fired spike + 0.6),
    // so input-spike x-positions are byte-for-byte identical across the two pages.
    // (All propagation dots also finish within this window: the latest tArrive is an
    // output spike, which is itself a fired-neuron time.)
    Tmax = 0;
    for (const n in S.st) if (S.st[n].fired) Tmax = Math.max(Tmax, S.st[n].tf);
    Tmax += 0.6;
    const sc = document.getElementById('scrub'); sc.max = Tmax.toFixed(2); sc.step = (Tmax / 600).toFixed(4);
    updateLutReadout();       // LUT side is constant for this input — set once here
    lastDone = null;          // force sim-side readout refresh
  }

  // ---- clock (single source of truth) ---------------------------------------
  let simT = 0, playing = false, speed = 1, lastTs = null, dirty = true, lastDone = null;
  const RATE = 2, FLASH = 0.35;

  // ---- graph ----------------------------------------------------------------
  function drawGraph() {
    const c = canv.animgraph; if (!c) return;
    const { ctx, w, h } = c; ctx.clearRect(0, 0, w, h);
    const padL = 30, padR = 34, padT = 26, padB = 14, R = 10;
    const pos = {};
    COLS.forEach((ids, ci) => {
      const cx = padL + (ci / (COLS.length - 1)) * (w - padL - padR);
      const u = h - padT - padB;
      ids.forEach((id, ri) => { pos[id] = { x: cx, y: padT + (ri + 0.5) * u / ids.length }; });
    });
    // (1) STATIC SKELETON — the full fixed wiring, faint, NEVER changes with input.
    // Guarantees every non-source node shows incoming AND every non-output node
    // shows outgoing edges, regardless of which synapses carried a spike this round.
    ctx.strokeStyle = css('--edge'); ctx.lineWidth = 1; ctx.globalAlpha = 0.14;
    for (const e of TOPO) {
      const a = pos[e.src], b = pos[e.dst]; if (!a || !b) continue;
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    }
    // (2) ACTIVE edges this input (carried a spike) — brighter overlay on the skeleton.
    ctx.strokeStyle = css('--muted'); ctx.globalAlpha = 0.5;
    const seen = {};
    for (const e of S.events) {
      const key = e.src + '>' + e.dst; if (seen[key]) continue; seen[key] = 1;
      const a = pos[e.src], b = pos[e.dst]; if (!a || !b) continue;
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    }
    ctx.globalAlpha = 1;
    // EVERY carried spike animates a dot. A near-zero real delay (input->detector
    // ramps have delay 0) would teleport, so give every hop a MIN on-screen travel
    // time; genuinely longer delays (START->CLOCK, row->output) travel longer.
    const MIN_TRAVEL = 0.6;
    for (const e of S.events) {
      const a = pos[e.src], b = pos[e.dst]; if (!a || !b) continue;
      const vspan = Math.max(e.tArrive - e.tDepart, MIN_TRAVEL);
      if (simT < e.tDepart || simT > e.tDepart + vspan) continue;
      const fr = (simT - e.tDepart) / vspan;                 // 0 at src, 1 at dst (left -> right)
      const dx = a.x + (b.x - a.x) * fr, dy = a.y + (b.y - a.y) * fr;
      ctx.fillStyle = e.weight < 0 ? css('--bad') : nodeColor(e.src);
      ctx.beginPath(); ctx.arc(dx, dy, 3.6, 0, 7); ctx.fill();
    }
    for (const id in pos) {
      const p = pos[id], s = S.st[id];
      const fired = s && s.fired && s.tf <= simT;
      const flashing = s && s.fired && Math.abs(simT - s.tf) < FLASH;
      const col = nodeColor(id);
      // membrane fill fraction from the REAL simulator trace: V(simT)/threshold.
      // Charges up to ~1 at the fire time, then drains over a short window after the spike.
      let frac = 0;
      const tr = S.traces[id], thr = S.thr[id], DRAIN = 0.5;
      if (tr && thr) frac = fired ? Math.max(0, 1 - (simT - s.tf) / DRAIN)
                                  : Math.max(0, Math.min(1, Vat(tr, simT) / thr));
      if (flashing) {
        const a = 1 - Math.abs(simT - s.tf) / FLASH;
        ctx.globalAlpha = 0.5 * a; ctx.fillStyle = col;
        ctx.beginPath(); ctx.arc(p.x, p.y, R + 7 * a, 0, 7); ctx.fill(); ctx.globalAlpha = 1;
      }
      // base disc (empty)
      ctx.beginPath(); ctx.arc(p.x, p.y, R, 0, 7); ctx.fillStyle = '#20262f'; ctx.fill();
      // membrane fill from the bottom — charging, or draining after a spike
      if (frac > 0.002) {
        ctx.save(); ctx.beginPath(); ctx.arc(p.x, p.y, R, 0, 7); ctx.clip();
        ctx.globalAlpha = 0.85; ctx.fillStyle = col;
        const fh = 2 * R * frac; ctx.fillRect(p.x - R, p.y + R - fh, 2 * R, fh);
        ctx.globalAlpha = 1; ctx.restore();
      }
      // border: coloured once the neuron has fired (persistent marker), dim otherwise
      ctx.beginPath(); ctx.arc(p.x, p.y, R, 0, 7);
      ctx.strokeStyle = fired ? col : css('--edge'); ctx.lineWidth = fired ? 2 : 1; ctx.stroke(); ctx.lineWidth = 1;
      ctx.fillStyle = frac > 0.55 ? '#05080d' : css('--muted'); ctx.font = 'bold 9px ui-monospace'; ctx.textAlign = 'center';
      ctx.fillText(id, p.x, p.y + 3); ctx.textAlign = 'left';
    }
    ctx.fillStyle = css('--accent'); ctx.font = 'bold 10px ui-monospace'; ctx.textAlign = 'center';
    ['START', 'INPUT', 'CLOCK', 'H', 'C', 'ROWS', 'OUT'].forEach((t, i) => {
      const cx = padL + (i / (COLS.length - 1)) * (w - padL - padR); ctx.fillText(t, cx, 12);
    });
    ctx.textAlign = 'left';
  }

  // ---- raster ---------------------------------------------------------------
  function drawRaster() {
    const c = canv.animraster; if (!c) return;
    const { ctx, w, h } = c; ctx.clearRect(0, 0, w, h);
    const rows = [];
    RGROUPS.forEach(([g, ids]) => { rows.push({ sep: g }); ids.forEach((id) => rows.push({ id })); });
    const padL = 150, padR = 20, padT = 14, padB = 24;   // match circuit.js raster exactly
    const X = (t) => padL + (t / Tmax) * (w - padL - padR);
    const top = padT, rowH = (h - padT - padB) / rows.length;
    ctx.font = '11px ui-monospace';
    for (let t = 0; t <= Tmax; t++) {
      ctx.strokeStyle = css('--edge'); ctx.globalAlpha = .3; ctx.beginPath(); ctx.moveTo(X(t), top); ctx.lineTo(X(t), h - padB); ctx.stroke(); ctx.globalAlpha = 1;
      ctx.fillStyle = css('--muted'); ctx.fillText('t=' + t, X(t) - 8, h - padB + 14);
    }
    rows.forEach((r, idx) => {
      const y = top + rowH * (idx + 0.5);
      if (r.sep) { ctx.fillStyle = css('--accent'); ctx.font = 'bold 9px ui-monospace'; ctx.fillText(r.sep, 6, y + 4); ctx.font = '11px ui-monospace'; return; }
      const s = S.st[r.id];
      ctx.fillStyle = css('--muted'); ctx.fillText(r.id, 12, y + 4);
      if (s && s.fired && s.tf <= simT) {
        ctx.strokeStyle = nodeColor(r.id); ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(X(s.tf), y - rowH * .34); ctx.lineTo(X(s.tf), y + rowH * .34); ctx.stroke(); ctx.lineWidth = 1;
      }
    });
    ctx.strokeStyle = css('--out'); ctx.lineWidth = 1.5; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(X(simT), top); ctx.lineTo(X(simT), h - padB); ctx.stroke(); ctx.setLineDash([]); ctx.lineWidth = 1;
  }

  // ---- readout (DOM writes kept OUT of the hot path) ------------------------
  function updateLutReadout() {
    const lutRow = LS.bitsToRow(LS.lutBits(m, x));
    fillBits('a_lutbits', LS.lutBits(m, x));
    document.getElementById('a_lutrow').textContent = lutRow;
    fillVec('a_lutvec', m.V[lutRow]);
  }
  function updateSimReadout(done) {
    const box = document.getElementById('result'); box.style.opacity = done ? '1' : '0.4';
    const bn = document.getElementById('abanner');
    if (!done) {
      fillBits('a_simbits', []); document.getElementById('a_simrow').textContent = '…';
      document.getElementById('a_simvec').innerHTML = '';
      bn.className = 'banner'; bn.textContent = 'playing…'; return;
    }
    const lutRow = LS.bitsToRow(LS.lutBits(m, x)), lutVec = m.V[lutRow];
    fillBits('a_simbits', S.bits); document.getElementById('a_simrow').textContent = S.row;
    fillVec('a_simvec', S.Ohat.map((v) => (v == null ? 0 : v)));
    const err = (S.Otrue && S.row === lutRow) ? Math.max.apply(null, S.Ohat.map((v, j) => Math.abs(v - lutVec[j]))) : Infinity;
    const match = err < 1e-9;
    bn.className = 'banner ' + (match ? 'ok' : 'bad');
    bn.textContent = match ? 'MATCH ✓ — emerged from the spiking simulation' : 'MISMATCH ✗';
  }

  function drawScene() {
    if (needResize) resizeAll();
    drawGraph(); drawRaster();
    document.getElementById('simtime').textContent = 't = ' + simT.toFixed(2) + ' / ' + Tmax.toFixed(2);
    const done = simT >= Tmax - 1e-9;
    if (done !== lastDone) { updateSimReadout(done); lastDone = done; }
  }

  // ---- THE single rAF loop --------------------------------------------------
  function loop(ts) {
    try {
      if (lastTs == null) lastTs = ts;
      let dt = (ts - lastTs) / 1000; lastTs = ts;
      if (dt > 0.1) dt = 0.1;                       // clamp big jumps (tab refocus)
      if (playing) {
        simT += dt * speed * RATE;
        if (simT >= Tmax) { simT = Tmax; playing = false; document.getElementById('play').textContent = '▶ Play'; }
        document.getElementById('scrub').value = simT;
        dirty = true;
      }
      if (dirty || needResize) { drawScene(); dirty = false; }
    } catch (err) { showErr((err && err.stack) || String(err)); }
    requestAnimationFrame(loop);                     // exactly one chain, started once below
  }

  // ---- controls (only set state; never start another loop) ------------------
  const slidersEl = document.getElementById('sliders');
  const valEls = [];
  for (let i = 0; i < D; i++) {
    const row = document.createElement('div');
    row.className = 'slider-row';
    row.innerHTML = `<label>x${i}</label><input type="range" min="-1" max="1" step="0.01" value="0" data-i="${i}"><span class="val">0.00</span>`;
    slidersEl.appendChild(row);
    const inp = row.querySelector('input'), val = row.querySelector('.val');
    valEls.push(val);
    inp.addEventListener('input', () => { x[i] = parseFloat(inp.value); val.textContent = x[i].toFixed(2); recompute(); simT = 0; playing = false; setPlay(false); dirty = true; });
  }
  function syncSliders() { slidersEl.querySelectorAll('input').forEach((inp) => { const i = +inp.dataset.i; inp.value = x[i]; valEls[i].textContent = x[i].toFixed(2); }); }
  const setPlay = (p) => { document.getElementById('play').textContent = p ? '❚❚ Pause' : '▶ Play'; };
  document.getElementById('rand').onclick = () => { x = x.map(() => +(2 * Math.random() - 1).toFixed(2)); syncSliders(); recompute(); simT = 0; playing = true; setPlay(true); dirty = true; };
  document.getElementById('reset').onclick = () => { x = new Array(D).fill(0); syncSliders(); recompute(); simT = 0; playing = false; setPlay(false); dirty = true; };
  document.getElementById('play').onclick = () => { if (simT >= Tmax) simT = 0; playing = !playing; setPlay(playing); dirty = true; };
  document.getElementById('restart').onclick = () => { simT = 0; playing = true; setPlay(true); dirty = true; };
  document.getElementById('speed').onchange = (e) => { speed = parseFloat(e.target.value); };
  document.getElementById('scrub').addEventListener('input', (e) => { playing = false; setPlay(false); simT = parseFloat(e.target.value); dirty = true; });
  window.addEventListener('resize', () => { needResize = true; });

  recompute();
  requestAnimationFrame(loop);   // <-- the ONLY place a loop is ever started
})();
