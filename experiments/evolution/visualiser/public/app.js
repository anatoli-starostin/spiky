// app.js — UI + drawing for the LUT <-> spiking visualiser (full 4-layer pipeline).
// Maths lives in lut_spiking.js (window.LUTSpiking). Built defensively: any fault
// shows in a visible top error-bar instead of silently blanking the page.
(function () {
  'use strict';

  function showErr(msg) {
    let bar = document.getElementById('errbar');
    if (!bar) { bar = document.createElement('div'); bar.id = 'errbar'; document.body.prepend(bar); }
    bar.style.display = 'block';
    bar.textContent = '⚠ Visualiser error — ' + msg;
    if (window.console) console.error(msg);
  }
  window.addEventListener('error', (e) => showErr((e.error && e.error.stack) || e.message));
  window.addEventListener('unhandledrejection', (e) => showErr('promise: ' + ((e.reason && e.reason.message) || e.reason)));

  const LS = window.LUTSpiking;
  if (!LS) { showErr('lut_spiking.js did not load (window.LUTSpiking is undefined)'); return; }

  const modelFull = LS.buildModel();
  const { D, K, Dout, ALPHA, BETA, T_READ, GAP, TC_OFF, ROWD, DBASE, KAPPA } = modelFull.cfg;
  // shared time axis: widest possible output spike = latest row fire + largest delay
  const minV = Math.min.apply(null, modelFull.V.map((row) => Math.min.apply(null, row)));
  const TMAX = (T_READ + GAP + TC_OFF + ROWD) + (DBASE - KAPPA * minV) + 0.4;
  let x = new Array(D).fill(0);
  let qbits = 17;
  const model = () => (qbits >= 17 ? modelFull : LS.quantizeModel(modelFull, qbits));
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const fmt = (v) => (v >= 0 ? ' ' : '') + v.toFixed(3);
  const bin = (a) => a.toString(2).padStart(K, '0');

  // ---- sliders --------------------------------------------------------------
  const slidersEl = document.getElementById('sliders');
  const valEls = [];
  for (let i = 0; i < D; i++) {
    const row = document.createElement('div');
    row.className = 'slider-row';
    row.innerHTML = `<label>x${i}</label><input type="range" min="-1" max="1" step="0.01" value="0" data-i="${i}"><span class="val">0.00</span>`;
    slidersEl.appendChild(row);
    const inp = row.querySelector('input'), val = row.querySelector('.val');
    valEls.push(val);
    inp.addEventListener('input', () => { x[i] = parseFloat(inp.value); val.textContent = x[i].toFixed(2); render(); });
  }
  function syncSliders() {
    slidersEl.querySelectorAll('input').forEach((inp) => { const i = +inp.dataset.i; inp.value = x[i]; valEls[i].textContent = x[i].toFixed(2); });
  }
  document.getElementById('rand').onclick = () => { x = x.map(() => +(2 * Math.random() - 1).toFixed(2)); syncSliders(); render(); };
  document.getElementById('reset').onclick = () => { x = new Array(D).fill(0); syncSliders(); render(); };
  const qslider = document.getElementById('qbits');
  qslider.addEventListener('input', () => { qbits = +qslider.value; render(); });

  // ---- canvas helper --------------------------------------------------------
  function fitCanvas(cv) {
    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth || (cv.parentElement && cv.parentElement.clientWidth) || 820;
    const h = +cv.getAttribute('height');
    cv.width = Math.max(1, Math.round(w * dpr)); cv.height = Math.round(h * dpr);
    const ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }
  const bitBox = (b) => `<span class="bit ${b ? 'on' : 'off'}">${b}</span>`;
  function fillBits(id, bits) { document.getElementById(id).innerHTML = bits.map(bitBox).join(''); }
  function fillVec(id, vec) {
    const mx = Math.max(1e-9, ...vec.map(Math.abs));
    document.getElementById(id).innerHTML = vec.map((v) => {
      const wpct = (Math.abs(v) / mx) * 50;
      return `<div class="small mono">${fmt(v)}</div><div class="vecbar"><i style="left:${v < 0 ? 50 - wpct : 50}%;width:${wpct}%;background:${v < 0 ? css('--bad') : css('--spike')}"></i></div>`;
    }).join('');
  }

  // ---- the 4-layer spike timeline ------------------------------------------
  function drawTimeline() {
    const cv = document.getElementById('timeline');
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const m = model(), P = LS.fullPipeline(m, x);
    const padL = 118, padR = 20, padT = 24, padB = 26;
    const X = (t) => padL + (t / TMAX) * (w - padL - padR);
    const nRows = D + K + (1 << K) + Dout;
    const top = padT, bot = h - padB, rowH = (bot - top) / nRows;
    const rowY = (r) => top + rowH * (r + 0.5);
    ctx.font = '11px ui-monospace';

    // time grid
    for (let t = 0; t <= TMAX; t++) {
      ctx.strokeStyle = css('--edge'); ctx.globalAlpha = .35;
      ctx.beginPath(); ctx.moveTo(X(t), top); ctx.lineTo(X(t), bot); ctx.stroke(); ctx.globalAlpha = 1;
      if (t % 1 === 0) { ctx.fillStyle = css('--muted'); ctx.fillText('t=' + t, X(t) - 8, bot + 15); }
    }
    // alpha, T_read, t_row verticals
    const vline = (t, color, dash, label) => {
      ctx.setLineDash(dash); ctx.strokeStyle = color; ctx.lineWidth = 1.6;
      ctx.beginPath(); ctx.moveTo(X(t), top - 6); ctx.lineTo(X(t), bot); ctx.stroke();
      ctx.setLineDash([]); ctx.lineWidth = 1; ctx.fillStyle = color; ctx.fillText(label, X(t) - 8, top - 9);
    };
    vline(ALPHA, css('--muted'), [4, 3], 'α');
    vline(T_READ, css('--ok'), [], 'T_read');
    vline(P.t_row, css('--row'), [2, 3], 't_row');

    // helper to draw one neuron row: label + baseline (+ optional spike)
    function neuron(r, label, labelColor, spikeT, spikeColor, note) {
      const y = rowY(r);
      ctx.fillStyle = labelColor; ctx.fillText(label, 6, y + 4);
      ctx.strokeStyle = css('--edge'); ctx.globalAlpha = .28; ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke(); ctx.globalAlpha = 1;
      if (spikeT != null) {
        ctx.strokeStyle = spikeColor; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(X(spikeT), y - rowH * .34); ctx.lineTo(X(spikeT), y + rowH * .34); ctx.stroke(); ctx.lineWidth = 1;
        ctx.fillStyle = spikeColor; ctx.fillText(note != null ? note : spikeT.toFixed(2), X(spikeT) + 6, y - 3);
      } else if (note) { ctx.fillStyle = css('--muted'); ctx.fillText(note, padL + 6, y + 4); }
    }
    // stage separators + captions
    const sep = (r, txt) => {
      const yy = top + rowH * r;
      ctx.strokeStyle = css('--edge'); ctx.beginPath(); ctx.moveTo(2, yy); ctx.lineTo(w - padR, yy); ctx.stroke();
      ctx.fillStyle = css('--accent'); ctx.font = 'bold 11px ui-monospace'; ctx.fillText(txt, 6, yy + 13); ctx.font = '11px ui-monospace';
    };

    let r = 0;
    sep(r, '① INPUT');
    for (let i = 0; i < D; i++, r++) neuron(r, 'input n' + i, css('--muted'), P.tin[i], css('--spike'));
    sep(r, '② DETECTORS = address bits');
    for (let k = 0; k < K; k++, r++) {
      const s = P.detectors[k];
      neuron(r, 'detector ' + k, css('--addr'), s.fired ? s.t_out : null, css('--addr'),
        s.fired ? 'spike → bit 1' : null);
      if (!s.fired) { const y = rowY(r); ctx.fillStyle = css('--muted'); ctx.fillText('silent → bit 0', X(T_READ) + 8, y + 4); }
    }
    sep(r, '③ ROW-SELECT (1-hot) — exactly one fires: row ' + P.r + ' = ' + bin(P.r));
    for (let a = 0; a < (1 << K); a++, r++) {
      const rw = P.rows[a];
      const lbl = 'row ' + a + '=' + bin(a);
      if (rw.fires) neuron(r, lbl, css('--row'), P.t_row, css('--row'), 'FIRES (3/3)');
      else neuron(r, lbl, css('--muted'), null, null, 'coincidence ' + rw.count + '/' + K);
    }
    sep(r, '④ OUTPUT — 4 spikes, latencies encode O[r]');
    for (let j = 0; j < Dout; j++, r++) {
      const o = P.outputs[j];
      neuron(r, 'output o' + j, css('--out'), o.t, css('--out'), 'ô=' + o.ohat.toFixed(2));
    }
  }

  // ---- centerpiece + stage tables + LUT ground truth ------------------------
  function drawTables() {
    const m = model(), P = LS.fullPipeline(m, x);
    const pre = LS.lutPreacts(m, x);
    const lutOut = m.V[P.r];

    // centerpiece: OUTPUT VECTORS
    fillBits('lutbits', P.bits); fillBits('spkbits', P.bits);
    document.getElementById('lutrow').textContent = P.r;
    document.getElementById('spkrow').textContent = P.r;
    fillVec('lutvec', lutOut); fillVec('spkvec', P.Ohat);
    const err = Math.max.apply(null, P.Ohat.map((v, j) => Math.abs(v - lutOut[j])));
    const match = err < 1e-9;
    const banner = document.getElementById('banner');
    banner.className = 'banner ' + (match ? 'ok' : 'bad');
    banner.textContent = match ? 'MATCH ✓  — decoded output ô == LUT output O[r] (exactly)' : 'MISMATCH ✗';
    document.getElementById('matchnote').textContent = match
      ? `Same 3 address bits → same row ${P.r} → the 4 output-spike latencies decode to O[${P.r}] to floating-point precision (max err ${err.toExponential(1)}).`
      : 'Unexpected mismatch at full precision — this should not happen.';

    // ① address
    const at = document.querySelector('#addr tbody'); at.innerHTML = '';
    P.detectors.forEach((s) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${s.k}</td><td class="mono">${fmt(s.margin)}</td><td>${s.fired ? '<span class="yes">spike</span>' : '<span class="no">—</span>'}</td><td>${bitBox(s.fired ? 1 : 0)}</td>`;
      at.appendChild(tr);
    });
    // ② row-select
    const rt = document.querySelector('#rowsel tbody'); rt.innerHTML = '';
    P.rows.forEach((rw) => {
      const tr = document.createElement('tr');
      if (rw.fires) tr.className = 'winrow';
      tr.innerHTML = `<td>${rw.a}=${bin(rw.a)}</td><td>${rw.count}/${K}</td><td>${rw.fires ? '<span class="yes">FIRES</span>' : '<span class="no">—</span>'}</td>`;
      rt.appendChild(tr);
    });
    // ③ output
    const ot = document.querySelector('#outtab tbody'); ot.innerHTML = '';
    P.outputs.forEach((o) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${o.j}</td><td class="mono eqA">${fmt(o.val)}</td><td class="mono">${(o.t - P.t_row).toFixed(3)}</td><td class="mono eqB">${fmt(o.ohat)}</td>`;
      ot.appendChild(tr);
    });
    // LUT ground truth (per-k)
    const lm = document.querySelector('#lutmath tbody'); lm.innerHTML = '';
    for (let k = 0; k < K; k++) {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${k}</td><td class="mono eqA">${fmt(pre[k])}</td><td>${bitBox(pre[k] > 0 ? 1 : 0)}</td>`;
      lm.appendChild(tr);
    }
  }

  // ---- pure spiking circuit: emergent-spike raster --------------------------
  function drawCircuit() {
    const cv = document.getElementById('circuit');
    if (!cv) return;
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const m = model(), S = LS.simulateCircuit(m, x);
    // ordered neuron rows, grouped by layer
    const groups = [
      ['START', [['START', css('--muted')]]],
      ['INPUT', Array.from({ length: D }, (_, i) => ['x' + i, css('--spike')])],
      ['CLOCK', [['CLK', css('--ok')]]],
      ['DETECTORS H (address)', Array.from({ length: K }, (_, k) => ['H' + k, css('--addr')])],
      ['COMPLEMENTS C (bit=0 lines)', Array.from({ length: K }, (_, k) => ['C' + k, css('--warn')])],
      ['ROW-SELECT r (1-of-8)', Array.from({ length: 1 << K }, (_, a) => ['r' + a, css('--row')])],
      ['OUTPUT o', Array.from({ length: Dout }, (_, j) => ['o' + j, css('--out')])],
    ];
    const rowsList = [];
    groups.forEach(([g, arr]) => { rowsList.push({ sep: g }); arr.forEach(([id, col]) => rowsList.push({ id, col })); });
    const nR = rowsList.length;
    // time axis from the actual emergent spikes
    let maxT = 0;
    for (const n in S.st) if (S.st[n].fired) maxT = Math.max(maxT, S.st[n].tf);
    maxT += 0.6;
    const padL = 150, padR = 20, padT = 14, padB = 24;
    const X = (t) => padL + (t / maxT) * (w - padL - padR);
    const top = padT, rowH = (h - padT - padB) / nR;
    ctx.font = '11px ui-monospace';
    for (let t = 0; t <= maxT; t++) {
      ctx.strokeStyle = css('--edge'); ctx.globalAlpha = .3; ctx.beginPath(); ctx.moveTo(X(t), top); ctx.lineTo(X(t), h - padB); ctx.stroke(); ctx.globalAlpha = 1;
      ctx.fillStyle = css('--muted'); ctx.fillText('t=' + t, X(t) - 8, h - padB + 15);
    }
    rowsList.forEach((r, idx) => {
      const y = top + rowH * (idx + 0.5);
      if (r.sep) { ctx.fillStyle = css('--accent'); ctx.font = 'bold 10px ui-monospace'; ctx.fillText(r.sep, 6, y + 4); ctx.font = '11px ui-monospace'; return; }
      const s = S.st[r.id];
      ctx.fillStyle = css('--muted'); ctx.fillText(r.id, 12, y + 4);
      ctx.strokeStyle = css('--edge'); ctx.globalAlpha = .22; ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke(); ctx.globalAlpha = 1;
      if (s && s.fired) {
        ctx.strokeStyle = r.col; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(X(s.tf), y - rowH * .34); ctx.lineTo(X(s.tf), y + rowH * .34); ctx.stroke(); ctx.lineWidth = 1;
        ctx.fillStyle = r.col; ctx.fillText(s.tf.toFixed(2), X(s.tf) + 5, y - 3);
      } else { ctx.fillStyle = '#3a4250'; ctx.fillText('· silent', padL + 6, y + 4); }
    });

    // read-back vs LUT ground truth
    const lutRow = LS.bitsToRow(LS.lutBits(m, x)), lutVec = m.V[lutRow];
    fillBits('clut_bits', LS.lutBits(m, x)); fillBits('csim_bits', S.bits);
    document.getElementById('clut_row').textContent = lutRow;
    document.getElementById('csim_row').textContent = S.row;
    fillVec('clut_vec', lutVec); fillVec('csim_vec', S.Ohat.map((v) => (v == null ? 0 : v)));
    const err = (S.Otrue && S.row === lutRow) ? Math.max.apply(null, S.Ohat.map((v, j) => Math.abs(v - lutVec[j]))) : Infinity;
    const match = err < 1e-9;
    const bn = document.getElementById('cbanner');
    bn.className = 'banner ' + (match ? 'ok' : 'bad');
    bn.textContent = match ? 'MATCH ✓  — everything emerged from the spiking simulation' : 'MISMATCH ✗';
    document.getElementById('cnote').textContent = match
      ? `Emergent spikes decode to the same 3 bits, the same winning row ${S.row}, and an output vector equal to O[${S.row}] to fp precision (max err ${err.toExponential(1)}). No margin was read anywhere — the detectors fired because the clock pulse pushed their integrated membrane past threshold.`
      : 'Unexpected mismatch — the simulation disagrees with the table.';
  }

  // ---- network topology graph ----------------------------------------------
  function drawTopology() {
    const cv = document.getElementById('topology');
    if (!cv) return;
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const m = model(), P = LS.fullPipeline(m, x);
    const padL = 46, padR = 52, padT = 30, padB = 18, R = 11;
    const col = [padL, padL + 0.34 * (w - padL - padR), padL + 0.68 * (w - padL - padR), w - padR];
    const ys = (n) => { const u = h - padT - padB; return Array.from({ length: n }, (_, i) => padT + (i + 0.5) * u / n); };
    const yIn = ys(D), yDet = ys(K), yRow = ys(1 << K), yOut = ys(Dout);
    const maxW = Math.max.apply(null, m.W.reduce((a, r) => a.concat(r.map(Math.abs)), [1e-6]));
    const tin = P.tin, tMin = Math.min.apply(null, tin), tMax = Math.max.apply(null, tin);

    // --- edges (drawn behind nodes) ---
    for (let k = 0; k < K; k++) {                       // input -> detector (weights)
      const fired = P.detectors[k].fired;
      for (let i = 0; i < D; i++) {
        const wv = m.W[k][i];
        ctx.strokeStyle = wv >= 0 ? css('--ok') : css('--bad');
        ctx.globalAlpha = fired ? Math.min(1, 0.22 + 0.78 * Math.abs(wv) / maxW) : 0.07;
        ctx.lineWidth = 0.5 + 3 * Math.abs(wv) / maxW;
        ctx.beginPath(); ctx.moveTo(col[0] + R, yIn[i]); ctx.lineTo(col[1] - R, yDet[k]); ctx.stroke();
      }
    }
    ctx.globalAlpha = 1; ctx.lineWidth = 1;
    for (let a = 0; a < (1 << K); a++) {                 // detector -> row (polarity)
      const win = P.rows[a].fires;
      for (let k = 0; k < K; k++) {
        const bit = (a >> (K - 1 - k)) & 1;
        ctx.setLineDash(bit ? [] : [3, 3]);
        ctx.strokeStyle = css('--row'); ctx.globalAlpha = win ? 0.9 : 0.06; ctx.lineWidth = win ? 2 : 1;
        ctx.beginPath(); ctx.moveTo(col[1] + R, yDet[k]); ctx.lineTo(col[2] - R, yRow[a]); ctx.stroke();
      }
    }
    ctx.setLineDash([]); ctx.globalAlpha = 1; ctx.lineWidth = 1;
    for (let a = 0; a < (1 << K); a++) {                 // row -> output (delays)
      const win = P.rows[a].fires;
      for (let j = 0; j < Dout; j++) {
        ctx.strokeStyle = css('--out');
        ctx.globalAlpha = win ? 0.9 : 0.05; ctx.lineWidth = win ? 2 : 1;
        ctx.beginPath(); ctx.moveTo(col[2] + R, yRow[a]); ctx.lineTo(col[3] - R, yOut[j]); ctx.stroke();
        if (win) {
          ctx.fillStyle = css('--out'); ctx.font = '9px ui-monospace';
          ctx.fillText('Δ' + P.outputs[j].delay.toFixed(2), (col[2] + col[3]) / 2 - 12, (yRow[a] + yOut[j]) / 2 - 2);
        }
      }
    }
    ctx.globalAlpha = 1; ctx.lineWidth = 1;

    // --- nodes (on top) ---
    function node(cx, cy, label, fill, textColor, sub) {
      ctx.beginPath(); ctx.arc(cx, cy, R, 0, 7); ctx.fillStyle = fill; ctx.fill();
      ctx.strokeStyle = css('--edge'); ctx.stroke();
      ctx.fillStyle = textColor; ctx.font = 'bold 10px ui-monospace'; ctx.textAlign = 'center';
      ctx.fillText(label, cx, cy + 3); ctx.textAlign = 'left';
      if (sub) { ctx.fillStyle = css('--muted'); ctx.font = '9px ui-monospace'; ctx.fillText(sub, cx + R + 4, cy + 3); }
    }
    for (let i = 0; i < D; i++) {
      const frac = tMax > tMin ? 1 - (tin[i] - tMin) / (tMax - tMin) : 1;   // earlier => brighter
      node(col[0], yIn[i], 'x' + i, `rgba(0,180,216,${0.35 + 0.65 * frac})`, '#001018', 't=' + tin[i].toFixed(2));
    }
    for (let k = 0; k < K; k++) {
      const s = P.detectors[k];
      node(col[1], yDet[k], 'H' + k, s.fired ? css('--addr') : '#20262f', s.fired ? '#1a0f16' : css('--muted'),
        'm=' + s.margin.toFixed(2) + ' →' + (s.fired ? 1 : 0));
    }
    for (let a = 0; a < (1 << K); a++) {
      const rw = P.rows[a];
      node(col[2], yRow[a], 'r' + a, rw.fires ? css('--row') : '#20262f', rw.fires ? '#100a1f' : css('--muted'),
        rw.fires ? 'WIN 3/3' : rw.count + '/' + K);
    }
    for (let j = 0; j < Dout; j++) node(col[3], yOut[j], 'o' + j, css('--out'), '#1a1204', P.outputs[j].ohat.toFixed(2));

    // layer captions
    ctx.fillStyle = css('--accent'); ctx.font = 'bold 11px ui-monospace'; ctx.textAlign = 'center';
    ['① INPUT', '② DETECTORS', '③ ROW-SELECT', '④ OUTPUT'].forEach((t, i) => ctx.fillText(t, col[i], 14));
    ctx.textAlign = 'left';
  }

  // ---- secondary: detector membranes ---------------------------------------
  function drawMembranes() {
    const cv = document.getElementById('membranes');
    if (cv.offsetParent === null) return;
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    const m = model(), th = LS.thetas(m);
    const padL = 52, padR = 16, gap = 12, TR = T_READ + 0.6;
    const subH = (h - gap * (K - 1)) / K;
    const X = (t) => padL + (t / TR) * (w - padL - padR);
    for (let k = 0; k < K; k++) {
      const y0 = k * (subH + gap);
      const prof = LS.membraneProfile(m, x, k);
      let vmin = th[k], vmax = th[k];
      for (const p of prof) { vmin = Math.min(vmin, p.v); vmax = Math.max(vmax, p.v); }
      const pad = (vmax - vmin) * 0.15 || 1; vmin -= pad; vmax += pad;
      const Y = (v) => y0 + subH - ((v - vmin) / (vmax - vmin)) * (subH - 8) - 4;
      ctx.strokeStyle = css('--edge'); ctx.strokeRect(padL, y0 + 2, w - padL - padR, subH - 4);
      ctx.fillStyle = css('--muted'); ctx.font = '11px ui-monospace'; ctx.fillText('V' + k + '(t)', 6, y0 + 14);
      ctx.setLineDash([5, 4]); ctx.strokeStyle = css('--warn');
      ctx.beginPath(); ctx.moveTo(padL, Y(th[k])); ctx.lineTo(w - padR, Y(th[k])); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = css('--warn'); ctx.fillText('θ' + k + '=' + th[k].toFixed(2), w - padR - 92, Y(th[k]) - 4);
      ctx.strokeStyle = css('--ok'); ctx.globalAlpha = .8; ctx.beginPath(); ctx.moveTo(X(T_READ), y0 + 2); ctx.lineTo(X(T_READ), y0 + subH - 2); ctx.stroke(); ctx.globalAlpha = 1;
      ctx.strokeStyle = css('--spk'); ctx.lineWidth = 2; ctx.beginPath();
      prof.forEach((p, j) => { const px = X(p.t), py = Y(p.v); j ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
      ctx.stroke(); ctx.lineWidth = 1;
      const vr = LS.membraneAtReadout(m, x, k);
      ctx.fillStyle = css('--spk'); ctx.beginPath(); ctx.arc(X(T_READ), Y(vr), 4.5, 0, 7); ctx.fill();
    }
  }

  // ---- secondary: quantization ---------------------------------------------
  function drawQuant() {
    const label = document.getElementById('qbitsLabel'), fid = document.getElementById('qfid');
    if (qbits >= 17) {
      label.textContent = 'full';
      fid.innerHTML = 'Full precision — the table is reproduced exactly. Slide left to quantize W, b and watch fidelity fall.';
    } else {
      label.textContent = qbits + ' bits';
      const f = LS.quantFidelity(modelFull, qbits, 3000, 999);
      fid.innerHTML = `At <b>${qbits} bits</b>: over 3000 random inputs the quantized table matches the full-precision one on <b>${(f.bit * 100).toFixed(2)}%</b> of bits and <b>${(f.row * 100).toFixed(2)}%</b> of rows.`;
    }
  }

  function render() {
    try { drawTimeline(); drawTopology(); drawCircuit(); drawTables(); drawMembranes(); drawQuant(); }
    catch (err) { showErr((err && err.stack) || String(err)); }
  }
  document.querySelectorAll('details').forEach((d) => d.addEventListener('toggle', render));
  window.addEventListener('resize', render);
  window.addEventListener('load', render);
  requestAnimationFrame(render);
  render();
})();
