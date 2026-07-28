// circuit.js — wires ONLY two panels for the minimal circuit.html page:
// (1) input sliders + LUT ground truth, (2) the pure event-driven spiking
// circuit (reusing LUTSpiking.simulateCircuit, unchanged). Defensive throughout.
(function () {
  'use strict';

  function showErr(msg) {
    let bar = document.getElementById('errbar');
    if (!bar) { bar = document.createElement('div'); bar.id = 'errbar'; document.body.prepend(bar); }
    bar.style.display = 'block';
    bar.textContent = '⚠ Circuit page error — ' + msg;
    if (window.console) console.error(msg);
  }
  window.addEventListener('error', (e) => showErr((e.error && e.error.stack) || e.message));
  window.addEventListener('unhandledrejection', (e) => showErr('promise: ' + ((e.reason && e.reason.message) || e.reason)));

  const LS = window.LUTSpiking;
  if (!LS) { showErr('lut_spiking.js did not load (window.LUTSpiking is undefined)'); return; }

  const m = LS.buildModel();
  const { D, K, Dout } = m.cfg;
  let x = new Array(D).fill(0);
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const fmt = (v) => (v >= 0 ? ' ' : '') + v.toFixed(3);
  const bitBox = (b) => `<span class="bit ${b ? 'on' : 'off'}">${b}</span>`;
  const fillBits = (id, bits) => { document.getElementById(id).innerHTML = bits.map(bitBox).join(''); };
  function fillVec(id, vec) {
    const mx = Math.max(1e-9, ...vec.map(Math.abs));
    document.getElementById(id).innerHTML = vec.map((v) => {
      const wp = (Math.abs(v) / mx) * 50;
      return `<div class="small mono">${fmt(v)}</div><div class="vecbar"><i style="left:${v < 0 ? 50 - wp : 50}%;width:${wp}%;background:${v < 0 ? css('--bad') : css('--spike')}"></i></div>`;
    }).join('');
  }
  function fitCanvas(cv) {
    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth || (cv.parentElement && cv.parentElement.clientWidth) || 820;
    const h = +cv.getAttribute('height');
    cv.width = Math.max(1, Math.round(w * dpr)); cv.height = Math.round(h * dpr);
    const ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }

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

  // ---- panel 1: LUT ground truth (+ input latencies) -----------------------
  // Takes the SAME simulateCircuit output S the raster uses, so the input times
  // shown here are literally the neurons' spike times — one source of truth.
  function drawLUT(S) {
    // input latency table (t_i = alpha - beta*x_i, straight from the sim)
    const lt = document.querySelector('#latencies tbody'); lt.innerHTML = '';
    for (let i = 0; i < D; i++) {
      const ti = S.st['x' + i].tf;                 // the actual input-neuron spike time
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>x${i}</td><td class="mono">${fmt(x[i])}</td><td class="mono">${ti.toFixed(3)}</td>`;
      lt.appendChild(tr);
    }
    // LUT ground truth
    const pre = LS.lutPreacts(m, x), bits = LS.lutBits(m, x), row = LS.bitsToRow(bits);
    const tb = document.querySelector('#lutmath tbody'); tb.innerHTML = '';
    for (let k = 0; k < K; k++) {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${k}</td><td class="mono">${fmt(pre[k])}</td><td>${bitBox(bits[k])}</td>`;
      tb.appendChild(tr);
    }
    document.getElementById('lut_row').textContent = row;
    fillVec('lut_out', m.V[row]);
  }

  // ---- panel 2: pure spiking circuit (emergent-spike raster) ----------------
  function drawCircuit(S) {
    const cv = document.getElementById('circuit');
    const { ctx, w, h } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, h);
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

    // read-back vs LUT
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
      ? `Emergent spikes decode to the same 3 bits, the same winning row ${S.row}, and an output vector equal to O[${S.row}] to fp precision (max err ${err.toExponential(1)}). No margin was read anywhere.`
      : 'Unexpected mismatch — the simulation disagrees with the table.';
  }

  function render() {
    try { const S = LS.simulateCircuit(m, x); drawLUT(S); drawCircuit(S); }  // one source of truth
    catch (err) { showErr((err && err.stack) || String(err)); }
  }
  window.addEventListener('resize', render);
  window.addEventListener('load', render);
  requestAnimationFrame(render);
  render();
})();
