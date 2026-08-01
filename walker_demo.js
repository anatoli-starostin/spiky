// inspector.js — dependency-free spnet inspector: linked graph / activity / raster.
(function () {
  'use strict';
  function showErr(msg) {
    let bar = document.getElementById('errbar');
    bar.style.display = 'block'; bar.textContent = '⚠ inspector error — ' + msg;
    if (window.console) console.error(msg);
  }
  window.addEventListener('error', (e) => showErr((e.error && e.error.stack) || e.message));

  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const LAYER_COL = { clock: () => css('--ok'), gate: () => '#c58af9', input: () => css('--spike'),
    decode: () => '#3fb6a8', detect: () => css('--addr'), hidden: () => '#c58af9',
    compl: () => css('--warn'), rows: () => css('--row'), output: () => css('--out') };
  const FLASH = 0.45, DOT_MIN = 0.6;

  let G = null, A = null;
  let byId = {}, fireTicks = {}, volt = {}, thr = {}, outSyn = {};
  let simT = 0, Tmax = 1, playing = false, speed = 1, lastTs = null, dirty = true;
  let sel = null, hover = null;

  // Walker2d LUT->spiking: our network + REAL spnet spike trace, in the inspector's own schema.
  const _gf = 'walker_graph.json';
  const _af = 'walker_activity.json';
  Promise.all([fetch(_gf).then(r => r.json()), fetch(_af).then(r => r.json())])
    .then(([g, a]) => { G = g; A = a; init(); }).catch(e => showErr('load: ' + e));

  // ---- activity loading (a single activity object, OR one variant of a bundle) ----
  function loadActivity(act) {
    fireTicks = {}; volt = {};
    for (const n of G.neurons) fireTicks[n.id] = [];
    for (const v of act.voltages) volt[v.neuron_id] = v.trace;
    for (const sp of act.spikes) (fireTicks[sp.neuron_id] || (fireTicks[sp.neuron_id] = [])).push(sp.tick);
    Tmax = act.t1;
    simT = 0; playing = false; dirty = true;
    const pb = document.getElementById('play'); if (pb) pb.textContent = '▶ Play';
    renderVariantInfo(act);
  }

  const ordStr = (a) => a.map((d) => 'o' + d).join(' > ');

  function renderVariantInfo(act) {
    const info = document.getElementById('variantInfo');
    if (!info || act.input === undefined) return;
    const rows = act.gt.map((g, o) => { const d = act.dec[o], e = Math.abs(g - d), ok = e < 0.15;
      return `<tr><td>a${o}</td><td class="mono">${g.toFixed(3)}</td><td class="mono">${d.toFixed(3)}</td><td class="mono">${e.toFixed(3)}</td><td>${ok ? '<span style="color:var(--ok)">✓</span>' : '<span style="color:var(--warn)">✕</span>'}</td></tr>`; }).join('');
    info.innerHTML =
      `<b>obs (17-dim)</b> = <span class="mono">[${act.input.join(', ')}]</span><br>`
      + `<b>action means — LUT oracle vs spiking</b> <span class="small" style="color:var(--muted)">(decoded from the 6 output-neuron spike times)</span>:`
      + `<table style="width:100%;font-size:12px;margin-top:4px"><tr><td>dim</td><td>oracle</td><td>spiking</td><td>|Δ|</td><td></td></tr>${rows}</table>`
      + (act.match ? '<span style="color:var(--ok)">✓ spiking output matches the trained LUT policy</span>'
                   : '<span style="color:var(--warn)">✕ one action dim differs (a rare boundary address-flip)</span>');
  }

  function setupVariantsUI() {
    if (!A.variants) return;
    const ctl = document.querySelector('.ctl');
    const bar = document.createElement('div');
    bar.className = 'ctl'; bar.style.marginTop = '8px'; bar.style.flexWrap = 'wrap';
    const lab = document.createElement('label'); lab.textContent = 'input ';
    const sel = document.createElement('select'); sel.id = 'variantSel';
    A.variants.forEach((v, i) => {
      const o = document.createElement('option');
      o.value = i; o.textContent = (i + 1) + ') ' + v.label; sel.appendChild(o);
    });
    sel.addEventListener('change', () => loadActivity(A.variants[+sel.value]));
    lab.appendChild(sel); bar.appendChild(lab);
    const info = document.createElement('div');
    info.id = 'variantInfo'; info.style.cssText = 'margin-top:8px;font-size:12.5px;line-height:1.8';
    ctl.parentNode.insertBefore(bar, ctl.nextSibling);
    bar.parentNode.insertBefore(info, bar.nextSibling);
  }

  function init() {
    for (const n of G.neurons) { byId[n.id] = n; thr[n.id] = n.spike_threshold; }
    for (const s of G.synapses) (outSyn[s.source] || (outSyn[s.source] = [])).push(s);
    setupVariantsUI();
    loadActivity(A.variants ? A.variants[0] : A);
    document.getElementById('glegend').innerHTML =
      G.layers.map(l => `<span style="color:${(LAYER_COL[l] ? LAYER_COL[l]() : '#999')}">■ ${l}</span>`).join(' · ')
      + ' · edges: <span style="color:#5b9dff">■ −w</span>/<span style="color:#ff6b6b">■ +w</span>, dashed = delay>1';
    document.title = 'Walker2d LUT → spiking (inspector)';
    setupCanvas(); bindControls();
    requestAnimationFrame(loop);
  }

  // ---- canvas (dpr-capped, size-once) ----
  const canv = {};
  function setupCanvas() {
    ['graph', 'raster'].forEach(id => {
      const cv = document.getElementById(id);
      canv[id] = { cv, ctx: cv.getContext('2d'), w: 0, h: 0, hCSS: +cv.getAttribute('height') };
    });
    resize();
    window.addEventListener('resize', () => { resize(); dirty = true; });
    document.addEventListener('visibilitychange', () => { if (!document.hidden) { lastTs = null; dirty = true; } });
    // interactions
    const g = canv.graph.cv;
    g.addEventListener('mousemove', (e) => { hover = pick(e); dirty = true; g.style.cursor = hover ? 'pointer' : 'default'; });
    g.addEventListener('click', (e) => { sel = pick(e); showPanel(); dirty = true; });
    const r = canv.raster.cv;
    let drag = false;
    const scrub = (e) => { const t = rasterTick(e); if (t != null) { simT = t; playing = false; setPlay(false); dirty = true; } };
    r.addEventListener('mousedown', (e) => { drag = true; scrub(e); });
    r.addEventListener('mousemove', (e) => { if (drag) scrub(e); });
    window.addEventListener('mouseup', () => { drag = false; });
  }
  function resize() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    for (const id in canv) {
      const c = canv[id];
      const w = Math.round(c.cv.clientWidth || 800), h = c.hCSS;
      if (w === c.w) continue;
      c.cv.style.height = h + 'px'; c.cv.width = Math.max(1, w * dpr); c.cv.height = h * dpr;
      c.ctx.setTransform(dpr, 0, 0, dpr, 0, 0); c.w = w; c.h = h;
    }
    layoutGraph();
  }

  // ---- graph layout: map node.x/node.y to canvas ----
  let pos = {};
  function layoutGraph() {
    const c = canv.graph; if (!c) return;
    const xs = G.neurons.map(n => n.x), ys = G.neurons.map(n => n.y);
    const minx = Math.min(...xs), maxx = Math.max(...xs), miny = Math.min(...ys), maxy = Math.max(...ys);
    const padL = 40, padR = 40, padT = 24, padB = 20;
    const sx = (c.w - padL - padR) / Math.max(1, maxx - minx), sy = (c.h - padT - padB) / Math.max(1, maxy - miny);
    pos = {};
    for (const n of G.neurons) pos[n.id] = { x: padL + (n.x - minx) * sx, y: padT + (n.y - miny) * sy };
  }

  const wmax = () => Math.max(1e-6, ...G.synapses.map(s => Math.abs(s.weight)));
  function edgeStyle(ctx, s) {
    const mag = Math.min(1, Math.abs(s.weight) / wmax());
    ctx.strokeStyle = s.weight < 0 ? '#5b9dff' : '#ff6b6b';
    ctx.globalAlpha = 0.15 + 0.5 * mag; ctx.lineWidth = 0.6 + 2.2 * mag;
    if (s.delay > 1) ctx.setLineDash([4, 3]); else ctx.setLineDash([]);
  }

  function firedBy(id, t) { const f = fireTicks[id] || []; for (const ft of f) if (Math.abs(t - ft) < FLASH) return true; return false; }
  function nodeFill(id, t) {
    const tr = volt[id]; if (!tr) return 0; const v = tr[Math.max(0, Math.min(tr.length - 1, Math.floor(t)))];
    const th = thr[id]; if (!(th > 0) || th > 1e5) { // huge threshold (outputs): scale by trace max
      const mx = Math.max(1e-6, ...tr.map(Math.abs)); return Math.max(-1, Math.min(1, v / mx));
    }
    return Math.max(-1, Math.min(1, v / th));
  }

  function drawGraph() {
    const c = canv.graph, ctx = c.ctx; ctx.clearRect(0, 0, c.w, c.h);
    // edges
    for (const s of G.synapses) {
      const a = pos[s.source], b = pos[s.target]; if (!a || !b) continue;
      const hot = sel && sel.kind === 'edge' && sel.o.id === s.id;
      edgeStyle(ctx, s); if (hot) { ctx.globalAlpha = 1; ctx.lineWidth += 1.5; }
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    }
    ctx.setLineDash([]); ctx.globalAlpha = 1;
    // travelling spike dots (source fired at T -> dot to target over delay)
    for (const s of G.synapses) {
      const a = pos[s.source], b = pos[s.target]; if (!a || !b) continue;
      for (const T of (fireTicks[s.source] || [])) {
        const span = Math.max(s.delay, DOT_MIN);
        if (simT < T || simT > T + span) continue;
        const fr = (simT - T) / span;
        ctx.fillStyle = s.weight < 0 ? '#5b9dff' : '#ff6b6b';
        ctx.beginPath(); ctx.arc(a.x + (b.x - a.x) * fr, a.y + (b.y - a.y) * fr, 3, 0, 7); ctx.fill();
      }
    }
    // nodes
    for (const n of G.neurons) {
      const p = pos[n.id], R = 11, col = (LAYER_COL[n.type] ? LAYER_COL[n.type]() : '#888');
      const flashing = firedBy(n.id, simT), frac = nodeFill(n.id, simT);
      if (flashing) { ctx.globalAlpha = 0.5; ctx.fillStyle = col; ctx.beginPath(); ctx.arc(p.x, p.y, R + 6, 0, 7); ctx.fill(); ctx.globalAlpha = 1; }
      ctx.beginPath(); ctx.arc(p.x, p.y, R, 0, 7); ctx.fillStyle = '#20262f'; ctx.fill();
      if (Math.abs(frac) > 0.02) {
        ctx.save(); ctx.beginPath(); ctx.arc(p.x, p.y, R, 0, 7); ctx.clip();
        ctx.globalAlpha = 0.8; ctx.fillStyle = frac >= 0 ? col : '#5b9dff';
        const fh = 2 * R * Math.min(1, Math.abs(frac)); ctx.fillRect(p.x - R, p.y + R - fh, 2 * R, fh);
        ctx.globalAlpha = 1; ctx.restore();
      }
      const hot = (hover && hover.kind === 'node' && hover.o.id === n.id) || (sel && sel.kind === 'node' && sel.o.id === n.id);
      ctx.beginPath(); ctx.arc(p.x, p.y, R, 0, 7); ctx.strokeStyle = hot ? '#fff' : col; ctx.lineWidth = hot ? 2.5 : 1.2; ctx.stroke();
      ctx.fillStyle = css('--ink'); ctx.font = 'bold 8px ui-monospace'; ctx.textAlign = 'center'; ctx.fillText(n.label, p.x, p.y + 3);
    }
    ctx.textAlign = 'left'; ctx.lineWidth = 1;
  }

  // ---- raster ----
  let rasterGeom = null;
  function drawRaster() {
    const c = canv.raster, ctx = c.ctx; ctx.clearRect(0, 0, c.w, c.h);
    const order = G.neurons.slice().sort((a, b) => a.col - b.col || a.y - b.y);
    const padL = 64, padR = 16, padT = 12, padB = 22;
    const n = order.length, rowH = (c.h - padT - padB) / n;
    const X = (t) => padL + (t / Math.max(1, Tmax)) * (c.w - padL - padR);
    rasterGeom = { padL, padR, X, order };
    ctx.font = '9px ui-monospace';
    for (let t = 0; t <= Tmax; t++) { ctx.strokeStyle = css('--edge'); ctx.globalAlpha = t % 5 === 0 ? .4 : .12; ctx.beginPath(); ctx.moveTo(X(t), padT); ctx.lineTo(X(t), c.h - padB); ctx.stroke(); ctx.globalAlpha = 1; if (t % 2 === 0) { ctx.fillStyle = css('--muted'); ctx.textAlign = 'center'; ctx.fillText(t, X(t), c.h - padB + 12); } }
    ctx.textAlign = 'left';
    for (let i = 0; i < n; i++) {
      const nd = order[i], y = padT + rowH * (i + 0.5), col = (LAYER_COL[nd.type] ? LAYER_COL[nd.type]() : '#888');
      ctx.fillStyle = col; ctx.font = '9px ui-monospace'; ctx.fillText(nd.label, 6, y + 3);
      for (const T of (fireTicks[nd.id] || [])) {
        ctx.fillStyle = col; ctx.beginPath(); ctx.arc(X(T), y, Math.min(4, rowH * 0.35), 0, 7); ctx.fill();
      }
    }
    // playhead
    ctx.strokeStyle = css('--out'); ctx.lineWidth = 1.5; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(X(simT), padT); ctx.lineTo(X(simT), c.h - padB); ctx.stroke(); ctx.setLineDash([]); ctx.lineWidth = 1;
  }
  function rasterTick(e) {
    if (!rasterGeom) return null; const c = canv.raster, rect = c.cv.getBoundingClientRect();
    const x = (e.clientX - rect.left); const { padL, padR } = rasterGeom;
    const t = (x - padL) / (c.w - padL - padR) * Tmax; return Math.max(0, Math.min(Tmax, t));
  }

  // ---- picking (graph) ----
  function pick(e) {
    const c = canv.graph, rect = c.cv.getBoundingClientRect(); const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    for (const n of G.neurons) { const p = pos[n.id]; if ((mx - p.x) ** 2 + (my - p.y) ** 2 <= 144) return { kind: 'node', o: n }; }
    let best = null, bd = 6;
    for (const s of G.synapses) { const a = pos[s.source], b = pos[s.target]; if (!a || !b) continue; const d = segDist(mx, my, a, b); if (d < bd) { bd = d; best = { kind: 'edge', o: s }; } }
    return best;
  }
  function segDist(px, py, a, b) {
    const dx = b.x - a.x, dy = b.y - a.y, L2 = dx * dx + dy * dy; if (L2 === 0) return Math.hypot(px - a.x, py - a.y);
    let t = ((px - a.x) * dx + (py - a.y) * dy) / L2; t = Math.max(0, Math.min(1, t));
    return Math.hypot(px - (a.x + t * dx), py - (a.y + t * dy));
  }
  function showPanel() {
    const side = document.getElementById('side'); if (!sel) { side.innerHTML = '<h3>Click a node or edge</h3>'; return; }
    if (sel.kind === 'node') {
      const n = sel.o, f = ['type', 'label', 'cf_2', 'cf_1', 'cf_0', 'a', 'b', 'c', 'd', 'spike_threshold'];
      side.innerHTML = `<h3 style="color:${(LAYER_COL[n.type] ? LAYER_COL[n.type]() : '#888')}">Neuron ${n.label} <span class="mono" style="color:var(--muted)">#${n.id}</span></h3>`
        + '<table>' + f.map(k => `<tr><td>${k}</td><td class="mono">${typeof n[k] === 'number' ? (Math.abs(n[k]) > 1e5 ? n[k].toExponential(1) : (+n[k]).toFixed(4)) : n[k]}</td></tr>`).join('') + '</table>'
        + `<p class="small">fires at ticks: <b class="mono">${(fireTicks[n.id] || []).join(', ') || '—'}</b></p>`;
    } else {
      const s = sel.o, sn = byId[s.source], tn = byId[s.target];
      side.innerHTML = `<h3>Synapse <span class="mono">${sn.label} → ${tn.label}</span></h3>`
        + '<table>' + [['weight', s.weight.toFixed(4)], ['delay', s.delay], ['synapse_meta_index', s.synapse_meta_index],
        ['learning_rate', s.learning_rate], ['min_weight', s.min_weight], ['max_weight', s.max_weight]]
          .map(([k, v]) => `<tr><td>${k}</td><td class="mono">${v}</td></tr>`).join('') + '</table>';
    }
  }

  // ---- loop + controls ----
  const setPlay = (p) => { document.getElementById('play').textContent = p ? '❚❚ Pause' : '▶ Play'; };
  function bindControls() {
    document.getElementById('play').onclick = () => { if (simT >= Tmax) simT = 0; playing = !playing; setPlay(playing); dirty = true; };
    document.getElementById('step').onclick = () => { playing = false; setPlay(false); simT = Math.min(Tmax, Math.floor(simT) + 1); dirty = true; };
    document.getElementById('restart').onclick = () => { simT = 0; playing = true; setPlay(true); dirty = true; };
    document.getElementById('speed').onchange = (e) => { speed = parseFloat(e.target.value); };
  }
  function loop(ts) {
    try {
      if (document.hidden) { lastTs = null; requestAnimationFrame(loop); return; }
      if (lastTs == null) lastTs = ts; let dt = (ts - lastTs) / 1000; lastTs = ts; if (dt > 0.1) dt = 0.1;
      if (playing) { simT += dt * speed * 3; if (simT >= Tmax) { simT = Tmax; playing = false; setPlay(false); } dirty = true; }
      if (dirty) { drawGraph(); drawRaster(); document.getElementById('tickinfo').textContent = 'tick ' + simT.toFixed(2) + ' / ' + Tmax; dirty = false; }
    } catch (err) { showErr((err && err.stack) || String(err)); }
    requestAnimationFrame(loop);
  }
})();
