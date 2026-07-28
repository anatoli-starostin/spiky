// lut_spiking.js
// ============================================================================
// The ENTIRE hyperplane-LUT -> spiking-network transfer, in plain readable JS.
// Nothing is hidden and nothing calls out to Python — this mirrors the CPU toy
// ~/projects/lut2spiking_toy/demo.py exactly.
//
//   LUT:       bit_k = 1[<W_k, x> + b_k > 0];  row = K bits as int;  out = V[row]
//
//   Spiking:   encode x as first-spike latencies   t_i = alpha - beta * x_i
//              hyperplane k = a leak-free IF neuron whose membrane is
//                  V_k(t) = sum_i W[k,i] * (t - t_i) * [t >= t_i]
//              read the sign at a FIXED time T_read > max latency. There
//                  V_k(T_read) = (T_read - alpha)*sum_i W[k,i] + beta*<W_k,x>
//              so with threshold  theta_k = (T_read - alpha)*sum_i W[k,i] - beta*b_k
//                  V_k(T_read) - theta_k = beta * (<W_k, x> + b_k)
//              and (beta>0) the spiking bit [V_k(T_read) > theta_k] == the LUT bit.
// ============================================================================
(function (global) {
  'use strict';

  // ---- seeded RNG (mulberry32) + standard normal (Box-Muller) --------------
  function mulberry32(seed) {
    return function () {
      let t = (seed += 0x6d2b79f5);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function makeNormal(rng) {
    let spare = null;
    return function () {
      if (spare !== null) { const s = spare; spare = null; return s; }
      let u, v, s;
      do { u = 2 * rng() - 1; v = 2 * rng() - 1; s = u * u + v * v; } while (s >= 1 || s === 0);
      const m = Math.sqrt((-2 * Math.log(s)) / s);
      spare = v * m;
      return u * m;
    };
  }

  // ---- default configuration -----------------------------------------------
  // GAP/EPS/C shape the OUTPUT-spike window: a fired detector emits a real spike
  // in (T_READ, T_READ+GAP]; larger readout margin -> earlier spike.
  const CFG = { D: 6, K: 3, Dout: 4, ALPHA: 3.0, BETA: 1.0, T_READ: 5.0, SEED: 0,
                GAP: 2.4, EPS: 0.2, C: 0.3,   // address-spike window
                TC_OFF: 0.2, ROWD: 0.2,       // complement-line time, row coincidence delay
                DBASE: 1.8, KAPPA: 0.35 };     // output-layer latency code

  // ---- build the fixed-seed toy model (W, b, V) ----------------------------
  function buildModel(cfg) {
    cfg = Object.assign({}, CFG, cfg || {});
    const rng = mulberry32(cfg.SEED);
    const N = makeNormal(rng);
    const { D, K, Dout } = cfg;
    const W = Array.from({ length: K }, () => Array.from({ length: D }, () => N()));
    const b = Array.from({ length: K }, () => N());
    const V = Array.from({ length: 1 << K }, () => Array.from({ length: Dout }, () => N()));
    return { W, b, V, cfg };
  }

  const dot = (a, c) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * c[i]; return s; };

  // ---- exact LUT -----------------------------------------------------------
  function lutPreacts(m, x) { return m.W.map((Wk, k) => dot(Wk, x) + m.b[k]); } // <W_k,x>+b_k
  function lutBits(m, x) { return lutPreacts(m, x).map((a) => (a > 0 ? 1 : 0)); }
  function bitsToRow(bits) { let r = 0; for (let k = 0; k < bits.length; k++) r = (r << 1) | bits[k]; return r; } // MSB-first
  function lutForward(m, x) { const bits = lutBits(m, x); const row = bitsToRow(bits); return { bits, row, out: m.V[row] }; }

  // ---- spiking analogue ----------------------------------------------------
  const latencies = (m, x) => x.map((xi) => m.cfg.ALPHA + m.cfg.BETA * xi); // t_i = alpha + beta*x_i (bigger x -> LATER)
  const sumW = (m) => m.W.map((Wk) => Wk.reduce((a, c) => a + c, 0));
  function thetas(m) {
    // new encoding (t_i = alpha + beta*x) negates the data term in V_k(T_read); with
    // detector weights Wd = -W below, theta_k = -(T_read-alpha)*sumW_k - beta*b_k keeps
    // V_k(T_read) - theta_k = beta*(<W_k,x> + b_k) exactly, so bit = sign(<W_k,x>+b_k).
    const { ALPHA, BETA, T_READ } = m.cfg, sw = sumW(m);
    return sw.map((s, k) => -(T_READ - ALPHA) * s - BETA * m.b[k]);
  }
  // detector membrane V_k(t) = sum_i Wd[k,i]*(t - t_i)*[t >= t_i], Wd = -W (see thetas)
  function membrane(m, x, k, t) {
    const tl = latencies(m, x); let v = 0;
    for (let i = 0; i < tl.length; i++) if (t >= tl[i]) v += -m.W[k][i] * (t - tl[i]); // Wd = -W
    return v;
  }
  const membraneAtReadout = (m, x, k) => membrane(m, x, k, m.cfg.T_READ);
  function spikingBits(m, x) { const th = thetas(m); return m.W.map((_, k) => (membraneAtReadout(m, x, k) > th[k] ? 1 : 0)); }
  function spikingForward(m, x) { const bits = spikingBits(m, x); const row = bitsToRow(bits); return { bits, row, out: m.V[row] }; }

  // ---- REAL output spikes --------------------------------------------------
  // Detector k emits a spike iff its readout margin  V_k(T_read) - theta_k > 0
  // (== beta*(<W_k,x>+b_k) > 0, i.e. exactly the LUT bit). The spike TIME is a
  // decreasing function of the margin, clamped into (T_read, T_read+GAP], so a
  // larger margin fires EARLIER. The set of firing detectors is the bit pattern;
  // their firing ORDER is the descending-margin order the LUT math predicts.
  function outputSpikes(m, x) {
    const th = thetas(m), { T_READ, GAP, EPS, C } = m.cfg;
    return m.W.map((_, k) => {
      const margin = membraneAtReadout(m, x, k) - th[k];   // = beta*(<W_k,x>+b_k)
      const fired = margin > 0;
      let t = T_READ + GAP - C * margin;
      t = Math.max(T_READ + EPS, Math.min(T_READ + GAP, t));
      return { k, margin, fired, t_out: fired ? t : null };
    });
  }
  // detectors that fired, ordered by spike time (earliest first)
  function spikeOrder(m, x) {
    return outputSpikes(m, x).filter((s) => s.fired).sort((a, b) => a.t_out - b.t_out).map((s) => s.k);
  }
  // detectors ordered by LUT margin (largest first) — the ground-truth order
  function marginOrder(m, x) {
    return outputSpikes(m, x).filter((s) => s.fired).sort((a, b) => b.margin - a.margin).map((s) => s.k);
  }

  // ---- the FULL 4-layer spiking pipeline -----------------------------------
  // input spikes -> detector/address spikes -> ONE row neuron fires (1-hot
  // spiking address decoder) -> Dout output neurons whose spike LATENCIES encode
  // the stored row vector O[r]. Decoded output equals O[r] to fp precision.
  function fullPipeline(m, x) {
    const { K, T_READ, GAP, TC_OFF, ROWD, DBASE, KAPPA } = m.cfg;
    const tin = latencies(m, x);                         // layer 1: input latencies
    const detectors = outputSpikes(m, x);               // layer 2: address bits
    const bits = detectors.map((s) => (s.fired ? 1 : 0));
    const r = bitsToRow(bits);

    // layer 3: honest 1-hot decoder. Each detector contributes ONE active
    // polarity line: a "1" line at its spike time if it fired, else a "0"
    // complement line at TC (fixed, just after the address window closes).
    const TC = T_READ + GAP + TC_OFF;
    const lines = detectors.map((s) =>
      s.fired ? { k: s.k, polarity: 1, t: s.t_out } : { k: s.k, polarity: 0, t: TC });
    // every row neuron a integrates the polarity lines matching its address
    // pattern; it fires only on full K-way coincidence (count === K).
    const rows = [];
    for (let a = 0; a < (1 << K); a++) {
      const abits = []; for (let k = 0; k < K; k++) abits.push((a >> (K - 1 - k)) & 1);
      let count = 0; for (let k = 0; k < K; k++) if (abits[k] === bits[k]) count++;
      rows.push({ a, abits, count, fires: count === K });
    }
    const t_row = Math.max.apply(null, lines.map((l) => l.t)) + ROWD;  // winner's coincidence time

    // layer 4: output neurons. The winner drives output j through a stored
    // per-(row,j) delay = DBASE - KAPPA*O[r][j]; bigger value -> earlier spike.
    const O = m.V[r];
    const outputs = O.map((val, j) => {
      const delay = DBASE - KAPPA * val;
      const t = t_row + delay;
      const ohat = (DBASE - (t - t_row)) / KAPPA;   // decode; == val exactly
      return { j, val, delay, t, ohat };
    });
    return { tin, detectors, bits, r, TC, lines, rows, t_row, outputs,
             Ohat: outputs.map((o) => o.ohat), Otrue: O };
  }

  // ---- piecewise-linear membrane profile (for plotting + causal readout) ---
  // Returns the breakpoints {t, v} of V_k(t) over [0, T_read]: the membrane is
  // exactly piecewise-linear with a kink at every spike arrival t_i.
  function membraneProfile(m, x, k) {
    const tl = latencies(m, x);
    const order = tl.map((_, i) => i).sort((a, c) => tl[a] - tl[c]); // arrival order
    const T = m.cfg.T_READ;
    const pts = [{ t: 0, v: 0 }]; // resting membrane before the first spike
    let v = 0, cum = 0, tprev = null;
    for (let j = 0; j < order.length; j++) {
      const i = order[j], ti = tl[i];
      if (tprev === null) { pts.push({ t: ti, v: 0 }); cum = -m.W[k][i]; tprev = ti; }        // Wd = -W
      else { v += cum * (ti - tprev); pts.push({ t: ti, v }); cum += -m.W[k][i]; tprev = ti; } // Wd = -W
    }
    v += cum * (T - tprev);            // final segment to the readout time
    pts.push({ t: T, v });
    return pts;
  }

  // ---- naive FIRST-CROSSING (causal) readout -------------------------------
  // A real IF neuron fires the instant its membrane first exceeds theta, using
  // only the spikes that have arrived so far. Because the cumulative ramp slope
  // can go negative, the membrane can overshoot theta early and drift back — so
  // this causal bit sometimes DIFFERS from the fixed-window bit. That is exactly
  // why the exact transfer needs the fixed-window (all-spikes-arrived) readout.
  // Returns { bit, crossT } where crossT is the first crossing time (or null).
  function firstCrossing(m, x, k) {
    const th = thetas(m)[k];
    const pts = membraneProfile(m, x, k);
    for (let j = 1; j < pts.length; j++) {
      const a = pts[j - 1], c = pts[j];
      const ga = a.v - th, gc = c.v - th;
      if (ga <= 0 && gc > 0) {                 // upward crossing inside this segment
        const frac = gc === ga ? 0 : -ga / (gc - ga);
        return { bit: 1, crossT: a.t + frac * (c.t - a.t) };
      }
    }
    // never rose above threshold — check the resting/initial side too
    return { bit: pts[0].v > th ? 1 : 0, crossT: null };
  }
  function firstCrossBits(m, x) { return m.W.map((_, k) => firstCrossing(m, x, k).bit); }

  // ---- PURE SPIKING CIRCUIT: a genuine event-driven simulator ---------------
  // Nothing is "logic on top": bits / row / output EMERGE from a network of
  // leak-free IF neurons wired by synapses (each with a weight AND a conduction
  // delay). Neurons: START, inputs x_i, CLOCK, detectors H_k, complements C_k,
  // row neurons r_a, outputs o_j. Verified to reproduce the LUT to fp precision.
  function simulateCircuit(m, x) {
    const { D, K, Dout, ALPHA, BETA, T_READ, DBASE, KAPPA } = m.cfg;
    const THETA = 50, TC = T_READ + 1.5, INH = -20, SYND = 0.05, OW = 5;
    const th = thetas(m);
    const syn = {};
    const add = (pre, post, kind, weight, delay) => { (syn[pre] || (syn[pre] = [])).push({ post, kind, weight, delay }); };
    for (let k = 0; k < K; k++) {
      for (let i = 0; i < D; i++) add('x' + i, 'H' + k, 'ramp', -m.W[k][i], 0);   // input ramps, Wd = -W
      add('CLK', 'H' + k, 'step', THETA - th[k], 0);                            // calibrated clock pulse
      add('CLK', 'C' + k, 'step', 1, TC - T_READ);                             // complement excitation
      add('H' + k, 'C' + k, 'step', INH, SYND);                                // feedforward inhibition
    }
    add('START', 'CLK', 'step', 1, T_READ);                                     // clock = fixed delay from start
    for (let a = 0; a < (1 << K); a++) {
      for (let k = 0; k < K; k++) {
        const bit = (a >> (K - 1 - k)) & 1;
        add((bit ? 'H' : 'C') + k, 'r' + a, 'step', 1, SYND);                  // polarity lines -> row
      }
      for (let j = 0; j < Dout; j++) add('r' + a, 'o' + j, 'step', OW, DBASE - KAPPA * m.V[a][j]); // value in the delay
    }
    const thr = { START: 0.5, CLK: 0.5 };
    for (let i = 0; i < D; i++) thr['x' + i] = 0.5;
    for (let k = 0; k < K; k++) { thr['H' + k] = THETA; thr['C' + k] = 0.5; }
    for (let a = 0; a < (1 << K); a++) thr['r' + a] = 2.5;                       // 3-way coincidence
    for (let j = 0; j < Dout; j++) thr['o' + j] = 0.5;
    const st = {}; for (const n in thr) st[n] = { V: 0, slope: 0, last: 0, fired: false, tf: null };
    // per-neuron piecewise-linear membrane trace: [{t, V}] breakpoints (for the fill viz)
    const traces = {}; for (const n in thr) traces[n] = [{ t: 0, V: 0 }];

    const ev = []; let seq = 0;
    const events = [];   // propagation trace: every spike that travelled a synapse
    const push = (t, post, kind, weight) => ev.push({ t, seq: seq++, post, kind, weight });
    function fire(n, t) {
      st[n].fired = true; st[n].tf = t;
      const outs = syn[n] || [];
      for (const s of outs) {
        push(t + s.delay, s.post, s.kind, s.weight);
        events.push({ src: n, dst: s.post, tDepart: t, tArrive: t + s.delay, kind: s.kind, weight: s.weight });
      }
    }
    fire('START', 0);
    traces['START'] = [{ t: 0, V: thr['START'] }];   // source: charged at t=0
    const tin = [];
    for (let i = 0; i < D; i++) {
      const ti = ALPHA + BETA * x[i]; tin.push(ti); fire('x' + i, ti);   // t_i = alpha + beta*x
      traces['x' + i] = [{ t: 0, V: 0 }, { t: ti, V: thr['x' + i] }];     // source: fires at t_i
    }
    // event loop: pop the earliest (t, seq); integrate ramps to t; apply; threshold
    while (ev.length) {
      let mi = 0;
      for (let e = 1; e < ev.length; e++) if (ev[e].t < ev[mi].t || (ev[e].t === ev[mi].t && ev[e].seq < ev[mi].seq)) mi = e;
      const cur = ev.splice(mi, 1)[0];
      const n = st[cur.post];
      if (n.fired) continue;
      n.V += n.slope * (cur.t - n.last); n.last = cur.t;
      if (cur.kind === 'ramp') n.slope += cur.weight; else n.V += cur.weight;
      traces[cur.post].push({ t: cur.t, V: n.V });   // breakpoint (V is piecewise-linear between these)
      if (n.V > thr[cur.post]) fire(cur.post, cur.t);
    }
    // DECODE FROM EMERGENT SPIKES (not from any analytic margin)
    const bits = []; for (let k = 0; k < K; k++) bits.push(st['H' + k].fired ? 1 : 0);
    let row = -1; for (let a = 0; a < (1 << K); a++) if (st['r' + a].fired) row = a;
    const t_row = row >= 0 ? st['r' + row].tf : null;
    const outputs = [];
    for (let j = 0; j < Dout; j++) {
      const tf = st['o' + j].tf;
      outputs.push({ j, tf, ohat: (tf != null && t_row != null) ? (DBASE - (tf - t_row)) / KAPPA : null });
    }
    return { st, tin, bits, row, t_row, TC, events, traces, thr,
             Ohat: outputs.map((o) => o.ohat), Otrue: row >= 0 ? m.V[row] : null, outputs };
  }

  // ---- static circuit topology (the FULL fixed wiring, no simulation) -------
  // Mirrors the synapse construction in simulateCircuit but returns just the edge
  // list (src, dst, kind) — a STABLE skeleton that never changes with the input.
  // Every non-source node has incoming edges; every non-output node has outgoing.
  function circuitTopology(m) {
    const { D, K, Dout, T_READ, DBASE, KAPPA } = m.cfg;
    const THETA = 50, INH = -20;              // must match simulateCircuit's constants
    const th = thetas(m);
    const edges = [];
    const add = (src, dst, kind, weight) => edges.push({ src, dst, kind, weight });
    for (let k = 0; k < K; k++) {
      for (let i = 0; i < D; i++) add('x' + i, 'H' + k, 'ramp', -m.W[k][i]);   // inputs -> detectors (Wd=-W)
      add('CLK', 'H' + k, 'step', THETA - th[k]);                              // clock gate pulse
      add('CLK', 'C' + k, 'step', 1);                                          // complement excitation
      add('H' + k, 'C' + k, 'step', INH);                                      // feedforward inhibition
    }
    add('START', 'CLK', 'step', 1);
    for (let a = 0; a < (1 << K); a++) {
      for (let k = 0; k < K; k++) {
        const bit = (a >> (K - 1 - k)) & 1;
        add((bit ? 'H' : 'C') + k, 'r' + a, 'step', 1);                        // polarity lines -> row
      }
      for (let j = 0; j < Dout; j++) add('r' + a, 'o' + j, 'step', DBASE - KAPPA * m.V[a][j]); // rows -> outputs (delay-coded value)
    }
    return edges;
  }

  // ---- weight quantization (the precision floor) ---------------------------
  function quantizeArray(A, bits) {
    if (bits >= 30) return A.slice();
    let mx = 0; for (const v of A) mx = Math.max(mx, Math.abs(v));
    if (mx === 0) return A.slice();
    const levels = Math.pow(2, bits - 1) - 1, scale = mx / levels;
    return A.map((v) => Math.round(v / scale) * scale);
  }
  function quantizeModel(m, bits) {
    return {
      W: m.W.map((Wk) => quantizeArray(Wk, bits)),
      b: quantizeArray(m.b, bits),
      V: m.V, // the stored values aren't part of the addressing
      cfg: m.cfg,
    };
  }
  // fidelity of a b-bit model vs full precision, over `n` random inputs
  function quantFidelity(mFull, bits, n, seed) {
    if (bits >= 30) return { bit: 1, row: 1 };
    const mQ = quantizeModel(mFull, bits);
    const rng = mulberry32(seed || 12345);
    let bitOK = 0, rowOK = 0, totBits = 0;
    for (let s = 0; s < n; s++) {
      const x = Array.from({ length: mFull.cfg.D }, () => 2 * rng() - 1);
      const bf = lutBits(mFull, x), bq = lutBits(mQ, x);
      for (let k = 0; k < bf.length; k++) { totBits++; if (bf[k] === bq[k]) bitOK++; }
      if (bitsToRow(bf) === bitsToRow(bq)) rowOK++;
    }
    return { bit: bitOK / totBits, row: rowOK / n };
  }

  global.LUTSpiking = {
    CFG, buildModel, dot,
    lutPreacts, lutBits, bitsToRow, lutForward,
    latencies, sumW, thetas, membrane, membraneAtReadout, membraneProfile,
    spikingBits, spikingForward, firstCrossing, firstCrossBits,
    outputSpikes, spikeOrder, marginOrder, fullPipeline, simulateCircuit, circuitTopology,
    quantizeModel, quantFidelity,
  };
})(typeof window !== 'undefined' ? window : globalThis);
