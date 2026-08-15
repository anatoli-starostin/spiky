import * as THREE from 'three'

// ---------------------------------------------------------------------------
// Scene / renderer / camera
// ---------------------------------------------------------------------------
const app = document.getElementById('app')
const renderer = new THREE.WebGLRenderer({ antialias: true })
renderer.setPixelRatio(Math.min(devicePixelRatio, 2))
renderer.shadowMap.enabled = true
app.appendChild(renderer.domElement)

const scene = new THREE.Scene()
scene.background = new THREE.Color(0x0e1116)
scene.fog = new THREE.Fog(0x0e1116, 12, 40)

const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 200)
camera.position.set(0, 1.6, 6)

scene.add(new THREE.HemisphereLight(0xbcd3ff, 0x20242c, 0.9))
const sun = new THREE.DirectionalLight(0xffffff, 1.1)
sun.position.set(4, 10, 6); sun.castShadow = true
sun.shadow.mapSize.set(1024, 1024)
sun.shadow.camera.left = -20; sun.shadow.camera.right = 20
sun.shadow.camera.top = 20; sun.shadow.camera.bottom = -20
scene.add(sun)

// Ground: a long strip along +x (the walking direction) + a grid for motion cues.
const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(400, 12),
  new THREE.MeshStandardMaterial({ color: 0x161b24, roughness: 1 }))
ground.rotation.x = -Math.PI / 2
ground.receiveShadow = true
scene.add(ground)
const grid = new THREE.GridHelper(400, 400, 0x2a3444, 0x1b2230)
grid.position.y = 0.001
scene.add(grid)

// ---------------------------------------------------------------------------
// Walker2d skeleton — box/capsule bones we reposition each frame from qpos.
// Approximate MJCF segment lengths (metres); signs of joints are approximate.
// ---------------------------------------------------------------------------
const LEN = { torso: 0.5, thigh: 0.45, shin: 0.5, foot: 0.19 }
const DEPTH = 0.13                 // left/right leg offset in z (pseudo-3D from a 2D walker)

function bone(radius, color) {
  const m = new THREE.Mesh(
    new THREE.CylinderGeometry(radius, radius, 1, 10),
    new THREE.MeshStandardMaterial({ color, roughness: 0.6, metalness: 0.05 }))
  m.castShadow = true
  scene.add(m)
  return m
}
const walker = new THREE.Group(); scene.add(walker)
const bones = {
  torso: bone(0.06, 0x6ea8ff),
  rThigh: bone(0.045, 0xffb454), rShin: bone(0.04, 0xffb454), rFoot: bone(0.04, 0xff8c42),
  lThigh: bone(0.045, 0x8affc1), lShin: bone(0.04, 0x8affc1), lFoot: bone(0.04, 0x42d392),
}
const head = new THREE.Mesh(new THREE.SphereGeometry(0.09, 16, 12),
  new THREE.MeshStandardMaterial({ color: 0x6ea8ff, roughness: 0.5 }))
head.castShadow = true; scene.add(head)

const UP = new THREE.Vector3(0, 1, 0)
const _a = new THREE.Vector3(), _b = new THREE.Vector3(), _d = new THREE.Vector3()
function placeBone(mesh, p1, p2, thick) {
  _a.set(p1[0], p1[1], p1[2]); _b.set(p2[0], p2[1], p2[2])
  _d.subVectors(_b, _a)
  const len = Math.max(_d.length(), 1e-4)
  mesh.position.copy(_a).addScaledVector(_d, 0.5)
  mesh.scale.set(1, len, 1)
  mesh.quaternion.setFromUnitVectors(UP, _d.clone().normalize())
  if (thick) mesh.scale.set(thick, len, thick)
}

// Forward kinematics in the MuJoCo x-z plane, then mapped to three (x, z=up, depthOffset).
// seg(from, ang, len): ang measured from vertical +z, positive tilts toward +x.
function seg(from, ang, len) { return [from[0] + Math.sin(ang) * len, from[1] + Math.cos(ang) * len] }

function poseFromQpos(q) {
  // q = [x, z, torsoRot, rThigh, rLeg, rFoot, lThigh, lLeg, lFoot]
  const x = q[0], z = q[1], rot = q[2]
  const hip = [x, z]
  const torsoTop = seg(hip, rot, LEN.torso)
  function leg(thigh, leg_, foot) {
    const thighA = rot + Math.PI - thigh
    const knee = seg(hip, thighA, LEN.thigh)
    const shinA = thighA - leg_
    const ankle = seg(knee, shinA, LEN.shin)
    const footA = shinA - Math.PI / 2 + foot
    const toe = seg(ankle, footA, LEN.foot)
    return { knee, ankle, toe }
  }
  const R = leg(q[3], q[4], q[5]), L = leg(q[6], q[7], q[8])
  return { hip, torsoTop, R, L, torsoX: x }
}

// map a 2D (mx, mz) point + depth -> three.js [x, y=up, z=depth]
const P = (p, d) => [p[0], p[1], d]

function renderPose(pose) {
  placeBone(bones.torso, P(pose.hip, 0), P(pose.torsoTop, 0))
  head.position.set(pose.torsoTop[0], pose.torsoTop[1] + 0.02, 0)
  const legs = [['r', pose.R, +DEPTH], ['l', pose.L, -DEPTH]]
  for (const [k, s, d] of legs) {
    const cap = k.toUpperCase()
    placeBone(bones[`${k}Thigh`], P(pose.hip, d), P(s.knee, d))
    placeBone(bones[`${k}Shin`], P(s.knee, d), P(s.ankle, d))
    placeBone(bones[`${k}Foot`], P(s.ankle, d), P(s.toe, d))
  }
}

// ---------------------------------------------------------------------------
// ACCURATE render path — real Walker2d MJCF geometry (gymnasium envs/mujoco/assets/walker2d.xml).
// Segments are the exact capsules; bodies placed by faithful planar (x-z) MuJoCo FK: every limb joint is a
// hinge about -y applied ABOUT its own anchor, torso pitch (rooty) about +y. qpos (radians) =
//   [x, z, torso_pitch, thighR, legR, footR, thighL, legL, footL].
// Capsule spec per body: r=radius, hl=half cylinder length, cap=[A,B] hemisphere-centre endpoints in body
// local x-z; bodyPos = body offset from parent; jointPos = hinge anchor in body local. Values read straight
// from walker2d.xml (capsule size="r hl"; foot capsule is rotated -90deg about y so it lies along -x).
// ---------------------------------------------------------------------------
const MJCF = {
  torso: { r: 0.05, hl: 0.20,  cap: [[0, 0.20], [0, -0.20]] },
  thigh: { r: 0.05, hl: 0.225, cap: [[0, 0.0],  [0, -0.45]], bodyPos: [0, -0.20], jointPos: [0, 0] },
  leg:   { r: 0.04, hl: 0.25,  cap: [[0, 0.25], [0, -0.25]], bodyPos: [0, -0.70], jointPos: [0, 0.25] },
  foot:  { r: 0.06, hl: 0.10,  cap: [[-0.20, 0.10], [0.0, 0.10]], bodyPos: [0.20, -0.35], jointPos: [-0.20, 0.10] },
}
const rotY = (a, v) => { const c = Math.cos(a), s = Math.sin(a); return [v[0] * c + v[1] * s, -v[0] * s + v[1] * c] }
const applyF = (f, p) => { const r = rotY(f.a, p); return [r[0] + f.p[0], r[1] + f.p[1]] }
const composeF = (parent, rel) => ({ a: parent.a + rel.a, p: applyF(parent, rel.p) })
// child frame relative to parent: translate by bodyPos, then rotate `angle` about the jointPos anchor.
function relF(bodyPos, jointPos, angle) {
  const rj = rotY(angle, jointPos)
  return { a: angle, p: [bodyPos[0] + jointPos[0] - rj[0], bodyPos[1] + jointPos[1] - rj[1]] }
}
function accurateFrames(q) {
  const torso = { p: [q[0], q[1]], a: q[2] }
  const oneLeg = (t, l, f) => {
    const thigh = composeF(torso, relF(MJCF.thigh.bodyPos, MJCF.thigh.jointPos, -t))  // hinge about -y
    const leg = composeF(thigh, relF(MJCF.leg.bodyPos, MJCF.leg.jointPos, -l))
    const foot = composeF(leg, relF(MJCF.foot.bodyPos, MJCF.foot.jointPos, -f))
    return { thigh, leg, foot }
  }
  return { torso, R: oneLeg(q[3], q[4], q[5]), L: oneLeg(q[6], q[7], q[8]) }
}
// Accurate bones are TRUE capsules of the real length (rigid segments) — position + orient only, no scaling,
// so the hemispherical caps stay undistorted and match MuJoCo. No head: the MJCF torso is just a capsule.
function capsuleBone(r, hl, color) {
  const m = new THREE.Mesh(new THREE.CapsuleGeometry(r, 2 * hl, 6, 14),
    new THREE.MeshStandardMaterial({ color, roughness: 0.55, metalness: 0.05 }))
  m.castShadow = true; scene.add(m); return m
}
const accBones = {
  torso: capsuleBone(MJCF.torso.r, MJCF.torso.hl, 0x6ea8ff),
  rThigh: capsuleBone(MJCF.thigh.r, MJCF.thigh.hl, 0xffb454),
  rShin:  capsuleBone(MJCF.leg.r,   MJCF.leg.hl,   0xffb454),
  rFoot:  capsuleBone(MJCF.foot.r,  MJCF.foot.hl,  0xff8c42),
  lThigh: capsuleBone(MJCF.thigh.r, MJCF.thigh.hl, 0x8affc1),
  lShin:  capsuleBone(MJCF.leg.r,   MJCF.leg.hl,   0x8affc1),
  lFoot:  capsuleBone(MJCF.foot.r,  MJCF.foot.hl,  0x42d392),
}
const approxMeshes = [bones.torso, bones.rThigh, bones.rShin, bones.rFoot, bones.lThigh, bones.lShin, bones.lFoot, head]
const accMeshes = Object.values(accBones)

function placeCapsule(mesh, aW, bW) {
  _a.set(aW[0], aW[1], aW[2]); _b.set(bW[0], bW[1], bW[2]); _d.subVectors(_b, _a)
  mesh.position.copy(_a).addScaledVector(_d, 0.5)
  mesh.quaternion.setFromUnitVectors(UP, _d.clone().normalize())   // no length scaling: segments are rigid
}
function placeCap(mesh, frame, geom, d) {
  const a = applyF(frame, geom.cap[0]), b = applyF(frame, geom.cap[1])
  placeCapsule(mesh, [a[0], a[1], d], [b[0], b[1], d])
}
function renderAccurate(q) {
  const F = accurateFrames(q)
  placeCap(accBones.torso, F.torso, MJCF.torso, 0)
  for (const [k, s, d] of [['r', F.R, +DEPTH], ['l', F.L, -DEPTH]]) {
    placeCap(accBones[`${k}Thigh`], s.thigh, MJCF.thigh, d)
    placeCap(accBones[`${k}Shin`],  s.leg,   MJCF.leg,   d)
    placeCap(accBones[`${k}Foot`],  s.foot,  MJCF.foot,  d)
  }
}

// Render-mode toggle (Accurate MJCF geometry vs the original Approximate one). Default: accurate.
let renderMode = 'accurate'
function applyRenderVisibility() {
  const acc = renderMode === 'accurate'
  for (const m of accMeshes) m.visible = acc
  for (const m of approxMeshes) m.visible = !acc
}
function renderFrame(q) {
  if (renderMode === 'accurate') renderAccurate(q)
  else renderPose(poseFromQpos(q))
}
applyRenderVisibility()

// ---------------------------------------------------------------------------
// State + smoothing + camera follow
// ---------------------------------------------------------------------------
// Render is DECOUPLED from the WS message rate: we keep the last two received qpos and interpolate between
// them every animation frame (rAF ~60fps), so a ~30 sps stream with network jitter still looks smooth.
let fromQ = null, toQ = null, interpStart = 0, interpDur = 1000 / 30
let emaInterval = 1000 / 30, lastMsg = 0, prevStep = -1
const curQ = new Array(9).fill(0)                    // reused each frame (no per-frame allocation)

function pushState(qpos, step, sps) {
  const now = performance.now()
  const nominal = 1000 / Math.max(1, sps || 30)      // the stream's real inter-frame interval at this sps
  const gap = lastMsg ? now - lastMsg : nominal
  // A gap far larger than the stream cadence means the server was SILENT (paused, or auto-stopped to the
  // zero idle) and has just resumed. Do NOT fold that idle gap into the smoothed interval: it would blow up
  // interpDur and play the next move in extreme slow motion (a 3-min idle -> ~36 s lerp). Reset to the
  // nominal cadence and snap the pose instead, so resume is immediate and smooth.
  const resumed = gap > Math.max(1200, nominal * 4)
  if (resumed) emaInterval = nominal
  else if (lastMsg) emaInterval += (gap - emaInterval) * 0.2   // smoothed inter-arrival interval
  lastMsg = now
  // On an episode reset OR a resume-from-idle the pose jumps discontinuously — snap instead of sliding.
  const reset = resumed || (prevStep >= 0 && (step <= prevStep || (toQ && Math.abs(qpos[0] - toQ[0]) > 0.4)))
  prevStep = step
  fromQ = reset ? qpos.slice() : curQ.slice()        // start from where we're currently drawn -> no snapping
  toQ = qpos
  interpStart = now
  interpDur = Math.max(16, emaInterval)              // aim to finish just as the next state arrives
}

let camX = 0, poseX = 0
const camOffset = new THREE.Vector3(0.5, 1.3, 5.5)

function animate() {
  requestAnimationFrame(animate)
  if (fromQ && toQ) {
    const a = interpDur > 0 ? Math.min(1, (performance.now() - interpStart) / interpDur) : 1
    for (let i = 0; i < 9; i++) curQ[i] = fromQ[i] + (toQ[i] - fromQ[i]) * a   // lerp root + joint angles
    renderFrame(curQ)                                    // dispatches to accurate (MJCF) or approximate path
    poseX = curQ[0]                                      // torso x drives the camera follow
  }
  camX += (poseX - camX) * 0.08                       // camera follows with smoothing
  camera.position.set(camX + camOffset.x, camOffset.y, camOffset.z)
  camera.lookAt(camX, 0.9, 0)
  renderer.render(scene, camera)
}
animate()

function resize() {
  const w = innerWidth, h = innerHeight
  renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix()
}
addEventListener('resize', resize); resize()

// ---------------------------------------------------------------------------
// WebSocket + UI
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id)
const conn = $('conn')
let ws = null, paused = false, serverFull = false
// Remembered user intent, re-applied after a (re)connect so a dropped socket doesn't silently strand
// the session on the server's default actor with unresponsive controls.
let lastActor = null, lastPaused = false, lastNoReset = false, lastShowSpikes = false, lastShowNetwork = false
let sockId = 0

function send(obj) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)) }

// ---- live spike raster (spiking-LUT actor only) ----
let spikeMeta = null
const spikeCanvas = $('spikeCanvas'), sctx = spikeCanvas ? spikeCanvas.getContext('2d') : null
function setupSpikeInfo() {
  if (spikeMeta) $('spikeInfo').textContent = `${spikeMeta.n_rows} neurons · ${spikeMeta.n_ticks} ticks/inference`
}
function bandColorFor(row) {
  if (spikeMeta) for (const b of spikeMeta.bands) if (row >= b.start && row < b.end) return b.color
  return '#8ab0d0'
}
// `a` = flat (row uint16, tick uint16) pairs for one act(). Redraw the whole raster each frame.
function drawSpikes(a) {
  if (!sctx || !spikeMeta) return
  const W = spikeCanvas.width, H = spikeCanvas.height
  const PADL = 92, PADR = 8, PADT = 6, PADB = 6
  const pw = W - PADL - PADR, ph = H - PADT - PADB
  const nt = spikeMeta.n_ticks
  sctx.clearRect(0, 0, W, H)
  sctx.textAlign = 'right'; sctx.textBaseline = 'middle'; sctx.font = '10px sans-serif'
  // Band-weighted vertical layout. A uniform row/nr mapping gave the dense S2-cells band
  // (~2048 of ~2889 rows) ~71% of the height and squeezed the 272-row green memory band into
  // ~9% (~0.13 px/row) — far thinner than the dot, so adjacent memory rows blurred together.
  // Instead cap the dense band(s) to a thin strip and share the rest by row count, so the
  // sparse, interesting bands (memory, rails, inputs, outputs) get ~1 px per row and the
  // per-neuron firing is legible. Rows stay linear within each band.
  const bands = spikeMeta.bands
  let sparseRows = 0, denseRows = 0
  for (const b of bands) { const n = b.end - b.start; if (n > 400) denseRows += n; else sparseRows += n }
  const denseH = denseRows ? Math.min(0.14 * ph, 90) : 0        // dense S2-cells: a capped strip
  const perRow = (ph - denseH) / Math.max(1, sparseRows)        // px/row for the sparse bands
  const top = {}, rh = {}
  let y = PADT
  for (const b of bands) {
    const n = b.end - b.start
    const h = (n > 400) ? denseH : n * perRow
    top[b.name] = y; rh[b.name] = n ? h / n : 0
    y += h
  }
  for (const b of bands) {                                      // stage bands: faint tint + label
    const y0 = top[b.name], y1 = top[b.name] + (b.end - b.start) * rh[b.name]
    sctx.fillStyle = b.color + '22'
    sctx.fillRect(PADL, y0, pw, Math.max(1, y1 - y0))
    sctx.fillStyle = '#b9c6d6'
    sctx.fillText(b.name, PADL - 6, (y0 + y1) / 2)
  }
  for (let i = 0; i < a.length; i += 2) {                       // one dot per fired neuron at its tick
    const row = a[i], tick = a[i + 1]
    let bt = null
    for (const b of bands) if (row >= b.start && row < b.end) { bt = b; break }
    if (!bt) continue
    const yy = top[bt.name] + (row - bt.start) * rh[bt.name]
    const dh = Math.max(1, Math.min(2.4, rh[bt.name]))          // dot height <= its row height (no overspill)
    sctx.fillStyle = bt.color
    sctx.fillRect(PADL + (tick / nt) * pw, yy, 2.2, dh)
  }
}

// ---- live NETWORK-GRAPH view (spiking-LUT actor only) — coexists with the raster ----
// Nodes are laid out in stage columns (inputs → S1 rails → S1 mem → S1 tie → S2 cells → S3 outputs).
// Static synapses are pre-rendered once to an offscreen canvas; a self-clocked loop sweeps a virtual
// tick 0..n_ticks and animates a dot along each synapse whose source fired (dot travels over the delay),
// so spikes visibly cascade left→right — the same idea, on the same tick axis, as the raster.
let netMeta = null, netEdges = null, netSrcIdx = null, netFire = null, netPos = null, netBg = null
let netSimT = 0, netLastTs = 0, netRunning = false, netRAF = 0
const netCanvas = $('netCanvas'), nctx = netCanvas ? netCanvas.getContext('2d') : null
const NET_TPS = 105, NET_END_PAD = 45, NET_FLASH = 5     // ticks/sec sweep, end pause, flash half-width

function netSetInfo() {
  if (netMeta) $('netInfo').textContent = `${netMeta.n_nodes} neurons · ${netMeta.n_edges} synapses · ${netMeta.n_ticks} ticks`
}
function netLayout() {
  if (!netMeta || !netCanvas) return
  const W = netCanvas.width, H = netCanvas.height, padL = 22, padR = 22, padT = 22, padB = 8
  const bands = netMeta.bands, nb = bands.length, N = netMeta.n_nodes
  const x = new Float32Array(N), y = new Float32Array(N)
  for (let b = 0; b < nb; b++) {
    const band = bands[b], size = band.end - band.start
    const cx = padL + (nb === 1 ? 0.5 : b / (nb - 1)) * (W - padL - padR)
    for (let r = band.start; r < band.end; r++) {
      x[r] = cx
      y[r] = padT + ((r - band.start) / Math.max(1, size - 1)) * (H - padT - padB)
    }
  }
  netPos = { x, y }
  netDrawBackground()
}
function netDrawBackground() {
  if (!netEdges || !netPos || !netCanvas) return
  const W = netCanvas.width, H = netCanvas.height
  netBg = document.createElement('canvas'); netBg.width = W; netBg.height = H
  const g = netBg.getContext('2d'); g.lineWidth = 0.5
  const { src, tgt, exc } = netEdges, n = src.length, { x, y } = netPos
  for (const isExc of [1, 0]) {                            // two batched passes (one strokeStyle each)
    g.strokeStyle = isExc ? 'rgba(255,107,107,0.05)' : 'rgba(91,157,255,0.13)'
    g.beginPath()
    for (let i = 0; i < n; i++) {
      if (exc[i] !== isExc) continue
      g.moveTo(x[src[i]], y[src[i]]); g.lineTo(x[tgt[i]], y[tgt[i]])
    }
    g.stroke()
  }
  g.fillStyle = '#9fb0c3'; g.font = '9px sans-serif'; g.textAlign = 'center'
  for (const b of netMeta.bands) g.fillText(b.name.replace(/^S\d /, ''), x[b.start], 12)
}
function onTopology(body) {                                // body = flat uint16 (src,tgt,delay,exc) quads
  if (!netMeta) return
  const n = (body.length / 4) | 0, N = netMeta.n_nodes
  const src = new Uint16Array(n), tgt = new Uint16Array(n), dly = new Uint16Array(n), exc = new Uint8Array(n)
  for (let i = 0; i < n; i++) { src[i] = body[i*4]; tgt[i] = body[i*4+1]; dly[i] = body[i*4+2]; exc[i] = body[i*4+3] }
  netEdges = { src, tgt, dly, exc }
  netSrcIdx = Array.from({ length: N }, () => [])
  for (let i = 0; i < n; i++) netSrcIdx[src[i]].push(i)
  netFire = new Int16Array(N).fill(-1)
  netLayout()
}
function netSetSpikes(a) {                                 // a = (row,tick) pairs; store firing tick per neuron
  if (!netFire) return
  netFire.fill(-1)
  for (let i = 0; i < a.length; i += 2) netFire[a[i]] = a[i + 1]
}
function netLoop(ts) {
  if (!netRunning) return
  netRAF = requestAnimationFrame(netLoop)
  if (!nctx || !netBg || !netPos || !netEdges || !netMeta) return
  const dt = netLastTs ? (ts - netLastTs) / 1000 : 0; netLastTs = ts
  netSimT += dt * NET_TPS
  if (netSimT > netMeta.n_ticks + NET_END_PAD) netSimT = 0
  const t = netSimT, W = netCanvas.width, H = netCanvas.height
  nctx.clearRect(0, 0, W, H); nctx.drawImage(netBg, 0, 0)
  const { x, y } = netPos, { src, tgt, dly, exc } = netEdges
  for (let r = 0; r < netFire.length; r++) {               // travelling spike dots on firing sources' edges
    const ft = netFire[r]; if (ft < 0) continue
    const ed = netSrcIdx[r]; if (!ed.length) continue
    for (let j = 0; j < ed.length; j++) {
      const i = ed[j], span = Math.max(dly[i], 4)
      if (t < ft || t > ft + span) continue
      const fr = (t - ft) / span, ax = x[src[i]], ay = y[src[i]]
      nctx.fillStyle = exc[i] ? '#ff8a8a' : '#7fb3ff'
      nctx.fillRect(ax + (x[tgt[i]] - ax) * fr - 1, ay + (y[tgt[i]] - ay) * fr - 1, 2.4, 2.4)
    }
  }
  for (let r = 0; r < netFire.length; r++) {               // flash nodes at their firing tick
    const ft = netFire[r]; if (ft < 0 || Math.abs(t - ft) > NET_FLASH) continue
    nctx.fillStyle = '#ffe08a'; nctx.fillRect(x[r] - 1.5, y[r] - 1.5, 3, 3)
  }
  nctx.fillStyle = '#9fb0c3'; nctx.font = '10px sans-serif'; nctx.textAlign = 'right'
  nctx.fillText('tick ' + Math.min(Math.floor(t), netMeta.n_ticks), W - 6, H - 4)
}
function netStart() { if (!netRunning) { netRunning = true; netLastTs = 0; netRAF = requestAnimationFrame(netLoop) } }
function netStop() { netRunning = false; if (netRAF) cancelAnimationFrame(netRAF); netRAF = 0 }

// ---- per-actor spiking-view adaptation ----
// The spike raster + network graph only make sense for actors that expose read_spikes (the server
// advertises which via the `spiking` list in its `actors` message + a `spiking` flag in `state`).
// On actor-switch we (a) clear both canvases + drop stale layout state so no frozen frame lingers,
// and (b) show/hide the spike+network controls & panels for the selected actor.
let spikingActors = null            // Set of actor names with the spike/network view; null until the server tells us
let curVizActor = null              // last actor we applied the viz UI for (avoid redundant work on every state msg)
function activeIsSpiking(name) { return !spikingActors || spikingActors.has(name) }   // unknown -> assume yes (don't hide)
function resetVizCanvases() {
  spikeMeta = null; netMeta = null; netEdges = null; netSrcIdx = null; netFire = null; netPos = null; netBg = null
  if (sctx) sctx.clearRect(0, 0, spikeCanvas.width, spikeCanvas.height)
  if (nctx) nctx.clearRect(0, 0, netCanvas.width, netCanvas.height)
}
function applyActorViz(name) {
  if (name == null || name === curVizActor) return
  curVizActor = name
  resetVizCanvases()                                   // kill any frozen frame from the previous actor
  const spk = activeIsSpiking(name)
  for (const id of ['showspikes', 'shownetwork']) {    // hide the toggles' rows when not applicable
    const cb = $(id); if (!cb) continue
    const row = cb.closest ? cb.closest('.row') : null
    if (row) row.style.display = spk ? 'flex' : 'none'
  }
  const note = $('vizNote'); if (note) note.style.display = spk ? 'none' : 'flex'
  if (!spk) {                                          // non-spiking: hide both panels, stop the net animation
    $('spikePanel').style.display = 'none'
    $('networkPanel').style.display = 'none'
    netStop()
  } else {                                             // spiking: restore panels per the user's checkbox state
    $('spikePanel').style.display = $('showspikes').checked ? 'block' : 'none'
    $('networkPanel').style.display = $('shownetwork').checked ? 'block' : 'none'
    if ($('shownetwork').checked) netStart()
  }
}

const overlay = $('overlay'), overlayMsg = $('overlayMsg')
function showOverlay(msg) { if (overlayMsg) overlayMsg.textContent = msg; if (overlay) overlay.style.display = 'flex' }
function hideOverlay() { if (overlay) overlay.style.display = 'none' }

function connect(url) {
  if (ws) { try { ws.close() } catch {} }
  serverFull = false; hideOverlay()
  conn.textContent = 'connecting…'; conn.className = ''
  // Bind every handler to THIS socket object (sk), and gate side-effects on `ws === sk` so a stale
  // socket (e.g. one that keeps delivering frames after a reconnect) can never drive the render or
  // schedule reconnects while the buttons' send() targets a different, current `ws`. Fixes the
  // "robot keeps walking but buttons dead" two-socket class of bug.
  const sk = new WebSocket(url); sk._id = ++sockId; ws = sk
  sk.binaryType = 'arraybuffer'                            // spike frames arrive as binary (ArrayBuffer)
  sk.onopen = () => { if (ws !== sk) return; conn.textContent = 'connected'; conn.className = 'ok'; hideOverlay() }
  sk.onclose = (e) => {
    if (ws !== sk) return                                  // a stale socket closing must NOT touch UI or reconnect
    conn.textContent = serverFull ? 'server full' : 'reconnecting…'; conn.className = 'bad'
    // When the server told us it's at capacity, do NOT auto-reconnect (no tight retry loop) — the
    // overlay's Retry button lets the visitor try again on their own terms. Otherwise reconnect.
    if (!serverFull) setTimeout(() => connect(url), 1500)
  }
  sk.onerror = () => { if (ws !== sk) return; conn.textContent = 'error'; conn.className = 'bad' }
  sk.onmessage = (ev) => {
    if (ws !== sk) return                                  // ignore frames from a superseded socket
    if (typeof ev.data !== 'string') {                     // binary frame: uint16 kind tag, then body
      const arr = new Uint16Array(ev.data), body = arr.subarray(1)
      if (arr[0] === 2) onTopology(body)                   // kind 2 = network topology (once)
      else { if (lastShowSpikes) drawSpikes(body); if (lastShowNetwork) netSetSpikes(body) }  // kind 1 = spikes
      return
    }
    const m = JSON.parse(ev.data)
    if (m.type === 'server_full') {                    // at capacity: friendly banner, stop reconnecting
      serverFull = true
      showOverlay(m.message || 'The demo server is at capacity, please come back later.')
      return
    }
    if (m.type === 'spike_meta') { spikeMeta = m; setupSpikeInfo(); return }   // raster layout (once)
    if (m.type === 'network_meta') { netMeta = m; netSetInfo(); return }       // graph layout (before topology)
    if (m.type === 'actor_changed') {                  // server auto-switched the model (auto-stop -> zero)
      const sel = $('actor')                           // reflect it in the selector so the user sees "zero".
      if (sel && m.actor) sel.value = m.actor          // programmatic set does NOT fire onchange -> not a user interaction
      $('actorName').textContent = m.actor
      applyActorViz(m.actor)                           // adapt the spike/network view to the (server-)switched actor
      return
    }
    if (m.type === 'actors') {
      const sel = $('actor'); sel.innerHTML = ''
      for (const name of m.actors) {
        const o = document.createElement('option'); o.value = o.textContent = name; sel.appendChild(o)
      }
      if (Array.isArray(m.spiking)) spikingActors = new Set(m.spiking)   // which actors have the spike/network view
      if (m.active) sel.value = m.active
      // Restore the user's chosen actor + toggles after a (re)connect (the fresh session starts on the
      // server default). No-op on the first connect (lastActor is null).
      if (lastActor && m.actors.includes(lastActor) && lastActor !== m.active) {
        sel.value = lastActor
        send({ cmd: 'actor', name: lastActor })
      }
      if (lastPaused) send({ cmd: 'pause', value: true })
      if (lastNoReset) send({ cmd: 'no_reset', value: true })
      if (lastShowSpikes) send({ cmd: 'show_spikes', value: true })
      if (lastShowNetwork) send({ cmd: 'show_network', value: true })
      curVizActor = null; applyActorViz(sel.value)         // set up the spike/network UI for the active actor
    } else if (m.type === 'state') {
      pushState(m.qpos, m.step, m.sps)                // update interpolation target (no geometry rebuild)
      $('env').textContent = m.env
      $('step').textContent = m.step
      $('reward').textContent = (+m.reward).toFixed(2)
      $('ret').textContent = (+m.return).toFixed(1)
      $('actorName').textContent = m.actor
      applyActorViz(m.actor)                         // no-op unless the active actor changed (guards on curVizActor)
      paused = !!m.paused                            // reflect server pause state
      $('pauseBtn').textContent = paused ? 'Resume' : 'Pause'
      $('pauseBtn').classList.toggle('active', paused)
      $('noreset').checked = !!m.no_reset            // reflect server free-fall state
    }
  }
}

$('restart').onclick = () => send({ cmd: 'restart' })
$('pauseBtn').onclick = () => {
  paused = !paused                                  // optimistic toggle; server echoes it back in state
  lastPaused = paused
  send({ cmd: 'pause', value: paused })
  $('pauseBtn').textContent = paused ? 'Resume' : 'Pause'
  $('pauseBtn').classList.toggle('active', paused)
}
$('renderBtn').onclick = () => {                          // live switch between accurate MJCF and approximate render
  renderMode = renderMode === 'accurate' ? 'approx' : 'accurate'
  applyRenderVisibility()
  $('renderBtn').textContent = 'Render: ' + (renderMode === 'accurate' ? 'Accurate' : 'Approx')
  $('renderBtn').classList.toggle('active', renderMode === 'accurate')
}

// Foldable controls panel — collapses to just the title bar so it doesn't overlap the robot on narrow screens.
const ui = document.getElementById('ui')
function setFold(collapsed) {
  ui.classList.toggle('collapsed', collapsed)
  $('foldBtn').innerHTML = collapsed ? '&#9656;' : '&#9662;'   // ▸ when collapsed, ▾ when expanded
  $('foldBtn').setAttribute('aria-label', collapsed ? 'Expand controls' : 'Collapse controls')
}
$('foldBtn').onclick = () => setFold(!ui.classList.contains('collapsed'))
setFold(innerWidth < 620)                                 // default: collapsed on narrow/mobile, expanded on wide
$('speed').oninput = (e) => { $('speedVal').textContent = e.target.value + '/s'; send({ cmd: 'speed', sps: +e.target.value }) }
$('actor').onchange = (e) => { lastActor = e.target.value; send({ cmd: 'actor', name: e.target.value }); applyActorViz(e.target.value) }
$('noreset').onchange = (e) => { lastNoReset = e.target.checked; send({ cmd: 'no_reset', value: e.target.checked }) }
$('showspikes').onchange = (e) => {
  lastShowSpikes = e.target.checked
  send({ cmd: 'show_spikes', value: e.target.checked })
  $('spikePanel').style.display = e.target.checked ? 'block' : 'none'
  if (!e.target.checked && sctx) sctx.clearRect(0, 0, spikeCanvas.width, spikeCanvas.height)
}
$('shownetwork').onchange = (e) => {
  lastShowNetwork = e.target.checked
  send({ cmd: 'show_network', value: e.target.checked })
  $('networkPanel').style.display = e.target.checked ? 'block' : 'none'
  if (e.target.checked) netStart(); else netStop()
}
$('srv').onchange = (e) => connect(e.target.value.trim())
$('overlayRetry').onclick = () => connect($('srv').value.trim())   // manual retry from the overload banner

// WebSocket URL resolution order:
//   1. window.WALKER2D_WS from config.js (set to wss://your-domain for a GitHub Pages build), else
//   2. the SAME host that served this page on port 8765 (local `http.server` / tailnet IP just works).
const configured = (typeof window !== 'undefined' && window.WALKER2D_WS) || ''
const defaultWs = configured || `ws://${location.hostname || 'localhost'}:8765`
$('srv').value = defaultWs
connect(defaultWs)
