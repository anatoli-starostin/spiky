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

function pushState(qpos, step) {
  const now = performance.now()
  if (lastMsg) emaInterval += ((now - lastMsg) - emaInterval) * 0.2   // smoothed inter-arrival interval
  lastMsg = now
  // On an episode reset the pose jumps discontinuously (root x snaps back) — snap instead of sliding.
  const reset = prevStep >= 0 && (step <= prevStep || (toQ && Math.abs(qpos[0] - toQ[0]) > 0.4))
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
let ws = null, paused = false

function send(obj) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)) }

function connect(url) {
  if (ws) { try { ws.close() } catch {} }
  conn.textContent = 'connecting…'; conn.className = ''
  ws = new WebSocket(url)
  ws.onopen = () => { conn.textContent = 'connected'; conn.className = 'ok' }
  ws.onclose = () => { conn.textContent = 'disconnected'; conn.className = 'bad'; setTimeout(() => connect(url), 1500) }
  ws.onerror = () => { conn.textContent = 'error'; conn.className = 'bad' }
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data)
    if (m.type === 'actors') {
      const sel = $('actor'); sel.innerHTML = ''
      for (const name of m.actors) {
        const o = document.createElement('option'); o.value = o.textContent = name; sel.appendChild(o)
      }
      if (m.active) sel.value = m.active
    } else if (m.type === 'state') {
      pushState(m.qpos, m.step)                       // update interpolation target (no geometry rebuild)
      $('env').textContent = m.env
      $('step').textContent = m.step
      $('reward').textContent = (+m.reward).toFixed(2)
      $('ret').textContent = (+m.return).toFixed(1)
      $('actorName').textContent = m.actor
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
$('actor').onchange = (e) => send({ cmd: 'actor', name: e.target.value })
$('noreset').onchange = (e) => send({ cmd: 'no_reset', value: e.target.checked })
$('srv').onchange = (e) => connect(e.target.value.trim())

// WebSocket URL resolution order:
//   1. window.WALKER2D_WS from config.js (set to wss://your-domain for a GitHub Pages build), else
//   2. the SAME host that served this page on port 8765 (local `http.server` / tailnet IP just works).
const configured = (typeof window !== 'undefined' && window.WALKER2D_WS) || ''
const defaultWs = configured || `ws://${location.hostname || 'localhost'}:8765`
$('srv').value = defaultWs
connect(defaultWs)
