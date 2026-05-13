/**
 * Skill Selector — compare view
 * Left: chat input + spatial vs semantic columns
 * Right: always-on Three.js pointcloud
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ── DOM refs ──────────────────────────────────────────────────────────────────

const $ = id => document.getElementById(id);
const queryInput      = $('query-input');
const searchBtn       = $('search-btn');
const compareEmpty    = $('compare-empty');
const compareCols     = $('compare-cols');
const spatialCards    = $('spatial-cards');
const semanticCards   = $('semantic-cards');
const domainTag       = $('domain-tag');
const mapCanvas       = $('map-canvas');
const mapTooltip      = $('map-tooltip');
const mapDomainLabel  = $('map-domain-label');
const loadingOverlay  = $('loading-overlay');
const skillCountLabel = $('skill-count-label');
const toggleWiresBtn  = $('toggle-wires-btn');
const modalOverlay    = $('modal-overlay');
const modalTitle      = $('modal-title');
const modalBadges     = $('modal-badges');
const modalDesc       = $('modal-description');
const modalContent    = $('modal-content');
const modalSourceLink = $('modal-source-link');
const modalDownloadLink = $('modal-download-link');

// ── State ─────────────────────────────────────────────────────────────────────

let currentDomain = null;
let isSearching   = false;
let lastSpatialResults = [];
let lastSemanticResults = [];

// ── Category colours ──────────────────────────────────────────────────────────

const CATEGORY_COLORS = [
  0x4A90D9, 0x6BC878, 0xE07AC8, 0xE8A23C,
  0x9B6BDB, 0x5EC8BC, 0xD96A6A, 0x8BBFD4,
  0xBBD45E, 0xD4845E, 0x7A9EDB, 0xC85E8B,
];
const catColorMap = {};
function getCatColor(cat) {
  if (!catColorMap[cat]) {
    const idx = Object.keys(catColorMap).length % CATEGORY_COLORS.length;
    catColorMap[cat] = CATEGORY_COLORS[idx];
  }
  return catColorMap[cat];
}
function colorToHex(c) { return '#' + c.toString(16).padStart(6, '0'); }

// ── Three.js setup ────────────────────────────────────────────────────────────

let renderer, scene, camera, controls;
let pointsGroup, wireGroup, queryGroup, lineGroup, domainWireGroup;
let allSkillPoints = [];
let allDomains     = [];
let resultSlugs    = new Set();
let wiresVisible   = false;
const SCALE = 10;

function initThree() {
  renderer = new THREE.WebGLRenderer({ canvas: mapCanvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x0F1923, 1);

  scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x0F1923, 0.016);

  camera = new THREE.PerspectiveCamera(55, 1, 0.1, 500);
  camera.position.set(0, 0, 38);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping  = true;
  controls.dampingFactor  = 0.07;
  controls.minDistance    = 2;
  controls.maxDistance    = 120;
  controls.autoRotate     = true;
  controls.autoRotateSpeed = 0.4;

  scene.add(new THREE.AmbientLight(0x8DA0BA, 0.6));
  const dir = new THREE.DirectionalLight(0xffffff, 0.8);
  dir.position.set(5, 10, 8);
  scene.add(dir);

  pointsGroup     = new THREE.Group();
  wireGroup       = new THREE.Group();
  domainWireGroup = new THREE.Group();
  queryGroup      = new THREE.Group();
  lineGroup       = new THREE.Group();
  wireGroup.visible = false;
  scene.add(pointsGroup, wireGroup, domainWireGroup, queryGroup, lineGroup);

  resizeRenderer();
  window.addEventListener('resize', resizeRenderer);

  renderer.domElement.addEventListener('mousemove', onMouseMove);
  renderer.domElement.addEventListener('click', onCanvasClick);

  animate();
}

function resizeRenderer() {
  if (!renderer) return;
  const panel = $('map-panel');
  const w = panel.clientWidth;
  const h = panel.clientHeight;
  renderer.setSize(w, h);
  if (camera) { camera.aspect = w / h; camera.updateProjectionMatrix(); }
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  const t = Date.now() / 1000;
  queryGroup.children.forEach(obj => {
    if (obj.userData.pulse) {
      obj.scale.setScalar(0.8 + 0.4 * (0.5 + 0.5 * Math.sin(t * Math.PI * 2)));
    }
  });
  renderer.render(scene, camera);
}

// ── Load pointcloud ───────────────────────────────────────────────────────────

async function loadPointcloud() {
  const res = await fetch('/api/pointcloud');
  if (!res.ok) { loadingOverlay.style.display = 'none'; return; }
  const data = await res.json();

  allSkillPoints = [];
  pointsGroup.clear();
  wireGroup.clear();

  const sizeRadius = { S: 0.08, M: 0.12, L: 0.18 };
  data.skills.forEach(s => {
    const geo  = new THREE.SphereGeometry(sizeRadius[s.size] || 0.1, 6, 6);
    const mat  = new THREE.MeshLambertMaterial({ color: getCatColor(s.category) });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(s.x * SCALE, s.z * SCALE, s.y * SCALE);
    mesh.userData = { name: s.name, category: s.category, size: s.size };
    pointsGroup.add(mesh);
    allSkillPoints.push({ mesh, name: s.name, category: s.category, size: s.size });
  });

  allDomains = data.domains;
  data.domains.forEach(d => {
    const geo  = new THREE.SphereGeometry((d.r60 || 1) * SCALE, 16, 12);
    const mat  = new THREE.MeshBasicMaterial({ color: getCatColor(d.name), wireframe: true, opacity: 0.12, transparent: true });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(d.cx * SCALE, d.cz * SCALE, d.cy * SCALE);
    wireGroup.add(mesh);
  });

  if (skillCountLabel) skillCountLabel.textContent = `${data.skills.length.toLocaleString()} skills`;
  loadingOverlay.style.display = 'none';
}

// ── Query visuals ─────────────────────────────────────────────────────────────

function clearQueryVisuals() {
  queryGroup.clear();
  lineGroup.clear();
  domainWireGroup.clear();
}

function showQueryPoints(queryPoints, spatialResults, semanticResults, domainName) {
  clearQueryVisuals();
  controls.autoRotate = false;

  const centroid = queryPoints.find(p => p.is_centroid) || queryPoints[0];
  if (!centroid) return;
  const origin = new THREE.Vector3(centroid.x * SCALE, centroid.z * SCALE, centroid.y * SCALE);

  // Amber pulsing sphere at query position
  queryPoints.forEach(qp => {
    const geo  = new THREE.SphereGeometry(0.28, 12, 10);
    const mat  = new THREE.MeshLambertMaterial({ color: 0xC89632, emissive: 0xC89632, emissiveIntensity: 0.6 });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(qp.x * SCALE, qp.z * SCALE, qp.y * SCALE);
    mesh.userData.pulse = true;
    queryGroup.add(mesh);
  });

  // Domain cluster indicator
  const matchedDomain = allDomains.find(d => d.name === domainName);
  let domainPos = null;
  if (matchedDomain) {
    domainPos = new THREE.Vector3(matchedDomain.cx * SCALE, matchedDomain.cz * SCALE, matchedDomain.cy * SCALE);
    // Bright line: query → domain centroid
    const geo  = new THREE.BufferGeometry().setFromPoints([origin, domainPos]);
    const mat  = new THREE.LineBasicMaterial({ color: 0xE8B840, linewidth: 3 });
    lineGroup.add(new THREE.Line(geo, mat));
    // Domain centroid sphere
    const dcGeo  = new THREE.SphereGeometry(0.35, 12, 10);
    const dcMat  = new THREE.MeshLambertMaterial({ color: getCatColor(domainName), emissive: 0x4A90D9, emissiveIntensity: 0.3 });
    const dcMesh = new THREE.Mesh(dcGeo, dcMat);
    dcMesh.position.copy(domainPos);
    queryGroup.add(dcMesh);
    // Domain wireframe
    const wGeo  = new THREE.SphereGeometry((matchedDomain.r60 || 1) * SCALE, 20, 14);
    const wMat  = new THREE.MeshBasicMaterial({ color: getCatColor(domainName), wireframe: true, opacity: 0.25, transparent: true });
    const wMesh = new THREE.Mesh(wGeo, wMat);
    wMesh.position.copy(domainPos);
    domainWireGroup.add(wMesh);

    mapDomainLabel.textContent = domainName;
    mapDomainLabel.classList.remove('hidden');
  } else {
    mapDomainLabel.classList.add('hidden');
  }

  // All unique result points (spatial + semantic combined, deduplicated)
  const allResults = [...spatialResults, ...semanticResults];
  const spatialNames  = new Set(spatialResults.map(r => r.name));
  const semanticNames = new Set(semanticResults.map(r => r.name));

  resultSlugs = new Set(allResults.map(r => r.name));

  allResults.forEach((r, i) => {
    const pt = r.point_3d;
    if (!pt) return;
    const inBoth   = spatialNames.has(r.name) && semanticNames.has(r.name);
    const colHex   = inBoth ? 0xffffff : (spatialNames.has(r.name) ? 0xC89632 : 0x4A90D9);
    const mat      = new THREE.LineBasicMaterial({ color: colHex, opacity: 0.35, transparent: true });
    const target   = new THREE.Vector3(pt.x * SCALE, pt.z * SCALE, pt.y * SCALE);
    const geo      = new THREE.BufferGeometry().setFromPoints([origin, target]);
    const line     = new THREE.Line(geo, mat);
    line.visible   = false;
    lineGroup.add(line);
    setTimeout(() => { line.visible = true; }, 80 * i);
  });

  // Highlight result skill spheres:
  //   amber  = spatial only
  //   blue   = semantic only
  //   white  = both
  allSkillPoints.forEach(sp => {
    const inSpa = spatialNames.has(sp.name);
    const inSem = semanticNames.has(sp.name);
    if (inSpa && inSem) {
      sp.mesh.material.emissive = new THREE.Color(0xffffff);
      sp.mesh.material.emissiveIntensity = 0.6;
    } else if (inSpa) {
      sp.mesh.material.emissive = new THREE.Color(0xC89632);
      sp.mesh.material.emissiveIntensity = 0.6;
    } else if (inSem) {
      sp.mesh.material.emissive = new THREE.Color(0x4A90D9);
      sp.mesh.material.emissiveIntensity = 0.6;
    } else {
      sp.mesh.material.emissive = new THREE.Color(0x000000);
      sp.mesh.material.emissiveIntensity = 0;
    }
  });

  // Fly camera
  const focusTarget = domainPos
    ? new THREE.Vector3().addVectors(origin, domainPos).multiplyScalar(0.5)
    : origin.clone();
  const dist    = domainPos ? origin.distanceTo(domainPos) * 1.6 + 8 : 20;
  const camPos  = focusTarget.clone().add(new THREE.Vector3(0, dist * 0.3, dist));
  flyCamera(camPos, focusTarget);
}

function flyCamera(newPos, newTarget, duration = 900) {
  const startPos    = camera.position.clone();
  const startTarget = controls.target.clone();
  const start       = Date.now();
  function tick() {
    const t    = Math.min((Date.now() - start) / duration, 1);
    const ease = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
    camera.position.lerpVectors(startPos, newPos, ease);
    controls.target.lerpVectors(startTarget, newTarget, ease);
    if (t < 1) requestAnimationFrame(tick);
  }
  tick();
}


// ── Raycasting ────────────────────────────────────────────────────────────────

const raycaster = new THREE.Raycaster();
raycaster.params.Points = { threshold: 0.3 };
const mouse = new THREE.Vector2();

function onMouseMove(e) {
  if (!renderer) return;
  const rect = mapCanvas.getBoundingClientRect();
  mouse.x =  ((e.clientX - rect.left) / rect.width)  * 2 - 1;
  mouse.y = -((e.clientY - rect.top)  / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObjects(allSkillPoints.map(s => s.mesh));
  if (hits.length > 0) {
    const sp = allSkillPoints.find(s => s.mesh === hits[0].object);
    if (sp) {
      const inSpa = lastSpatialResults.some(r => r.name === sp.name);
      const inSem = lastSemanticResults.some(r => r.name === sp.name);
      const tag   = inSpa && inSem ? ' · both' : inSpa ? ' · spatial' : inSem ? ' · semantic' : '';
      mapTooltip.innerHTML = `<div class="tooltip-name">${sp.name}${tag}</div><div class="tooltip-desc">${sp.category} · ${sp.size || ''}</div>`;
      mapTooltip.style.left = (e.clientX - rect.left + 14) + 'px';
      mapTooltip.style.top  = (e.clientY - rect.top  + 14) + 'px';
      mapTooltip.classList.add('visible');
      mapCanvas.style.cursor = 'pointer';
      return;
    }
  }
  mapTooltip.classList.remove('visible');
  mapCanvas.style.cursor = 'grab';
}

function onCanvasClick(e) {
  if (!renderer) return;
  const rect = mapCanvas.getBoundingClientRect();
  mouse.x =  ((e.clientX - rect.left) / rect.width)  * 2 - 1;
  mouse.y = -((e.clientY - rect.top)  / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObjects(allSkillPoints.map(s => s.mesh));
  if (hits.length > 0) {
    const sp = allSkillPoints.find(s => s.mesh === hits[0].object);
    if (sp) openSkillModal(sp.name);
  }
}

// ── Compare search ────────────────────────────────────────────────────────────

async function doCompare() {
  const query = queryInput.value.trim();
  if (!query || isSearching) return;

  isSearching = true;
  searchBtn.disabled = true;
  searchBtn.innerHTML = '<i class="ph ph-circle-notch ph-spin"></i>';

  try {
    const res = await fetch('/api/compare', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ query }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    lastSpatialResults  = data.spatial  || [];
    lastSemanticResults = data.semantic || [];
    currentDomain       = data.domain   || null;

    renderCompare(data);
    showQueryPoints(data.query_points, lastSpatialResults, lastSemanticResults, currentDomain);

    if (currentDomain) {
      domainTag.textContent = `Cluster: ${currentDomain}`;
      domainTag.classList.remove('hidden');
    } else {
      domainTag.classList.add('hidden');
    }

    compareEmpty.classList.add('hidden');
    compareCols.classList.remove('hidden');
  } catch (err) {
    console.error(err);
  } finally {
    isSearching = false;
    searchBtn.disabled = false;
    searchBtn.innerHTML = '<i class="ph ph-arrow-right"></i>';
  }
}

function renderCompare(data) {
  const spatialNames  = new Set(data.spatial.map(r => r.name));
  const semanticNames = new Set(data.semantic.map(r => r.name));

  spatialCards.innerHTML  = '';
  semanticCards.innerHTML = '';

  data.spatial.forEach(r => {
    const inBoth = semanticNames.has(r.name);
    spatialCards.appendChild(makeCompareCard(r, 'spatial', r.score, inBoth));
  });
  if (data.spatial.length === 0) {
    spatialCards.innerHTML = '<p class="no-results">No domain matched — try a more specific query.</p>';
  }

  data.semantic.forEach(r => {
    const inBoth = spatialNames.has(r.name);
    semanticCards.appendChild(makeCompareCard(r, 'semantic', r.score, inBoth));
  });
}

function makeCompareCard(skill, type, score, inBoth) {
  const card = document.createElement('div');
  card.className = `compare-card${inBoth ? ' in-both' : ''}`;

  const scoreLabel  = type === 'spatial'
    ? `${(score * 100).toFixed(1)}% match`
    : `${(score * 100).toFixed(1)}% cosine`;
  const overlapBadge = inBoth
    ? `<span class="overlap-badge"><i class="ph ph-intersect"></i> also semantic</span>`
    : '';
  const alsoSpatial  = type === 'semantic' && inBoth
    ? `<span class="overlap-badge"><i class="ph ph-intersect"></i> also spatial</span>`
    : '';
  const badge = type === 'spatial' ? overlapBadge : alsoSpatial;

  const tagHtml = (skill.tags || [])
    .map(t => `<span class="cc-tag">${escHtml(t)}</span>`)
    .join('');

  card.innerHTML = `
    <div class="cc-name">${escHtml(skill.name)}</div>
    <div class="cc-desc">${escHtml((skill.description || '').slice(0, 120))}${skill.description?.length > 120 ? '…' : ''}</div>
    ${tagHtml ? `<div class="cc-tags">${tagHtml}</div>` : ''}
    <div class="cc-footer">
      <span class="cc-score ${type}">${scoreLabel}</span>
      ${badge}
      <span class="cc-repo">${escHtml(skill.source_repo || '')}</span>
    </div>
  `;
  card.addEventListener('click', () => openSkillModal(skill.name));
  return card;
}

// ── Skill modal ───────────────────────────────────────────────────────────────

async function openSkillModal(name) {
  const res = await fetch('/api/skills/' + encodeURIComponent(name));
  if (!res.ok) return;
  const skill = await res.json();

  modalTitle.textContent = skill.name;
  modalBadges.innerHTML  = `
    <span class="badge" style="background:var(--surface-2);color:var(--fg-muted);border:1px solid var(--border)">${skill.source_repo}</span>
    <span class="badge badge-size-${skill.size}">${skill.size}</span>
    <span class="badge" style="background:var(--surface-2);color:var(--fg-muted);border:1px solid var(--border)">Tier ${skill.embed_tier}</span>
  `;
  modalDesc.textContent    = skill.description;
  modalContent.innerHTML   = skill.body ? renderMarkdown(skill.body) : '<em style="color:var(--fg-muted)">No body content.</em>';
  modalSourceLink.href     = skill.url || '#';
  const rawUrl             = skill.url
    ? skill.url.replace('github.com', 'raw.githubusercontent.com').replace('/tree/', '/') + '/SKILL.md'
    : '#';
  modalDownloadLink.href     = rawUrl;
  modalDownloadLink.download = `${skill.name}.SKILL.md`;
  modalOverlay.classList.remove('hidden');
}

function closeModal() { modalOverlay.classList.add('hidden'); }

// ── Minimal Markdown renderer ─────────────────────────────────────────────────

function renderMarkdown(md) {
  return md
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/^### (.+)$/gm,  '<h3>$1</h3>')
    .replace(/^## (.+)$/gm,   '<h2>$1</h2>')
    .replace(/^# (.+)$/gm,    '<h1>$1</h1>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g,     '<em>$1</em>')
    .replace(/`([^`]+)`/g,     '<code>$1</code>')
    .replace(/^- (.+)$/gm,    '<li>$1</li>')
    .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>')
    .replace(/\n\n/g, '</p><p>')
    .replace(/^(?!<[hul])/gm, '<p>')
    .replace(/<p><\/p>/g, '');
}

function escHtml(s) {
  return (s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}


// ── Event listeners ───────────────────────────────────────────────────────────

searchBtn.addEventListener('click', doCompare);
queryInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) doCompare();
});

toggleWiresBtn.addEventListener('click', () => {
  wiresVisible = !wiresVisible;
  wireGroup.visible = wiresVisible;
});

$('modal-close').addEventListener('click', closeModal);
modalOverlay.addEventListener('click', e => { if (e.target === modalOverlay) closeModal(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

// Example query buttons
document.querySelectorAll('.example-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    queryInput.value = btn.textContent;
    doCompare();
  });
});

// ── Init ──────────────────────────────────────────────────────────────────────

async function init() {
  initThree();
  await loadPointcloud();
}

init();

// Expose for Playwright automation
window.__openSkillModal = openSkillModal;
window.__doCompare = (q) => { queryInput.value = q; return doCompare(); };
