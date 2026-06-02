/* ─────────────────────────────────────────────────────────────────────
   CricketLytics Dashboard — app.js
   Handles: file upload/drag-drop, form submission, loading animation,
            results rendering (donut chart, angle gauges, timeline, tips)
   ───────────────────────────────────────────────────────────────────── */

'use strict';

const API_BASE = window.location.origin;

// ── Shot colour map (matches shot_classifier.py SHOT_COLORS, CSS-friendly) ──
const SHOT_COLORS = {
  "Stance (Ready)":    '#6b7280',
  "Forward Defensive": '#d97706',
  "Drive":             '#10b981',
  "Lofted Drive":      '#34d399',
  "Pull / Hook Shot":  '#3b82f6',
  "Sweep Shot":        '#a855f7',
  "Cut Shot":          '#ef4444',
  "Back Foot Punch":   '#06b6d4',
};

const SHOT_EMOJIS = {
  "Stance (Ready)":    '🧍',
  "Forward Defensive": '🛡️',
  "Drive":             '🏏',
  "Lofted Drive":      '🚀',
  "Pull / Hook Shot":  '🔄',
  "Sweep Shot":        '🧹',
  "Cut Shot":          '✂️',
  "Back Foot Punch":   '👊',
};

const ANGLE_LABELS = {
  front_knee:  'Front Knee',
  back_knee:   'Back Knee',
  front_elbow: 'Elbow (Top Hand)',
};

const ANGLE_COLORS = {
  front_knee:  '#00d4ff',
  back_knee:   '#7c3aed',
  front_elbow: '#10b981',
};

// Ideal ranges [min, max] in degrees for colour feedback
const ANGLE_IDEALS = {
  front_knee:  [130, 175],
  back_knee:   [130, 175],
  front_elbow: [90, 145],
};

const TIP_ICONS = ['💡', '⚡', '🎯', '🔍', '📐', '🏋️', '⚙️', '✅'];

// Loading step durations (ms)
const STEP_DELAYS = [0, 1800, 4000, 6500, 9000];

// ── DOM refs ─────────────────────────────────────────────────────────
const form          = document.getElementById('analysis-form');
const dropZone      = document.getElementById('drop-zone');
const fileInput     = document.getElementById('file-input');
const filePreview   = document.getElementById('file-preview');
const fileName      = document.getElementById('file-name');
const removeFile    = document.getElementById('remove-file');
const analyzeBtn    = document.getElementById('analyze-btn');
const btnLabel      = document.getElementById('btn-label');
const uploadSection = document.getElementById('upload-section');
const loadingSection= document.getElementById('loading-section');
const resultsSection= document.getElementById('results-section');
const newAnalysisBtn= document.getElementById('new-analysis-btn');
const errorBanner   = document.getElementById('error-banner');
const errorMsg      = document.getElementById('error-msg');
const errorClose    = document.getElementById('error-close');
const loadingMsg    = document.getElementById('loading-msg');

let selectedFile   = null;
let timelineChart  = null;
let _slideshowTimer = null;   // setInterval for annotated frame flipbook

// ── File selection ────────────────────────────────────────────────────
function setFile(file) {
  if (!file) return;
  selectedFile = file;
  fileName.textContent = file.name;
  filePreview.hidden = false;
  dropZone.querySelector('.drop-zone__icon').style.display = 'none';
  dropZone.querySelector('.drop-zone__label').style.display = 'none';
  dropZone.querySelector('.drop-zone__sub').style.display = 'none';
  dropZone.querySelector('label').style.display = 'none';
  checkFormReady();
}

function clearFile() {
  selectedFile = null;
  fileInput.value = '';
  filePreview.hidden = true;
  dropZone.querySelector('.drop-zone__icon').style.display = '';
  dropZone.querySelector('.drop-zone__label').style.display = '';
  dropZone.querySelector('.drop-zone__sub').style.display = '';
  dropZone.querySelector('label').style.display = '';
  checkFormReady();
}

fileInput.addEventListener('change', () => { if (fileInput.files[0]) setFile(fileInput.files[0]); });
removeFile.addEventListener('click', (e) => { e.stopPropagation(); clearFile(); });

// ── Drag & drop ───────────────────────────────────────────────────────
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && (file.type.startsWith('video/') || /\.(mp4|avi)$/i.test(file.name))) {
    setFile(file);
  } else {
    showError('Please drop a valid video file (MP4 or AVI).');
  }
});
dropZone.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') fileInput.click(); });

// ── Form ready check ──────────────────────────────────────────────────
function checkFormReady() {
  const hand = document.getElementById('select-hand').value;
  const view = document.getElementById('select-view').value;
  analyzeBtn.disabled = !(selectedFile && hand && view);
}

document.getElementById('select-hand').addEventListener('change', checkFormReady);
document.getElementById('select-view').addEventListener('change', checkFormReady);

// ── Loading step animator ─────────────────────────────────────────────
function animateLoadingSteps() {
  const steps = document.querySelectorAll('.step');
  const msgs = [
    'Initialising MediaPipe Holistic pipeline…',
    'Processing video frames…',
    'Computing joint angles per frame…',
    'Classifying batting shots…',
    'Aggregating posture coaching tips…',
  ];

  steps.forEach(s => { s.classList.remove('active', 'done'); });

  STEP_DELAYS.forEach((delay, i) => {
    setTimeout(() => {
      if (i > 0) steps[i - 1]?.classList.replace('active', 'done');
      steps[i]?.classList.add('active');
      loadingMsg.textContent = msgs[i] || msgs[msgs.length - 1];
    }, delay);
  });
}

// ── Form submission ───────────────────────────────────────────────────
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!selectedFile) return;

  const hand          = document.getElementById('select-hand').value;
  const view          = document.getElementById('select-view').value;
  const intendedShot  = document.getElementById('select-shot').value || 'auto';

  const fd = new FormData();
  fd.append('video',         selectedFile, selectedFile.name);
  fd.append('hand',          hand);
  fd.append('view',          view);
  fd.append('intended_shot', intendedShot);

  // Show loading
  uploadSection.hidden  = true;
  loadingSection.hidden = false;
  resultsSection.hidden = true;
  animateLoadingSteps();

  try {
    const res = await fetch(`${API_BASE}/api/analyze`, { method: 'POST', body: fd });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Server error ${res.status}`);
    }
    const data = await res.json();
    renderResults(data);
  } catch (err) {
    loadingSection.hidden = true;
    uploadSection.hidden  = false;
    showError(err.message || 'Analysis failed. Is the server running?');
  }
});

// ── New analysis button ───────────────────────────────────────────────
newAnalysisBtn.addEventListener('click', () => {
  resultsSection.hidden = true;
  uploadSection.hidden  = false;
  clearFile();
  _stopFrameSlideshow();
  // Destroy old charts
  if (timelineChart) { timelineChart.destroy();  timelineChart = null; }
});

// ── Error banner ──────────────────────────────────────────────────────
function showError(msg) {
  errorMsg.textContent = msg;
  errorBanner.hidden   = false;
}
errorClose.addEventListener('click', () => { errorBanner.hidden = true; });

// ── Render results ────────────────────────────────────────────────────
function renderResults(data) {
  loadingSection.hidden = false;

  // Small delay so the last loading step can be seen as "done"
  setTimeout(() => {
    loadingSection.hidden = true;
    resultsSection.hidden = false;
    _populateResults(data);
  }, 600);
}

function _populateResults(data) {
  // ── Meta line ─────────────────────────────────────────────────────
  const handLabel = data.hand === 'right' ? 'Right-handed' : 'Left-handed';
  const viewLabel = data.view.charAt(0).toUpperCase() + data.view.slice(1) + ' view';
  const shotLabel = data.intended_shot === 'auto' ? 'Auto-detect' : data.intended_shot;
  document.getElementById('results-meta').textContent =
    `${handLabel} · ${viewLabel} · Intended: ${shotLabel}`;

  // ── Stat cards ─────────────────────────────────────────────────────
  document.getElementById('val-frames').textContent =
    `${data.frames_analyzed.toLocaleString()} / ${data.total_frames.toLocaleString()}`;
  const framePct = data.total_frames > 0
    ? Math.round((data.frames_analyzed / data.total_frames) * 100)
    : 0;
  setTimeout(() => {
    document.getElementById('bar-frames').style.width = `${framePct}%`;
  }, 200);

  const dom = data.dominant_shot || 'Unknown';
  document.getElementById('val-shot').textContent = `${SHOT_EMOJIS[dom] || '🏏'} ${dom}`;
  const pill = document.getElementById('shot-pill');
  pill.textContent = dom;
  pill.style.background = hexToRgba(SHOT_COLORS[dom] || '#6b7280', 0.15);
  pill.style.color = SHOT_COLORS[dom] || '#6b7280';
  pill.style.borderColor = hexToRgba(SHOT_COLORS[dom] || '#6b7280', 0.3);


  // ── Annotated frame slideshow ───────────────────────────────────
  _startFrameSlideshow(data.preview_frames || []);

  // ── Angle gauges ──────────────────────────────────────────────────
  _renderAngleGauges(data.average_angles);

  // ── Angle timeline ────────────────────────────────────────────────
  _renderTimeline(data.angle_timeline || []);

  // ── Coaching tips ─────────────────────────────────────────────────
  _renderTips(data.coaching_tips || [], data.intended_shot);
}

// ── Annotated frame slideshow (base64 frames from JSON) ──────────────
function _startFrameSlideshow(frames) {
  const feedImg     = document.getElementById('analysis-feed');
  const placeholder = document.getElementById('feed-placeholder');
  const controls    = document.getElementById('feed-controls');
  const badge       = document.getElementById('feed-badge');
  const counter     = document.getElementById('feed-frame-counter');

  _stopFrameSlideshow();

  if (!frames || frames.length === 0) {
    placeholder.hidden = false;
    feedImg.hidden     = true;
    controls.hidden    = true;
    badge.textContent  = 'no frames';
    return;
  }

  placeholder.hidden = true;
  feedImg.hidden     = false;
  controls.hidden    = false;
  badge.textContent  = `${frames.length} frames`;

  let idx = 0;
  feedImg.src = `data:image/jpeg;base64,${frames[0]}`;
  counter.textContent = `Frame 1 / ${frames.length}`;

  // Cycle through frames automatically
  _slideshowTimer = setInterval(() => {
    idx = (idx + 1) % frames.length;
    feedImg.src = `data:image/jpeg;base64,${frames[idx]}`;
    counter.textContent = `Frame ${idx + 1} / ${frames.length}`;
  }, 800);   // advance every 800 ms
}

function _stopFrameSlideshow() {
  if (_slideshowTimer) {
    clearInterval(_slideshowTimer);
    _slideshowTimer = null;
  }
  const feedImg     = document.getElementById('analysis-feed');
  const placeholder = document.getElementById('feed-placeholder');
  const controls    = document.getElementById('feed-controls');
  const badge       = document.getElementById('feed-badge');
  if (feedImg)     { feedImg.hidden = true; feedImg.src = ''; }
  if (placeholder) { placeholder.hidden = false; }
  if (controls)    { controls.hidden = true; }
  if (badge)       { badge.textContent = 'pose overlay'; }
}

// ── Angle gauges ─────────────────────────────────────────────────────
function _renderAngleGauges(avgAngles) {
  const grid = document.getElementById('gauges-grid');
  grid.innerHTML = '';

  const keys = ['front_knee', 'back_knee', 'front_elbow'].filter(k => avgAngles[k] !== undefined);
  if (keys.length === 0) {
    grid.innerHTML = '<p style="color:var(--c-text-2);font-size:.85rem">No angle data available for this video.</p>';
    return;
  }

  keys.forEach((key, i) => {
    const val    = avgAngles[key];
    const label  = ANGLE_LABELS[key] || key;
    const color  = ANGLE_COLORS[key] || '#00d4ff';
    const [lo, hi] = ANGLE_IDEALS[key] || [90, 180];
    const pct    = Math.min(100, Math.max(0, ((val - 80) / (190 - 80)) * 100));
    const inRange= val >= lo && val <= hi;

    const row = document.createElement('div');
    row.className = 'gauge-row';
    row.style.animationDelay = `${i * 0.12}s`;
    row.innerHTML = `
      <span class="gauge-label">${label}</span>
      <div class="gauge-bar-wrap">
        <div class="gauge-bar" data-pct="${pct}"
             style="background: linear-gradient(90deg, ${color}88, ${color}); width: 0;"></div>
      </div>
      <span class="gauge-value" style="color:${inRange ? '#10b981' : '#f59e0b'}">${val}°</span>
    `;
    grid.appendChild(row);

    // Animate bar after insertion
    setTimeout(() => {
      row.querySelector('.gauge-bar').style.width = `${pct}%`;
    }, 200 + i * 100);
  });

  // Ideal range note
  const note = document.createElement('p');
  note.style.cssText = 'font-size:.72rem;color:var(--c-text-3);margin-top:8px;';
  note.textContent = '🟢 Green = within typical range · 🟡 Amber = outside typical range';
  grid.appendChild(note);
}

// ── Angle timeline chart ─────────────────────────────────────────────
function _renderTimeline(timeline) {
  if (timelineChart) { timelineChart.destroy(); timelineChart = null; }

  const keys = ['front_knee', 'back_knee', 'front_elbow'].filter(k =>
    timeline.some(t => t[k] !== undefined)
  );

  if (!timeline.length || !keys.length) {
    document.getElementById('card-timeline').hidden = true;
    return;
  }
  document.getElementById('card-timeline').hidden = false;

  const labels = timeline.map(t => `F${t.frame}`);
  const datasets = keys.map(k => ({
    label:       ANGLE_LABELS[k] || k,
    data:        timeline.map(t => t[k] ?? null),
    borderColor: ANGLE_COLORS[k],
    backgroundColor: hexToRgba(ANGLE_COLORS[k], 0.08),
    fill:        false,
    tension:     0.4,
    pointRadius: 2,
    pointHoverRadius: 5,
    borderWidth: 2,
    spanGaps:    true,
  }));

  const ctx = document.getElementById('timeline-chart').getContext('2d');
  timelineChart = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: '#94a3b8', font: { size: 12 }, boxWidth: 14 },
        },
        tooltip: {
          backgroundColor: 'rgba(6,11,20,0.92)',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          titleColor: '#e2e8f0',
          bodyColor: '#94a3b8',
        },
      },
      scales: {
        x: {
          ticks: { color: '#475569', maxTicksLimit: 10 },
          grid:  { color: 'rgba(255,255,255,0.04)' },
        },
        y: {
          min: 60, max: 200,
          ticks: { color: '#475569', callback: (v) => `${v}°` },
          grid:  { color: 'rgba(255,255,255,0.04)' },
          title: { display: true, text: 'Angle (°)', color: '#475569', font: { size: 11 } },
        },
      },
      animation: { duration: 800 },
    },
  });
}

// ── Coaching tips ─────────────────────────────────────────────────────
function _renderTips(tips, intendedShot) {
  const grid = document.getElementById('tips-grid');
  grid.innerHTML = '';

  const badge = document.getElementById('tips-mode-badge');
  badge.textContent = intendedShot && intendedShot !== 'auto'
    ? `${intendedShot} mode` : 'auto mode';

  tips.forEach((tip, i) => {
    const card = document.createElement('div');
    card.className = 'tip-card';
    card.style.animationDelay = `${i * 0.07}s`;
    card.innerHTML = `
      <span class="tip-icon">${TIP_ICONS[i % TIP_ICONS.length]}</span>
      <span class="tip-text">${tip}</span>
    `;
    grid.appendChild(card);
  });

  if (tips.length === 0) {
    grid.innerHTML = '<div class="tip-card"><span class="tip-icon">✅</span><span class="tip-text">No specific coaching cues — great technique detected!</span></div>';
  }
}

// ── Helper: hex → rgba ────────────────────────────────────────────────
function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1,3),16);
  const g = parseInt(hex.slice(3,5),16);
  const b = parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${alpha})`;
}
