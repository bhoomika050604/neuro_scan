// NeuroScan — Frontend Logic
'use strict';

const API = {
  speech:     '/predict/speech',
  gait:       '/predict/gait',
  gait_form:  '/predict/gait_form',
  biomarkers: '/predict/biomarkers',
  clinical:   '/predict/clinical',
  mri:        '/predict/mri',
  fused:      '/predict/fused',
};

const state = {
  predictions:   {},
  speechBlob:    null,
  gaitBlob:      null,
  mediaRecorder: null,
  recordInterval: null,
  recordSeconds:  0,
};

// ── Safe getElementById ─────────────────────────────────────────
function $id(id) { return document.getElementById(id); }

// ── Tab Switching ───────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    const panel = $id('tab-' + btn.dataset.tab);
    if (panel) panel.classList.add('active');
  });
});

// ── Loading ─────────────────────────────────────────────────────
const overlay     = $id('loading-overlay');
const loadingText = $id('loading-text');
function showLoading(msg = 'Analyzing...') {
  if (loadingText) loadingText.textContent = msg;
  if (overlay) overlay.classList.remove('hidden');
}
function hideLoading() {
  if (overlay) overlay.classList.add('hidden');
}

// ── Result Renderer ─────────────────────────────────────────────
function renderResult(containerId, data) {
  const el = $id(containerId);
  if (!el) return;
  el.classList.remove('hidden');

  if (data.error && !data.probability) {
    el.innerHTML = `<div class="result-title">Error</div><p style="color:var(--red)">${data.error}</p>`;
    return;
  }

  const prob      = data.probability ?? 0;
  const pct       = Math.round(prob * 100);
  const risk      = data.risk_level || 'Unknown';
  const riskClass = risk === 'Low' ? 'risk-low' : risk === 'High' ? 'risk-high' : 'risk-moderate';

  const featuresHtml = Object.entries(data.features || {}).map(([k, v]) => `
    <div class="feat-item">
      <div class="feat-key">${k}</div>
      <div class="feat-val">${v}</div>
    </div>`).join('');

  const findingsHtml = (data.key_findings || []).map(f => `<li>${f}</li>`).join('');

  el.innerHTML = `
    <div class="result-title">Analysis Result — ${(data.modality || '').toUpperCase()}</div>
    <div class="result-row">
      <div class="result-prob">${pct}%</div>
      <div class="result-risk ${riskClass}">${risk} Risk</div>
      <div style="font-size:0.8rem;color:var(--text2)">
        Confidence: ${Math.round((data.confidence || 0.8) * 100)}%
        ${data.prediction ? ' · Prediction: <strong style="color:var(--accent)">' + data.prediction + '</strong>' : ''}
      </div>
    </div>
    <div class="result-features">${featuresHtml}</div>
    <div style="font-size:0.8rem;color:var(--text2);margin-bottom:6px">Key Findings:</div>
    <ul class="findings-list">${findingsHtml}</ul>
    <div class="result-model">
      Model: ${data.model || 'N/A'} · Accuracy: ${data.accuracy || 'N/A'}
      ${data.auc ? ' · AUC: ' + data.auc : ''}
    </div>`;
}

// ── Modality Status ──────────────────────────────────────────────
function setModalityReady(modality, prob) {
  const card = $id('ms-' + modality);
  if (!card) return;
  card.classList.add('ready');
  card.querySelector('.ms-status').textContent = Math.round((prob || 0) * 100) + '% risk';
}
function updateFuseButton() {
  const count = Object.keys(state.predictions).length;
  const btn   = $id('fuse-btn');
  if (!btn) return;
  btn.disabled = count < 1;
  btn.textContent = count > 0
    ? `Run Multimodal Fusion (${count} modalit${count === 1 ? 'y' : 'ies'} ready)`
    : 'Run Multimodal Fusion Analysis';
}

// ══════════════════════════════════════════════════════════════
// ── SPEECH ────────────────────────────────────────────────────
// ══════════════════════════════════════════════════════════════

setupDropZone('speech-drop', 'speech-file', 'speech-file-info', 'speech-upload-btn',
  ['wav','mp3','ogg','webm','m4a','txt','csv']);

const speechUploadBtn = $id('speech-upload-btn');
if (speechUploadBtn) {
  speechUploadBtn.addEventListener('click', async () => {
    const file = $id('speech-file').files[0];
    if (!file) return;
    showLoading('Extracting vocal features...');
    const fd = new FormData();
    fd.append('file', file);
    try {
      const res  = await fetch(API.speech, { method: 'POST', body: fd });
      const data = await res.json();
      state.predictions.speech = data;
      renderResult('speech-result', data);
      setModalityReady('speech', data.probability);
      updateFuseButton();
    } catch(e) { alert('Speech error: ' + e.message); }
    finally { hideLoading(); }
  });
}

// Speech Recording
let speechChunks = [];
const speechRecBtn        = $id('speech-record-btn');
const speechStopBtn       = $id('speech-stop-btn');
const speechAnalyzeRecBtn = $id('speech-analyze-rec-btn');
const speechTimer         = $id('speech-timer');
const waveEl              = document.querySelector('#speech-waveform .wave-bars');

if (speechRecBtn) {
  speechRecBtn.addEventListener('click', async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      state.mediaRecorder = new MediaRecorder(stream);
      speechChunks = [];
      state.mediaRecorder.ondataavailable = e => { if (e.data.size > 0) speechChunks.push(e.data); };
      state.mediaRecorder.onstop = () => {
        state.speechBlob = new Blob(speechChunks, { type: 'audio/webm' });
        if (speechAnalyzeRecBtn) speechAnalyzeRecBtn.disabled = false;
      };
      state.mediaRecorder.start(100);
      speechRecBtn.classList.add('recording');
      speechRecBtn.disabled = true;
      if (speechStopBtn) speechStopBtn.disabled = false;
      if (speechTimer) speechTimer.classList.remove('hidden');
      if (waveEl) waveEl.classList.add('active');
      state.recordSeconds = 0;
      state.recordInterval = setInterval(() => {
        state.recordSeconds++;
        const m = String(Math.floor(state.recordSeconds / 60)).padStart(2, '0');
        const s = String(state.recordSeconds % 60).padStart(2, '0');
        if (speechTimer) speechTimer.textContent = `${m}:${s}`;
      }, 1000);
    } catch(e) { alert('Microphone access denied: ' + e.message); }
  });
}

if (speechStopBtn) {
  speechStopBtn.addEventListener('click', () => {
    if (state.mediaRecorder && state.mediaRecorder.state !== 'inactive') {
      state.mediaRecorder.stop();
      state.mediaRecorder.stream.getTracks().forEach(t => t.stop());
    }
    clearInterval(state.recordInterval);
    if (speechRecBtn) { speechRecBtn.classList.remove('recording'); speechRecBtn.disabled = false; }
    speechStopBtn.disabled = true;
    if (waveEl) waveEl.classList.remove('active');
  });
}

if (speechAnalyzeRecBtn) {
  speechAnalyzeRecBtn.addEventListener('click', async () => {
    if (!state.speechBlob) return;
    showLoading('Analyzing recorded speech...');
    const reader = new FileReader();
    reader.onload = async () => {
      const b64 = reader.result.split(',')[1];
      try {
        const res  = await fetch(API.speech, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ audio_data: b64 })
        });
        const data = await res.json();
        state.predictions.speech = data;
        renderResult('speech-result', data);
        setModalityReady('speech', data.probability);
        updateFuseButton();
      } catch(e) { alert('Error: ' + e.message); }
      finally { hideLoading(); }
    };
    reader.readAsDataURL(state.speechBlob);
  });
}

// ══════════════════════════════════════════════════════════════
// ── GAIT ──────────────────────────────────────────────────────
// ══════════════════════════════════════════════════════════════

// Severity slider
const gaitSevSlider = $id('gait-severity');
if (gaitSevSlider) {
  gaitSevSlider.addEventListener('input', () => {
    const valEl = $id('gait-sev-val');
    if (valEl) valEl.textContent = gaitSevSlider.value;
  });
}

// Gait form submit
const gaitFormBtn = $id('gait-form-btn');
if (gaitFormBtn) {
  gaitFormBtn.addEventListener('click', async () => {
    const speed = $id('gait-speed') ? $id('gait-speed').value : '';
    if (!speed) { alert('Please enter at least Gait Speed.'); return; }
    showLoading('Running gait model...');
    const payload = {
      age:          $id('gait-age')      ? $id('gait-age').value      : '',
      height_m:     $id('gait-height')   ? $id('gait-height').value   : '',
      weight_kg:    $id('gait-weight')   ? $id('gait-weight').value   : '',
      gender:       $id('gait-gender')   ? $id('gait-gender').value   : 'm',
      gait_speed_ms: speed,
      severity:     $id('gait-severity') ? $id('gait-severity').value : '0',
    };
    try {
      const res  = await fetch(API.gait_form, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      state.predictions.gait = data;
      renderResult('gait-result', data);
      setModalityReady('gait', data.probability);
      updateFuseButton();
    } catch(e) { alert('Error: ' + e.message); }
    finally { hideLoading(); }
  });
}

// Gait CSV upload
setupDropZone('gait-drop', 'gait-file', 'gait-file-info', 'gait-upload-btn', ['csv','txt','tsv']);

const gaitUploadBtn = $id('gait-upload-btn');
if (gaitUploadBtn) {
  gaitUploadBtn.addEventListener('click', async () => {
    const file = $id('gait-file') ? $id('gait-file').files[0] : null;
    if (!file) return;
    showLoading('Processing gait CSV...');
    const fd = new FormData();
    fd.append('file', file);
    try {
      const res  = await fetch(API.gait, { method: 'POST', body: fd });
      const data = await res.json();
      state.predictions.gait = data;
      renderResult('gait-result', data);
      setModalityReady('gait', data.probability);
      updateFuseButton();
    } catch(e) { alert('Error: ' + e.message); }
    finally { hideLoading(); }
  });
}

// Gait Camera
let gaitChunks = [];
const gaitCamStartBtn = $id('gait-cam-start');
const gaitRecBtn      = $id('gait-rec-btn');
const gaitStopBtn     = $id('gait-stop-btn');
const gaitAnalyzeBtn  = $id('gait-analyze-rec-btn');
const gaitVideo       = $id('gait-camera');
const gaitPlaceholder = $id('gait-cam-placeholder');
let gaitStream = null;

if (gaitCamStartBtn) {
  gaitCamStartBtn.addEventListener('click', async () => {
    try {
      gaitStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      if (gaitVideo) { gaitVideo.srcObject = gaitStream; gaitVideo.style.display = 'block'; }
      if (gaitPlaceholder) gaitPlaceholder.style.display = 'none';
      if (gaitRecBtn) gaitRecBtn.disabled = false;
      gaitCamStartBtn.disabled = true;
    } catch(e) { alert('Camera access denied: ' + e.message); }
  });
}
if (gaitRecBtn) {
  gaitRecBtn.addEventListener('click', () => {
    if (!gaitStream) return;
    const mr = new MediaRecorder(gaitStream);
    gaitChunks = [];
    mr.ondataavailable = e => { if (e.data.size > 0) gaitChunks.push(e.data); };
    mr.onstop = () => {
      state.gaitBlob = new Blob(gaitChunks, { type: 'video/webm' });
      if (gaitAnalyzeBtn) gaitAnalyzeBtn.disabled = false;
    };
    mr.start(100);
    state.mediaRecorder = mr;
    gaitRecBtn.disabled = true;
    gaitRecBtn.classList.add('recording');
    if (gaitStopBtn) gaitStopBtn.disabled = false;
  });
}
if (gaitStopBtn) {
  gaitStopBtn.addEventListener('click', () => {
    if (state.mediaRecorder && state.mediaRecorder.state !== 'inactive') state.mediaRecorder.stop();
    if (gaitRecBtn) { gaitRecBtn.disabled = false; gaitRecBtn.classList.remove('recording'); }
    gaitStopBtn.disabled = true;
  });
}
if (gaitAnalyzeBtn) {
  gaitAnalyzeBtn.addEventListener('click', async () => {
    if (!state.gaitBlob) return;
    showLoading('Analyzing gait video...');
    const reader = new FileReader();
    reader.onload = async () => {
      const b64 = reader.result.split(',')[1];
      try {
        const res  = await fetch(API.gait, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ video_data: b64 })
        });
        const data = await res.json();
        state.predictions.gait = data;
        renderResult('gait-result', data);
        setModalityReady('gait', data.probability);
        updateFuseButton();
      } catch(e) { alert('Error: ' + e.message); }
      finally { hideLoading(); }
    };
    reader.readAsDataURL(state.gaitBlob);
  });
}

// ══════════════════════════════════════════════════════════════
// ── BIOMARKERS ────────────────────────────────────────────────
// ══════════════════════════════════════════════════════════════

const bioSubmitBtn = $id('bio-submit-btn');
if (bioSubmitBtn) {
  bioSubmitBtn.addEventListener('click', async () => {
    const payload = {
      alpha_synuclein: $id('bio-alpha-syn') ? $id('bio-alpha-syn').value : '',
      dj1_protein:     $id('bio-dj1')       ? $id('bio-dj1').value       : '',
      nfl:             $id('bio-nfl')        ? $id('bio-nfl').value        : '',
      gfap:            $id('bio-gfap')       ? $id('bio-gfap').value       : '',
      uric_acid:       $id('bio-uric')       ? $id('bio-uric').value       : '',
      tnf_alpha:       $id('bio-tnf')        ? $id('bio-tnf').value        : '',
      il6:             $id('bio-il6')        ? $id('bio-il6').value        : '',
      dopac:           $id('bio-dopac')      ? $id('bio-dopac').value      : '',
    };
    Object.keys(payload).forEach(k => { if (!payload[k]) delete payload[k]; });
    if (Object.keys(payload).length === 0) { alert('Please enter at least one biomarker value.'); return; }
    showLoading('Running biomarker analysis...');
    try {
      const res  = await fetch(API.biomarkers, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      state.predictions.biomarkers = data;
      renderResult('bio-result', data);
      setModalityReady('biomarkers', data.probability);
      updateFuseButton();
    } catch(e) { alert('Error: ' + e.message); }
    finally { hideLoading(); }
  });
}

// ══════════════════════════════════════════════════════════════
// ── CLINICAL ──────────────────────────────────────────────────
// ══════════════════════════════════════════════════════════════

['tremor','rigidity','brady','postural'].forEach(id => {
  const el = $id(`cl-${id}`);
  if (el) el.addEventListener('input', () => {
    const valEl = $id(`${id}-val`);
    if (valEl) valEl.textContent = el.value;
  });
});

const clSubmitBtn = $id('cl-submit-btn');
if (clSubmitBtn) {
  clSubmitBtn.addEventListener('click', async () => {
    const g = id => $id(id);
    const payload = {
      age:                    g('cl-age')       ? g('cl-age').value       : 60,
      sex:                    g('cl-sex')       ? g('cl-sex').value       : 'male',
      symptom_duration_years: g('cl-duration')  ? g('cl-duration').value  : 0,
      updrs_total:            g('cl-updrs')     ? g('cl-updrs').value     : 0,
      family_history:         g('cl-family')    && g('cl-family').checked    ? 'yes' : 'no',
      pesticide_exposure:     g('cl-pesticide') && g('cl-pesticide').checked ? 'yes' : 'no',
      head_trauma:            g('cl-trauma')    && g('cl-trauma').checked    ? 'yes' : 'no',
      smoking:                g('cl-smoking')   && g('cl-smoking').checked   ? 'yes' : 'no',
      tremor:                 g('cl-tremor')    ? g('cl-tremor').value    : 0,
      rigidity:               g('cl-rigidity')  ? g('cl-rigidity').value  : 0,
      bradykinesia:           g('cl-brady')     ? g('cl-brady').value     : 0,
      postural_instability:   g('cl-postural')  ? g('cl-postural').value  : 0,
      micrographia:           g('cl-micro')     && g('cl-micro').checked     ? 'yes' : 'no',
      hypophonia:             g('cl-hypo')      && g('cl-hypo').checked      ? 'yes' : 'no',
      masked_face:            g('cl-mask')      && g('cl-mask').checked      ? 'yes' : 'no',
      anosmia:                g('cl-anosmia')   && g('cl-anosmia').checked   ? 'yes' : 'no',
      rem_sleep_disorder:     g('cl-rem')       && g('cl-rem').checked       ? 'yes' : 'no',
      constipation:           g('cl-constip')   && g('cl-constip').checked   ? 'yes' : 'no',
      depression:             g('cl-depress')   && g('cl-depress').checked   ? 'yes' : 'no',
      cognitive_decline:      g('cl-cogdecline')&& g('cl-cogdecline').checked? 'yes' : 'no',
      orthostatic_hypotension:g('cl-ortho')     && g('cl-ortho').checked     ? 'yes' : 'no',
    };
    showLoading('Processing clinical assessment...');
    try {
      const res  = await fetch(API.clinical, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      state.predictions.clinical = data;
      renderResult('cl-result', data);
      setModalityReady('clinical', data.probability);
      updateFuseButton();
    } catch(e) { alert('Error: ' + e.message); }
    finally { hideLoading(); }
  });
}

// ══════════════════════════════════════════════════════════════
// ── MRI ───────────────────────────────────────────────────────
// ══════════════════════════════════════════════════════════════

setupDropZone('mri-drop', 'mri-file', 'mri-file-info', 'mri-submit-btn',
  ['nii','gz','dcm','png','jpg','jpeg']);

const mriFileInput = $id('mri-file');
if (mriFileInput) {
  mriFileInput.addEventListener('change', () => {
    const file = mriFileInput.files[0];
    if (!file) return;
    const ext = file.name.split('.').pop().toLowerCase();
    if (['png','jpg','jpeg'].includes(ext)) {
      const reader = new FileReader();
      reader.onload = e => {
        const preview = $id('mri-preview');
        const wrap    = $id('mri-preview-wrap');
        if (preview) preview.src = e.target.result;
        if (wrap)    wrap.classList.remove('hidden');
      };
      reader.readAsDataURL(file);
    } else {
      const wrap = $id('mri-preview-wrap');
      if (wrap) wrap.classList.add('hidden');
    }
  });
}

const mriSubmitBtn = $id('mri-submit-btn');
if (mriSubmitBtn) {
  mriSubmitBtn.addEventListener('click', async () => {
    const file = mriFileInput ? mriFileInput.files[0] : null;
    if (!file) return;
    showLoading('Analyzing brain MRI...');
    const fd = new FormData();
    fd.append('file', file);
    try {
      const res  = await fetch(API.mri, { method: 'POST', body: fd });
      const data = await res.json();
      state.predictions.mri = data;
      renderResult('mri-result', data);
      setModalityReady('mri', data.probability);
      updateFuseButton();
    } catch(e) { alert('Error: ' + e.message); }
    finally { hideLoading(); }
  });
}

// ══════════════════════════════════════════════════════════════
// ── FUSION ────────────────────────────────────────────────────
// ══════════════════════════════════════════════════════════════

const fuseBtn = $id('fuse-btn');
if (fuseBtn) {
  fuseBtn.addEventListener('click', async () => {
    showLoading('Running multimodal fusion...');
    try {
      const res  = await fetch(API.fused, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(state.predictions)
      });
      const data = await res.json();
      renderFusion(data);
    } catch(e) { alert('Fusion error: ' + e.message); }
    finally { hideLoading(); }
  });
}

function renderFusion(data) {
  const el = $id('fusion-result');
  if (!el) return;
  el.classList.remove('hidden');

  const prob = data.fused_probability || 0;
  const pct  = Math.round(prob * 100);
  const risk = data.risk_level || 'Unknown';
  const riskClass = risk === 'Low' ? 'risk-low' : risk === 'High' ? 'risk-high' : 'risk-moderate';

  const gaugePct = $id('gauge-pct');
  if (gaugePct) gaugePct.textContent = pct + '%';
  drawGauge(pct, risk);

  const badge = $id('fusion-risk-badge');
  if (badge) { badge.textContent = risk + ' Risk'; badge.className = 'fusion-risk-badge ' + riskClass; }
  const interp = $id('fusion-interpretation');
  if (interp) interp.textContent = data.interpretation || '';
  const rec = $id('fusion-recommendation');
  if (rec) rec.textContent = data.recommendation || '';
  const disc = $id('fusion-disclaimer');
  if (disc) disc.textContent = data.disclaimer || '';

  const barsEl = $id('fusion-bars');
  if (barsEl) {
    const summary = data.modality_summary || {};
    barsEl.innerHTML = Object.entries(summary).map(([mod, info]) => {
      const p = Math.round((info.probability || 0) * 100);
      return `
        <div class="fusion-bar-item">
          <div class="bar-header">
            <span>${mod.charAt(0).toUpperCase()+mod.slice(1)}</span>
            <span style="color:var(--accent);font-family:var(--font-mono)">${p}%</span>
          </div>
          <div class="bar-track">
            <div class="bar-fill" style="width:0%" data-target="${p}"></div>
          </div>
        </div>`;
    }).join('');
    requestAnimationFrame(() => {
      document.querySelectorAll('.bar-fill').forEach(bar => {
        bar.style.width = bar.dataset.target + '%';
      });
    });
  }
}

function drawGauge(pct, risk) {
  const canvas = $id('gauge-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const cx = 110, cy = 115, r = 90;
  ctx.clearRect(0, 0, 220, 130);
  ctx.beginPath();
  ctx.arc(cx, cy, r, Math.PI, 0, false);
  ctx.strokeStyle = '#1e1e38';
  ctx.lineWidth = 16;
  ctx.stroke();
  const color = risk === 'Low' ? '#4ade80' : risk === 'High' ? '#f87171' : '#fbbf24';
  const angle = Math.PI + (pct / 100) * Math.PI;
  ctx.beginPath();
  ctx.arc(cx, cy, r, Math.PI, angle, false);
  ctx.strokeStyle = color;
  ctx.lineWidth = 16;
  ctx.lineCap = 'round';
  ctx.stroke();
  for (let i = 0; i <= 10; i++) {
    const a = Math.PI + (i / 10) * Math.PI;
    const x1 = cx + (r - 22) * Math.cos(a), y1 = cy + (r - 22) * Math.sin(a);
    const x2 = cx + (r - 14) * Math.cos(a), y2 = cy + (r - 14) * Math.sin(a);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
    ctx.strokeStyle = '#2a2a45'; ctx.lineWidth = 2; ctx.stroke();
  }
}

// ── Drop Zone Helper ────────────────────────────────────────────
function setupDropZone(dropId, inputId, infoId, btnId, allowedExts) {
  const drop  = $id(dropId);
  const input = $id(inputId);
  const info  = $id(infoId);
  const btn   = $id(btnId);
  if (!drop || !input) return;

  drop.addEventListener('click', e => { if (!e.target.closest('label')) input.click(); });
  drop.addEventListener('dragover',  e => { e.preventDefault(); drop.classList.add('dragover'); });
  drop.addEventListener('dragleave', ()  => drop.classList.remove('dragover'));
  drop.addEventListener('drop', e => {
    e.preventDefault(); drop.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file, input, info, btn, allowedExts);
  });
  input.addEventListener('change', () => {
    if (input.files[0]) handleFileSelect(input.files[0], input, info, btn, allowedExts);
  });
}

function handleFileSelect(file, input, info, btn, allowedExts) {
  const ext = file.name.split('.').pop().toLowerCase();
  if (!allowedExts.includes(ext)) {
    alert(`Invalid file type.\nAllowed: ${allowedExts.join(', ')}`);
    return;
  }
  try {
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
  } catch(e) { /* Safari fallback — input already changed */ }

  const size = (file.size / 1024 / 1024).toFixed(2);
  if (info)  { info.textContent = `📎 ${file.name} · ${size} MB`; info.classList.remove('hidden'); }
  if (btn)   btn.disabled = false;
}
