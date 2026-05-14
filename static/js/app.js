// ── Sample data ─────────────────────────────────────────────────────────────
// SAMPLES is injected by the template into a global variable before this script loads.

function loadSample(idx) {
  const s = SAMPLES[idx].data;
  document.getElementById('f_step').value     = s.step;
  document.getElementById('f_type').value     = s.type;
  document.getElementById('f_amount').value   = s.amount;
  document.getElementById('f_oldOrig').value  = s.oldbalanceOrg;
  document.getElementById('f_newOrig').value  = s.newbalanceOrig;
  document.getElementById('f_oldDest').value  = s.oldbalanceDest;
  document.getElementById('f_newDest').value  = s.newbalanceDest;
  document.getElementById('f_nameDest').value = s.nameDest || '';
  document.getElementById('f_flagged').value  = s.isFlaggedFraud || 0;

  document.querySelectorAll('.sample-btn').forEach((b,i) =>
    b.classList.toggle('active', i === idx)
  );
}

// ── Predict ──────────────────────────────────────────────────────────────────
async function runPredict(e) {
  e.preventDefault();
  setLoading(true);
  clearError();

  const fd = new FormData(document.getElementById('txForm'));

  try {
    const resp = await fetch('/predict-form', { method:'POST', body:fd });
    const data = await resp.json();

    if (!resp.ok || data.error) {
      showError(data.error || 'Prediction failed.');
      return;
    }
    renderResult(data);
  } catch(err) {
    showError('Network error: ' + err.message);
  } finally {
    setLoading(false);
  }
}

// ── Render result ─────────────────────────────────────────────────────────────
function renderResult(d) {
  document.getElementById('placeholder').style.display = 'none';
  const panel = document.getElementById('resultPanel');
  panel.classList.add('show');

  // Verdict
  const box   = document.getElementById('verdictBox');
  const fraud = d.is_fraud;
  box.className = 'verdict ' + (fraud ? 'fraud' : 'legit');
  document.getElementById('verdictIcon').textContent  = fraud ? '\u{1F6A8}' : '\u2705';
  document.getElementById('verdictTitle').textContent = fraud ? 'FRAUD DETECTED' : 'LEGITIMATE';
  document.getElementById('verdictSub').textContent   =
    fraud
      ? 'Risk: ' + d.risk_level + '  \u00b7  Confidence: ' + d.confidence
      : 'Safe transaction  \u00b7  Confidence: ' + d.confidence;

  // Gauge
  const pct  = d.fraud_probability_pct;
  const thr  = d.threshold * 100;
  const fill = document.getElementById('gaugeFill');
  fill.className = 'gauge-fill ' + (fraud ? 'danger' : 'safe');
  setTimeout(function() { fill.style.width = Math.min(pct,100) + '%'; }, 60);
  document.getElementById('gaugeVal').textContent  = pct.toFixed(2) + '%';
  document.getElementById('gaugeThr').style.left   = thr + '%';
  document.getElementById('thrLabel').textContent  = 'Threshold ' + (thr).toFixed(1) + '%';

  // Badges
  const rb = document.getElementById('riskBadge');
  rb.textContent = d.risk_level;
  rb.className   = 'risk-badge risk-' + d.risk_level;
  document.getElementById('confBadge').textContent = d.confidence;

  // Features
  const HIGH_RISK_FEATS = ['errorBalanceOrig','errorBalanceDest','orig_balance_zeroed','amount_ratio_orig'];
  const grid = document.getElementById('featGrid');
  grid.innerHTML = '';
  for (const [k,v] of Object.entries(d.features)) {
    const hi  = HIGH_RISK_FEATS.includes(k) && Math.abs(v) > 0.01;
    const val = Number.isInteger(v) ? v : v.toFixed(3);
    grid.innerHTML +=
      '<div class="feat-chip ' + (hi ? 'hi' : '') + '">' +
        '<span class="fname">' + k + '</span>' +
        '<span class="fval">' + val + '</span>' +
      '</div>';
  }

  // Raw JSON
  document.getElementById('rawJson').textContent =
    JSON.stringify({
      fraud_probability: d.fraud_probability,
      is_fraud: d.is_fraud,
      risk_level: d.risk_level,
      confidence: d.confidence,
      threshold: d.threshold,
    }, null, 2);
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function setLoading(on) {
  document.getElementById('btnPredict').disabled   = on;
  document.getElementById('btnText').textContent   = on ? 'Analyzing\u2026' : 'Run Inference';
  document.getElementById('btnSpinner').style.display = on ? 'block' : 'none';
}
function showError(msg) {
  const t = document.getElementById('errorToast');
  t.textContent = '\u26a0 ' + msg;
  t.classList.add('show');
}
function clearError() {
  document.getElementById('errorToast').classList.remove('show');
}

// Auto-load first sample on page load
document.addEventListener('DOMContentLoaded', function() {
  loadSample(0);
});
