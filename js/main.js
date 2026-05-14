const slides  = document.querySelectorAll('.slide');
const dots    = document.querySelectorAll('.nav-dot');
const bar     = document.getElementById('progressBar');
const counter = document.getElementById('slideCounter');
const hint    = document.querySelector('.key-hint');
let currentSlide = 0;

// ── Reveal on scroll ─────────────────────────────────────────────────────────
const revealObs = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) e.target.classList.add('visible');
  });
}, { threshold: 0.12 });

document.querySelectorAll('.reveal').forEach(el => revealObs.observe(el));

// ── Active slide tracking ────────────────────────────────────────────────────
const slideObs = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      const idx = Array.from(slides).indexOf(e.target);
      if (idx >= 0) {
        currentSlide = idx;
        updateDots(idx);
        updateProgress(idx);
        updateCounter(idx);
        if (idx > 0 && hint) hint.classList.add('hidden');
      }
    }
  });
}, { threshold: 0.5 });

slides.forEach(s => slideObs.observe(s));

function updateDots(idx) {
  dots.forEach((d, i) => d.classList.toggle('active', i === idx));
}
function updateProgress(idx) {
  bar.style.width = ((idx + 1) / slides.length) * 100 + '%';
}
function updateCounter(idx) {
  if (counter) counter.textContent = (idx + 1) + ' / ' + slides.length;
}

// ── Dot click ────────────────────────────────────────────────────────────────
dots.forEach((d, i) => {
  d.addEventListener('click', () => {
    slides[i].scrollIntoView({ behavior: 'smooth' });
  });
});

// ── Keyboard ─────────────────────────────────────────────────────────────────
document.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowDown' || e.key === 'PageDown' || e.key === ' ') {
    e.preventDefault();
    slides[Math.min(currentSlide + 1, slides.length - 1)].scrollIntoView({ behavior: 'smooth' });
  } else if (e.key === 'ArrowUp' || e.key === 'PageUp') {
    e.preventDefault();
    slides[Math.max(currentSlide - 1, 0)].scrollIntoView({ behavior: 'smooth' });
  } else if (e.key === 'Home') {
    e.preventDefault();
    slides[0].scrollIntoView({ behavior: 'smooth' });
  } else if (e.key === 'End') {
    e.preventDefault();
    slides[slides.length - 1].scrollIntoView({ behavior: 'smooth' });
  }
});

// ── Counter animation ────────────────────────────────────────────────────────
const counterObs = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting && !e.target.dataset.counted) {
      e.target.dataset.counted = 'true';
      animateCounter(e.target);
    }
  });
}, { threshold: 0.5 });

document.querySelectorAll('[data-count]').forEach(el => counterObs.observe(el));

function formatNumber(n, decimals) {
  if (decimals > 0) return n.toFixed(decimals);
  if (n >= 1000000) return (n / 1000000).toFixed(1).replace(/\.0$/, '') + 'M';
  if (n >= 1000) return Math.round(n).toLocaleString('en-US');
  return Math.round(n).toString();
}

function animateCounter(el) {
  const target = parseFloat(el.dataset.count);
  const suffix = el.dataset.suffix || '';
  const decimals = (el.dataset.decimals || '0') | 0;
  const duration = 1400;
  const start = performance.now();

  function tick(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = eased * target;
    el.textContent = formatNumber(current, decimals) + suffix;
    if (progress < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

// ── Init ─────────────────────────────────────────────────────────────────────
updateDots(0);
updateProgress(0);
updateCounter(0);
