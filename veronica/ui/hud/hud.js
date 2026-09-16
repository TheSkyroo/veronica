(() => {
  // Golden holo-globe palette. `tint`, when set, overrides the base gold hues
  // (front/highlight/back) for that state; `speed` multiplies each shell's
  // base rotation speed; `alpha` scales overall wireframe/particle opacity.
  const PALETTE = {
    idle:       {speed:0.6, alpha:0.7,  glow:'rgba(255,170,60,.20)'},
    warming:    {speed:0.6, alpha:0.55, glow:'rgba(170,170,175,.18)', grey:true},
    listening:  {speed:1.0, alpha:1.0,  glow:'rgba(255,195,90,.45)',  micBoost:true},
    thinking:   {speed:2.5, alpha:1.0,  glow:'rgba(255,190,80,.50)',  flicker:true},
    speaking:   {speed:1.0, alpha:1.0,  glow:'rgba(255,190,80,.50)',  voiceBoost:true},
    followup:   {speed:0.8, alpha:0.85, glow:'rgba(255,190,80,.32)'},
    confirming: {speed:1.0, alpha:1.0,  glow:'rgba(255,138,60,.45)',  tint:{front:'#ff8a3c', highlight:'#ffc199', back:'#8a3d10'}},
    error:      {speed:0.0, alpha:1.0,  glow:'rgba(255,90,90,.40)',   tint:{front:'#ff5a5a', highlight:'#ffb0b0', back:'#7a1f1f'}},
  };
  const GOLD = {front:'#f2c37a', highlight:'#ffe6b0', back:'#6b4a1c'};
  const model = {state:'idle', heard:'', reply:'', tool:null, prompt:'', mic:0, ready:true,
                 voice:null, voiceStart:0, confirmStart:null, confirmTimeoutMs:8000};
  const MAX_REPLY_LEN = 220;
  let micSmooth = 0, replyQueue = [], typing = false, replyGen = 0, pendingTimeout = null;
  let replySentences = [];

  const $ = id => document.getElementById(id);
  const heardEl = $('heard').querySelector('.msg');
  const replyEl = $('reply').querySelector('.msg');
  const toolEl = $('tool').querySelector('.msg');
  const detailEl = $('tool').querySelector('.detail');
  const badgeEl = $('tool').querySelector('.badge');
  const promptEl = $('prompt').querySelector('.msg');
  const hintEl = $('hint').querySelector('.msg');

  function clearReply() {
    replyGen++;
    replyQueue = [];
    typing = false;
    if (pendingTimeout !== null) { clearTimeout(pendingTimeout); pendingTimeout = null; }
  }

  function clearTurn() {
    model.heard = ''; model.reply = ''; replySentences = []; clearReply();
    replyEl.textContent = ''; heardEl.textContent = '';
    model.tool = null; badgeEl.className = 'badge'; badgeEl.textContent = ''; toolEl.textContent = ''; detailEl.textContent = '';
    model.prompt = ''; promptEl.textContent = '';
    hintEl.textContent = '';
  }

  function typeNext() {
    if (typing || replyQueue.length === 0) return;
    typing = true;
    const gen = replyGen;
    const s = replyQueue.shift();
    const start = replyEl.textContent.length ? replyEl.textContent + ' ' : '';
    let i = 0;
    const step = () => {
      if (gen !== replyGen) { typing = false; return; }
      try {
        i = Math.min(s.length, i + 2);           // ~40 chars/s at 20 fps ticks
        replyEl.textContent = start + s.slice(0, i);
        if (i < s.length) { pendingTimeout = setTimeout(step, 50); return; }
      } catch (e) {
        typing = false; pendingTimeout = null; console.error('hud typewriter step failed', e); typeNext(); return;
      }
      typing = false; pendingTimeout = null;
      typeNext();
    };
    step();
  }

  const hud = {
    push(ev) {
      const payload = ev && ev.payload;
      const kind = ev && ev.kind;
      try {
        switch (kind) {
          case 'state':
            if (payload === 'listening' && model.state !== 'followup') {
              clearTurn();
            }
            if (payload === 'confirming') model.confirmStart = null;
            model.state = payload; break;
          case 'heard': {
            // A new user utterance (including a follow-up, which never
            // passes through 'listening') starts a fresh turn: clear the
            // previous reply/tool state so it doesn't bleed into this one.
            clearTurn();
            model.heard = payload || ''; heardEl.textContent = model.heard;
            break;
          }
          case 'sentence': {
            const s = String(payload ?? '');
            if (!s) break;
            replySentences.push(s);
            let joined = replySentences.join(' ');
            while (joined.length > MAX_REPLY_LEN && replySentences.length > 1) {
              replySentences.shift();
              joined = replySentences.join(' ');
            }
            model.reply = joined; replyQueue.push(s); typeNext(); break;
          }
          case 'tool': {
            const t = payload && typeof payload === 'object' ? payload : {};
            const decision = t.decision || '';
            const summary = t.summary || '';
            model.tool = t; badgeEl.className = 'badge ' + decision;
            badgeEl.textContent = {auto:'⚡', ask:'?', allowed:'✓', declined:'✕'}[decision] || '';
            toolEl.textContent = summary.length > 60 ? summary.slice(0, 59) + '…' : summary;
            // The final allowed/declined event doesn't repeat `detail` — keep
            // whatever the preceding 'ask' event already put there instead
            // of blanking it out.
            if (typeof t.detail === 'string') detailEl.textContent = t.detail;
            if (decision === 'ask') {
              hintEl.textContent = 'say "yes" or "no"';
              model.confirmTimeoutMs = (+t.timeout_ms) || 8000;
              // Countdown starts here (once the question has actually been
              // spoken and we're about to start listening), not when the
              // 'confirming' state was entered.
              model.confirmStart = performance.now();
            } else {
              hintEl.textContent = '';
              model.prompt = ''; promptEl.textContent = '';
            }
            break;
          }
          case 'prompt': {
            // The confirmation question, spoken right before we start
            // listening. Shown immediately, with the hint beneath it; the
            // countdown arc itself doesn't start until the 'tool' ask event.
            const s = String(payload ?? '');
            model.prompt = s; promptEl.textContent = s;
            hintEl.textContent = s ? 'say "yes" or "no"' : '';
            break;
          }
          case 'mic': model.mic = Math.max(0, Math.min(1, +payload || 0)); break;
          case 'voice': model.voice = payload; model.voiceStart = performance.now(); break;
          case 'warm': model.ready = !!(payload && payload.ready); break;
        }
      } catch (err) {
        console.error('hud.push failed', err);
      }
    },
    state() { return {state:model.state, heard:model.heard, reply:model.reply, tool:model.tool, mic:model.mic, ready:model.ready}; },
    setVisible(visible) {
      visible = !!visible;
      if (!visible) {
        rafActive = false;
        if (rafId) { cancelAnimationFrame(rafId); rafId = 0; }
        return;
      }
      if (rafActive) return;
      rafActive = true;
      if (!rafId) { t0 = performance.now(); rafId = requestAnimationFrame(frame); }
    },
  };
  window.hud = hud;

  // ---- orb renderer: JARVIS-style golden wireframe holo-globe --------------
  const SIZE = 170, CX = SIZE / 2, CY = SIZE / 2, R = 66;
  const canvas = $('orb'), ctx = canvas.getContext('2d');
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  canvas.width = SIZE * dpr; canvas.height = SIZE * dpr; ctx.scale(dpr, dpr);

  // Deterministic PRNG (mulberry32) so the "circuit gap" pattern on each
  // dashed ring/arc is stable frame to frame instead of re-randomized.
  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function sphPoint(lat, lon) {
    return {x: Math.cos(lat) * Math.cos(lon), y: Math.sin(lat), z: Math.cos(lat) * Math.sin(lon)};
  }
  function rotX(p, a) {
    const c = Math.cos(a), s = Math.sin(a);
    return {x: p.x, y: p.y * c - p.z * s, z: p.y * s + p.z * c};
  }
  function rotY(p, a) {
    const c = Math.cos(a), s = Math.sin(a);
    return {x: p.x * c + p.z * s, y: p.y, z: -p.x * s + p.z * c};
  }

  const STEP_DEG = 4;

  // Short, sparse gap runs (~10% coverage, 1-2 segments per gap) so rings
  // and arcs still read as continuous circles rather than a scribble.
  function buildGapMask(n, rand) {
    const mask = new Array(n).fill(false);
    let i = 0;
    while (i < n) {
      if (rand() < 0.07) {
        const runLen = 1 + (rand() < 0.5 ? 0 : 1);
        for (let j = 0; j < runLen && i < n; j++, i++) mask[i] = true;
      } else {
        i++;
      }
    }
    return mask;
  }

  function buildShellCurves(seed) {
    const rand = mulberry32(seed);
    const curves = [];
    const N_LAT = 9, N_LON = 12;
    for (let i = 0; i < N_LAT; i++) { // 9 latitude rings, evenly spaced, poles skipped
      const latDeg = -80 + (160 / (N_LAT - 1)) * i;
      const lat = latDeg * Math.PI / 180;
      const pts = [];
      for (let lonDeg = 0; lonDeg <= 360; lonDeg += STEP_DEG) pts.push(sphPoint(lat, lonDeg * Math.PI / 180));
      curves.push({pts, gaps: buildGapMask(pts.length, rand)});
    }
    for (let i = 0; i < N_LON; i++) { // 12 longitude arcs
      const lon = (i / N_LON) * Math.PI * 2;
      const pts = [];
      for (let latDeg = -90; latDeg <= 90; latDeg += STEP_DEG) pts.push(sphPoint(latDeg * Math.PI / 180, lon));
      curves.push({pts, gaps: buildGapMask(pts.length, rand)});
    }
    return curves;
  }

  // shell0 = outermost, shell2 = innermost. shell2 spins about a different
  // axis (X instead of Y) so the three layers visibly slide past each other.
  const SHELLS = [
    {factor: 1.00, tilt: 0.35, baseSpeed: 0.25, curves: buildShellCurves(1), angle: 0, axis: 'y'},
    {factor: 0.78, tilt: -0.60, baseSpeed: -0.40, curves: buildShellCurves(2), angle: 0, axis: 'y'},
    {factor: 0.55, tilt: 0.55, baseSpeed: 0.60, curves: buildShellCurves(3), angle: 0, axis: 'x'},
  ];

  const N_PARTICLES = 120;
  const particles = Array.from({length: N_PARTICLES}, (_, i) => {
    const rnd = mulberry32(1000 + i);
    return {base: sphPoint(Math.asin(rnd() * 2 - 1), rnd() * Math.PI * 2), size: 0.8 + rnd() * 0.4};
  });

  const N_SPARKS = 24;
  const sparkRand = mulberry32(777);
  const sparks = Array.from({length: N_SPARKS}, () => spawnSpark(-sparkRand() * 2));
  function spawnSpark(t0offset) {
    return {
      dir: sphPoint(Math.asin(sparkRand() * 2 - 1), sparkRand() * Math.PI * 2),
      born: performance.now() + (t0offset || 0) * 1000,
      dur: 1000 + sparkRand() * 1000,
    };
  }

  let t0 = performance.now(), rafId = 0;
  let rafActive = !(typeof document !== 'undefined' && document.visibilityState === 'hidden');

  if (typeof document !== 'undefined' && 'visibilityState' in document) {
    document.addEventListener('visibilitychange', () => {
      const visible = document.visibilityState !== 'hidden';
      if (!visible) {
        rafActive = false;
        if (rafId) { cancelAnimationFrame(rafId); rafId = 0; }
        return;
      }
      if (rafActive) return;
      rafActive = true;
      if (!rafId) { t0 = performance.now(); rafId = requestAnimationFrame(frame); }
    });
  }

  function voiceLevel(now) {
    const v = model.voice; if (!v || !v.levels || !v.levels.length) return 0;
    const idx = Math.floor((now - model.voiceStart) / (v.step_ms || 50));
    return idx < v.levels.length ? v.levels[idx] : 0;
  }

  function greyOf(hex) {
    // desaturate a hex color toward mid-grey for the 'warming' state.
    const n = parseInt(hex.slice(1), 16);
    const r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255;
    const lum = 0.3 * r + 0.59 * g + 0.11 * b;
    const mix = c => Math.round(c * 0.25 + lum * 0.75);
    return `rgb(${mix(r)},${mix(g)},${mix(b)})`;
  }

  function project(p, factor) {
    return {x: CX + p.x * R * factor, y: CY + p.y * R * factor, z: p.z};
  }

  function shellTransform(shell, base) {
    return shell.axis === 'x'
      ? rotX(rotY(base, shell.tilt), shell.angle)
      : rotY(rotX(base, shell.tilt), shell.angle);
  }

  function frame(now) {
    rafId = 0;
    const dt = Math.min(0.1, (now - t0) / 1000); t0 = now;
    const pal = PALETTE[model.state] || PALETTE.idle;
    micSmooth += (model.mic - micSmooth) * 0.25;
    const vLevel = model.state === 'speaking' ? voiceLevel(now) : 0;

    for (const shell of SHELLS) {
      shell.angle += dt * shell.baseSpeed * pal.speed;
    }

    ctx.clearRect(0, 0, SIZE, SIZE);

    let colors = pal.tint ? pal.tint : GOLD;
    if (pal.grey) colors = {front: greyOf(GOLD.front), highlight: greyOf(GOLD.highlight), back: greyOf(GOLD.back)};

    // soft radial glow, behind everything, normal blending — brightest at
    // the very center, fully transparent by 0.9R.
    let glowAlphaBoost = 1;
    if (pal.voiceBoost) glowAlphaBoost = 1 + vLevel * 0.6;
    else if (pal.flicker) glowAlphaBoost = 0.8 + 0.4 * Math.sin(now / 130);
    const glow = ctx.createRadialGradient(CX, CY, 0, CX, CY, R * 0.9);
    glow.addColorStop(0, 'rgba(255,170,60,.45)'); glow.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.globalAlpha = Math.min(1, glowAlphaBoost);
    ctx.fillStyle = glow; ctx.fillRect(0, 0, SIZE, SIZE);
    ctx.globalAlpha = 1;

    // ---- wireframe shells (source-over) ----
    for (let si = 0; si < SHELLS.length; si++) {
      const shell = SHELLS[si];
      let factor = shell.factor;
      if (pal.micBoost && si === 0) factor *= 1 + 0.15 * micSmooth;
      const extraLine = pal.voiceBoost ? 1.2 * vLevel : 0;

      // faint filled disc behind the wireframe so the sphere reads as a body
      ctx.globalAlpha = 1;
      ctx.fillStyle = 'rgba(255,180,80,0.04)';
      ctx.beginPath(); ctx.arc(CX, CY, R * factor, 0, Math.PI * 2); ctx.fill();

      const frontPath = new Path2D(), backPath = new Path2D();
      for (const curve of shell.curves) {
        const proj = curve.pts.map(base => project(shellTransform(shell, base), factor));
        for (let i = 0; i < proj.length - 1; i++) {
          if (curve.gaps[i] || curve.gaps[i + 1]) continue;
          const a = proj[i], b = proj[i + 1];
          const path = (a.z + b.z) >= 0 ? frontPath : backPath;
          path.moveTo(a.x, a.y); path.lineTo(b.x, b.y);
        }
      }
      // back-facing first, front-facing drawn on top of it
      ctx.globalAlpha = pal.alpha * 0.18;
      ctx.strokeStyle = colors.back;
      ctx.lineWidth = 0.45 + extraLine;
      ctx.stroke(backPath);
      ctx.globalAlpha = pal.alpha * 0.9;
      ctx.strokeStyle = colors.front;
      ctx.lineWidth = 0.8 + extraLine;
      ctx.stroke(frontPath);
    }
    ctx.globalAlpha = 1;

    // ---- additive layer: particles, sparks, hub ----
    ctx.globalCompositeOperation = 'lighter';
    const shell0 = SHELLS[0];

    // particles: front-facing only, so they don't muddy the wireframe
    ctx.fillStyle = colors.highlight;
    ctx.globalAlpha = pal.alpha * 0.6;
    for (const q of particles) {
      const proj = project(shellTransform(shell0, q.base), shell0.factor);
      if (proj.z <= 0.1) continue;
      ctx.beginPath(); ctx.arc(proj.x, proj.y, q.size, 0, Math.PI * 2); ctx.fill();
    }

    ctx.fillStyle = colors.highlight;
    for (const spark of sparks) {
      let age = now - spark.born;
      if (age > spark.dur) { Object.assign(spark, spawnSpark(0)); age = now - spark.born; }
      if (age < 0) continue;
      const frac = age / spark.dur;
      const rr = 1.0 + 0.35 * frac;
      const proj = project(shellTransform(shell0, spark.dir), rr);
      ctx.globalAlpha = pal.alpha * 0.5 * (1 - frac);
      ctx.beginPath(); ctx.arc(proj.x, proj.y, 0.8, 0, Math.PI * 2); ctx.fill();
    }

    // central hub — the "eye": glow + bright ring + counter-rotating spokes
    const hubAngle = -shell0.angle * 1.4;
    let hubAlpha = pal.alpha;
    if (pal.flicker) hubAlpha *= 0.55 + 0.45 * Math.sin(now / 60);
    ctx.globalAlpha = hubAlpha;
    const hubGlow = ctx.createRadialGradient(CX, CY, 0, CX, CY, 20);
    hubGlow.addColorStop(0, colors.highlight); hubGlow.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = hubGlow;
    ctx.beginPath(); ctx.arc(CX, CY, 20, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = colors.highlight;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(CX, CY, 8, 0, Math.PI * 2); ctx.stroke();
    for (let i = 0; i < 6; i++) {
      const a = hubAngle + (i / 6) * Math.PI * 2;
      ctx.beginPath();
      ctx.moveTo(CX + Math.cos(a) * 8, CY + Math.sin(a) * 8);
      ctx.lineTo(CX + Math.cos(a) * 16, CY + Math.sin(a) * 16);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';

    // confirming: "?" glyph, always; countdown arc only once the question
    // has actually been spoken and the 'tool' ask event set confirmStart —
    // before that (model.confirmStart === null) there's nothing to count
    // down yet.
    if (model.state === 'confirming') {
      if (model.confirmStart !== null) {
        const frac = Math.max(0, 1 - (now - model.confirmStart) / model.confirmTimeoutMs);
        ctx.beginPath(); ctx.arc(CX, CY, 58, -Math.PI / 2, -Math.PI / 2 + frac * Math.PI * 2);
        ctx.lineWidth = 3; ctx.strokeStyle = colors.front; ctx.stroke();
      }
      ctx.fillStyle = '#fff'; ctx.font = 'bold 26px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('?', CX, CY + 1);
    }
    // warming spinner
    if (model.state === 'warming') {
      ctx.beginPath(); ctx.arc(CX, CY, 58, shell0.angle * 2, shell0.angle * 2 + Math.PI * 0.6);
      ctx.lineWidth = 3; ctx.strokeStyle = colors.front; ctx.stroke();
    }
    if (rafActive && rafId === 0) rafId = requestAnimationFrame(frame);
  }
  if (rafActive && rafId === 0) rafId = requestAnimationFrame(frame);
})();
