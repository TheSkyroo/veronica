(() => {
  const model = {state:'idle', heard:'', reply:'', tool:null, prompt:'', mic:0, ready:true,
                 voice:null, voiceStart:0, confirmStart:null, confirmTimeoutMs:8000};
  const MAX_REPLY_LEN = 220;
  let micSmooth = 0, replyQueue = [], typing = false, replyGen = 0, pendingTimeout = null;
  let replySentences = [];

  const STATUS_LABELS = {
    idle: '', warming: 'Warming up…', listening: 'Listening…', thinking: 'Thinking…',
    speaking: 'Speaking', followup: 'Listening…', confirming: 'Say yes or no',
    paused: 'Paused — say continue', error: 'Error',
  };

  const $ = id => document.getElementById(id);
  const heardRowEl = $('heard');
  const replyRowEl = $('reply');
  const heardEl = heardRowEl.querySelector('.msg');
  const replyEl = replyRowEl.querySelector('.msg');
  const actionEl = $('action');
  const toolBoxEl = $('tool');
  const toolTitleEl = $('tool-title');
  const detailEl = toolBoxEl.querySelector('.detail');
  const badgeEl = toolBoxEl.querySelector('.badge');
  const pillEl = toolBoxEl.querySelector('.pill');
  const promptRowEl = $('prompt');
  const promptEl = promptRowEl.querySelector('.msg');
  const countdownEl = document.querySelector('.countdown');
  const countdownBarEl = countdownEl.querySelector('i');
  const hintRowEl = $('hint');
  const hintEl = hintRowEl.querySelector('.msg');
  const statusEl = $('status');
  const statusLabelEl = statusEl.querySelector('.label');
  const statusLevelEl = statusEl.querySelector('.level i');
  statusEl.dataset.state = 'idle';
  const brainEl = $('brain');
  const captionEl = $('caption');
  const captionMsgEl = captionEl.querySelector('.msg');
  captionEl.dataset.state = 'idle';

  // Mini mode's single-line caption: whatever set it last wins (see hud.push
  // below for the precedence between partial/final transcript, "Thinking…",
  // the current spoken sentence, and the confirmation prompt).
  function setCaption(text, opts) {
    opts = opts || {};
    captionMsgEl.textContent = text || '';
    captionMsgEl.classList.toggle('partial', !!opts.partial);
    captionMsgEl.classList.toggle('prompt', !!opts.prompt);
    captionEl.classList.toggle('hidden', !text);
  }

  const PILL_TEXT = {auto: 'auto', ask: 'waiting', allowed: 'done', declined: 'declined', redirected: 'redirected', preapproved: 'pre-approved', limit: 'limit'};

  // Hide an empty bubble/row (no awkward blank box in the card) and show it
  // once it has content.
  function setBubble(rowEl, msgEl, text) {
    msgEl.textContent = text;
    rowEl.classList.toggle('hidden', !text);
  }

  function clearReply() {
    replyGen++;
    replyQueue = [];
    typing = false;
    if (pendingTimeout !== null) { clearTimeout(pendingTimeout); pendingTimeout = null; }
  }

  function clearTurn() {
    model.heard = ''; model.reply = ''; replySentences = []; clearReply();
    setBubble(heardRowEl, heardEl, ''); heardEl.classList.remove('partial');
    setBubble(replyRowEl, replyEl, '');
    model.tool = null;
    badgeEl.className = 'badge'; badgeEl.textContent = '';
    toolTitleEl.textContent = ''; detailEl.textContent = '';
    pillEl.className = 'pill'; pillEl.textContent = '';
    actionEl.classList.add('hidden');
    model.prompt = ''; promptEl.textContent = ''; promptRowEl.classList.add('hidden');
    hintEl.textContent = ''; hintRowEl.classList.add('hidden');
    countdownEl.classList.add('hidden');
    countdownBarEl.style.transition = 'none'; countdownBarEl.style.width = '100%';
    setCaption('');
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
        replyRowEl.classList.toggle('hidden', replyEl.textContent.length === 0);
        if (i < s.length) { pendingTimeout = setTimeout(step, 50); return; }
      } catch (e) {
        typing = false; pendingTimeout = null; console.error('hud typewriter step failed', e); typeNext(); return;
      }
      typing = false; pendingTimeout = null;
      typeNext();
    };
    step();
  }

  // Icon-render mode (?icon=1): used by scripts/make_icon.py to screenshot
  // the orb alone (no card/text/caption) at a large size for the app icon.
  // Applied as a body class so hud.css can hide the card chrome and scale
  // the orb canvas to fill the viewport via CSS; the renderer below uses a
  // 512-logical / 1024-pixel backing store in this mode, so make_icon.py
  // screenshots a 1024 px viewport at deviceScaleFactor 1.
  const ICON_MODE = (() => {
    try {
      return new URLSearchParams(location.search).get('icon') === '1';
    } catch (e) { return false; }
  })();
  if (ICON_MODE) document.body.classList.add('icon-mode');

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
            model.state = payload;
            statusEl.dataset.state = payload;
            captionEl.dataset.state = payload;
            statusLabelEl.textContent = STATUS_LABELS[payload] || '';
            if (payload === 'thinking') setCaption('Thinking…');
            else if (payload === 'error') setCaption('Error');
            break;
          case 'heard_partial': {
            const s = String(payload ?? '');
            setBubble(heardRowEl, heardEl, s);
            heardEl.classList.add('partial');
            setCaption(s, {partial: true});
            break;
          }
          case 'heard': {
            // A new user utterance (including a follow-up, which never
            // passes through 'listening') starts a fresh turn: clear the
            // previous reply/tool state so it doesn't bleed into this one.
            clearTurn();
            model.heard = payload || '';
            setBubble(heardRowEl, heardEl, model.heard);
            heardEl.classList.remove('partial');
            setCaption(model.heard);
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
            model.reply = joined; replyQueue.push(s); typeNext();
            // Caption shows only the sentence currently being spoken, set
            // instantly (no typewriter) rather than the accumulated reply.
            setCaption(s);
            break;
          }
          case 'tool': {
            const t = payload && typeof payload === 'object' ? payload : {};
            const decision = t.decision || '';
            const summary = t.summary || '';
            model.tool = t;
            actionEl.classList.remove('hidden');
            badgeEl.className = 'badge ' + decision;
            badgeEl.textContent = {auto:'⚡', ask:'?', allowed:'✓', declined:'✕', redirected:'↪', preapproved:'⚡', limit:'⏳'}[decision] || '';
            toolTitleEl.textContent = summary.length > 60 ? summary.slice(0, 59) + '…' : summary;
            // The final allowed/declined/redirected event doesn't repeat `detail` — keep
            // whatever the preceding 'ask' event already put there instead
            // of blanking it out.
            if (typeof t.detail === 'string') detailEl.textContent = t.detail;
            pillEl.className = 'pill ' + (PILL_TEXT[decision] || '');
            pillEl.textContent = PILL_TEXT[decision] || '';
            if (decision === 'ask') {
              // No "say yes or no" text here: the #status label already
              // says "Say yes or no" while confirming, so the hint row is
              // reserved for the question itself (set by the 'prompt' event
              // below), not a duplicate of the status label.
              model.confirmTimeoutMs = (+t.timeout_ms) || 8000;
              // Countdown starts here (once the question has actually been
              // spoken and we're about to start listening), not when the
              // 'confirming' state was entered.
              model.confirmStart = performance.now();
              countdownEl.classList.remove('hidden');
              countdownBarEl.style.transition = 'none';
              countdownBarEl.style.width = '100%';
              // Force a reflow so the width reset above is applied before the
              // transition below kicks in, otherwise the browser may coalesce
              // both style writes into a single paint and skip the shrink.
              void countdownBarEl.offsetWidth;
              countdownBarEl.style.transition = 'width ' + model.confirmTimeoutMs + 'ms linear';
              countdownBarEl.style.width = '0%';
              if (model.prompt) setCaption(model.prompt + ' · say yes or no', {prompt: true});
            } else {
              model.prompt = ''; promptEl.textContent = ''; promptRowEl.classList.add('hidden');
              hintEl.textContent = ''; hintRowEl.classList.add('hidden');
              countdownEl.classList.add('hidden');
            }
            break;
          }
          case 'prompt': {
            // The confirmation question, spoken right before we start
            // listening. Shown immediately in the prompt row, alongside a
            // short hint on how to answer (the status label already reads
            // "Say yes or no" while confirming, but the hint row spells out
            // the exact words expected). The countdown bar itself doesn't
            // start until the 'tool' ask event.
            const s = String(payload ?? '');
            model.prompt = s;
            promptEl.textContent = s;
            promptRowEl.classList.toggle('hidden', !s);
            hintEl.textContent = s ? 'say "yes" or "no"' : '';
            hintRowEl.classList.toggle('hidden', !s);
            if (s) setCaption(s, {prompt: true});
            break;
          }
          case 'mic': model.mic = Math.max(0, Math.min(1, +payload || 0)); break;
          case 'voice': model.voice = payload; model.voiceStart = performance.now(); break;
          case 'warm': model.ready = !!(payload && payload.ready); break;
          case 'hud': {
            // Which brain is answering ("Codex", "Claude (for Codex)" while
            // standing in). The mode/config payloads are handled by the
            // window itself and never reach here.
            if (payload && typeof payload === 'object' && typeof payload.backend === 'string') {
              brainEl.textContent = payload.backend ? 'Brain: ' + payload.backend : '';
            }
            break;
          }
        }
      } catch (err) {
        console.error('hud.push failed', err);
      }
    },
    state() {
      return {state:model.state, heard:model.heard, reply:model.reply, tool:model.tool, mic:model.mic, ready:model.ready,
              particles:activeCount(), intensity:cfg.intensity};
    },
    setMode(mode) {
      document.body.classList.toggle('mini', mode === 'mini');
      applyMode();
    },
    // Live orb config from settings: {particles: 500..8000, intensity: 0.2..2}.
    // Rebuilds the particle buffers only when the active count changes.
    configure(c) {
      c = c && typeof c === 'object' ? c : {};
      if (c.particles !== undefined && c.particles !== null && isFinite(+c.particles)) {
        cfg.particles = Math.round(Math.max(500, Math.min(8000, +c.particles)));
      }
      if (c.intensity !== undefined && c.intensity !== null && isFinite(+c.intensity)) {
        cfg.intensity = Math.max(0.2, Math.min(2, +c.intensity));
      }
      ensureParticles();
    },
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

  if (ICON_MODE) {
    hud.push({kind: 'state', payload: 'speaking'});
    // a synthetic envelope so the speaking spokes are in the icon
    const lv = []; for (let i = 0; i < 200; i++) lv.push(0.35 + 0.55 * Math.abs(Math.sin(i * 0.9) * Math.sin(i * 0.23)));
    hud.push({kind: 'voice', payload: {step_ms: 50, levels: lv}});
  }


  // ---- orb renderer: particle orb -------------------------------------------
  // A few thousand glow sprites on the surface of a sphere (Fibonacci
  // distribution), rotated + jittered on the CPU into typed arrays every
  // frame and drawn additively. All per-state look parameters (rotation,
  // swirl, radius, brightness, squeeze, three colours) live in `target` and
  // are eased into `cur` (~400 ms), so state changes never pop.
  const canvas = $('orb'), ctx = canvas.getContext('2d');
  // Icon mode renders a 512-logical / 1024-pixel canvas (CSS scales it to
  // the viewport) so the sphere/sprite proportions match the 170 px HUD.
  const dpr = ICON_MODE ? 2 : Math.max(1, window.devicePixelRatio || 1);
  const reducedMq = (() => {
    try { return window.matchMedia('(prefers-reduced-motion: reduce)'); } catch (e) { return null; }
  })();
  const reducedMotion = () => !!(reducedMq && reducedMq.matches);

  const cfg = {particles: 4000, intensity: ICON_MODE ? 1.4 : 1.0};   // icon: stronger glow at small sizes
  const FULL_SIZE = 170, MINI_SIZE = 64, ICON_SIZE = 512;
  let SIZE = FULL_SIZE, CX = SIZE / 2, CY = SIZE / 2, R = 66, SCALE = 1;

  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const hex = h => { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; };
  const grey = c => { const l = 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]; return c.map(v => v * 0.2 + l * 0.8); };
  const GOLD1 = hex('#f0c36c'), GOLD2 = hex('#ffb454'), BLUE = hex('#7ad0ff');

  // Parameter vector layout (Float32Array): see P_* indices.
  const P_ROT = 0, P_SWIRL = 1, P_BREATHE = 2, P_RADIUS = 3, P_BRIGHT = 4, P_SQUEEZE = 5,
        P_C1 = 6, P_C2 = 9, P_C3 = 12, P_RING = 15, P_LEN = 16;
  function params(o) {
    const v = new Float32Array(P_LEN);
    v[P_ROT] = o.rot; v[P_SWIRL] = o.swirl; v[P_BREATHE] = o.breathe; v[P_RADIUS] = o.radius;
    v[P_BRIGHT] = o.bright; v[P_SQUEEZE] = o.squeeze; v[P_RING] = o.ring || 0;
    for (let i = 0; i < 3; i++) { v[P_C1 + i] = o.c1[i]; v[P_C2 + i] = o.c2[i]; v[P_C3 + i] = o.c3[i]; }
    return v;
  }
  const STATES = {
    idle:       params({rot:0.15, swirl:0,   breathe:0.03, radius:1.0, bright:0.42, squeeze:1, c1:GOLD1, c2:GOLD2, c3:BLUE}),
    warming:    params({rot:0.10, swirl:0,   breathe:0,    radius:1.0, bright:0.36, squeeze:1,
                        c1:grey(GOLD1), c2:grey(GOLD2), c3:grey(BLUE)}),
    listening:  params({rot:0.25, swirl:0,   breathe:0,    radius:1.0, bright:0.66, squeeze:1, c1:GOLD1, c2:GOLD2, c3:BLUE}),
    followup:   params({rot:0.22, swirl:0,   breathe:0,    radius:1.0, bright:0.60, squeeze:1, c1:GOLD1, c2:GOLD2, c3:BLUE}),
    thinking:   params({rot:0.45, swirl:1.0, breathe:0,    radius:1.0, bright:0.72, squeeze:1,
                        c1:GOLD1, c2:hex('#ff9a3c'), c3:BLUE}),
    speaking:   params({rot:0.25, swirl:0,   breathe:0,    radius:1.0, bright:0.72, squeeze:1, c1:GOLD1, c2:GOLD2, c3:BLUE}),
    confirming: params({rot:0.20, swirl:0,   breathe:0,    radius:0.8, bright:0.68, squeeze:0.35, ring:1,
                        c1:hex('#ff9a5c'), c2:hex('#ff8a3c'), c3:hex('#ffc9a0')}),
    error:      params({rot:0.05, swirl:0,   breathe:0,    radius:1.0, bright:0.72, squeeze:1,
                        c1:hex('#ff6a6a'), c2:hex('#ff5a5a'), c3:hex('#ffb0b0')}),
  };
  const cur = new Float32Array(STATES.idle);
  let targetVec = STATES.idle;

  // ---- particle buffers (rebuilt only when the active count changes) ----
  let N = 0;
  let base, jw, jp, scat, kind, sz, px, py, pz, bucketIdx, bucketOf;
  const NB = 6;                                  // depth/alpha buckets
  const bucketCount = new Int32Array(NB), bucketStart = new Int32Array(NB + 1);
  const bucketAlpha = new Float32Array(NB);
  for (let b = 0; b < NB; b++) bucketAlpha[b] = 0.10 + 0.90 * Math.pow(b / (NB - 1), 1.6);

  function activeCount() {
    const mini = document.body.classList.contains('mini');
    return Math.max(1, mini ? Math.round(cfg.particles / 4) : cfg.particles);
  }
  function buildParticles(n) {
    N = n;
    base = new Float32Array(n * 3); jw = new Float32Array(n); jp = new Float32Array(n); scat = new Float32Array(n);
    kind = new Uint8Array(n); sz = new Uint8Array(n);
    px = new Float32Array(n); py = new Float32Array(n); pz = new Float32Array(n);
    bucketIdx = new Uint16Array(n); bucketOf = new Uint8Array(n);
    const rnd = mulberry32(42);
    const GA = Math.PI * (3 - Math.sqrt(5));
    // Fibonacci spacing keeps the surface evenly covered; a random nudge of
    // about one spacing per point hides the lattice (otherwise the spiral
    // reads as a moiré grid when it rotates).
    const nudge = Math.sqrt(4 * Math.PI / n) * 0.8;
    for (let i = 0; i < n; i++) {
      const y0 = 1 - (i + 0.5) * 2 / n, r0 = Math.sqrt(Math.max(0, 1 - y0 * y0)), th = GA * i;
      let x = Math.cos(th) * r0 + (rnd() - 0.5) * nudge, y = y0 + (rnd() - 0.5) * nudge, z = Math.sin(th) * r0 + (rnd() - 0.5) * nudge;
      const inv = 1 / Math.sqrt(x * x + y * y + z * z);
      base[3 * i] = x * inv; base[3 * i + 1] = y * inv; base[3 * i + 2] = z * inv;
      jw[i] = 0.5 + rnd() * 1.3; jp[i] = rnd() * Math.PI * 2; scat[i] = 0.25 + rnd() * 0.75;
      const k = rnd(); kind[i] = k < 0.08 ? 2 : (k < 0.5 ? 0 : 1);
      const s = rnd(); sz[i] = s < 0.62 ? 0 : (s < 0.92 ? 1 : 2);
      if (kind[i] === 2 && sz[i] > 1) sz[i] = 1;   // accents stay small
    }
  }
  function ensureParticles() {
    const n = activeCount();
    if (n !== N) buildParticles(n);
  }

  // ---- sprites: pre-rendered radial glows, 3 kinds x 3 sizes ----
  const sprites = [[], [], []];
  let glowSprite = null;
  const spriteKey = new Int16Array(9).fill(-1);
  let spriteScale = 1;
  function makeSprite(rgb, radius, coreMix, edgeAlpha) {
    const devR = Math.max(1, Math.ceil(radius * dpr));
    const s = devR * 2 + 2;
    const c = document.createElement('canvas'); c.width = c.height = s;
    const g = c.getContext('2d');
    const core = rgb.map(v => Math.round(v + (255 - v) * coreMix));
    const mid = rgb.map(Math.round);
    const grad = g.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, devR);
    grad.addColorStop(0, `rgba(${core[0]},${core[1]},${core[2]},1)`);
    grad.addColorStop(0.3, `rgba(${mid[0]},${mid[1]},${mid[2]},${edgeAlpha})`);
    grad.addColorStop(0.6, `rgba(${mid[0]},${mid[1]},${mid[2]},${edgeAlpha * 0.26})`);
    grad.addColorStop(1, `rgba(${mid[0]},${mid[1]},${mid[2]},0)`);
    g.fillStyle = grad; g.fillRect(0, 0, s, s);
    return {c, w: s / dpr, h: s / (2 * dpr)};
  }
  function buildSprites(force) {
    let changed = force;
    for (let i = 0; i < 9; i++) {
      const q = Math.round(cur[P_C1 + i] / 6);       // quantised: rebuild only on visible change
      if (q !== spriteKey[i]) { spriteKey[i] = q; changed = true; }
    }
    if (!changed) return;
    const radii = [0.95, 1.4, 2.0];
    for (let k = 0; k < 3; k++) {
      const rgb = [cur[P_C1 + 3 * k], cur[P_C1 + 3 * k + 1], cur[P_C1 + 3 * k + 2]];
      for (let s = 0; s < 3; s++) sprites[k][s] = makeSprite(rgb, radii[s] * spriteScale, 0.42, 0.9);
    }
    // body glow: the sphere's soft interior, drawn once behind the particles
    const c2 = [cur[P_C2], cur[P_C2 + 1], cur[P_C2 + 2]];
    glowSprite = makeSprite(c2, R * 0.95, 0.15, 0.32);
  }

  // ---- HUD ring: faint circle + ticks (static path) and a slow arc ----
  let ringPath = null;
  function buildRing() {
    ringPath = new Path2D();
    const rr = R * 1.16;
    ringPath.arc(CX, CY, rr, 0, Math.PI * 2);
    for (let i = 0; i < 48; i++) {
      const a = (i / 48) * Math.PI * 2, len = (i % 12 === 0) ? 3.2 : 1.6;
      const s = rr + 1.5;
      ringPath.moveTo(CX + Math.cos(a) * s, CY + Math.sin(a) * s);
      ringPath.lineTo(CX + Math.cos(a) * (s + len * SCALE), CY + Math.sin(a) * (s + len * SCALE));
    }
  }

  // Logical canvas size follows the mode (170 full / 64 mini; icon mode keeps
  // 170 and lets CSS + a high devicePixelRatio scale it up), so sprites stay
  // 1-2 px on screen in every mode instead of being downsampled.
  function applyMode() {
    const mini = document.body.classList.contains('mini');
    SIZE = ICON_MODE ? ICON_SIZE : (mini ? MINI_SIZE : FULL_SIZE);
    CX = CY = SIZE / 2; R = SIZE * (66 / 170); SCALE = SIZE / FULL_SIZE;
    spriteScale = Math.max(0.55, SCALE) * (ICON_MODE ? 1.2 : 1);
    canvas.width = Math.round(SIZE * dpr); canvas.height = Math.round(SIZE * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ensureParticles();
    buildSprites(true);
    buildRing();
  }

  // ---- orbit rings (faint great circles, drawn as polylines) ----
  const ORBITS = [
    // [tilt about X, spin multiplier, phase]
    [0.55, 1.0, 0.0],
    [-0.9, 0.6, 1.9],
    [1.25, -0.8, 3.7],
  ];
  const ORB_SEG = 72;
  const orbX = new Float32Array(ORB_SEG + 1), orbY = new Float32Array(ORB_SEG + 1), orbZ = new Float32Array(ORB_SEG + 1);
  function drawOrbitRings(rr, rot, tiltC, tiltS, squeeze, bright) {
    const c2r = cur[P_C2], c2g = cur[P_C2 + 1], c2b = cur[P_C2 + 2];
    ctx.strokeStyle = css(c2r, c2g, c2b, 1);
    ctx.lineWidth = Math.max(0.6, 1.0 * SCALE);
    const gain = Math.min(1.4, cfg.intensity) * Math.min(1, 0.55 + bright * 0.6);
    for (let k = 0; k < ORBITS.length; k++) {
      const tilt = ORBITS[k][0], a0 = rot * ORBITS[k][1] + ORBITS[k][2];
      const tc = Math.cos(tilt), ts = Math.sin(tilt), ca = Math.cos(a0), sa = Math.sin(a0);
      for (let i = 0; i <= ORB_SEG; i++) {
        const t = (i / ORB_SEG) * Math.PI * 2;
        // circle in the XZ plane, tilted about X, then spun about Y
        let x = Math.cos(t), y = 0, z = Math.sin(t);
        const y1 = y * tc - z * ts, z1 = y * ts + z * tc;
        const x2 = x * ca + z1 * sa, z2 = -x * sa + z1 * ca;
        const y2 = y1 * squeeze;
        const ty = y2 * tiltC - z2 * tiltS, tz = y2 * tiltS + z2 * tiltC;
        orbX[i] = CX + x2 * rr; orbY[i] = CY + ty * rr; orbZ[i] = tz;
      }
      // two passes: back half (dim) then front half (brighter), split by depth
      for (let pass = 0; pass < 2; pass++) {
        ctx.globalAlpha = (pass === 0 ? 0.16 : 0.5) * gain;
        ctx.beginPath();
        let open = false;
        for (let i = 0; i <= ORB_SEG; i++) {
          const front = orbZ[i] >= 0;
          if (front === (pass === 1)) {
            if (!open) { ctx.moveTo(orbX[i], orbY[i]); open = true; } else ctx.lineTo(orbX[i], orbY[i]);
          } else open = false;
        }
        ctx.stroke();
      }
    }
  }

  // ---- transient effects ----
  const ripples = [{born: -1}, {born: -1}, {born: -1}, {born: -1}];
  const RIPPLE_MS = 900;
  let lastRipple = 0, prevMic = 0;
  const N_SPOKES = 36;
  const spokeHist = new Float32Array(N_SPOKES);
  let spokeHead = 0, lastSpoke = 0, vSmooth = 0;
  let lastState = 'idle', errorAt = -1, animT = 0, rotA = 0, ringA = 0, lastDraw = -1e9;

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

  // Bench/diagnostics hook (scripts/orb_bench.py): per-frame draw time.
  const stats = {frameMs: 0, frames: 0, totalMs: 0, reset() { this.frames = 0; this.totalMs = 0; this.frameMs = 0; }};
  window.__hud = stats;

  function css(r, g, b, a) { return `rgba(${r | 0},${g | 0},${b | 0},${a})`; }

  function frame(now) {
    rafId = 0;
    // idle: 30 fps is plenty for a slow breathe; everything else runs at 60.
    if (model.state === 'idle' && now - lastDraw < 30) {
      if (rafActive) rafId = requestAnimationFrame(frame);
      return;
    }
    const tStart = performance.now();
    const dt = Math.min(0.1, (now - t0) / 1000); t0 = now; lastDraw = now;
    const reduced = reducedMotion();
    const state = model.state;

    if (state !== lastState) {
      targetVec = STATES[state] || STATES.idle;
      if (state === 'error') errorAt = now;
      lastState = state;
    }
    // ease every parameter toward its target: ~95% of the way in 400 ms
    const k = 1 - Math.exp(-dt / 0.13);
    for (let i = 0; i < P_LEN; i++) cur[i] += (targetVec[i] - cur[i]) * k;
    buildSprites(false);

    micSmooth += (model.mic - micSmooth) * 0.25;
    statusLevelEl.style.width = (Math.max(0, Math.min(1, micSmooth)) * 100) + '%';
    const listening = state === 'listening' || state === 'followup';
    const vRaw = state === 'speaking' ? voiceLevel(now) : 0;
    vSmooth += (vRaw - vSmooth) * 0.45;

    if (!reduced) { animT += dt; rotA += cur[P_ROT] * dt; ringA += 0.25 * dt; }

    // per-state modulation of radius / brightness
    let radius = cur[P_RADIUS] * (1 + cur[P_BREATHE] * Math.sin(animT * Math.PI / 2));   // ~4 s breathe
    let bright = cur[P_BRIGHT];
    if (listening) { radius *= 1 + 0.12 * micSmooth; bright += 0.4 * micSmooth; }
    if (state === 'speaking') { radius *= 1 + 0.12 * vSmooth; bright += 0.35 * vSmooth; }
    if (state === 'confirming') bright *= 0.75 + 0.25 * Math.sin(now / 1000 * Math.PI * 2);
    // error: burst outward for ~250 ms, reassemble by ~800 ms
    let scatter = 0, flash = 0;
    if (state === 'error' && errorAt >= 0) {
      const te = (now - errorAt) / 1000;
      scatter = te < 0.25 ? te / 0.25 : Math.max(0, 1 - (te - 0.25) / 0.55);
      flash = Math.max(0, 1 - te / 0.3);
    }
    bright *= cfg.intensity;
    const swirl = reduced ? 0 : cur[P_SWIRL];
    const squeeze = cur[P_SQUEEZE], ring = cur[P_RING];
    const tiltC = Math.cos(0.38), tiltS = Math.sin(0.38);
    const RR = R * radius;

    // ---- project particles (hot loop; no allocations) ----
    for (let b = 0; b < NB; b++) bucketCount[b] = 0;
    const cosR = Math.cos(rotA), sinR = Math.sin(rotA);
    for (let i = 0; i < N; i++) {
      const bx = base[3 * i], by = base[3 * i + 1], bz = base[3 * i + 2];
      const ph = animT * jw[i] + jp[i];
      const sJ = Math.sin(ph);
      let r = 1 + 0.02 * sJ + scatter * scat[i] * 0.55;
      let x, z;
      if (swirl !== 0) {
        // vortex: differential rotation about Y by latitude + a travelling
        // wave, i.e. a cheap curl-ish flow along the surface
        const a = rotA + swirl * (0.9 * Math.sin(3 * by + animT * 1.3) + 0.45 * Math.sin(2.5 * bx + animT * 0.8));
        const c = Math.cos(a), s = Math.sin(a);
        x = bx * c + bz * s; z = -bx * s + bz * c;
      } else {
        x = bx * cosR + bz * sinR; z = -bx * sinR + bz * cosR;
      }
      let y = by * squeeze + 0.012 * Math.cos(ph * 0.7);   // drifting noise
      if (ring > 0.001) {
        // confirming: pull points out toward the equator's rim so the
        // squeezed sphere reads as a torus/ring rather than a flat blob
        const rr = Math.sqrt(x * x + z * z) + 1e-4;
        const want = 0.84 + 0.16 * rr;
        const g = 1 + ring * (want / rr - 1);
        x *= g; z *= g;
      }
      // tilt the spin axis toward the viewer a little
      const ty = y * tiltC - z * tiltS, tz = y * tiltS + z * tiltC;
      px[i] = CX + x * RR * r; py[i] = CY + ty * RR * r; pz[i] = tz;
      const u = (tz + 1) * 0.5;
      const b = (u * (NB - 1) + 0.5) | 0;
      bucketOf[i] = b; bucketCount[b]++;
    }
    bucketStart[0] = 0;
    for (let b = 0; b < NB; b++) { bucketStart[b + 1] = bucketStart[b] + bucketCount[b]; bucketCount[b] = bucketStart[b]; }
    for (let i = 0; i < N; i++) bucketIdx[bucketCount[bucketOf[i]]++] = i;

    // ---- draw ----
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1;
    ctx.clearRect(0, 0, SIZE, SIZE);

    // body glow behind everything
    if (glowSprite) {
      ctx.globalAlpha = Math.min(1, bright * 0.7);
      const gw = glowSprite.w * radius;
      ctx.drawImage(glowSprite.c, CX - gw / 2, CY - gw / 2, gw, gw);
    }

    // HUD ring: thin, ticked, with a slow arc; deliberately faint
    if (ringPath) {
      const c1r = cur[P_C1], c1g = cur[P_C1 + 1], c1b = cur[P_C1 + 2];
      ctx.lineWidth = Math.max(0.5, 0.6 * SCALE);
      ctx.strokeStyle = css(c1r, c1g, c1b, 1);
      ctx.globalAlpha = 0.11 * Math.min(1.4, cfg.intensity);
      ctx.stroke(ringPath);
      ctx.globalAlpha = 0.22 * Math.min(1.4, cfg.intensity);
      ctx.lineWidth = Math.max(0.7, 1.0 * SCALE);
      ctx.beginPath(); ctx.arc(CX, CY, R * 1.16, ringA, ringA + 1.1); ctx.stroke();
      ctx.beginPath(); ctx.arc(CX, CY, R * 1.16, ringA + Math.PI, ringA + Math.PI + 0.35); ctx.stroke();
    }

    // orbit rings: a few faint great circles on the sphere, each tilted its
    // own way and spinning with (or against) the particles — structure
    // behind the dust. Back halves are dimmer so they read as 3D.
    ctx.globalCompositeOperation = 'lighter';
    drawOrbitRings(RR * 1.1, rotA, tiltC, tiltS, squeeze, bright);

    // particles, additive, back (dim) to front (bright)
    for (let b = 0; b < NB; b++) {
      const a = Math.min(1, bright * bucketAlpha[b]);
      if (a < 0.045) continue;   // far side at idle: invisible, not worth 10% of the draw calls
      ctx.globalAlpha = a;
      const end = bucketStart[b + 1];
      const back = b < NB / 2 ? 1 : 0;
      for (let j = bucketStart[b]; j < end; j++) {
        const i = bucketIdx[j];
        let s = sz[i] - back; if (s < 0) s = 0;
        const sp = sprites[kind[i]][s];
        ctx.drawImage(sp.c, px[i] - sp.h, py[i] - sp.h, sp.w, sp.w);
      }
    }

    // listening: ripples spawned on mic peaks
    if (listening && !reduced && micSmooth > 0.22 && micSmooth > prevMic + 0.025 && now - lastRipple > 220) {
      for (let i = 0; i < ripples.length; i++) {
        if (ripples[i].born < 0 || now - ripples[i].born > RIPPLE_MS) { ripples[i].born = now; lastRipple = now; break; }
      }
    }
    prevMic = micSmooth;
    ctx.lineWidth = Math.max(0.6, 0.8 * SCALE);
    for (let i = 0; i < ripples.length; i++) {
      const rp = ripples[i];
      if (rp.born < 0) continue;
      const f = (now - rp.born) / RIPPLE_MS;
      if (f >= 1 || reduced) { rp.born = -1; continue; }
      ctx.globalAlpha = 0.4 * (1 - f) * (1 - f) * Math.min(1.5, cfg.intensity);
      ctx.strokeStyle = css(cur[P_C3], cur[P_C3 + 1], cur[P_C3 + 2], 1);
      ctx.beginPath(); ctx.arc(CX, CY, Math.min(SIZE / 2 - 1, RR * (1.02 + 0.22 * f)), 0, Math.PI * 2); ctx.stroke();
    }

    // speaking: radial spokes whose lengths trace the recent level history
    if (state === 'speaking') {
      if (now - lastSpoke > 45) { spokeHist[spokeHead] = vSmooth; spokeHead = (spokeHead + 1) % N_SPOKES; lastSpoke = now; }
      // one path + one stroke for all spokes (a stroke per spoke is ~2 ms in
      // WebKit); flicker is expressed as per-spoke length instead of alpha
      ctx.strokeStyle = css(cur[P_C1], cur[P_C1 + 1], cur[P_C1 + 2], 1);
      ctx.lineWidth = Math.max(0.5, 0.7 * SCALE);
      const r0 = RR * 1.05;
      ctx.globalAlpha = Math.min(1, (0.22 + 0.3 * vSmooth) * cfg.intensity);
      ctx.beginPath();
      for (let s = 0; s < N_SPOKES; s++) {
        const lvl = spokeHist[(spokeHead + s) % N_SPOKES];
        if (lvl < 0.02) continue;
        const a = ringA * 0.5 + (s / N_SPOKES) * Math.PI * 2;
        const len = (1.5 + 9 * lvl) * (0.8 + 0.2 * Math.sin(now / 70 + s * 1.7)) * SCALE;
        ctx.moveTo(CX + Math.cos(a) * r0, CY + Math.sin(a) * r0);
        ctx.lineTo(CX + Math.cos(a) * (r0 + len), CY + Math.sin(a) * (r0 + len));
      }
      ctx.stroke();
    } else if (spokeHist[spokeHead] !== 0) {
      spokeHist.fill(0);
    }

    // error: brief red flash over the body
    if (flash > 0) {
      ctx.globalAlpha = 0.35 * flash;
      ctx.fillStyle = css(255, 80, 80, 1);
      ctx.beginPath(); ctx.arc(CX, CY, RR * 1.1, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';

    // confirming: "?" glyph, always; countdown arc only once the question
    // has actually been spoken and the 'tool' ask event set confirmStart.
    if (state === 'confirming') {
      const stroke = css(cur[P_C1], cur[P_C1 + 1], cur[P_C1 + 2], 1);
      if (model.confirmStart !== null) {
        const frac = Math.max(0, 1 - (now - model.confirmStart) / model.confirmTimeoutMs);
        ctx.beginPath(); ctx.arc(CX, CY, R * 0.88, -Math.PI / 2, -Math.PI / 2 + frac * Math.PI * 2);
        ctx.lineWidth = 3 * SCALE; ctx.strokeStyle = stroke; ctx.stroke();
      }
      ctx.fillStyle = '#fff'; ctx.font = 'bold ' + Math.round(26 * SCALE) + 'px system-ui';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('?', CX, CY + 1);
    }
    // warming spinner
    if (state === 'warming') {
      const a0 = animT * 2.4;
      ctx.beginPath(); ctx.arc(CX, CY, R * 0.88, a0, a0 + Math.PI * 0.6);
      ctx.lineWidth = 3 * SCALE; ctx.strokeStyle = css(cur[P_C1], cur[P_C1 + 1], cur[P_C1 + 2], 1); ctx.stroke();
    }

    const ms = performance.now() - tStart;
    stats.frameMs = ms; stats.frames++; stats.totalMs += ms;
    if (rafActive && rafId === 0) rafId = requestAnimationFrame(frame);
  }

  applyMode();
  if (rafActive && rafId === 0) rafId = requestAnimationFrame(frame);
})();
