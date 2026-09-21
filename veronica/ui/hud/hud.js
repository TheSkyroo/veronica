(() => {
  const model = {state:'idle', heard:'', reply:'', tool:null, prompt:'', mic:0, ready:true,
                 voice:null, voiceStart:0, confirmStart:null, confirmTimeoutMs:8000};
  const MAX_REPLY_LEN = 220;
  let micSmooth = 0, replyQueue = [], typing = false, replyGen = 0, pendingTimeout = null;
  let replySentences = [];

  const STATUS_LABELS = {
    idle: '', warming: 'Warming up…', listening: 'Listening…', thinking: 'Thinking…',
    speaking: 'Speaking', followup: 'Listening…', confirming: 'Say yes or no', error: 'Error',
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
              particles:activeCount(), intensity:cfg.intensity, detail:cfg.detail};
    },
    setMode(mode) {
      document.body.classList.toggle('mini', mode === 'mini');
      applyMode();
    },
    // Live orb config from settings: {particles: 100..2000 (specks),
    // intensity: 0.2..2, detail: 0.5..1.5}. Rebuilds the geometry only when
    // the active speck count or the detail level changes.
    configure(c) {
      c = c && typeof c === 'object' ? c : {};
      if (c.particles !== undefined && c.particles !== null && isFinite(+c.particles)) {
        cfg.particles = Math.round(Math.max(100, Math.min(2000, +c.particles)));
      }
      if (c.intensity !== undefined && c.intensity !== null && isFinite(+c.intensity)) {
        cfg.intensity = Math.max(0.2, Math.min(2, +c.intensity));
      }
      if (c.detail !== undefined && c.detail !== null && isFinite(+c.detail)) {
        cfg.detail = Math.max(0.5, Math.min(1.5, +c.detail));
      }
      ensureGeometry();
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


  // ---- orb renderer: hologram orb ---------------------------------------------
  // A J.A.R.V.I.S.-style hologram globe: three nested spherical shells, each a
  // set of great-circle rings at fixed pseudo-random tilts, drawn as
  // *segmented* traces (lit runs with gaps, tick marks, chip blocks, a few
  // radial connectors between shells), a dense bright core (two bold rings,
  // a glow disc, drifting specks) and a sparse scatter of twinkling specks
  // in the volume between the shells. Everything is drawn additively with a
  // wide low-alpha pass under each stroke for bloom. Ring geometry is
  // precomputed once (unit-sphere points per ring, Float32Array), projected
  // per frame with a rotation about a tilted axis, and stroked once per
  // shell per depth bucket (front/back), never per segment. All per-state
  // look parameters live in `target` and are eased into `cur` (~400 ms).
  const canvas = $('orb'), ctx = canvas.getContext('2d');
  // Icon mode renders a 512-logical / 1024-pixel canvas (CSS scales it to
  // the viewport) so the proportions match the 170 px HUD.
  const dpr = ICON_MODE ? 2 : Math.max(1, window.devicePixelRatio || 1);
  const reducedMq = (() => {
    try { return window.matchMedia('(prefers-reduced-motion: reduce)'); } catch (e) { return null; }
  })();
  const reducedMotion = () => !!(reducedMq && reducedMq.matches);

  // particles = speck count (100..2000), intensity = glow multiplier (0.2..2),
  // detail = rings-per-shell and lit-fraction multiplier (0.5..1.5)
  const cfg = {particles: 400, intensity: ICON_MODE ? 1.4 : 1.0, detail: 1.0};
  const FULL_SIZE = 170, MINI_SIZE = 64, ICON_SIZE = 512;
  let SIZE = FULL_SIZE, CX = SIZE / 2, CY = SIZE / 2, R = 66, SCALE = 1, LW = 1;

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
  const CORE = hex('#ffd27a'), MID = hex('#ff9a3c'), RIM = hex('#d8641e'), COOL = hex('#8fd3ff');

  // Parameter vector layout (Float32Array): see P_* indices. c1/c2/c3 are the
  // inner/mid/outer shell colours (core / mid / rim of the palette).
  const P_ROT = 0, P_BREATHE = 1, P_RADIUS = 2, P_BRIGHT = 3, P_SQUEEZE = 4, P_RING = 5,
        P_LIT = 6, P_MARCH = 7, P_INNER = 8, P_C1 = 9, P_C2 = 12, P_C3 = 15, P_LEN = 18;
  function params(o) {
    const v = new Float32Array(P_LEN);
    v[P_ROT] = o.rot; v[P_BREATHE] = o.breathe; v[P_RADIUS] = o.radius; v[P_BRIGHT] = o.bright;
    v[P_SQUEEZE] = o.squeeze; v[P_RING] = o.ring || 0; v[P_LIT] = o.lit === undefined ? 0.55 : o.lit;
    v[P_MARCH] = o.march || 0; v[P_INNER] = o.inner || 1;
    for (let i = 0; i < 3; i++) { v[P_C1 + i] = o.c1[i]; v[P_C2 + i] = o.c2[i]; v[P_C3 + i] = o.c3[i]; }
    return v;
  }
  const STATES = {
    idle:       params({rot:0.12, breathe:0.02, radius:1.0, bright:0.50, squeeze:1, c1:CORE, c2:MID, c3:RIM}),
    warming:    params({rot:0.10, breathe:0,    radius:1.0, bright:0.40, squeeze:1,
                        c1:grey(CORE), c2:grey(MID), c3:grey(RIM)}),
    listening:  params({rot:0.22, breathe:0,    radius:1.0, bright:0.62, squeeze:1, c1:CORE, c2:MID, c3:RIM}),
    followup:   params({rot:0.20, breathe:0,    radius:1.0, bright:0.58, squeeze:1, c1:CORE, c2:MID, c3:RIM}),
    thinking:   params({rot:0.22, breathe:0,    radius:1.0, bright:0.70, squeeze:1, march:1, inner:3,
                        c1:hex('#ffc46a'), c2:MID, c3:hex('#ff9a3c')}),
    speaking:   params({rot:0.22, breathe:0,    radius:1.0, bright:0.70, squeeze:1, c1:CORE, c2:MID, c3:RIM}),
    confirming: params({rot:0.20, breathe:0,    radius:0.8, bright:0.66, squeeze:0.35, ring:1,
                        c1:hex('#ffd0a0'), c2:hex('#ff9a5c'), c3:hex('#ff8a3c')}),
    error:      params({rot:0.05, breathe:0,    radius:1.0, bright:0.72, squeeze:1,
                        c1:hex('#ffb0b0'), c2:hex('#ff6a6a'), c3:hex('#ff4a4a')}),
  };
  const cur = new Float32Array(STATES.idle);
  let targetVec = STATES.idle;

  // ---- geometry (rebuilt only when mode / detail / speck count changes) ----
  const SHELL_R = [0.55, 0.78, 1.0];
  const SHELL_SPEED = [-1.6, 1.25, 1.0];          // inner counter-rotates, faster
  const RINGS_BASE = [6, 8, 10];                   // rings per shell at detail 1 (inner..outer)
  const CORE_SPECKS = 200, MAX_CONNECTORS = 12;   // core: everything at r <= 0.35
  let nShells = 3, litScale = 1;
  // rings: flat point buffers + per-ring descriptors
  let rings = [];        // {shell, off, n, thr:Float32Array(n), feat:Uint16Array, ftype:Uint8Array, dir}
  let NP = 0, base, px, py, pz, scat, segCode;
  let conns = [];        // {shell, ring, seg}
  // core rings (fixed): two bold rings inside r <= 0.35
  const coreRings = [];
  let NC = 0, cbase, cpx, cpy, cpz;
  // specks: volume (between shells) + core
  let NS = 0, sbase, sw, sp, skind, ssz, spx, spy, spz, sbucket;
  let NCS = 0, csbase, csw, csp, cspx, cspy;
  const geomKey = {mode: '', detail: -1, specks: -1};

  function ringBasis(rnd, out, axis) {
    // unit normal -> two orthonormal tangents. With `axis`, the normal is
    // kept near the plane perpendicular to it (a meridian family sharing
    // poles, which is what makes a set of great circles read as a globe
    // cage rather than a tangle); otherwise it's uniformly random.
    let nx, ny, nz;
    if (axis) {
      const a = rnd() * Math.PI * 2, wob = (rnd() - 0.5) * 0.45;     // +-13 deg off the equatorial plane
      // tangents of the axis
      const [ax, ay, az] = axis;
      let tx, ty, tz;
      if (Math.abs(ay) < 0.9) { tx = az; ty = 0; tz = -ax; } else { tx = 0; ty = -az; tz = ay; }
      const il = 1 / Math.sqrt(tx * tx + ty * ty + tz * tz); tx *= il; ty *= il; tz *= il;
      const bx = ay * tz - az * ty, by = az * tx - ax * tz, bz = ax * ty - ay * tx;
      const cw = Math.cos(wob), sw = Math.sin(wob);
      nx = (tx * Math.cos(a) + bx * Math.sin(a)) * cw + ax * sw;
      ny = (ty * Math.cos(a) + by * Math.sin(a)) * cw + ay * sw;
      nz = (tz * Math.cos(a) + bz * Math.sin(a)) * cw + az * sw;
    } else {
      const z = rnd() * 2 - 1, a = rnd() * Math.PI * 2, r = Math.sqrt(1 - z * z);
      nx = r * Math.cos(a); ny = r * Math.sin(a); nz = z;
    }
    // u = normalize(n x (0,1,0)) unless n ~ y, then n x (1,0,0)
    let ux, uy, uz;
    if (Math.abs(ny) < 0.9) { ux = nz; uy = 0; uz = -nx; } else { ux = 0; uy = -nz; uz = ny; }
    const il = 1 / Math.sqrt(ux * ux + uy * uy + uz * uz); ux *= il; uy *= il; uz *= il;
    const vx = ny * uz - nz * uy, vy = nz * ux - nx * uz, vz = nx * uy - ny * ux;
    out[0] = ux; out[1] = uy; out[2] = uz; out[3] = vx; out[4] = vy; out[5] = vz;
  }

  function buildGeometry() {
    const mini = document.body.classList.contains('mini');
    nShells = mini ? 1 : 3;
    litScale = cfg.detail;
    const rnd = mulberry32(31);
    const B = new Float64Array(6);
    rings = []; conns = [];
    // pass 1: descriptors + point count
    let off = 0;
    for (let s = 0; s < 3; s++) {
      // per-shell axis, tilted 25..45 deg off the view direction so the
      // meridian family reads as nested ellipses sharing two poles
      const tilt = 0.45 + rnd() * 0.35, aa = rnd() * Math.PI * 2;
      const axis = [Math.sin(tilt) * Math.cos(aa), Math.sin(tilt) * Math.sin(aa), Math.cos(tilt)];
      const nr = Math.max(3, Math.min(15, Math.round(RINGS_BASE[s] * cfg.detail)));
      for (let j = 0; j < nr; j++) {
        const n = 60 + Math.floor(rnd() * 81);
        const d = {shell: s, off, n, thr: new Float32Array(n), feat: null, ftype: null, dir: (j & 1) ? 1 : -1,
                   B: new Float64Array(6)};
        ringBasis(rnd, B, j < Math.ceil(nr * 0.75) ? axis : null); d.B.set(B);
        // segmented mask: runs of 4..14 lit (outer shell 6..18: its long arcs
        // are what sells the globe), gaps of 3..10. thr < litFrac means lit;
        // lit segments get thr in [0, .5), gap segments fill in from the run
        // ends toward the gap middle as the lit fraction rises.
        let i = 0, lit = rnd() < 0.5;
        d.runScat = new Float32Array(n);
        const runMin = s === 2 ? 6 : 4, runSpan = s === 2 ? 13 : 11;
        while (i < n) {
          const len = lit ? runMin + Math.floor(rnd() * runSpan) : 3 + Math.floor(rnd() * 8);
          const end = Math.min(n, i + len);
          const rs = 0.3 + rnd() * 0.7;                   // error shatter: whole runs fly, not points
          for (let k = i; k < end; k++) {
            d.runScat[k] = rs;
            if (lit) d.thr[k] = rnd() * 0.5;
            else {
              const dist = Math.min(k - i, end - 1 - k) / Math.max(1, (end - i) / 2);   // 0 at edges, 1 mid
              d.thr[k] = 0.55 + 0.42 * dist + rnd() * 0.03;
            }
          }
          i = end; lit = !lit;
        }
        // trace features: ticks + chips at random segments (~1 per 9 segments)
        const nf = Math.max(2, Math.round(n / 9));
        d.feat = new Uint16Array(nf); d.ftype = new Uint8Array(nf); d.fsize = new Float32Array(nf);
        for (let f = 0; f < nf; f++) {
          d.feat[f] = Math.floor(rnd() * n);
          d.ftype[f] = rnd() < 0.55 ? 0 : 1;                       // 0 tick, 1 chip
          d.fsize[f] = d.ftype[f] === 0 ? 2 + rnd() * 2 : 3 + rnd() * 3;   // px at 170
        }
        rings.push(d);
        off += n;
      }
    }
    NP = off;
    base = new Float32Array(NP * 3); px = new Float32Array(NP); py = new Float32Array(NP); pz = new Float32Array(NP);
    scat = new Float32Array(NP); segCode = new Uint8Array(NP);
    for (const d of rings) {
      const [ux, uy, uz, vx, vy, vz] = d.B;
      for (let i = 0; i < d.n; i++) {
        const th = i / d.n * Math.PI * 2, c = Math.cos(th), s = Math.sin(th), k = 3 * (d.off + i);
        base[k] = ux * c + vx * s; base[k + 1] = uy * c + vy * s; base[k + 2] = uz * c + vz * s;
        scat[d.off + i] = d.runScat[i];
      }
    }
    // radial connectors from shell s to s+1 (max 12 on screen)
    const nConn = mini ? 0 : MAX_CONNECTORS;
    for (let c = 0; c < nConn; c++) {
      const shell = c % 2, cand = rings.filter(r => r.shell === shell);
      const rg = cand[Math.floor(rnd() * cand.length)];
      conns.push({ring: rg, seg: Math.floor(rnd() * rg.n)});
    }
    // core rings: two bold rings at r = 0.30 / 0.23
    coreRings.length = 0;
    const CN = 72;
    NC = CN * 2; cbase = new Float32Array(NC * 3); cpx = new Float32Array(NC); cpy = new Float32Array(NC); cpz = new Float32Array(NC);
    for (let r = 0; r < 2; r++) {
      ringBasis(rnd, B);
      const rad = r === 0 ? 0.27 : 0.20;
      coreRings.push({off: r * CN, n: CN, rad});
      for (let i = 0; i < CN; i++) {
        const th = i / CN * Math.PI * 2, c = Math.cos(th), s = Math.sin(th), k = 3 * (r * CN + i);
        cbase[k] = (B[0] * c + B[3] * s) * rad; cbase[k + 1] = (B[1] * c + B[4] * s) * rad; cbase[k + 2] = (B[2] * c + B[5] * s) * rad;
      }
    }
    // volume specks: between the shells, never on them
    const srnd = mulberry32(99);
    NS = activeCount();
    sbase = new Float32Array(NS * 3); sw = new Float32Array(NS); sp = new Float32Array(NS);
    skind = new Uint8Array(NS); ssz = new Float32Array(NS);
    spx = new Float32Array(NS); spy = new Float32Array(NS); spz = new Float32Array(NS); sbucket = new Uint8Array(NS);
    const bands = mini ? [[0.40, 0.95]] : [[0.38, 0.52], [0.58, 0.75], [0.81, 0.97]];
    for (let i = 0; i < NS; i++) {
      const band = bands[Math.floor(srnd() * bands.length)];
      const rr = band[0] + srnd() * (band[1] - band[0]);
      const z = srnd() * 2 - 1, a = srnd() * Math.PI * 2, q = Math.sqrt(1 - z * z);
      sbase[3 * i] = q * Math.cos(a) * rr; sbase[3 * i + 1] = q * Math.sin(a) * rr; sbase[3 * i + 2] = z * rr;
      sw[i] = 0.8 + srnd() * 2.2; sp[i] = srnd() * Math.PI * 2;
      skind[i] = srnd() < 0.06 ? 1 : 0;
      ssz[i] = 0.9 + srnd() * 0.5;
    }
    // core specks: inside r < 0.33, drifting
    NCS = mini ? 40 : CORE_SPECKS;
    csbase = new Float32Array(NCS * 3); csw = new Float32Array(NCS); csp = new Float32Array(NCS);
    cspx = new Float32Array(NCS); cspy = new Float32Array(NCS);
    for (let i = 0; i < NCS; i++) {
      const rr = 0.33 * Math.cbrt(srnd());
      const z = srnd() * 2 - 1, a = srnd() * Math.PI * 2, q = Math.sqrt(1 - z * z);
      csbase[3 * i] = q * Math.cos(a) * rr; csbase[3 * i + 1] = q * Math.sin(a) * rr; csbase[3 * i + 2] = z * rr;
      csw[i] = 0.5 + srnd() * 1.5; csp[i] = srnd() * Math.PI * 2;
    }
    geomKey.mode = mini ? 'mini' : 'full'; geomKey.detail = cfg.detail; geomKey.specks = NS;
  }
  function activeCount() {
    const mini = document.body.classList.contains('mini');
    return Math.max(1, mini ? Math.round(cfg.particles * 0.15) : cfg.particles);
  }
  function ensureGeometry() {
    const mini = document.body.classList.contains('mini');
    if (geomKey.mode !== (mini ? 'mini' : 'full') || geomKey.detail !== cfg.detail || geomKey.specks !== activeCount()) buildGeometry();
  }

  // ---- core glow disc: pre-rendered radial gradient (alpha .35 -> 0) ----
  let coreGlow = null, glowKey = -1;
  function buildGlow(force) {
    const q = Math.round(cur[P_C1] / 6) * 1000 + Math.round(cur[P_C2 + 1] / 6);
    if (!force && q === glowKey) return;
    glowKey = q;
    const rad = R * 0.42, devR = Math.max(2, Math.ceil(rad * dpr)), s = devR * 2 + 2;
    const c = document.createElement('canvas'); c.width = c.height = s;
    const g = c.getContext('2d');
    const c1 = [cur[P_C1] | 0, cur[P_C1 + 1] | 0, cur[P_C1 + 2] | 0], c2 = [cur[P_C2] | 0, cur[P_C2 + 1] | 0, cur[P_C2 + 2] | 0];
    const grad = g.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, devR);
    grad.addColorStop(0, `rgba(255,${Math.min(255, c1[1] + 40)},${Math.min(255, c1[2] + 80)},0.35)`);
    grad.addColorStop(0.25, `rgba(${c1[0]},${c1[1]},${c1[2]},0.28)`);
    grad.addColorStop(0.6, `rgba(${c2[0]},${c2[1]},${c2[2]},0.10)`);
    grad.addColorStop(1, `rgba(${c2[0]},${c2[1]},${c2[2]},0)`);
    g.fillStyle = grad; g.fillRect(0, 0, s, s);
    coreGlow = {c, w: s / dpr};
  }

  // ---- HUD frame: faint circle + 48 ticks (static path) and two slow arcs ----
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

  // Logical canvas size follows the mode (170 full / 64 mini / 512 icon).
  function applyMode() {
    const mini = document.body.classList.contains('mini');
    SIZE = ICON_MODE ? ICON_SIZE : (mini ? MINI_SIZE : FULL_SIZE);
    CX = CY = SIZE / 2; R = SIZE * (66 / 170); SCALE = SIZE / FULL_SIZE;
    LW = Math.max(0.55, SCALE) * (ICON_MODE ? 1.15 : 1);    // line-width scale; icon: thicker traces
    canvas.width = Math.round(SIZE * dpr); canvas.height = Math.round(SIZE * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ensureGeometry();
    buildGlow(true);
    buildRing();
  }

  // ---- transient effects ----
  const ripples = [{born: -1}, {born: -1}, {born: -1}, {born: -1}];
  const RIPPLE_MS = 900;
  let lastRipple = 0, prevMic = 0;
  const N_SPOKES = 24;
  const spokeHist = new Float32Array(N_SPOKES);
  let spokeHead = 0, lastSpoke = 0, vSmooth = 0;
  let lastState = 'idle', errorAt = -1, animT = 0, ringA = 0, lastDraw = -1e9, marchPhase = 0;
  const shellA = new Float32Array(3);
  let coreA = 0;

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
    // rAF timestamps can predate the push's performance.now(): clamp at 0,
    // and never let a bad level (undefined/NaN) poison the eased parameters.
    const idx = Math.max(0, Math.floor((now - model.voiceStart) / (v.step_ms || 50)));
    const lv = idx < v.levels.length ? +v.levels[idx] : 0;
    return isFinite(lv) ? Math.max(0, Math.min(1, lv)) : 0;
  }

  // Bench/diagnostics hook (scripts/orb_bench.py): per-frame draw time.
  const stats = {frameMs: 0, frames: 0, totalMs: 0, reset() { this.frames = 0; this.totalMs = 0; this.frameMs = 0; }};
  window.__hud = stats;

  function css(r, g, b, a) { return `rgba(${r | 0},${g | 0},${b | 0},${a})`; }
  const tiltC = Math.cos(0.38), tiltS = Math.sin(0.38);

  // Stroke the current path twice: a wide faint pass (fake bloom) + the line.
  // A real ctx.filter blur was benched in WKWebView and is far over budget
  // (tens of ms per frame), so bloom is faked with width x3 at low alpha.
  function bloomStroke(width, alpha) {
    ctx.lineWidth = width * 3; ctx.globalAlpha = Math.min(1, alpha * 0.14); ctx.stroke();
    ctx.lineWidth = width; ctx.globalAlpha = Math.min(1, alpha); ctx.stroke();
  }

  function frame(now) {
    rafId = 0;
    // idle: 30 fps is plenty for a slow drift; everything else runs at 60.
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
    buildGlow(false);

    micSmooth += (model.mic - micSmooth) * 0.25;
    statusLevelEl.style.width = (Math.max(0, Math.min(1, micSmooth)) * 100) + '%';
    const listening = state === 'listening' || state === 'followup';
    const vRaw = state === 'speaking' ? voiceLevel(now) : 0;
    vSmooth += (vRaw - vSmooth) * 0.45;

    if (!reduced) {
      animT += dt; ringA += 0.25 * dt;
      for (let s = 0; s < 3; s++) shellA[s] += cur[P_ROT] * SHELL_SPEED[s] * (s === 0 ? cur[P_INNER] : 1) * dt;
      coreA += cur[P_ROT] * 2.2 * cur[P_INNER] * dt;
      marchPhase += 1.5 * cur[P_MARCH] * dt;
    }

    // per-state modulation of radius / brightness / lit fraction
    let radius = cur[P_RADIUS] * (1 + cur[P_BREATHE] * Math.sin(animT * Math.PI / 2));   // ~4 s breathe
    let bright = cur[P_BRIGHT];
    let lit = cur[P_LIT], litOuter = cur[P_LIT];
    if (listening) { radius *= 1 + 0.10 * micSmooth; bright += 0.35 * micSmooth; litOuter += 0.30 * micSmooth; }
    if (state === 'speaking') { radius *= 1 + 0.10 * vSmooth; bright += 0.30 * vSmooth; lit += 0.25 * vSmooth; litOuter = lit; }
    if (state === 'confirming') bright *= 0.75 + 0.25 * Math.sin(now / 1000 * Math.PI * 2);
    // error: shatter outward for ~250 ms, return by ~800 ms
    let scatter = 0, flash = 0;
    if (state === 'error' && errorAt >= 0) {
      const te = (now - errorAt) / 1000;
      scatter = te < 0.25 ? te / 0.25 : Math.max(0, 1 - (te - 0.25) / 0.55);
      flash = Math.max(0, 1 - te / 0.3);
    }
    bright *= cfg.intensity;
    lit = Math.min(0.97, lit * litScale); litOuter = Math.min(0.97, litOuter * litScale);
    const squeeze = cur[P_SQUEEZE], ring = cur[P_RING];
    const RR = R * radius;
    const march = Math.floor(marchPhase);

    // ---- project ring points (hot loop; no allocations) ----
    for (let s = 0; s < 3; s++) {
      const cosR = Math.cos(shellA[s]), sinR = Math.sin(shellA[s]), sr = RR * SHELL_R[s];
      for (const d of rings) {
        if (d.shell !== s) continue;
        const end = d.off + d.n;
        for (let i = d.off; i < end; i++) {
          const bx = base[3 * i], by = base[3 * i + 1], bz = base[3 * i + 2];
          let x = bx * cosR + bz * sinR, z = -bx * sinR + bz * cosR, y = by * squeeze;
          if (ring > 0.001) {
            const rr = Math.sqrt(x * x + z * z) + 1e-4, want = 0.84 + 0.16 * rr;
            const g = 1 + ring * (want / rr - 1); x *= g; z *= g;
          }
          const r = 1 + scatter * scat[i] * 0.4;
          const ty = y * tiltC - z * tiltS, tz = y * tiltS + z * tiltC;
          px[i] = CX + x * sr * r; py[i] = CY + ty * sr * r; pz[i] = tz;
        }
      }
    }
    // per-segment code: 0 off, 1 back, 2 front (segment i runs point i -> i+1)
    for (const d of rings) {
      const lf = d.shell === nShells - 1 || d.shell === 2 ? litOuter : lit;
      const off = d.off, n = d.n, thr = d.thr, sh = march * d.dir;
      for (let i = 0; i < n; i++) {
        let j = (i + sh) % n; if (j < 0) j += n;
        segCode[off + i] = thr[j] < lf ? (pz[off + i] > 0 ? 2 : 1) : 0;
      }
    }
    // core rings
    {
      const c = Math.cos(coreA), s = Math.sin(coreA);
      for (let i = 0; i < NC; i++) {
        const bx = cbase[3 * i], by = cbase[3 * i + 1], bz = cbase[3 * i + 2];
        const x = bx * c + bz * s, z = -bx * s + bz * c;
        const ty = by * tiltC - z * tiltS, tz = by * tiltS + z * tiltC;
        cpx[i] = CX + x * RR; cpy[i] = CY + ty * RR; cpz[i] = tz;
      }
    }
    // specks: rotate with the outer shell, twinkle (alpha noise) on animT
    {
      const c = Math.cos(shellA[2] * 0.7), s = Math.sin(shellA[2] * 0.7);
      for (let i = 0; i < NS; i++) {
        const bx = sbase[3 * i], by = sbase[3 * i + 1], bz = sbase[3 * i + 2];
        let x = bx * c + bz * s, z = -bx * s + bz * c, y = by * squeeze;
        if (ring > 0.001) {
          const rr = Math.sqrt(x * x + z * z) + 1e-4, want = 0.84 + 0.16 * rr;
          const g = 1 + ring * (want / rr - 1); x *= g; z *= g;
        }
        const r = 1 + scatter * (0.4 + 0.6 * sw[i] / 3) * 0.6;
        const ty = y * tiltC - z * tiltS, tz = y * tiltS + z * tiltC;
        spx[i] = CX + x * RR * r; spy[i] = CY + ty * RR * r; spz[i] = tz;
        const tw = 0.5 + 0.5 * Math.sin(animT * sw[i] + sp[i]);      // 0..1 twinkle
        // bucket: kind*4 + depth*2 + twinkle-level(0/1); dim twinkle skipped
        sbucket[i] = tw < 0.25 ? 255 : skind[i] * 4 + (tz > 0 ? 2 : 0) + (tw > 0.7 ? 1 : 0);
      }
      const cc = Math.cos(-coreA * 0.6), cs = Math.sin(-coreA * 0.6);
      for (let i = 0; i < NCS; i++) {
        const bx = csbase[3 * i] + 0.02 * Math.sin(animT * csw[i] + csp[i]), by = csbase[3 * i + 1] + 0.02 * Math.cos(animT * csw[i] * 0.8 + csp[i]), bz = csbase[3 * i + 2];
        const x = bx * cc + bz * cs, z = -bx * cs + bz * cc;
        const ty = by * tiltC - z * tiltS;
        cspx[i] = CX + x * RR; cspy[i] = CY + ty * RR;
      }
    }

    // ---- draw ----
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1;
    ctx.clearRect(0, 0, SIZE, SIZE);
    ctx.globalCompositeOperation = 'lighter';
    ctx.lineCap = 'butt';

    const c1 = css(cur[P_C1], cur[P_C1 + 1], cur[P_C1 + 2], 1);
    const c2 = css(cur[P_C2], cur[P_C2 + 1], cur[P_C2 + 2], 1);
    const c3 = css(cur[P_C3], cur[P_C3 + 1], cur[P_C3 + 2], 1);
    const shellCol = [c1, c2, c3];
    const B = Math.min(1, bright * 1.8);              // idle 0.5 -> 0.9
    const A_FRONT = 0.9 * B, A_BACK = 0.3 * B;
    const W_FRONT = 1.4 * LW, W_BACK = 0.85 * LW;

    // HUD frame: thin, ticked, with two slow arcs; deliberately faint
    if (ringPath) {
      ctx.lineWidth = Math.max(0.5, 0.6 * SCALE);
      ctx.strokeStyle = c2;
      ctx.globalAlpha = 0.11 * Math.min(1.4, cfg.intensity);
      ctx.stroke(ringPath);
      ctx.globalAlpha = 0.22 * Math.min(1.4, cfg.intensity);
      ctx.lineWidth = Math.max(0.7, 1.0 * SCALE);
      ctx.beginPath(); ctx.arc(CX, CY, R * 1.16, ringA, ringA + 1.1); ctx.stroke();
      ctx.beginPath(); ctx.arc(CX, CY, R * 1.16, ringA + Math.PI, ringA + Math.PI + 0.35); ctx.stroke();
    }

    // core glow disc (behind everything); the core fades with the torus morph
    const coreB = B * (1 - 0.85 * ring);
    if (coreGlow) {
      ctx.globalAlpha = Math.min(1, coreB * 1.1);
      const gw = coreGlow.w * radius;
      ctx.drawImage(coreGlow.c, CX - gw / 2, CY - gw / 2, gw, gw);
    }

    // shells: back bucket then front bucket; each = one path per shell,
    // stroked twice (bloom + line). Ticks / chips ride along per bucket.
    const shellFirst = nShells === 1 ? 2 : 0;
    for (let bucket = 1; bucket <= 2; bucket++) {
      const alpha0 = bucket === 2 ? A_FRONT : A_BACK, width = bucket === 2 ? W_FRONT : W_BACK;
      for (let s = shellFirst; s < 3; s++) {
        const alpha = s === 0 ? alpha0 * 0.7 : alpha0;    // inner shell: its rings stack additively
        ctx.strokeStyle = shellCol[s]; ctx.fillStyle = shellCol[s];
        ctx.beginPath();
        let any = false;
        for (const d of rings) {
          if (d.shell !== s) continue;
          const off = d.off, n = d.n;
          let cont = false;
          for (let i = 0; i < n; i++) {
            if (segCode[off + i] !== bucket) { cont = false; continue; }
            const a = off + i, b = off + ((i + 1) % n);
            if (!cont) { ctx.moveTo(px[a], py[a]); cont = true; }
            ctx.lineTo(px[b], py[b]); any = true;
          }
        }
        if (any) bloomStroke(width, alpha);
        // traces: tick marks (radial) + chip blocks (tangent boxes) on lit segments
        ctx.beginPath();
        let anyTick = false, anyChip = false;
        for (const d of rings) {
          if (d.shell !== s) continue;
          for (let f = 0; f < d.feat.length; f++) {
            const i = d.feat[f]; if (segCode[d.off + i] !== bucket) continue;
            const a = d.off + i, b = d.off + ((i + 1) % d.n);
            const x = px[a], y = py[a];
            if (d.ftype[f] === 0) {
              // tick: short radial line outward from the orb centre
              let dx = x - CX, dy = y - CY; const l = Math.sqrt(dx * dx + dy * dy) || 1; dx /= l; dy /= l;
              const len = d.fsize[f] * SCALE;
              ctx.moveTo(x, y); ctx.lineTo(x + dx * len, y + dy * len); anyTick = true;
            }
          }
        }
        if (anyTick) { ctx.lineWidth = Math.max(0.6, 0.9 * LW); ctx.globalAlpha = alpha * 0.85; ctx.stroke(); }
        ctx.beginPath();
        for (const d of rings) {
          if (d.shell !== s) continue;
          for (let f = 0; f < d.feat.length; f++) {
            const i = d.feat[f]; if (d.ftype[f] !== 1 || segCode[d.off + i] !== bucket) continue;
            const a = d.off + i, b = d.off + ((i + 1) % d.n);
            let tx = px[b] - px[a], ty = py[b] - py[a]; const l = Math.sqrt(tx * tx + ty * ty) || 1; tx /= l; ty /= l;
            const w = d.fsize[f] * SCALE, h = w * 0.5, nx = -ty * h / 2, ny = tx * h / 2;
            const x = px[a], y = py[a];
            ctx.moveTo(x + nx, y + ny); ctx.lineTo(x + tx * w + nx, y + ty * w + ny);
            ctx.lineTo(x + tx * w - nx, y + ty * w - ny); ctx.lineTo(x - nx, y - ny); ctx.closePath();
            anyChip = true;
          }
        }
        if (anyChip) { ctx.globalAlpha = alpha * 0.6; ctx.fill(); }
      }
      // radial connectors from a lit segment on shell s to the same direction on shell s+1
      if (conns.length) {
        ctx.beginPath();
        let anyC = false;
        for (const cn of conns) {
          const d = cn.ring, i = d.off + cn.seg;
          if (segCode[i] !== bucket) continue;
          const ratio = SHELL_R[d.shell + 1] / SHELL_R[d.shell];
          ctx.moveTo(px[i], py[i]); ctx.lineTo(CX + (px[i] - CX) * ratio, CY + (py[i] - CY) * ratio); anyC = true;
        }
        if (anyC) { ctx.strokeStyle = c2; ctx.lineWidth = Math.max(0.5, 0.7 * LW); ctx.globalAlpha = alpha0 * 0.55; ctx.stroke(); }
      }
      // specks in this depth bucket: 2 twinkle levels x 2 kinds, one fill each
      for (let kind = 0; kind < 2; kind++) {
        ctx.fillStyle = kind ? css(COOL[0], COOL[1], COOL[2], 1) : c1;
        for (let lv = 0; lv < 2; lv++) {
          const want = kind * 4 + (bucket === 2 ? 2 : 0) + lv;
          ctx.beginPath();
          let anyS = false;
          for (let i = 0; i < NS; i++) {
            if (sbucket[i] !== want) continue;
            const sz = ssz[i] * Math.max(0.6, SCALE) * (bucket === 2 ? 1 : 0.8);
            ctx.rect(spx[i] - sz / 2, spy[i] - sz / 2, sz, sz); anyS = true;
          }
          if (anyS) { ctx.globalAlpha = Math.min(1, (bucket === 2 ? 0.95 : 0.4) * B * (lv ? 1 : 0.55) * (state === 'speaking' ? 0.7 + 0.6 * vSmooth : 1)); ctx.fill(); }
        }
      }
      // core (between back and front so front rings pass over it)
      if (bucket === 1) {
        ctx.strokeStyle = c1;
        ctx.beginPath();
        for (const cr of coreRings) {
          for (let i = 0; i < cr.n; i++) {
            const a = cr.off + i, b = cr.off + ((i + 1) % cr.n);
            if (i === 0) ctx.moveTo(cpx[a], cpy[a]);
            ctx.lineTo(cpx[b], cpy[b]);
          }
        }
        bloomStroke(2.2 * LW, Math.min(1, coreB * 0.85));
        ctx.fillStyle = c1;
        ctx.beginPath();
        const csz = Math.max(0.7, 1.0 * SCALE);
        for (let i = 0; i < NCS; i++) ctx.rect(cspx[i] - csz / 2, cspy[i] - csz / 2, csz, csz);
        ctx.globalAlpha = Math.min(1, 0.75 * coreB); ctx.fill();
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
      ctx.strokeStyle = css(COOL[0], COOL[1], COOL[2], 1);
      ctx.beginPath(); ctx.arc(CX, CY, Math.min(SIZE / 2 - 1, RR * (1.02 + 0.22 * f)), 0, Math.PI * 2); ctx.stroke();
    }

    // speaking: radial spokes whose lengths trace the recent level history
    if (state === 'speaking') {
      if (now - lastSpoke > 45) { spokeHist[spokeHead] = vSmooth; spokeHead = (spokeHead + 1) % N_SPOKES; lastSpoke = now; }
      if (vSmooth > 0.05) {
        // one path + one stroke for all spokes (a stroke per spoke is ~2 ms in WebKit)
        ctx.strokeStyle = c1;
        ctx.lineWidth = Math.max(0.5, 0.6 * SCALE);
        const r0 = RR * 1.05;
        ctx.globalAlpha = Math.min(1, (0.2 + 0.3 * vSmooth) * cfg.intensity);
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
      }
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
      if (model.confirmStart !== null) {
        const frac = Math.max(0, 1 - (now - model.confirmStart) / model.confirmTimeoutMs);
        ctx.beginPath(); ctx.arc(CX, CY, R * 0.88, -Math.PI / 2, -Math.PI / 2 + frac * Math.PI * 2);
        ctx.lineWidth = 3 * SCALE; ctx.strokeStyle = c1; ctx.stroke();
      }
      ctx.font = 'bold ' + Math.round(26 * SCALE) + 'px system-ui';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.lineJoin = 'round'; ctx.lineWidth = 3 * SCALE; ctx.strokeStyle = 'rgba(0,0,0,0.55)';
      ctx.strokeText('?', CX, CY + 1);
      ctx.fillStyle = '#fff'; ctx.fillText('?', CX, CY + 1);
    }
    // warming spinner
    if (state === 'warming') {
      const a0 = animT * 2.4;
      ctx.beginPath(); ctx.arc(CX, CY, R * 0.88, a0, a0 + Math.PI * 0.6);
      ctx.lineWidth = 3 * SCALE; ctx.strokeStyle = c1; ctx.stroke();
    }

    const ms = performance.now() - tStart;
    stats.frameMs = ms; stats.frames++; stats.totalMs += ms;
    if (rafActive && rafId === 0) rafId = requestAnimationFrame(frame);
  }

  applyMode();
  if (rafActive && rafId === 0) rafId = requestAnimationFrame(frame);
})();
